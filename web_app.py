#!/usr/bin/env python3
"""
Pinecone Agent Web Interface
Flask-based web UI for Pinecone vector database operations with RAG support.
"""

# SSL Certificate configuration (must be done before importing httpx/urllib3)
import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

import logging
import unicodedata
from datetime import datetime, timezone
from urllib.parse import urlparse, urljoin

from flask import Flask, flash, redirect, render_template, request, send_from_directory, url_for
from flask_login import LoginManager, current_user, login_required, login_user, logout_user
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', os.urandom(24).hex())

# ========================================
# Database (SQLite via SQLAlchemy)
# ========================================
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'instance', 'app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

from models import db, User, SocialAccount
db.init_app(app)

# ========================================
# Flask-Login
# ========================================
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = '로그인이 필요합니다.'


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# ========================================
# OAuth (Google + Kakao)
# ========================================
from services.oauth import oauth, init_oauth
init_oauth(app)

# Import shared configuration
from services.domain_config import DOCUMENTS_PATH, DOMAIN_CONFIG

# Register API blueprints and CORS
from api import init_api
init_api(app)

# Create database tables
with app.app_context():
    os.makedirs(os.path.join(basedir, 'instance'), exist_ok=True)
    db.create_all()


# ========================================
# Auth Helper
# ========================================

def _is_safe_redirect(target):
    """Validate that redirect target is same-origin (prevents open redirect)."""
    if not target:
        return False
    ref_url = urlparse(request.host_url)
    test_url = urlparse(urljoin(request.host_url, target))
    return test_url.scheme in ('http', 'https') and ref_url.netloc == test_url.netloc


def _handle_oauth_callback(provider, user_info):
    """Process OAuth callback: find or create user, update social account, login.

    Args:
        provider: 'google' or 'kakao'
        user_info: dict with keys 'id', 'email', 'name', 'picture'
    """
    provider_user_id = str(user_info['id'])
    # 이메일이 없는 경우 (카카오 비즈앱 미전환 등) provider ID로 대체
    email = user_info.get('email') or f'{provider}_{provider_user_id}@oauth.local'

    # Check if social account already exists
    social = SocialAccount.query.filter_by(
        provider=provider, provider_user_id=provider_user_id,
    ).first()

    if social:
        user = social.user
        social.access_token = user_info.get('access_token')
        social.refresh_token = user_info.get('refresh_token')
        social.provider_data = user_info.get('raw')
    else:
        # Find existing user by email or create new
        user = User.query.filter_by(email=email).first()
        if not user:
            user = User(
                email=email,
                name=user_info.get('name') or email.split('@')[0],
                profile_image=user_info.get('picture'),
            )
            db.session.add(user)
            db.session.flush()

        social = SocialAccount(
            user_id=user.id,
            provider=provider,
            provider_user_id=provider_user_id,
            access_token=user_info.get('access_token'),
            refresh_token=user_info.get('refresh_token'),
            provider_data=user_info.get('raw'),
        )
        db.session.add(social)

    # Update user profile if newer info available
    if user_info.get('name'):
        user.name = user_info['name']
    if user_info.get('picture'):
        user.profile_image = user_info['picture']
    user.last_login = datetime.now(timezone.utc)

    db.session.commit()
    login_user(user)
    next_page = request.args.get('next', '')
    if not _is_safe_redirect(next_page):
        next_page = url_for('home')
    return redirect(next_page)


# ========================================
# Login / Logout Routes
# ========================================

@app.route('/login')
def login():
    """Login page with social login buttons."""
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    return render_template('login.html')


@app.route('/login/google')
def login_google():
    """Start Google OAuth flow."""
    redirect_uri = url_for('callback_google', _external=True)
    return oauth.google.authorize_redirect(redirect_uri)


@app.route('/callback/google')
def callback_google():
    """Handle Google OAuth callback."""
    try:
        token = oauth.google.authorize_access_token()
        user_info_resp = oauth.google.get(
            'https://www.googleapis.com/oauth2/v3/userinfo',
        )
        profile = user_info_resp.json()

        return _handle_oauth_callback('google', {
            'id': profile.get('sub'),
            'email': profile.get('email'),
            'name': profile.get('name'),
            'picture': profile.get('picture'),
            'access_token': token.get('access_token'),
            'refresh_token': token.get('refresh_token'),
            'raw': profile,
        })
    except Exception:
        logging.exception('Google OAuth callback failed')
        flash('Google 로그인에 실패했습니다. 다시 시도해주세요.')
        return redirect(url_for('login'))


@app.route('/login/kakao')
def login_kakao():
    """Start Kakao OAuth flow."""
    redirect_uri = url_for('callback_kakao', _external=True)
    return oauth.kakao.authorize_redirect(redirect_uri)


@app.route('/callback/kakao')
def callback_kakao():
    """Handle Kakao OAuth callback."""
    try:
        token = oauth.kakao.authorize_access_token()
        # Fetch user profile from Kakao API
        resp = oauth.kakao.get('https://kapi.kakao.com/v2/user/me')
        profile = resp.json()

        kakao_account = profile.get('kakao_account', {})
        kakao_profile = kakao_account.get('profile', {})

        return _handle_oauth_callback('kakao', {
            'id': profile.get('id'),
            'email': kakao_account.get('email'),
            'name': kakao_profile.get('nickname'),
            'picture': kakao_profile.get('profile_image_url'),
            'access_token': token.get('access_token'),
            'refresh_token': token.get('refresh_token'),
            'raw': profile,
        })
    except Exception:
        logging.exception('Kakao OAuth callback failed')
        flash('Kakao 로그인에 실패했습니다. 다시 시도해주세요.')
        return redirect(url_for('login'))


@app.route('/logout', methods=['POST'])
def logout():
    """Log out the current user (POST only to prevent CSRF via GET)."""
    logout_user()
    return redirect(url_for('home'))


# ========================================
# HTML Page Routes (protected)
# ========================================

@app.route('/')
def home():
    """Home page with domain selection (public)."""
    return render_template('home.html')


@app.route('/mypage')
@login_required
def mypage():
    """User profile page."""
    social_accounts = SocialAccount.query.filter_by(user_id=current_user.id).all()
    return render_template('mypage.html', social_accounts=social_accounts)


@app.route('/semiconductor')
def semiconductor():
    """Semiconductor domain page."""
    config = DOMAIN_CONFIG['semiconductor']
    return render_template('domain.html', domain='semiconductor', config=config)


@app.route('/laborlaw')
def laborlaw():
    """Labor law domain page."""
    config = DOMAIN_CONFIG['laborlaw']
    return render_template('domain.html', domain='laborlaw', config=config)


@app.route('/field-training')
def field_training():
    """Field training safety domain page."""
    config = DOMAIN_CONFIG['field-training']
    return render_template('domain.html', domain='field-training', config=config)


@app.route('/msds')
def msds():
    """MSDS chemical information page."""
    config = DOMAIN_CONFIG['msds']
    return render_template('msds.html', domain='msds', config=config)


@app.route('/documents/<path:filepath>')
def serve_document(filepath):
    """Serve document images and files."""
    # Normalize unicode for macOS (NFD filesystem) compatibility
    filepath = unicodedata.normalize('NFC', filepath)
    full_path = DOCUMENTS_PATH / filepath
    if not full_path.exists():
        filepath_nfd = unicodedata.normalize('NFD', filepath)
        full_path_nfd = DOCUMENTS_PATH / filepath_nfd
        if full_path_nfd.exists():
            filepath = filepath_nfd
    return send_from_directory(DOCUMENTS_PATH, filepath)


if __name__ == '__main__':
    # Check environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        exit(1)
    if not os.getenv("PINECONE_API_KEY"):
        print("Error: PINECONE_API_KEY not set")
        exit(1)

    print("\U0001f680 Pinecone Agent Web Interface")
    print("=" * 40)
    print(f"Index: {os.getenv('PINECONE_INDEX_NAME', 'document-index')}")
    print("=" * 40)
    print("\n\U0001f310 http://localhost:5001 에서 접속하세요\n")

    app.run(debug=True, host='0.0.0.0', port=5001)
