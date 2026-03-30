#!/usr/bin/env python3
"""
SafeFactory Web Interface
Flask-based web UI for Pinecone vector database operations with RAG support.
"""

# SSL Certificate configuration (must be done before importing httpx/urllib3)
import atexit
import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

import logging
import unicodedata
from datetime import datetime, timezone
from urllib.parse import urlparse, urljoin

from flask import Flask, abort, flash, redirect, render_template, request, send_from_directory, url_for
from flask_login import LoginManager, current_user, login_required, login_user, logout_user
from sqlalchemy.exc import IntegrityError
from flask_wtf.csrf import CSRFProtect
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# ========================================
# Reverse Proxy Support (Nginx/Apache)
# ========================================
# When behind a reverse proxy, trust X-Forwarded-* headers so that
# url_for(_external=True) generates the correct public URL (scheme + host).
from werkzeug.middleware.proxy_fix import ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

secret_key = os.getenv('SECRET_KEY')
if not secret_key:
    if os.getenv('FLASK_ENV') in ('production', 'prod'):
        raise ValueError('SECRET_KEY must be set in production')
    import secrets
    secret_key = secrets.token_hex(32)
    app.logger.warning('SECRET_KEY not set — using random key (sessions will reset on restart)')
app.config['SECRET_KEY'] = secret_key
app.config['PREFERRED_URL_SCHEME'] = 'https'

# ========================================
# CSRF Protection
# ========================================
csrf = CSRFProtect(app)

# ========================================
# Database (SQLite via SQLAlchemy)
# ========================================
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'instance', 'app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB request limit

from models import db, User, SocialAccount, seed_categories, seed_system_settings, ensure_provider_settings
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


@login_manager.unauthorized_handler
def unauthorized_api():
    """Return JSON 401 for API requests instead of redirecting to login page."""
    if request.path.startswith('/api/'):
        from api.response import error_response
        return error_response('로그인이 필요합니다.', 401)
    flash(login_manager.login_message)
    return redirect(url_for(login_manager.login_view, next=request.url))


# ========================================
# OAuth (Google + Kakao)
# ========================================
from services.oauth import oauth, init_oauth
init_oauth(app)

# Import shared configuration
from services.domain_config import DOCUMENTS_PATH, DOMAIN_CONFIG
from services.major_config import MAJOR_CONFIG, DEFAULT_MAJOR, get_major_config, get_all_major_keys

# Register API blueprints and CORS (pass csrf so compat routes are exempted)
app.jinja_env.globals['now'] = lambda: datetime.now(timezone.utc)

from api import init_api
init_api(app, csrf=csrf)

# Exempt API blueprint from CSRF (JSON APIs, not form submissions)
from api.v1 import v1_bp
csrf.exempt(v1_bp)

# Create database tables
with app.app_context():
    os.makedirs(os.path.join(basedir, 'instance'), exist_ok=True)
    db.create_all()
    seed_categories()
    seed_system_settings()
    ensure_provider_settings()

    # Migration: add 'major' column to users table if not present
    from sqlalchemy import inspect as sa_inspect
    inspector = sa_inspect(db.engine)
    user_columns = [c['name'] for c in inspector.get_columns('users')]
    if 'major' not in user_columns:
        with db.engine.connect() as conn:
            conn.execute(db.text(
                "ALTER TABLE users ADD COLUMN major VARCHAR(50) DEFAULT 'semiconductor'"
            ))
            conn.commit()
        logging.info("[Migration] Added 'major' column to users table")

# Register explicit shutdown for singleton httpx clients
from services.singletons import shutdown_all
atexit.register(shutdown_all)


# ========================================
# Auth Helper
# ========================================

def _is_safe_redirect(target):
    """Validate that redirect target is same-origin (prevents open redirect)."""
    if not target:
        return False
    # Reject protocol-relative URLs and backslash tricks
    if target.startswith('//') or target.startswith('\\'):
        return False
    ref_url = urlparse(request.host_url)
    test_url = urlparse(urljoin(request.host_url, target))
    return test_url.scheme in ('http', 'https') and ref_url.netloc == test_url.netloc


def _sanitize_image_url(url):
    """Allow only http/https image URLs to prevent javascript: XSS."""
    if not url:
        return None
    parsed = urlparse(url)
    if parsed.scheme in ('http', 'https'):
        return url
    return None


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

    real_email = user_info.get('email')  # None if provider didn't supply one

    if social:
        user = social.user
        social.access_token = user_info.get('access_token')
        social.refresh_token = user_info.get('refresh_token')
        social.provider_data = user_info.get('raw')

        # Upgrade placeholder email when a real one becomes available
        if real_email and user.email.endswith('@oauth.local'):
            try:
                existing_by_email = User.query.filter_by(email=real_email).first()
                if existing_by_email and existing_by_email.id != user.id:
                    # Merge: move social accounts to the real-email user, delete placeholder
                    for sa in user.social_accounts:
                        sa.user_id = existing_by_email.id
                    db.session.delete(user)
                    db.session.flush()
                    user = existing_by_email
                else:
                    user.email = real_email
                    db.session.flush()
            except IntegrityError:
                db.session.rollback()
                # Re-query both after rollback to get fresh state
                social = SocialAccount.query.filter_by(
                    provider=provider, provider_user_id=provider_user_id,
                ).first()
                user = User.query.filter_by(email=real_email).first() \
                    or (social.user if social else None)
                if not user:
                    logging.error('OAuth merge failed: no user found after rollback')
                    flash('로그인 처리 중 오류가 발생했습니다. 다시 시도해주세요.')
                    return redirect(url_for('login'))
    else:
        # Find existing user by email or create new
        user = User.query.filter_by(email=email).first()
        if not user:
            user = User(
                email=email,
                name=user_info.get('name') or email.split('@')[0],
                profile_image=_sanitize_image_url(user_info.get('picture')),
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
    sanitized_picture = _sanitize_image_url(user_info.get('picture'))
    if sanitized_picture:
        user.profile_image = sanitized_picture
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
    if not hasattr(oauth, 'google'):
        flash('Google 로그인이 설정되지 않았습니다.')
        return redirect(url_for('login'))
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
    if not hasattr(oauth, 'kakao'):
        flash('Kakao 로그인이 설정되지 않았습니다.')
        return redirect(url_for('login'))
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
    """Home page — major-aware dashboard."""
    if current_user.is_authenticated and current_user.major:
        major_key = current_user.major
    else:
        major_key = DEFAULT_MAJOR
    major_cfg = get_major_config(major_key)
    return render_template('home.html', domains=DOMAIN_CONFIG,
                           major_key=major_key, major_config=major_cfg)


@app.route('/mypage')
@login_required
def mypage():
    """User profile page with major selection."""
    social_accounts = SocialAccount.query.filter_by(user_id=current_user.id).all()
    return render_template('mypage.html', social_accounts=social_accounts,
                           major_config=MAJOR_CONFIG, current_major=current_user.major or DEFAULT_MAJOR)


@app.route('/history')
@login_required
def history():
    """Search history page."""
    return render_template('history.html')


@app.route('/my-documents')
@login_required
def my_documents():
    """User's bookmarked documents page."""
    return render_template('my_documents.html')


@app.route('/learn')
def learn():
    """Major-based unified learning environment."""
    if current_user.is_authenticated and current_user.major:
        major_key = current_user.major
    else:
        major_key = DEFAULT_MAJOR

    config = get_major_config(major_key)
    domain_style_config = {
        'title': config['name'],
        'namespace': config['namespaces']['primary'],
        'major': major_key,
        'icon': config['icon'],
        'color': config['color'],
        'gradient': config['gradient'],
        'description': config['description'],
        'sample_questions': config['sample_questions'],
    }
    return render_template('domain.html', domain='learn', config=domain_style_config, major=major_key)


# Legacy domain routes → redirect to /learn
@app.route('/semiconductor')
@app.route('/field-training')
@app.route('/safeguide')
@app.route('/search')
def legacy_domain_redirect():
    """Redirect old domain routes to unified /learn."""
    return redirect(url_for('learn'))


@app.route('/msds')
def msds():
    """MSDS chemical information page (cross-major common tool)."""
    config = DOMAIN_CONFIG['msds']
    return render_template('msds.html', domain='msds', config=config)


@app.route('/questions')
def questions():
    """Shared questions browsing page."""
    return render_template('questions.html')


@app.route('/wordcloud')
def wordcloud():
    """Shared questions word cloud page."""
    return render_template('wordcloud.html')


@app.route('/community')
def community():
    """Community bulletin board page."""
    return render_template('community.html')


@app.route('/news')
def news():
    """Labor safety news board page."""
    return render_template('news.html')


@app.route('/admin')
@login_required
def admin_page():
    """Admin dashboard (admin role required, checked via JS + API)."""
    if current_user.role != 'admin':
        flash('접근 권한이 없습니다.')
        return redirect(url_for('home'))
    return render_template('admin.html')


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
            full_path = full_path_nfd

    # Prevent symlink traversal outside DOCUMENTS_PATH
    try:
        resolved = full_path.resolve()
        docs_resolved = DOCUMENTS_PATH.resolve()
        if not str(resolved).startswith(str(docs_resolved) + os.sep) and resolved != docs_resolved:
            abort(403)
    except (OSError, ValueError):
        abort(403)

    return send_from_directory(DOCUMENTS_PATH, filepath)


if __name__ == '__main__':
    # Check environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        exit(1)
    if not os.getenv("PINECONE_API_KEY"):
        print("Error: PINECONE_API_KEY not set")
        exit(1)

    print("\U0001f680 SafeFactory Web Interface")
    print("=" * 40)
    print(f"Index: {os.getenv('PINECONE_INDEX_NAME', 'document-index')}")
    print("=" * 40)
    print("\n\U0001f310 http://localhost:5001 에서 접속하세요\n")

    app.run(debug=True, host='127.0.0.1', port=int(os.getenv('PORT', 5001)))
