"""OAuth 2.0 client configuration for Google and Kakao."""

import os

from authlib.integrations.flask_client import OAuth

oauth = OAuth()


def init_oauth(app):
    """Register Google and Kakao OAuth clients on the Flask app."""
    oauth.init_app(app)

    # ------------------------------------------------------------------
    # Google OAuth 2.0
    # ------------------------------------------------------------------
    google_id = os.getenv('GOOGLE_CLIENT_ID')
    google_secret = os.getenv('GOOGLE_CLIENT_SECRET')
    if not google_id or not google_secret:
        app.logger.warning('Google OAuth credentials not configured — Google login will be unavailable')

    oauth.register(
        name='google',
        client_id=google_id,
        client_secret=google_secret,
        server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
        client_kwargs={'scope': 'openid email profile'},
    )

    # ------------------------------------------------------------------
    # Kakao OAuth 2.0
    # ------------------------------------------------------------------
    kakao_id = os.getenv('KAKAO_CLIENT_ID')
    if not kakao_id:
        app.logger.warning('Kakao OAuth credentials not configured — Kakao login will be unavailable')

    oauth.register(
        name='kakao',
        client_id=kakao_id,
        client_secret=os.getenv('KAKAO_CLIENT_SECRET', ''),
        authorize_url='https://kauth.kakao.com/oauth/authorize',
        access_token_url='https://kauth.kakao.com/oauth/token',
        # 카카오는 token 요청 시 client_id를 POST body로 전송해야 함
        token_endpoint_auth_method='client_secret_post',
        # account_email은 비즈앱 전환 후에만 사용 가능
        client_kwargs={'scope': 'profile_nickname profile_image'},
    )
