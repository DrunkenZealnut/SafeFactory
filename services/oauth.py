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
    if google_id and google_secret:
        oauth.register(
            name='google',
            client_id=google_id,
            client_secret=google_secret,
            server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
            client_kwargs={'scope': 'openid email profile'},
        )
    else:
        app.logger.warning('Google OAuth credentials not configured — Google login will be unavailable')

    # ------------------------------------------------------------------
    # Kakao OAuth 2.0
    # ------------------------------------------------------------------
    kakao_id = os.getenv('KAKAO_CLIENT_ID')
    kakao_secret = os.getenv('KAKAO_CLIENT_SECRET')
    if kakao_id and kakao_secret:
        oauth.register(
            name='kakao',
            client_id=kakao_id,
            client_secret=kakao_secret,
            authorize_url='https://kauth.kakao.com/oauth/authorize',
            access_token_url='https://kauth.kakao.com/oauth/token',
            token_endpoint_auth_method='client_secret_post',
            client_kwargs={'scope': 'profile_nickname profile_image'},
        )
    elif kakao_id:
        app.logger.warning('KAKAO_CLIENT_SECRET not set — Kakao login requires client_secret since 2024')
    else:
        app.logger.warning('Kakao OAuth credentials not configured — Kakao login will be unavailable')
