"""Authentication API endpoints."""

from flask_login import current_user, login_required, logout_user

from api.v1 import v1_bp
from api.response import success_response


@v1_bp.route('/auth/me', methods=['GET'])
@login_required
def api_auth_me():
    """Return current authenticated user info."""
    return success_response(data=current_user.to_dict())


@v1_bp.route('/auth/logout', methods=['POST'])
@login_required
def api_auth_logout():
    """Log out the current user via API."""
    logout_user()
    return success_response(message='로그아웃되었습니다.')
