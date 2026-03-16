"""User profile API — major selection/query."""

from flask import request
from flask_login import current_user, login_required

from api.v1 import v1_bp
from api.response import success_response, error_response
from api import rate_limit
from models import db
from services.major_config import MAJOR_CONFIG, DEFAULT_MAJOR, get_major_config, get_all_major_keys


@v1_bp.route('/user/major', methods=['POST'])
@login_required
@rate_limit("30 per minute")
def set_user_major():
    """Set the current user's major."""
    data = request.get_json(silent=True)
    if not data:
        return error_response('요청 데이터가 없습니다.', 400)

    major = data.get('major', '').strip()
    if not major or major not in MAJOR_CONFIG:
        return error_response(f"Invalid major key: '{major}'", 400)

    current_user.major = major
    db.session.commit()

    config = get_major_config(major)
    return success_response(data={
        'major': major,
        'major_name': config['name'],
    })


@v1_bp.route('/user/major', methods=['GET'])
@login_required
def get_user_major():
    """Get the current user's major."""
    major = current_user.major or DEFAULT_MAJOR
    config = get_major_config(major)
    return success_response(data={
        'major': major,
        'major_name': config['name'],
        'major_config': {
            'icon': config['icon'],
            'color': config['color'],
            'description': config['description'],
        },
    })


@v1_bp.route('/majors', methods=['GET'])
def list_majors():
    """List all available majors."""
    majors = []
    for key in get_all_major_keys():
        config = MAJOR_CONFIG[key]
        majors.append({
            'key': key,
            'name': config['name'],
            'short_name': config['short_name'],
            'icon': config['icon'],
            'color': config['color'],
            'description': config['description'],
        })
    return success_response(data={
        'majors': majors,
        'default': DEFAULT_MAJOR,
    })
