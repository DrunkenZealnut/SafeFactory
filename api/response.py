"""Standard API response helpers."""

from flask import jsonify


def success_response(data=None, message=None):
    """Return a standard success JSON response."""
    resp = {'success': True}
    if data is not None:
        resp['data'] = data
    if message:
        resp['message'] = message
    return jsonify(resp)


def error_response(error, status_code=400, details=None):
    """Return a standard error JSON response."""
    resp = {'success': False, 'error': error}
    if details:
        resp['details'] = details
    return jsonify(resp), status_code
