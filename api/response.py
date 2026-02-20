"""Standard API response helpers."""

from datetime import datetime, timezone

from flask import g, jsonify

API_VERSION = 'v1'


def escape_like(value):
    r"""Escape SQL LIKE wildcard characters (%, _) in user input."""
    return value.replace('\\', r'\\').replace('%', r'\%').replace('_', r'\_')


def _build_meta():
    """Build response meta dict with timestamp, version, and request_id."""
    return {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'version': API_VERSION,
        'request_id': getattr(g, 'request_id', None),
    }


def success_response(data=None, message=None):
    """Return a standard success JSON response."""
    resp = {'success': True}
    if data is not None:
        resp['data'] = data
    if message:
        resp['message'] = message
    resp['meta'] = _build_meta()
    return jsonify(resp)


def error_response(error, status_code=400, details=None):
    """Return a standard error JSON response."""
    resp = {'success': False, 'error': error}
    if details:
        resp['details'] = details
    resp['meta'] = _build_meta()
    return jsonify(resp), status_code
