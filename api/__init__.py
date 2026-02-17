"""API package initialization with CORS, request tracing, and Blueprint registration."""

import os
import uuid

from flask import g, request
from flask_cors import CORS

# ------------------------------------------------------------------
# Rate Limiter – standard init_app pattern
# ------------------------------------------------------------------
# flask-limiter is optional; provide a no-op helper when unavailable.
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address

    limiter = Limiter(key_func=get_remote_address)
    _has_limiter = True
except ImportError:
    limiter = None
    _has_limiter = False


def rate_limit(limit_string):
    """Rate-limit decorator; no-op when flask-limiter is unavailable or disabled.

    Usage in route modules::

        from api import rate_limit

        @v1_bp.route('/search', methods=['POST'])
        @rate_limit("30 per minute")
        def api_search():
            ...
    """
    if limiter is not None:
        return limiter.limit(limit_string)
    return lambda f: f


def init_api(app, csrf=None):
    """Register API blueprints, CORS, request-ID tracing, and rate limiting."""

    # ------------------------------------------------------------------
    # CORS – configurable via CORS_ORIGINS env var (comma-separated)
    # ------------------------------------------------------------------
    cors_origins = os.environ.get('CORS_ORIGINS', '').strip()
    origins = [o.strip() for o in cors_origins.split(',') if o.strip()] if cors_origins else '*'

    CORS(app, resources={
        r"^/api/.+": {
            "origins": origins,
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization", "X-Request-ID"]
        }
    })

    # ------------------------------------------------------------------
    # Request-ID tracing
    # ------------------------------------------------------------------
    @app.before_request
    def _set_request_id():
        g.request_id = request.headers.get('X-Request-ID') or uuid.uuid4().hex[:12]

    @app.after_request
    def _add_request_id_header(response):
        request_id = getattr(g, 'request_id', None)
        if request_id:
            response.headers['X-Request-ID'] = request_id
        return response

    # ------------------------------------------------------------------
    # Blueprint registration
    # ------------------------------------------------------------------
    from api.v1 import v1_bp
    app.register_blueprint(v1_bp)

    from api.compat import register_compat_routes
    register_compat_routes(app, csrf=csrf)

    # ------------------------------------------------------------------
    # Rate Limiting (opt-in via RATE_LIMIT_ENABLED env var)
    #
    # NOTE: memory:// storage is per-worker — rate limits are NOT shared
    # across Gunicorn workers. For production with multiple workers, use
    # a shared backend such as Redis:
    #   RATE_LIMIT_STORAGE_URI=redis://localhost:6379
    # ------------------------------------------------------------------
    if _has_limiter:
        enabled = os.environ.get(
            'RATE_LIMIT_ENABLED', ''
        ).lower() in ('true', '1', 'yes')
        app.config.setdefault('RATELIMIT_ENABLED', enabled)
        app.config.setdefault(
            'RATELIMIT_STORAGE_URI',
            os.environ.get('RATE_LIMIT_STORAGE_URI', 'memory://'),
        )
        app.config.setdefault('RATELIMIT_DEFAULT', '60 per minute')
        limiter.init_app(app)
    elif os.environ.get('RATE_LIMIT_ENABLED', '').lower() in ('true', '1', 'yes'):
        import logging
        logging.warning("[Rate Limit] flask-limiter not installed, skipping rate limiting")
