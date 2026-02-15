"""API package initialization with CORS, request tracing, and Blueprint registration."""

import os
import uuid

from flask import g, request
from flask_cors import CORS


def init_api(app):
    """Register API blueprints, CORS, request-ID tracing, and rate limiting."""

    # ------------------------------------------------------------------
    # CORS – configurable via CORS_ORIGINS env var (comma-separated)
    # ------------------------------------------------------------------
    cors_origins = os.environ.get('CORS_ORIGINS', '').strip()
    origins = [o.strip() for o in cors_origins.split(',') if o.strip()] if cors_origins else '*'

    CORS(app, resources={
        r"/api/.*": {
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
    # Blueprint registration (must come before rate limiting)
    # ------------------------------------------------------------------
    from api.v1 import v1_bp
    app.register_blueprint(v1_bp)

    from api.compat import register_compat_routes
    register_compat_routes(app)

    # ------------------------------------------------------------------
    # Rate Limiting (opt-in via RATE_LIMIT_ENABLED env var)
    # ------------------------------------------------------------------
    if os.environ.get('RATE_LIMIT_ENABLED', '').lower() in ('true', '1', 'yes'):
        try:
            from flask_limiter import Limiter
            from flask_limiter.util import get_remote_address

            limiter = Limiter(
                get_remote_address,
                app=app,
                default_limits=["60 per minute"],
                storage_uri=os.environ.get('RATE_LIMIT_STORAGE_URI', 'memory://'),
            )
            # Stricter limits for LLM-backed endpoints
            for endpoint in ('v1.api_ask', 'v1.api_ask_stream'):
                fn = app.view_functions.get(endpoint)
                if fn:
                    app.view_functions[endpoint] = limiter.limit("20 per minute")(fn)
            fn = app.view_functions.get('v1.api_search')
            if fn:
                app.view_functions['v1.api_search'] = limiter.limit("30 per minute")(fn)
        except ImportError:
            import logging
            logging.warning("[Rate Limit] flask-limiter not installed, skipping rate limiting")
