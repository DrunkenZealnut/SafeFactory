"""API package initialization with CORS and Blueprint registration."""

from flask_cors import CORS


def init_api(app):
    """Register API blueprints and configure CORS."""
    CORS(app, resources={
        r"/api/.*": {
            "origins": "*",
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })

    from api.v1 import v1_bp
    app.register_blueprint(v1_bp)

    from api.compat import register_compat_routes
    register_compat_routes(app)
