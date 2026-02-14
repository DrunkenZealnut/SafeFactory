"""API v1 Blueprint definition."""

from flask import Blueprint

v1_bp = Blueprint('v1', __name__, url_prefix='/api/v1')

# Import route modules to register their endpoints on v1_bp
from api.v1 import index_ops  # noqa: E402, F401
from api.v1 import search     # noqa: E402, F401
from api.v1 import msds       # noqa: E402, F401
