"""API v1 Blueprint definition."""

from flask import Blueprint

v1_bp = Blueprint('v1', __name__, url_prefix='/api/v1')

# Import route modules to register their endpoints on v1_bp
from api.v1 import index_ops   # noqa: E402, F401
from api.v1 import search      # noqa: E402, F401
from api.v1 import msds        # noqa: E402, F401
from api.v1 import health      # noqa: E402, F401
from api.v1 import auth        # noqa: E402, F401
from api.v1 import community   # noqa: E402, F401
from api.v1 import admin       # noqa: E402, F401
from api.v1 import news        # noqa: E402, F401
from api.v1 import bookmarks   # noqa: E402, F401
from api.v1 import questions   # noqa: E402, F401
from api.v1 import user        # noqa: E402, F401
from api.v1 import graph       # noqa: E402, F401
from api.v1 import ncs_browse  # noqa: E402, F401
from api.v1 import feedback    # noqa: E402, F401
