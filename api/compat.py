"""Backward-compatible /api/* route registration.

Registers the same v1 view functions under the legacy /api/* paths
so existing frontend code continues to work without changes.
"""


def register_compat_routes(app, csrf=None):
    """Register legacy /api/* routes pointing to the same v1 view functions.

    Args:
        app: Flask application instance.
        csrf: Optional CSRFProtect instance. When provided, POST view
              functions are explicitly exempted from CSRF checks (they
              are JSON API endpoints, not form submissions).
    """
    from api.v1.index_ops import api_stats, api_namespaces, api_sources, api_delete
    from api.v1.search import api_search, api_ask, api_ask_stream, api_pdf_resolve
    from api.v1.msds import msds_search, msds_detail, msds_identify
    from api.v1.health import api_health, api_domains
    from api.v1.calculator import api_calculate_wage, api_calculate_insurance
    from api.v1.auth import api_auth_me, api_auth_logout

    # Collect POST view functions for CSRF exemption
    post_views = []

    def _add(rule, endpoint, view_func, **kwargs):
        app.add_url_rule(rule, endpoint, view_func, **kwargs)
        if 'POST' in kwargs.get('methods', []):
            post_views.append(view_func)

    # Index operations
    _add('/api/stats', 'compat_stats', api_stats, methods=['GET'])
    _add('/api/namespaces', 'compat_namespaces', api_namespaces, methods=['GET'])
    _add('/api/sources', 'compat_sources', api_sources, methods=['GET'])
    _add('/api/delete', 'compat_delete', api_delete, methods=['POST'])

    # Search & RAG
    _add('/api/search', 'compat_search', api_search, methods=['POST'])
    _add('/api/ask', 'compat_ask', api_ask, methods=['POST'])
    _add('/api/ask/stream', 'compat_ask_stream', api_ask_stream, methods=['POST'])
    _add('/api/pdf/resolve', 'compat_pdf_resolve', api_pdf_resolve, methods=['POST'])

    # MSDS
    _add('/api/msds/search', 'compat_msds_search', msds_search, methods=['POST'])
    _add('/api/msds/detail', 'compat_msds_detail', msds_detail, methods=['POST'])
    _add('/api/msds/identify', 'compat_msds_identify', msds_identify, methods=['POST'])

    # Health & configuration
    _add('/api/health', 'compat_health', api_health, methods=['GET'])
    _add('/api/domains', 'compat_domains', api_domains, methods=['GET'])

    # Calculator
    _add('/api/calculate/wage', 'compat_calc_wage', api_calculate_wage, methods=['POST'])
    _add('/api/calculate/insurance', 'compat_calc_insurance', api_calculate_insurance, methods=['POST'])

    # Auth
    _add('/api/auth/me', 'compat_auth_me', api_auth_me, methods=['GET'])
    _add('/api/auth/logout', 'compat_auth_logout', api_auth_logout, methods=['POST'])

    # Exempt all POST view functions from CSRF
    if csrf is not None:
        for view_func in post_views:
            csrf.exempt(view_func)
