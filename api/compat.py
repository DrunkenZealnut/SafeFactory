"""Backward-compatible /api/* route registration.

Registers the same v1 view functions under the legacy /api/* paths
so existing frontend code continues to work without changes.
"""


def register_compat_routes(app):
    """Register legacy /api/* routes pointing to the same v1 view functions."""
    from api.v1.index_ops import api_stats, api_namespaces, api_sources, api_delete
    from api.v1.search import api_search, api_ask, api_ask_stream
    from api.v1.msds import msds_search, msds_detail, msds_identify
    from api.v1.health import api_health, api_domains
    from api.v1.calculator import api_calculate_wage, api_calculate_insurance

    # Index operations
    app.add_url_rule('/api/stats', 'compat_stats', api_stats)
    app.add_url_rule('/api/namespaces', 'compat_namespaces', api_namespaces)
    app.add_url_rule('/api/sources', 'compat_sources', api_sources)
    app.add_url_rule('/api/delete', 'compat_delete', api_delete, methods=['POST'])

    # Search & RAG
    app.add_url_rule('/api/search', 'compat_search', api_search, methods=['POST'])
    app.add_url_rule('/api/ask', 'compat_ask', api_ask, methods=['POST'])
    app.add_url_rule('/api/ask/stream', 'compat_ask_stream', api_ask_stream, methods=['POST'])

    # MSDS
    app.add_url_rule('/api/msds/search', 'compat_msds_search', msds_search, methods=['POST'])
    app.add_url_rule('/api/msds/detail', 'compat_msds_detail', msds_detail, methods=['POST'])
    app.add_url_rule('/api/msds/identify', 'compat_msds_identify', msds_identify, methods=['POST'])

    # Health & configuration
    app.add_url_rule('/api/health', 'compat_health', api_health)
    app.add_url_rule('/api/domains', 'compat_domains', api_domains)

    # Calculator
    app.add_url_rule('/api/calculate/wage', 'compat_calc_wage', api_calculate_wage, methods=['POST'])
    app.add_url_rule('/api/calculate/insurance', 'compat_calc_insurance', api_calculate_insurance, methods=['POST'])
