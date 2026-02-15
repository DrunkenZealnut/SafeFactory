"""Health check and domain configuration endpoints."""

import os
import sys
import time

from api.v1 import v1_bp
from api.response import success_response, error_response
from services.domain_config import DOMAIN_CONFIG

_start_time = time.time()


@v1_bp.route('/health', methods=['GET'])
def api_health():
    """Health check endpoint for monitoring and native app connectivity."""
    services = {}

    # Check Pinecone connectivity with actual server query
    try:
        from services.singletons import get_agent
        agent = get_agent()
        if agent:
            try:
                agent.get_stats()
                services['pinecone'] = 'connected'
            except Exception:
                services['pinecone'] = 'degraded'
        else:
            services['pinecone'] = 'unavailable'
    except Exception:
        services['pinecone'] = 'error'

    # Check OpenAI API key
    services['openai'] = 'configured' if os.getenv('OPENAI_API_KEY') else 'not_configured'

    overall = 'healthy' if all(
        v in ('connected', 'configured') for v in services.values()
    ) else 'degraded'

    return success_response(data={
        'status': overall,
        'services': services,
        'uptime': round(time.time() - _start_time),
        'python_version': sys.version.split()[0],
        'version': 'v1',
    })


@v1_bp.route('/domains', methods=['GET'])
def api_domains():
    """Return domain configuration for native app dynamic rendering."""
    return success_response(data={
        'domains': DOMAIN_CONFIG,
        'count': len(DOMAIN_CONFIG),
    })
