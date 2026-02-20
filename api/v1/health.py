"""Health check and domain configuration endpoints."""

import os
import sys
import time

from api.v1 import v1_bp
from api.response import success_response, API_VERSION
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

    # Check Gemini API key (used for RAG answer generation)
    services['gemini'] = 'configured' if os.getenv('GEMINI_API_KEY') else 'not_configured'

    # Check Anthropic API key (optional)
    if os.getenv('ANTHROPIC_API_KEY'):
        services['anthropic'] = 'configured'

    overall = 'healthy' if all(
        v in ('connected', 'configured') for v in services.values()
    ) else 'degraded'

    return success_response(data={
        'status': overall,
        'services': services,
        'uptime': round(time.time() - _start_time),
        'python_version': sys.version.split()[0],
        'version': API_VERSION,
    })


@v1_bp.route('/llm-info', methods=['GET'])
def api_llm_info():
    """Return current LLM model configuration (public, no auth required)."""
    from services.settings import get_setting
    return success_response(data={
        'answer_model': get_setting('llm_answer_model'),
        'answer_provider': get_setting('llm_answer_provider'),
        'answer_temperature': get_setting('llm_answer_temperature'),
        'embedding_model': get_setting('embedding_model'),
        'reranker_type': get_setting('reranker_type'),
    })


@v1_bp.route('/domains', methods=['GET'])
def api_domains():
    """Return domain configuration for native app dynamic rendering."""
    from models import Category
    try:
        categories = Category.query.filter_by(is_active=True).order_by(Category.sort_order).all()
    except Exception:
        categories = []

    community_config = {
        'title': '커뮤니티',
        'icon': '💬',
        'color': '#7c4dff',
        'color_rgb': '124, 77, 255',
        'description': '질문, 정보공유, 자유로운 소통 공간',
        'type': 'community',
        'route': '/community',
        'api_base': '/api/v1/community',
        'categories': [c.to_dict() for c in categories],
        'features': ['질문/답변', '정보공유', '자유게시판', '공지사항'],
    }

    return success_response(data={
        'domains': DOMAIN_CONFIG,
        'community': community_config,
        'count': len(DOMAIN_CONFIG) + 1,
    })
