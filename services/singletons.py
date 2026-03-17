"""Lazy-initialized singleton instances for shared services."""

import os
import logging
import threading

_lock = threading.RLock()

_agent = None
_openai_client = None
_law_drf_client = None
_query_enhancer = None
_context_optimizer = None
_reranker = None
_hybrid_searcher = None
_uploader = None
_pinecone_client = None
_gemini_client = None
_anthropic_client = None


def _parse_float_env(name, default):
    """Parse a float from an environment variable, falling back to *default*."""
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except (ValueError, TypeError):
        logging.warning("환경 변수 %s의 값 '%s'이(가) 유효하지 않습니다. 기본값 %s을(를) 사용합니다.", name, raw, default)
        return default


def _require_env(name):
    """Return env var value or raise with a clear message."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"환경 변수 {name}이(가) 설정되지 않았습니다. .env 파일을 확인하세요.")
    return value


def get_gemini_client():
    """Get or create Google Gemini client for answer generation."""
    global _gemini_client
    instance = _gemini_client
    if instance is None:
        from google import genai
        with _lock:
            if _gemini_client is None:
                _gemini_client = genai.Client(
                    api_key=_require_env("GEMINI_API_KEY")
                )
            instance = _gemini_client
    return instance


def get_anthropic_client():
    """Get or create Anthropic Claude client for answer generation."""
    global _anthropic_client
    instance = _anthropic_client
    if instance is None:
        import anthropic
        with _lock:
            if _anthropic_client is None:
                _anthropic_client = anthropic.Anthropic(
                    api_key=_require_env("ANTHROPIC_API_KEY")
                )
            instance = _anthropic_client
    return instance


def get_openai_client():
    """Get or create OpenAI client with SSL certificate configuration."""
    global _openai_client
    instance = _openai_client
    if instance is None:
        import certifi
        import httpx
        from openai import OpenAI
        with _lock:
            if _openai_client is None:
                http_client = httpx.Client(verify=certifi.where())
                _openai_client = OpenAI(
                    api_key=_require_env("OPENAI_API_KEY"),
                    http_client=http_client,
                    timeout=_parse_float_env("OPENAI_TIMEOUT", 60.0),
                )
            instance = _openai_client
    return instance


def get_agent():
    """Get or create the PineconeAgent instance."""
    global _agent
    instance = _agent
    if instance is None:
        # Import outside lock to avoid deadlock with Python's import lock
        from src.agent import PineconeAgent
        with _lock:
            if _agent is None:
                _agent = PineconeAgent(
                    openai_api_key=_require_env("OPENAI_API_KEY"),
                    pinecone_api_key=_require_env("PINECONE_API_KEY"),
                    pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "document-index"),
                    create_index_if_not_exists=False
                )
            instance = _agent
    return instance


_PROVIDER_ENV_KEYS = {
    'openai': 'OPENAI_API_KEY',
    'gemini': 'GEMINI_API_KEY',
    'anthropic': 'ANTHROPIC_API_KEY',
}


def get_query_enhancer():
    """Get or create QueryEnhancer instance."""
    global _query_enhancer
    instance = _query_enhancer
    if instance is None:
        from services.settings import get_setting
        from src.query_enhancer import QueryEnhancer
        with _lock:
            if _query_enhancer is None:
                provider = get_setting('llm_query_provider', 'openai')
                model = get_setting('llm_query_model', 'gpt-4o-mini')
                env_key = _PROVIDER_ENV_KEYS.get(provider, 'OPENAI_API_KEY')
                _query_enhancer = QueryEnhancer(
                    _require_env(env_key), model=model, provider=provider,
                )
            instance = _query_enhancer
    return instance


def get_context_optimizer():
    """Get or create ContextOptimizer instance."""
    global _context_optimizer
    instance = _context_optimizer
    if instance is None:
        from services.settings import get_setting
        from src.context_optimizer import ContextOptimizer
        with _lock:
            if _context_optimizer is None:
                provider = get_setting('llm_context_provider', 'openai')
                model = get_setting('llm_context_model', 'gpt-4o-mini')
                env_key = _PROVIDER_ENV_KEYS.get(provider, 'OPENAI_API_KEY')
                _context_optimizer = ContextOptimizer(
                    _require_env(env_key), model=model, provider=provider,
                )
            instance = _context_optimizer
    return instance


def get_pinecone_client():
    """Get or create standalone Pinecone client.

    Creates the client directly instead of going through get_agent() to avoid
    unnecessary initialization overhead and potential circular dependency risks.
    """
    global _pinecone_client
    instance = _pinecone_client
    if instance is None:
        from pinecone import Pinecone
        with _lock:
            if _pinecone_client is None:
                _pinecone_client = Pinecone(api_key=_require_env("PINECONE_API_KEY"))
            instance = _pinecone_client
    return instance


def get_reranker_instance():
    """Get or create Reranker instance (prefers Pinecone Inference API)."""
    global _reranker
    instance = _reranker
    if instance is None:
        from src.reranker import get_reranker
        with _lock:
            if _reranker is None:
                pc = get_pinecone_client()
                _reranker = get_reranker(use_cross_encoder=True, pinecone_client=pc)
            instance = _reranker
    return instance


def get_hybrid_searcher_instance():
    """Get or create HybridSearcher instance."""
    global _hybrid_searcher
    instance = _hybrid_searcher
    if instance is None:
        from src.hybrid_searcher import get_hybrid_searcher
        with _lock:
            if _hybrid_searcher is None:
                _hybrid_searcher = get_hybrid_searcher()
            instance = _hybrid_searcher
    return instance


def get_law_drf_client():
    """Get or create LawDrfClient singleton.

    Returns None if LAW_OC env var is not set (DRF API not available).
    """
    global _law_drf_client
    instance = _law_drf_client
    if instance is None:
        oc = os.getenv('LAW_OC', '')
        if not oc:
            return None
        from services.law_drf_client import LawDrfClient
        with _lock:
            if _law_drf_client is None:
                _law_drf_client = LawDrfClient(oc=oc)
            instance = _law_drf_client
    return instance


def get_uploader():
    """Get or create PineconeUploader for stats."""
    global _uploader
    instance = _uploader
    if instance is None:
        from src.pinecone_uploader import PineconeUploader
        with _lock:
            if _uploader is None:
                _uploader = PineconeUploader(
                    api_key=_require_env("PINECONE_API_KEY"),
                    index_name=os.getenv("PINECONE_INDEX_NAME", "document-index"),
                    create_if_not_exists=False
                )
            instance = _uploader
    return instance


# ---------------------------------------------------------------------------
# Cache invalidation (called from admin settings endpoint)
# ---------------------------------------------------------------------------

def _close_if_possible(instance):
    """Call close() on an instance if available, swallowing errors."""
    if instance is not None and hasattr(instance, 'close'):
        try:
            instance.close()
        except Exception:
            pass


def invalidate_query_enhancer():
    """Reset QueryEnhancer so it is re-created with the latest model setting."""
    from src.query_enhancer import clear_enhancement_cache
    global _query_enhancer
    with _lock:
        old, _query_enhancer = _query_enhancer, None
    _close_if_possible(old)
    clear_enhancement_cache()


def invalidate_context_optimizer():
    """Reset ContextOptimizer so it is re-created with the latest model setting."""
    global _context_optimizer
    with _lock:
        old, _context_optimizer = _context_optimizer, None
    _close_if_possible(old)


def invalidate_reranker():
    """Reset Reranker so it is re-created with the latest setting."""
    global _reranker
    with _lock:
        old, _reranker = _reranker, None
    _close_if_possible(old)


_graph_searcher = None
_graph_searcher_lock = threading.RLock()


def get_graph_searcher():
    """Return (or lazily create) the singleton GraphSearcher instance."""
    global _graph_searcher
    if _graph_searcher is None:
        with _graph_searcher_lock:
            if _graph_searcher is None:
                from services.graph_searcher import GraphSearcher
                _graph_searcher = GraphSearcher()
    return _graph_searcher


def invalidate_graph_searcher():
    """Reset the GraphSearcher singleton (clears entity cache)."""
    global _graph_searcher
    with _graph_searcher_lock:
        _graph_searcher = None


_community_searcher = None


def get_community_searcher():
    """Return (or lazily create) the singleton CommunitySearcher instance."""
    global _community_searcher
    if _community_searcher is None:
        with _lock:
            if _community_searcher is None:
                from services.community_searcher import CommunitySearcher
                _community_searcher = CommunitySearcher()
    return _community_searcher


def invalidate_community_searcher():
    """Reset the CommunitySearcher singleton (clears summary cache)."""
    global _community_searcher
    with _lock:
        if _community_searcher is not None:
            _community_searcher.invalidate_cache()
        _community_searcher = None


def shutdown_all():
    """Close all singleton instances that hold resources.

    Call this at application shutdown (e.g. via atexit) to ensure
    httpx.Client connections are properly released.
    """
    from src.query_enhancer import shutdown_enhancement_executor
    with _lock:
        to_close = [_query_enhancer, _context_optimizer, _reranker]
    for inst in to_close:
        _close_if_possible(inst)
    shutdown_enhancement_executor()
