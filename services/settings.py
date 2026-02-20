"""Thread-safe system settings accessor with in-memory cache."""

import logging
import threading
import time

_cache: dict[str, str] = {}
_cache_lock = threading.Lock()
_cache_ts: float = 0.0
_CACHE_TTL = 60  # seconds
_FAILURE_TTL = 5  # shorter TTL after refresh failure

# Defaults matching current hardcoded values (fallback when no DB row exists)
DEFAULTS = {
    'llm_answer_model': 'gpt-4o-mini',
    'llm_answer_provider': 'openai',
    'llm_answer_temperature': '0.3',
    'llm_query_model': 'gpt-4o-mini',
    'llm_query_provider': 'openai',
    'llm_context_model': 'gpt-4o-mini',
    'llm_context_provider': 'openai',
    'embedding_model': 'text-embedding-3-small',
    'reranker_type': 'pinecone',
}


def get_setting(key: str, default: str | None = None) -> str | None:
    """Get a system setting value.

    Lookup order: DB cache → DEFAULTS → *default* parameter.
    """
    _maybe_refresh_cache()
    with _cache_lock:
        val = _cache.get(key)
    if val is not None:
        return val
    return DEFAULTS.get(key, default)


def invalidate_cache() -> None:
    """Force cache refresh on next ``get_setting`` call."""
    global _cache_ts
    with _cache_lock:
        _cache_ts = 0.0


def _maybe_refresh_cache() -> None:
    """Reload settings from DB when TTL has expired."""
    global _cache, _cache_ts
    now = time.monotonic()
    if (now - _cache_ts) < _CACHE_TTL:
        return
    with _cache_lock:
        # Double-check after acquiring the lock (recapture time to avoid stale value)
        if (time.monotonic() - _cache_ts) < _CACHE_TTL:
            return
        try:
            from models import SystemSetting
            rows = SystemSetting.query.all()
            _cache = {r.key: r.value for r in rows}
            _cache_ts = time.monotonic()
        except Exception as e:
            logging.warning("Failed to refresh settings cache (using defaults): %s", e)
            _cache_ts = time.monotonic() - _CACHE_TTL + _FAILURE_TTL  # retry after 5s
