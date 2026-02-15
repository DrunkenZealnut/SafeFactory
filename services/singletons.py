"""Lazy-initialized singleton instances for shared services."""

import os
import logging
import threading

_lock = threading.Lock()

_agent = None
_openai_client = None
_query_enhancer = None
_context_optimizer = None
_reranker = None
_hybrid_searcher = None
_uploader = None


def _require_env(name):
    """Return env var value or raise with a clear message."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"환경 변수 {name}이(가) 설정되지 않았습니다. .env 파일을 확인하세요.")
    return value


def get_openai_client():
    """Get or create OpenAI client with SSL certificate configuration."""
    global _openai_client
    if _openai_client is None:
        with _lock:
            if _openai_client is None:
                import certifi
                import httpx
                from openai import OpenAI
                http_client = httpx.Client(verify=certifi.where())
                _openai_client = OpenAI(
                    api_key=_require_env("OPENAI_API_KEY"),
                    http_client=http_client,
                    timeout=60.0
                )
    return _openai_client


def get_agent():
    """Get or create the PineconeAgent instance."""
    global _agent
    if _agent is None:
        with _lock:
            if _agent is None:
                from src.agent import PineconeAgent
                _agent = PineconeAgent(
                    openai_api_key=_require_env("OPENAI_API_KEY"),
                    pinecone_api_key=_require_env("PINECONE_API_KEY"),
                    pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "document-index"),
                    create_index_if_not_exists=False
                )
    return _agent


def get_query_enhancer():
    """Get or create QueryEnhancer instance."""
    global _query_enhancer
    if _query_enhancer is None:
        with _lock:
            if _query_enhancer is None:
                from src.query_enhancer import QueryEnhancer
                _query_enhancer = QueryEnhancer(_require_env("OPENAI_API_KEY"))
    return _query_enhancer


def get_context_optimizer():
    """Get or create ContextOptimizer instance."""
    global _context_optimizer
    if _context_optimizer is None:
        with _lock:
            if _context_optimizer is None:
                from src.context_optimizer import ContextOptimizer
                _context_optimizer = ContextOptimizer(_require_env("OPENAI_API_KEY"))
    return _context_optimizer


def get_pinecone_client():
    """Get Pinecone client from agent or create standalone."""
    try:
        agent = get_agent()
        return agent.pinecone_uploader.pc
    except Exception as e:
        logging.warning(f"Failed to get Pinecone client from agent: {e}")
        try:
            from pinecone import Pinecone
            return Pinecone(api_key=_require_env("PINECONE_API_KEY"))
        except Exception as e2:
            logging.error(f"Failed to create standalone Pinecone client: {e2}")
            return None


def get_reranker_instance():
    """Get or create Reranker instance (prefers Pinecone Inference API)."""
    global _reranker
    if _reranker is None:
        with _lock:
            if _reranker is None:
                from src.reranker import get_reranker
                pc = get_pinecone_client()
                _reranker = get_reranker(use_cross_encoder=True, pinecone_client=pc)
    return _reranker


def get_hybrid_searcher_instance():
    """Get or create HybridSearcher instance."""
    global _hybrid_searcher
    if _hybrid_searcher is None:
        with _lock:
            if _hybrid_searcher is None:
                from src.hybrid_searcher import get_hybrid_searcher
                _hybrid_searcher = get_hybrid_searcher()
    return _hybrid_searcher


def get_uploader():
    """Get or create PineconeUploader for stats."""
    global _uploader
    if _uploader is None:
        with _lock:
            if _uploader is None:
                from src.pinecone_uploader import PineconeUploader
                _uploader = PineconeUploader(
                    api_key=_require_env("PINECONE_API_KEY"),
                    index_name=os.getenv("PINECONE_INDEX_NAME", "document-index"),
                    create_if_not_exists=False
                )
    return _uploader
