"""Semantic Cache — returns cached answers for semantically similar queries.

Uses SQLite for persistence and numpy for cosine similarity search.
Designed to speed up frequently-asked questions from vocational students.
"""

import hashlib
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_DB_PATH = Path(__file__).parent.parent / 'instance' / 'semantic_cache.db'
_DEFAULT_TTL = 3600       # 1 hour (general queries)
_FAQ_TTL = 86400          # 24 hours (FAQ-type queries)
_MAX_ENTRIES = 1000
_SIMILARITY_THRESHOLD = 0.95


class SemanticCache:
    """In-process semantic cache with SQLite backend and numpy vector search."""

    def __init__(self, db_path: Path = _DB_PATH):
        self._db_path = db_path
        self._lock = threading.Lock()
        # In-memory embedding index for fast lookup: {namespace: [(embedding, cache_key), ...]}
        self._index: Dict[str, List] = {}
        self._init_db()
        self._load_index()

    # ------------------------------------------------------------------
    # Database setup
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path), timeout=5)

    def _init_db(self):
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS semantic_cache (
                cache_key   TEXT PRIMARY KEY,
                query_text  TEXT NOT NULL,
                embedding   BLOB NOT NULL,
                namespace   TEXT NOT NULL,
                response    TEXT NOT NULL,
                created_at  REAL NOT NULL,
                ttl         INTEGER NOT NULL,
                hit_count   INTEGER DEFAULT 0,
                is_faq      INTEGER DEFAULT 0
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sc_ns ON semantic_cache(namespace)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sc_ts ON semantic_cache(created_at)")
        conn.commit()
        conn.close()

    def _load_index(self):
        """Load embeddings into memory for fast cosine similarity."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT cache_key, namespace, embedding, created_at, ttl FROM semantic_cache"
        ).fetchall()
        conn.close()

        now = time.time()
        index: Dict[str, List] = {}
        for key, ns, emb_blob, created_at, ttl in rows:
            if now - created_at > ttl:
                continue  # skip expired
            emb = np.frombuffer(emb_blob, dtype=np.float32).copy()
            index.setdefault(ns, []).append((emb, key))
        self._index = index

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(self, query_embedding: np.ndarray, namespace: str) -> Optional[Dict[str, Any]]:
        """Find a cached response for a semantically similar query.

        Args:
            query_embedding: Float32 numpy array (1-D).
            namespace: Pinecone namespace to scope the search.

        Returns:
            Cached response dict, or None on miss.
        """
        entries = self._index.get(namespace)
        if not entries:
            return None

        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)

        best_sim = 0.0
        best_key = None
        for emb, key in entries:
            e_norm = emb / (np.linalg.norm(emb) + 1e-10)
            sim = float(np.dot(q_norm, e_norm))
            if sim > best_sim:
                best_sim = sim
                best_key = key

        if best_sim < _SIMILARITY_THRESHOLD or best_key is None:
            return None

        # Fetch from DB and validate TTL
        conn = self._get_conn()
        row = conn.execute(
            "SELECT response, created_at, ttl FROM semantic_cache WHERE cache_key = ?",
            (best_key,),
        ).fetchone()

        if not row:
            conn.close()
            return None

        resp_json, created_at, ttl = row
        if time.time() - created_at > ttl:
            # Expired — remove
            conn.execute("DELETE FROM semantic_cache WHERE cache_key = ?", (best_key,))
            conn.commit()
            conn.close()
            self._remove_from_index(namespace, best_key)
            return None

        # Update hit count
        conn.execute(
            "UPDATE semantic_cache SET hit_count = hit_count + 1 WHERE cache_key = ?",
            (best_key,),
        )
        conn.commit()
        conn.close()

        logging.info("[SemanticCache] HIT (sim=%.4f, key=%s) ns=%s", best_sim, best_key[:8], namespace)
        return json.loads(resp_json)

    def store(self, query_text: str, query_embedding: np.ndarray,
              namespace: str, response: Dict[str, Any], is_faq: bool = False):
        """Store a response in the cache.

        Args:
            query_text: Original query text.
            query_embedding: Float32 numpy array (1-D).
            namespace: Pinecone namespace.
            response: Full response dict to cache.
            is_faq: If True, uses longer TTL.
        """
        with self._lock:
            self._enforce_limit()

            cache_key = hashlib.md5(f"{query_text}:{namespace}".encode()).hexdigest()
            ttl = _FAQ_TTL if is_faq else _DEFAULT_TTL
            emb_blob = query_embedding.astype(np.float32).tobytes()

            conn = self._get_conn()
            conn.execute("""
                INSERT OR REPLACE INTO semantic_cache
                (cache_key, query_text, embedding, namespace, response, created_at, ttl, is_faq)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cache_key, query_text, emb_blob, namespace,
                json.dumps(response, ensure_ascii=False),
                time.time(), ttl, int(is_faq),
            ))
            conn.commit()
            conn.close()

            # Update memory index
            emb_copy = query_embedding.astype(np.float32).copy()
            self._index.setdefault(namespace, []).append((emb_copy, cache_key))
            logging.info("[SemanticCache] STORED key=%s ns=%s faq=%s", cache_key[:8], namespace, is_faq)

    def invalidate_namespace(self, namespace: str):
        """Remove all cached entries for a namespace."""
        conn = self._get_conn()
        conn.execute("DELETE FROM semantic_cache WHERE namespace = ?", (namespace,))
        conn.commit()
        conn.close()
        self._index.pop(namespace, None)
        logging.info("[SemanticCache] Invalidated namespace: %s", namespace)

    def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics for admin panel."""
        conn = self._get_conn()
        total = conn.execute("SELECT COUNT(*) FROM semantic_cache").fetchone()[0]
        total_hits = conn.execute("SELECT COALESCE(SUM(hit_count), 0) FROM semantic_cache").fetchone()[0]
        faq_count = conn.execute("SELECT COUNT(*) FROM semantic_cache WHERE is_faq = 1").fetchone()[0]
        by_ns = conn.execute(
            "SELECT namespace, COUNT(*), COALESCE(SUM(hit_count), 0) FROM semantic_cache GROUP BY namespace"
        ).fetchall()
        conn.close()
        return {
            'total_entries': total,
            'total_hits': total_hits,
            'faq_entries': faq_count,
            'max_entries': _MAX_ENTRIES,
            'by_namespace': {ns: {'entries': c, 'hits': h} for ns, c, h in by_ns},
        }

    def cleanup_expired(self):
        """Remove expired entries from DB and memory index."""
        conn = self._get_conn()
        conn.execute("DELETE FROM semantic_cache WHERE (created_at + ttl) < ?", (time.time(),))
        conn.commit()
        conn.close()
        self._load_index()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enforce_limit(self):
        """Evict oldest/least-hit entries when at capacity."""
        conn = self._get_conn()
        count = conn.execute("SELECT COUNT(*) FROM semantic_cache").fetchone()[0]
        if count >= _MAX_ENTRIES:
            delete_count = _MAX_ENTRIES // 5
            conn.execute("""
                DELETE FROM semantic_cache WHERE cache_key IN (
                    SELECT cache_key FROM semantic_cache
                    ORDER BY hit_count ASC, created_at ASC
                    LIMIT ?
                )
            """, (delete_count,))
            conn.commit()
            self._load_index()
        conn.close()

    def _remove_from_index(self, namespace: str, cache_key: str):
        """Remove a single entry from the memory index."""
        entries = self._index.get(namespace)
        if entries:
            self._index[namespace] = [(e, k) for e, k in entries if k != cache_key]
