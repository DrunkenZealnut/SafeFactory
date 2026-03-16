"""Online graph search service — matches query entities and traverses the Knowledge Graph."""

import json
import logging
import threading
from dataclasses import dataclass, field

from sqlalchemy import text

from models import db, KGEntity, KGEntityChunk
from services.graph_config import get_graph_config

logger = logging.getLogger(__name__)


@dataclass
class GraphResult:
    """A chunk discovered through graph traversal."""
    chunk_vector_id: str
    entity_path: list[str] = field(default_factory=list)
    relation_path: list[str] = field(default_factory=list)
    hop_distance: int = 0
    graph_score: float = 0.0


def _normalize(text: str) -> str:
    return text.lower().replace(' ', '').strip()


class GraphSearcher:
    """Matches query entities and traverses the KG to find related chunks."""

    def __init__(self):
        self._entity_cache: dict[str, dict[str, int]] = {}
        self._cache_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, namespace: str,
               hop_depth: int | None = None,
               max_results: int | None = None) -> list[GraphResult]:
        """Run graph-based search for the given query.

        Returns a list of GraphResult with chunk IDs and scores. Returns an
        empty list when graph is disabled for the namespace or no entities match.
        """
        config = get_graph_config(namespace)
        if not config.get('enabled', False):
            return []

        hop_depth = hop_depth or config.get('hop_depth', 2)
        max_results = max_results or config.get('max_graph_results', 10)

        # Step 1: match query to known entities
        matched_ids = self._match_query_entities(query, namespace)
        if not matched_ids:
            return []

        # Step 2: traverse graph
        traversal = self._traverse_graph(matched_ids, namespace, hop_depth)
        if not traversal:
            return []

        # Step 3: fetch related chunks and score
        results = self._collect_chunk_results(traversal, namespace, max_results)
        logger.info(
            "[Graph Search] query=%s, matched_entities=%d, traversed=%d, chunks=%d",
            query[:40], len(matched_ids), len(traversal), len(results),
        )
        return results

    def invalidate_cache(self, namespace: str | None = None):
        """Clear the entity cache."""
        with self._cache_lock:
            if namespace:
                self._entity_cache.pop(namespace, None)
            else:
                self._entity_cache.clear()

    # ------------------------------------------------------------------
    # Entity matching
    # ------------------------------------------------------------------

    def _match_query_entities(self, query: str, namespace: str) -> list[int]:
        """Find entities whose name or alias appears in the query."""
        cache = self._load_entity_cache(namespace)
        query_norm = _normalize(query)
        matched = set()
        for name_norm, entity_id in cache.items():
            if len(name_norm) >= 2 and name_norm in query_norm:
                matched.add(entity_id)
        return list(matched)

    def _load_entity_cache(self, namespace: str) -> dict[str, int]:
        """Load namespace entity cache (name_normalized → entity_id)."""
        with self._cache_lock:
            if namespace in self._entity_cache:
                return self._entity_cache[namespace]

        rows = (
            db.session.query(KGEntity.id, KGEntity.name_normalized, KGEntity.aliases_json)
            .filter_by(namespace=namespace)
            .all()
        )
        cache: dict[str, int] = {}
        for eid, name_norm, aliases_json in rows:
            cache[name_norm] = eid
            for alias in json.loads(aliases_json or '[]'):
                anorm = _normalize(alias)
                if anorm:
                    cache[anorm] = eid

        with self._cache_lock:
            self._entity_cache[namespace] = cache
        return cache

    # ------------------------------------------------------------------
    # Graph traversal (Recursive CTE)
    # ------------------------------------------------------------------

    def _traverse_graph(self, entity_ids: list[int], namespace: str,
                        max_hops: int = 2) -> list[dict]:
        """N-hop graph walk using SQLite recursive CTE."""
        if not entity_ids:
            return []

        placeholders = ','.join(str(int(eid)) for eid in entity_ids)

        sql = text(f"""
            WITH RECURSIVE graph_walk AS (
                SELECT
                    e.id,
                    e.name,
                    0 AS hop,
                    e.name AS path,
                    '' AS relation_path,
                    1.0 AS path_confidence
                FROM kg_entities e
                WHERE e.id IN ({placeholders}) AND e.namespace = :ns

                UNION ALL

                SELECT
                    e2.id,
                    e2.name,
                    gw.hop + 1,
                    gw.path || ' > ' || e2.name,
                    gw.relation_path || CASE WHEN gw.relation_path = '' THEN '' ELSE ',' END || r.relation_type,
                    gw.path_confidence * r.confidence
                FROM graph_walk gw
                JOIN kg_relations r ON r.source_id = gw.id AND r.namespace = :ns
                JOIN kg_entities e2 ON e2.id = r.target_id
                WHERE gw.hop < :max_hops
            )
            SELECT DISTINCT id, name, hop, path, relation_path, path_confidence
            FROM graph_walk
            ORDER BY hop ASC, path_confidence DESC
            LIMIT 50
        """)

        rows = db.session.execute(sql, {'ns': namespace, 'max_hops': max_hops}).fetchall()
        return [dict(row._mapping) for row in rows]

    # ------------------------------------------------------------------
    # Chunk collection
    # ------------------------------------------------------------------

    def _collect_chunk_results(self, traversal: list[dict],
                               namespace: str, max_results: int) -> list[GraphResult]:
        """Map traversed entities to chunks and compute scores."""
        entity_ids = [e['id'] for e in traversal]
        if not entity_ids:
            return []

        chunk_rows = (
            db.session.query(KGEntityChunk)
            .filter(KGEntityChunk.entity_id.in_(entity_ids), KGEntityChunk.namespace == namespace)
            .all()
        )

        entity_map = {e['id']: e for e in traversal}
        seen: set[str] = set()
        results: list[GraphResult] = []

        for ec in chunk_rows:
            if ec.chunk_vector_id in seen:
                continue
            seen.add(ec.chunk_vector_id)

            e_info = entity_map.get(ec.entity_id, {})
            hop = e_info.get('hop', 0)
            conf = e_info.get('path_confidence', 1.0)
            score = conf / (1.0 + hop * 0.5)

            path_str = e_info.get('path', '')
            rel_str = e_info.get('relation_path', '')

            results.append(GraphResult(
                chunk_vector_id=ec.chunk_vector_id,
                entity_path=path_str.split(' > ') if path_str else [],
                relation_path=[r for r in rel_str.split(',') if r],
                hop_distance=hop,
                graph_score=round(score, 4),
            ))

        results.sort(key=lambda r: r.graph_score, reverse=True)
        return results[:max_results]
