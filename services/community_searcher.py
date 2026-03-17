"""Online global search service using community summaries."""

import logging
import threading

from models import db, KGCommunity

logger = logging.getLogger(__name__)


class CommunitySearcher:
    """Global search using pre-computed community summaries."""

    def __init__(self):
        self._summary_cache: dict[str, list[dict]] = {}
        self._cache_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, namespace: str,
               max_communities: int = 10) -> dict:
        """Run global search on community summaries.

        Returns dict with 'answer_context', 'communities_used', 'community_titles'.
        Returns empty context when no communities are available.
        """
        relevant = self._select_relevant_communities(query, namespace, max_communities)
        if not relevant:
            return {'answer_context': '', 'communities_used': 0, 'community_titles': []}

        mapped = self._map_communities(relevant)
        context = self._reduce_results(mapped)

        return {
            'answer_context': context,
            'communities_used': len(relevant),
            'community_titles': [c['title'] for c in relevant],
        }

    def invalidate_cache(self, namespace: str | None = None):
        """Clear summary cache."""
        with self._cache_lock:
            if namespace:
                self._summary_cache.pop(namespace, None)
            else:
                self._summary_cache.clear()

    # ------------------------------------------------------------------
    # Community selection
    # ------------------------------------------------------------------

    def _select_relevant_communities(self, query: str, namespace: str,
                                     max_count: int) -> list[dict]:
        """Select communities relevant to the query using keyword overlap."""
        summaries = self._load_summaries(namespace)
        if not summaries:
            return []

        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored = []
        for comm in summaries:
            text = f"{comm['title']} {comm['summary']}".lower()
            text_words = set(text.split())
            overlap = len(query_words & text_words)
            # For short queries (1-2 words) or any overlap, include the community
            if overlap > 0 or len(query_words) <= 2:
                scored.append((overlap, comm))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [comm for _, comm in scored[:max_count]]

    def _load_summaries(self, namespace: str) -> list[dict]:
        """Load community summaries with caching."""
        with self._cache_lock:
            if namespace in self._summary_cache:
                return self._summary_cache[namespace]

        communities = (
            KGCommunity.query
            .filter_by(namespace=namespace)
            .filter(KGCommunity.summary.isnot(None))
            .all()
        )
        result = [
            {
                'id': c.id,
                'title': c.title or '',
                'summary': c.summary or '',
                'member_count': c.member_count,
            }
            for c in communities
        ]

        with self._cache_lock:
            self._summary_cache[namespace] = result
        return result

    # ------------------------------------------------------------------
    # Map-Reduce
    # ------------------------------------------------------------------

    def _map_communities(self, communities: list[dict]) -> list[str]:
        """Extract text from each community summary."""
        return [
            f"[{c['title']}] {c['summary']}"
            for c in communities
        ]

    def _reduce_results(self, mapped: list[str]) -> str:
        """Combine mapped results into a unified context string."""
        context_parts = []
        for i, text in enumerate(mapped, 1):
            context_parts.append(f"### 커뮤니티 {i}\n{text}")
        return '\n\n'.join(context_parts)
