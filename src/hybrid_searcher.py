"""
Hybrid Searcher Module
Combines BM25 keyword search with vector search using Reciprocal Rank Fusion (RRF).
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Try to import rank_bm25
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logging.warning("rank-bm25 not installed. Hybrid search disabled.")


@dataclass
class HybridResult:
    """Result from hybrid search."""
    content: str
    source_file: str
    vector_score: float
    bm25_score: float
    rrf_score: float
    metadata: Dict[str, Any]


class HybridSearcher:
    """
    Combines BM25 lexical search with vector semantic search.

    Uses Reciprocal Rank Fusion (RRF) to merge results from both methods.
    """

    def __init__(
        self,
        rrf_k: int = 60,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5
    ):
        """
        Initialize the HybridSearcher.

        Args:
            rrf_k: RRF constant (higher = more weight to lower ranks)
            vector_weight: Weight for vector search results (0-1)
            bm25_weight: Weight for BM25 results (0-1)
        """
        self.rrf_k = rrf_k
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.bm25_index = None
        self.corpus = []
        self.doc_map = {}  # Maps corpus index to document

    # Korean particles/postpositions to strip from tokens
    _KOREAN_PARTICLES = re.compile(
        r'(은|는|이|가|을|를|의|에|에서|으로|로|와|과|도|만|까지|부터|에게|한테|께)$'
    )

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize Korean/English mixed text with particle stripping.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        # Extract Korean, English, and numeric tokens
        tokens = re.findall(r'[가-힣]+|[a-zA-Z]+|[0-9]+', text.lower())
        # Strip Korean particles and filter short tokens
        stripped = []
        for t in tokens:
            cleaned = self._KOREAN_PARTICLES.sub('', t)
            if len(cleaned) > 1:
                stripped.append(cleaned)
        return stripped

    def build_index(self, documents: List[Dict[str, Any]]) -> None:
        """
        Build BM25 index from documents.

        Args:
            documents: List of documents with 'content' or metadata.content
        """
        if not BM25_AVAILABLE:
            logging.warning("BM25 not available, skipping index build")
            return

        self.corpus = []
        self.doc_map = {}

        for i, doc in enumerate(documents):
            content = doc.get('metadata', {}).get('content', '')
            if not content:
                content = doc.get('content', '')

            if content:
                tokens = self._tokenize(content)
                self.corpus.append(tokens)
                self.doc_map[i] = doc

        if self.corpus:
            self.bm25_index = BM25Okapi(self.corpus)
            logging.info(f"Built BM25 index with {len(self.corpus)} documents")
        else:
            logging.warning("No documents to index")

    def get_bm25_scores(self, query: str) -> Dict[str, float]:
        """Return {doc_id: bm25_score} map for all indexed documents.

        Uses the document 'id' field as key. Documents without an 'id' are
        skipped. Returns an empty dict when no BM25 index is available.
        """
        if not self.bm25_index:
            return {}
        query_tokens = self._tokenize(query)
        scores = self.bm25_index.get_scores(query_tokens)
        score_map = {}
        for idx, doc in self.doc_map.items():
            doc_id = doc.get('id')
            if doc_id:
                score_map[doc_id] = round(scores[idx], 4)
        return score_map

    def bm25_search(
        self,
        query: str,
        top_k: int = 20
    ) -> List[Tuple[int, float]]:
        """
        Search using BM25.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of (doc_index, score) tuples
        """
        if not self.bm25_index:
            return []

        query_tokens = self._tokenize(query)
        scores = self.bm25_index.get_scores(query_tokens)

        # Get top-k indices with scores
        indexed_scores = [(i, score) for i, score in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        return indexed_scores[:top_k]

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Tuple[int, float]]
    ) -> List[Dict[str, Any]]:
        """
        Merge results using Reciprocal Rank Fusion.

        RRF Score = sum(1 / (k + rank)) for each ranking list

        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search

        Returns:
            Merged and reranked results
        """
        rrf_scores = {}

        # Process vector search results
        for rank, doc in enumerate(vector_results, start=1):
            # Create unique identifier from content hash
            content = doc.get('metadata', {}).get('content', '')
            doc_id = hash(content[:200])  # Use first 200 chars for ID

            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {
                    'doc': doc,
                    'vector_rank': rank,
                    'bm25_rank': None,
                    'rrf_score': 0
                }
            rrf_scores[doc_id]['vector_rank'] = rank
            rrf_scores[doc_id]['rrf_score'] += self.vector_weight / (self.rrf_k + rank)

        # Process BM25 results
        for rank, (idx, score) in enumerate(bm25_results, start=1):
            doc = self.doc_map.get(idx)
            if not doc:
                continue

            content = doc.get('metadata', {}).get('content', '')
            doc_id = hash(content[:200])

            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {
                    'doc': doc,
                    'vector_rank': None,
                    'bm25_rank': rank,
                    'rrf_score': 0
                }
            else:
                rrf_scores[doc_id]['bm25_rank'] = rank

            rrf_scores[doc_id]['rrf_score'] += self.bm25_weight / (self.rrf_k + rank)

        # Sort by RRF score
        sorted_results = sorted(
            rrf_scores.values(),
            key=lambda x: x['rrf_score'],
            reverse=True
        )

        # Return documents with added RRF metadata
        result = []
        for item in sorted_results:
            doc = item['doc'].copy()
            doc['rrf_score'] = item['rrf_score']
            doc['vector_rank'] = item['vector_rank']
            doc['bm25_rank'] = item['bm25_rank']
            result.append(doc)

        return result

    def search(
        self,
        query: str,
        vector_results: List[Dict[str, Any]],
        top_k: int = 10,
        build_index: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector and BM25 results.

        Args:
            query: Search query
            vector_results: Results from vector search
            top_k: Number of results to return
            build_index: Whether to rebuild BM25 index from vector results

        Returns:
            Hybrid search results sorted by RRF score
        """
        if not vector_results:
            return []

        # Build BM25 index from vector results if needed
        if build_index and BM25_AVAILABLE:
            self.build_index(vector_results)

        # Get BM25 results
        bm25_results = self.bm25_search(query, top_k=len(vector_results))

        if not bm25_results:
            # If BM25 not available, return vector results with added metadata
            for i, doc in enumerate(vector_results):
                doc['rrf_score'] = doc.get('score', 0)
                doc['vector_rank'] = i + 1
                doc['bm25_rank'] = None
            return vector_results[:top_k]

        # Merge using RRF
        merged = self._reciprocal_rank_fusion(vector_results, bm25_results)

        return merged[:top_k]

    def search_with_keyword_boost(
        self,
        query: str,
        vector_results: List[Dict[str, Any]],
        keywords: List[str],
        top_k: int = 10,
        keyword_boost: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search with additional keyword boosting.

        Args:
            query: Search query
            vector_results: Results from vector search
            keywords: Keywords to boost
            top_k: Number of results
            keyword_boost: Boost factor per keyword match

        Returns:
            Search results with keyword boosting
        """
        # First, do standard hybrid search
        results = self.search(query, vector_results, top_k=top_k)

        if not keywords:
            return results

        # Apply keyword boosting
        keywords_lower = [k.lower() for k in keywords]

        for doc in results:
            content = doc.get('metadata', {}).get('content', '').lower()
            boost = 0

            for keyword in keywords_lower:
                if keyword in content:
                    boost += keyword_boost
                    # Extra boost for title/header matches
                    if content.startswith(keyword):
                        boost += keyword_boost

            doc['keyword_boost'] = boost
            doc['boosted_score'] = doc.get('rrf_score', 0) + boost

        # Re-sort by boosted score
        results.sort(key=lambda x: x.get('boosted_score', 0), reverse=True)

        return results[:top_k]


class SimpleHybridSearcher:
    """
    Fallback hybrid searcher when BM25 is not available.
    Uses simple keyword matching instead.
    """

    def __init__(self, keyword_weight: float = 0.3):
        self.keyword_weight = keyword_weight

    def search(
        self,
        query: str,
        vector_results: List[Dict[str, Any]],
        top_k: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Simple hybrid search using keyword matching.
        """
        if not vector_results:
            return []

        query_words = set(query.lower().split())

        for doc in vector_results:
            content = doc.get('metadata', {}).get('content', '').lower()
            content_words = set(content.split())

            # Calculate keyword overlap
            overlap = len(query_words & content_words)
            keyword_score = overlap / len(query_words) if query_words else 0

            # Combine scores
            vector_score = doc.get('score', 0)
            doc['keyword_score'] = keyword_score
            doc['hybrid_score'] = (
                (1 - self.keyword_weight) * vector_score +
                self.keyword_weight * keyword_score
            )
            doc['rrf_score'] = doc['hybrid_score']

        # Sort by hybrid score
        sorted_results = sorted(
            vector_results,
            key=lambda x: x.get('hybrid_score', 0),
            reverse=True
        )

        return sorted_results[:top_k]


def get_hybrid_searcher(**kwargs) -> Any:
    """
    Factory function to get appropriate hybrid searcher.

    Returns:
        HybridSearcher instance
    """
    if BM25_AVAILABLE:
        return HybridSearcher(**kwargs)
    else:
        logging.info("Using simple keyword-based hybrid search")
        return SimpleHybridSearcher()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test documents
    test_docs = [
        {"metadata": {"content": "CVD는 화학 기상 증착 공정입니다."}, "score": 0.9},
        {"metadata": {"content": "PVD는 물리 기상 증착 공정입니다."}, "score": 0.85},
        {"metadata": {"content": "반도체 제조에서 CVD가 널리 사용됩니다."}, "score": 0.8},
        {"metadata": {"content": "CVD 공정의 장점은 균일한 박막 형성입니다."}, "score": 0.75},
        {"metadata": {"content": "PECVD는 플라즈마 enhanced CVD입니다."}, "score": 0.7},
    ]

    query = "CVD 공정이란?"

    # Test hybrid search
    searcher = get_hybrid_searcher()

    print("=== Hybrid Search Test ===")
    results = searcher.search(query, test_docs, top_k=3)

    for i, doc in enumerate(results):
        content = doc.get('metadata', {}).get('content', '')[:50]
        rrf_score = doc.get('rrf_score', 0)
        vector_rank = doc.get('vector_rank', 'N/A')
        bm25_rank = doc.get('bm25_rank', 'N/A')
        print(f"{i+1}. RRF: {rrf_score:.4f} (V:{vector_rank}, B:{bm25_rank}) - {content}...")
