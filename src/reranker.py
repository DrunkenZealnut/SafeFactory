"""
Reranker Module
Implements cross-encoder reranking and MMR for diversity.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Try to import sentence-transformers for cross-encoder
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    logging.warning("sentence-transformers not installed. Cross-encoder reranking disabled.")


class Reranker:
    """
    Reranks retrieved documents using cross-encoder models.

    Cross-encoders process query-document pairs together, providing
    more accurate relevance scores than bi-encoders (separate embeddings).
    """

    # Available cross-encoder models
    MODELS = {
        "ms-marco-minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "ms-marco-electra": "cross-encoder/ms-marco-electra-base",
        "multilingual": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"  # Multilingual
    }

    def __init__(
        self,
        model_name: str = "multilingual",
        device: str = "cpu",
        max_length: int = 512
    ):
        """
        Initialize the Reranker.

        Args:
            model_name: Cross-encoder model to use
            device: Device for inference (cpu/cuda)
            max_length: Maximum sequence length
        """
        self.device = device
        self.max_length = max_length
        self.model = None

        if CROSS_ENCODER_AVAILABLE:
            try:
                model_path = self.MODELS.get(model_name, model_name)
                self.model = CrossEncoder(
                    model_path,
                    max_length=max_length,
                    device=device
                )
                logging.info(f"Loaded cross-encoder model: {model_path}")
            except Exception as e:
                logging.error(f"Failed to load cross-encoder model: {e}")
                self.model = None

    def rerank(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        top_k: int = 10,
        content_key: str = "content"
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using cross-encoder.

        Args:
            query: User query
            docs: List of documents to rerank
            top_k: Number of documents to return
            content_key: Key for document content

        Returns:
            Reranked list of documents
        """
        if not docs:
            return []

        if not self.model:
            logging.warning("Cross-encoder not available, returning original order")
            return docs[:top_k]

        # Prepare query-document pairs
        pairs = []
        for doc in docs:
            content = doc.get(content_key) or doc.get('metadata', {}).get('content', '')
            if content:
                # Truncate content to fit max_length
                content = content[:self.max_length * 4]  # Approximate char limit
                pairs.append([query, content])
            else:
                pairs.append([query, ""])

        try:
            # Get cross-encoder scores
            scores = self.model.predict(pairs, show_progress_bar=False)

            # Add rerank scores to documents
            for i, doc in enumerate(docs):
                doc['rerank_score'] = float(scores[i])
                doc['original_score'] = doc.get('score', 0)

            # Sort by rerank score
            reranked = sorted(docs, key=lambda x: x.get('rerank_score', 0), reverse=True)

            return reranked[:top_k]

        except Exception as e:
            logging.error(f"Reranking failed: {e}")
            return docs[:top_k]

    def mmr(
        self,
        query_embedding: List[float],
        docs: List[Dict[str, Any]],
        doc_embeddings: List[List[float]],
        top_k: int = 10,
        lambda_val: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Maximal Marginal Relevance (MMR) for diversity.

        Balances relevance and diversity in the selected documents.

        Args:
            query_embedding: Query embedding vector
            docs: List of documents
            doc_embeddings: List of document embeddings
            top_k: Number of documents to return
            lambda_val: Balance between relevance (1) and diversity (0)

        Returns:
            Selected documents with diversity
        """
        if not docs or not doc_embeddings:
            return []

        if len(docs) != len(doc_embeddings):
            logging.error("Number of docs and embeddings must match")
            return docs[:top_k]

        query_embedding = np.array(query_embedding)
        doc_embeddings = np.array(doc_embeddings)

        # Normalize embeddings
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        doc_norms = doc_embeddings / (np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-8)

        # Compute query-document similarities
        query_doc_sim = np.dot(doc_norms, query_norm)

        selected_indices = []
        remaining_indices = list(range(len(docs)))

        while len(selected_indices) < top_k and remaining_indices:
            mmr_scores = []

            for idx in remaining_indices:
                # Relevance to query
                relevance = query_doc_sim[idx]

                # Maximum similarity to already selected documents
                if selected_indices:
                    selected_embeddings = doc_norms[selected_indices]
                    similarities = np.dot(selected_embeddings, doc_norms[idx])
                    max_sim = np.max(similarities)
                else:
                    max_sim = 0

                # MMR score
                mmr_score = lambda_val * relevance - (1 - lambda_val) * max_sim
                mmr_scores.append((idx, mmr_score))

            # Select document with highest MMR score
            best_idx = max(mmr_scores, key=lambda x: x[1])[0]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        # Return selected documents with MMR scores
        result = []
        for i, idx in enumerate(selected_indices):
            doc = docs[idx].copy()
            doc['mmr_rank'] = i + 1
            doc['mmr_score'] = query_doc_sim[idx]
            result.append(doc)

        return result

    def hybrid_rerank(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        top_k: int = 10,
        rerank_weight: float = 0.7,
        original_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Combine cross-encoder and original scores for final ranking.

        Args:
            query: User query
            docs: List of documents
            top_k: Number of documents to return
            rerank_weight: Weight for cross-encoder score
            original_weight: Weight for original retrieval score

        Returns:
            Documents sorted by combined score
        """
        # First, rerank with cross-encoder
        reranked = self.rerank(query, docs, top_k=len(docs))

        if not self.model:
            return reranked[:top_k]

        # Normalize scores
        rerank_scores = [d.get('rerank_score', 0) for d in reranked]
        original_scores = [d.get('original_score', d.get('score', 0)) for d in reranked]

        # Min-max normalization
        def normalize(scores):
            if not scores:
                return []
            min_s, max_s = min(scores), max(scores)
            if max_s == min_s:
                return [0.5] * len(scores)
            return [(s - min_s) / (max_s - min_s) for s in scores]

        norm_rerank = normalize(rerank_scores)
        norm_original = normalize(original_scores)

        # Compute combined scores
        for i, doc in enumerate(reranked):
            doc['combined_score'] = (
                rerank_weight * norm_rerank[i] +
                original_weight * norm_original[i]
            )

        # Sort by combined score
        result = sorted(reranked, key=lambda x: x.get('combined_score', 0), reverse=True)

        return result[:top_k]


class PineconeReranker:
    """
    Reranker using Pinecone Inference API.
    Uses bge-reranker-v2-m3 (multilingual, 1024 tokens) for high-quality
    cross-encoder reranking without local GPU/CPU overhead.
    """

    def __init__(self, pinecone_client, model: str = "bge-reranker-v2-m3"):
        self.pc = pinecone_client
        self.model = model

    def rerank(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        top_k: int = 10,
        content_key: str = "content"
    ) -> List[Dict[str, Any]]:
        if not docs:
            return []

        # Extract content from each doc
        documents = []
        for doc in docs:
            content = doc.get(content_key) or doc.get('metadata', {}).get('content', '')
            # bge-reranker-v2-m3 supports up to 1024 tokens (~3000 chars for Korean)
            documents.append(content[:3000] if content else "")

        try:
            result = self.pc.inference.rerank(
                model=self.model,
                query=query,
                documents=documents,
                top_n=min(top_k, len(docs)),
                return_documents=False
            )

            # Map scores back to original docs
            reranked = []
            for item in result.data:
                idx = item['index']
                doc = docs[idx].copy() if isinstance(docs[idx], dict) else dict(docs[idx])
                doc['rerank_score'] = float(item['score'])
                doc['original_score'] = doc.get('score', 0)
                reranked.append(doc)

            return reranked

        except Exception as e:
            logging.error(f"Pinecone reranking failed: {e}")
            return docs[:top_k]

    def hybrid_rerank(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        top_k: int = 10,
        rerank_weight: float = 0.7,
        original_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Combine Pinecone rerank scores with original retrieval scores."""
        reranked = self.rerank(query, docs, top_k=len(docs))
        if not reranked or not any(d.get('rerank_score') for d in reranked):
            return reranked[:top_k]

        rerank_scores = [d.get('rerank_score', 0) for d in reranked]
        original_scores = [d.get('original_score', d.get('score', 0)) for d in reranked]

        def normalize(scores):
            if not scores:
                return []
            min_s, max_s = min(scores), max(scores)
            if max_s == min_s:
                return [0.5] * len(scores)
            return [(s - min_s) / (max_s - min_s) for s in scores]

        norm_rerank = normalize(rerank_scores)
        norm_original = normalize(original_scores)

        for i, doc in enumerate(reranked):
            doc['combined_score'] = (
                rerank_weight * norm_rerank[i] +
                original_weight * norm_original[i]
            )

        result = sorted(reranked, key=lambda x: x.get('combined_score', 0), reverse=True)
        return result[:top_k]


class LightweightReranker:
    """
    Lightweight reranker using keyword matching when cross-encoder is unavailable.
    """

    def __init__(self):
        pass

    def rerank(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Rerank using keyword overlap scoring.

        Args:
            query: User query
            docs: List of documents
            top_k: Number of documents to return

        Returns:
            Reranked documents
        """
        if not docs:
            return []

        query_words = set(query.lower().split())

        for doc in docs:
            content = doc.get('metadata', {}).get('content', '')
            content_words = set(content.lower().split())

            # Count exact keyword matches
            exact_matches = len(query_words & content_words)

            # Count partial matches (substrings)
            partial_matches = sum(
                1 for qw in query_words
                for cw in content_words
                if qw in cw or cw in qw
            )

            # Combine with original score
            original_score = doc.get('score', 0)
            keyword_score = (exact_matches * 2 + partial_matches) / (len(query_words) + 1)

            doc['keyword_score'] = keyword_score
            doc['combined_score'] = original_score * 0.6 + keyword_score * 0.4

        # Sort by combined score
        sorted_docs = sorted(docs, key=lambda x: x.get('combined_score', 0), reverse=True)

        return sorted_docs[:top_k]


def get_reranker(
    use_cross_encoder: bool = True,
    model_name: str = "multilingual",
    pinecone_client=None
) -> Any:
    """
    Factory function to get appropriate reranker.

    Priority: PineconeReranker > local CrossEncoder > LightweightReranker

    Args:
        use_cross_encoder: Whether to use cross-encoder
        model_name: Cross-encoder model name
        pinecone_client: Pinecone client instance for API-based reranking

    Returns:
        Reranker instance
    """
    use_local = os.environ.get('USE_LOCAL_RERANKER', '').lower() in ('true', '1', 'yes')

    # Try Pinecone Inference reranker first (unless forced local)
    if pinecone_client and not use_local:
        try:
            reranker = PineconeReranker(pinecone_client)
            # Quick validation: check if inference API is accessible
            logging.info("Using Pinecone Inference reranker (bge-reranker-v2-m3)")
            return reranker
        except Exception as e:
            logging.warning(f"Pinecone reranker init failed: {e}, falling back to local")

    # Fallback to local cross-encoder
    if use_cross_encoder and CROSS_ENCODER_AVAILABLE:
        logging.info("Using local cross-encoder reranker")
        return Reranker(model_name=model_name)

    logging.info("Using lightweight keyword-based reranker")
    return LightweightReranker()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test documents
    test_docs = [
        {"metadata": {"content": "CVD는 화학 기상 증착 공정입니다."}, "score": 0.9},
        {"metadata": {"content": "PVD는 물리 기상 증착 공정입니다."}, "score": 0.85},
        {"metadata": {"content": "반도체 제조에서 CVD가 널리 사용됩니다."}, "score": 0.8},
        {"metadata": {"content": "CVD 공정의 장점은 균일한 박막 형성입니다."}, "score": 0.75},
    ]

    query = "CVD 공정이란?"

    # Test reranker
    reranker = get_reranker(use_cross_encoder=True)

    print("=== Reranking Test ===")
    reranked = reranker.rerank(query, test_docs, top_k=3)

    for i, doc in enumerate(reranked):
        content = doc.get('metadata', {}).get('content', '')[:50]
        score = doc.get('rerank_score', doc.get('combined_score', 'N/A'))
        print(f"{i+1}. Score: {score:.4f if isinstance(score, float) else score} - {content}...")
