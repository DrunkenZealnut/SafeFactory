"""
Context Optimizer Module
Implements context optimization techniques for improved LLM responses.
"""

import os
import re
import certifi
import httpx
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from openai import OpenAI

# Set SSL certificate environment variables
os.environ.setdefault('SSL_CERT_FILE', certifi.where())
os.environ.setdefault('REQUESTS_CA_BUNDLE', certifi.where())


@dataclass
class OptimizedDoc:
    """Represents an optimized document chunk."""
    content: str
    source_file: str
    score: float
    relevance_score: float  # 0-1 relevance to query
    metadata: Dict[str, Any]


class ContextOptimizer:
    """
    Optimizes retrieved context for better LLM responses.

    Techniques:
    - Deduplication: Remove semantically similar content
    - Relevant extraction: Extract only relevant sentences
    - Reordering: Optimize document order for LLM attention
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        similarity_threshold: float = 0.85,
        model: str = "gpt-4o-mini",
        provider: str = "openai",
    ):
        """
        Initialize the ContextOptimizer.

        Args:
            openai_api_key: API key for the configured provider (optional)
            similarity_threshold: Threshold for considering content as duplicate
            model: Model for relevance extraction
            provider: LLM provider ('openai', 'gemini', 'anthropic')
        """
        self.similarity_threshold = similarity_threshold
        self.model = model
        self.provider = provider

        self._has_client = False

        if not openai_api_key:
            return

        if provider == 'gemini':
            from google import genai
            self._gemini = genai.Client(api_key=openai_api_key)
            self._has_client = True
        elif provider == 'anthropic':
            import anthropic
            self._anthropic = anthropic.Anthropic(api_key=openai_api_key)
            self._has_client = True
        else:
            self._http_client = httpx.Client(verify=certifi.where())
            self.client = OpenAI(api_key=openai_api_key, http_client=self._http_client, timeout=60.0)
            self._has_client = True

    def close(self):
        """Close the underlying HTTP client."""
        if hasattr(self, '_http_client'):
            try:
                self._http_client.close()
            except Exception:
                pass

    def _chat_complete(self, messages, temperature=0.1, max_tokens=500):
        """Dispatch chat completion to the configured provider."""
        if self.provider == 'gemini':
            from google.genai import types
            system_msg = next((m['content'] for m in messages if m['role'] == 'system'), '')
            user_msg = next((m['content'] for m in messages if m['role'] == 'user'), '')
            config = types.GenerateContentConfig(
                system_instruction=system_msg or None,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            resp = self._gemini.models.generate_content(
                model=self.model, contents=user_msg, config=config,
            )
            return resp.text
        elif self.provider == 'anthropic':
            system_msg = next((m['content'] for m in messages if m['role'] == 'system'), '')
            user_msgs = [m for m in messages if m['role'] != 'system']
            resp = self._anthropic.messages.create(
                model=self.model,
                system=system_msg or None,
                messages=user_msgs,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.content[0].text
        else:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content

    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """
        Compute simple text similarity using Jaccard index.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1)
        """
        # Tokenize
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _compute_ngram_similarity(self, text1: str, text2: str, n: int = 3) -> float:
        """
        Compute n-gram based similarity.

        Args:
            text1: First text
            text2: Second text
            n: N-gram size

        Returns:
            Similarity score (0-1)
        """
        def get_ngrams(text: str, n: int) -> set:
            text = text.lower()
            return set(text[i:i+n] for i in range(len(text) - n + 1))

        ngrams1 = get_ngrams(text1, n)
        ngrams2 = get_ngrams(text2, n)

        if not ngrams1 or not ngrams2:
            return 0.0

        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)

        return intersection / union if union > 0 else 0.0

    def deduplicate(
        self,
        docs: List[Dict[str, Any]],
        content_key: str = "content"
    ) -> List[Dict[str, Any]]:
        """
        Remove semantically similar documents.

        Uses combined Jaccard and n-gram similarity to identify duplicates.
        Keeps the document with the higher score.

        Args:
            docs: List of documents with content and metadata
            content_key: Key for content in document dict

        Returns:
            Deduplicated list of documents
        """
        if not docs:
            return []

        # Sort by best available score (rerank > combined > rrf > score) to keep highest scored duplicates
        sorted_docs = sorted(
            docs,
            key=lambda x: x.get('rerank_score', x.get('combined_score', x.get('rrf_score', x.get('score', 0)))),
            reverse=True
        )

        unique_docs = []
        seen_contents = []

        for doc in sorted_docs:
            content = doc.get(content_key, '') or doc.get('metadata', {}).get('content', '')

            if not content:
                continue

            is_duplicate = False
            for seen_content in seen_contents:
                # Combine different similarity metrics
                jaccard_sim = self._compute_text_similarity(content, seen_content)
                ngram_sim = self._compute_ngram_similarity(content, seen_content)
                combined_sim = (jaccard_sim + ngram_sim) / 2

                if combined_sim >= self.similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_docs.append(doc)
                seen_contents.append(content)

        return unique_docs

    def extract_relevant_sentences(
        self,
        query: str,
        doc_content: str,
        max_sentences: int = 5
    ) -> str:
        """
        Extract only the most relevant sentences from a document.

        Uses LLM to identify and extract relevant sentences.

        Args:
            query: User query
            doc_content: Document content
            max_sentences: Maximum number of sentences to extract

        Returns:
            Extracted relevant content
        """
        if not self._has_client:
            # Fallback: return truncated content
            return doc_content[:1000]

        prompt = f"""다음 문서에서 질문과 가장 관련있는 문장들만 추출해주세요.

질문: {query}

문서:
{doc_content[:2000]}

규칙:
1. 최대 {max_sentences}개의 문장만 추출
2. 가장 관련성 높은 문장 순서대로 정렬
3. 추출된 문장들만 반환 (설명 없이)
4. 관련 없는 내용이면 "관련 내용 없음" 반환

관련 문장들:"""

        try:
            extracted = self._chat_complete(
                [{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
            ).strip()

            # If no relevant content found
            if "관련 내용 없음" in extracted or not extracted:
                return doc_content[:500]

            return extracted

        except Exception as e:
            print(f"Sentence extraction failed: {e}")
            return doc_content[:1000]

    def reorder_for_llm(
        self,
        docs: List[Dict[str, Any]],
        strategy: str = "lost_in_middle"
    ) -> List[Dict[str, Any]]:
        """
        Reorder documents to optimize LLM attention.

        The "Lost in the Middle" phenomenon shows LLMs pay more attention
        to content at the beginning and end. This reordering places
        the most relevant content in those positions.

        Args:
            docs: List of documents sorted by relevance
            strategy: Reordering strategy
                - "lost_in_middle": Best at start and end
                - "best_first": Keep original order (highest first)
                - "best_last": Reverse order (highest last)

        Returns:
            Reordered list of documents
        """
        if not docs or len(docs) <= 2:
            return docs

        if strategy == "best_first":
            return docs

        if strategy == "best_last":
            return list(reversed(docs))

        if strategy == "lost_in_middle":
            # Interleave: best at start and end, worst in middle
            n = len(docs)
            reordered = [None] * n

            # Place even-indexed items from the start
            # Place odd-indexed items from the end
            start_idx = 0
            end_idx = n - 1

            for i, doc in enumerate(docs):
                if i % 2 == 0:
                    reordered[start_idx] = doc
                    start_idx += 1
                else:
                    reordered[end_idx] = doc
                    end_idx -= 1

            return [doc for doc in reordered if doc is not None]

        return docs

    def compute_relevance_scores(
        self,
        query: str,
        docs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Compute relevance scores for documents using keyword matching.

        Args:
            query: User query
            docs: List of documents

        Returns:
            Documents with added relevance_score field
        """
        query_words = set(query.lower().split())

        for doc in docs:
            content = doc.get('metadata', {}).get('content', '')
            if not content:
                doc['relevance_score'] = 0.0
                continue

            content_words = set(content.lower().split())
            overlap = len(query_words & content_words)
            doc['relevance_score'] = overlap / len(query_words) if query_words else 0.0

        return docs

    def optimize(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        dedupe: bool = True,
        extract_relevant: bool = False,
        reorder: bool = True,
        max_docs: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Apply all optimization techniques to the retrieved documents.

        Args:
            query: User query
            docs: Retrieved documents
            dedupe: Whether to deduplicate
            extract_relevant: Whether to extract relevant sentences
            reorder: Whether to reorder for LLM attention
            max_docs: Maximum documents to return

        Returns:
            Optimized list of documents
        """
        if not docs:
            return []

        result = docs.copy()

        # Step 1: Deduplicate
        if dedupe:
            result = self.deduplicate(result)

        # Step 2: Limit to max_docs
        result = result[:max_docs]

        # Step 3: Extract relevant sentences (expensive, optional)
        if extract_relevant and self._has_client:
            for doc in result:
                content = doc.get('metadata', {}).get('content', '')
                if content:
                    extracted = self.extract_relevant_sentences(query, content)
                    doc['metadata']['original_content'] = content
                    doc['metadata']['content'] = extracted

        # Step 4: Compute relevance scores
        result = self.compute_relevance_scores(query, result)

        # Step 5: Reorder for LLM attention
        if reorder:
            result = self.reorder_for_llm(result, strategy="lost_in_middle")

        return result


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    optimizer = ContextOptimizer(api_key)

    # Test documents
    test_docs = [
        {"content": "CVD는 화학 기상 증착 공정입니다.", "score": 0.9, "metadata": {"source": "doc1"}},
        {"content": "CVD 공정은 화학 기상 증착 방식입니다.", "score": 0.85, "metadata": {"source": "doc2"}},
        {"content": "PVD는 물리 기상 증착 공정입니다.", "score": 0.8, "metadata": {"source": "doc3"}},
        {"content": "반도체 제조에서 CVD가 사용됩니다.", "score": 0.75, "metadata": {"source": "doc4"}},
    ]

    print("=== Deduplication Test ===")
    deduplicated = optimizer.deduplicate(test_docs)
    print(f"Original: {len(test_docs)}, After dedup: {len(deduplicated)}")
    for doc in deduplicated:
        print(f"  - {doc['content'][:50]}...")

    print("\n=== Reordering Test ===")
    reordered = optimizer.reorder_for_llm(test_docs, strategy="lost_in_middle")
    print("Reordered scores:", [doc['score'] for doc in reordered])
