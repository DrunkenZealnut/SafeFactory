"""
Context Generator Module
Generates contextual prefixes for chunks using LLM (Anthropic Contextual Retrieval).

Each chunk gets a 2-3 sentence description of its role within the full document,
which is prepended before embedding and BM25 indexing. This reduces retrieval
failure rate by 49-67% according to Anthropic's research.

Uses prompt caching: the document portion is cached across calls for same-document
chunks, reducing cost by ~90%.
"""

import hashlib
import logging
import os
import sqlite3
import time
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Domain-specific contextual retrieval prompts
DOMAIN_CONTEXT_PROMPTS = {
    'laborlaw': (
        "이 청크가 어떤 법률/시행령/판례의 어떤 조항이나 쟁점에 해당하며, "
        "문서 전체에서 어떤 맥락인지 검색 품질 향상을 위해 간결하게 설명해주세요. "
        "법률명과 조항 번호를 반드시 포함하세요."
    ),
    'semiconductor': (
        "이 청크가 어떤 반도체 공정/학습단위에 해당하며, "
        "문서 전체에서 어떤 맥락인지 검색 품질 향상을 위해 간결하게 설명해주세요. "
        "NCS 능력단위명과 학습 주제를 포함하세요."
    ),
    'msds': (
        "이 청크가 어떤 화학물질의 어떤 안전 정보(유해성/취급/응급조치 등)에 해당하는지 "
        "검색 품질 향상을 위해 간결하게 설명해주세요. "
        "물질명과 CAS 번호를 포함하세요."
    ),
    'field-training': (
        "이 청크가 어떤 현장 안전교육/카드북의 어떤 주제에 해당하며, "
        "문서 전체에서 어떤 맥락인지 검색 품질 향상을 위해 간결하게 설명해주세요. "
        "장비명 또는 위험요인 유형을 포함하세요."
    ),
    'safeguide': (
        "이 청크가 어떤 안전보건 가이드의 어떤 주제에 해당하며, "
        "문서 전체에서 어떤 맥락인지 검색 품질 향상을 위해 간결하게 설명해주세요. "
        "안전 규정명 또는 관련 법조항을 포함하세요."
    ),
}

DEFAULT_CONTEXT_PROMPT = (
    "이 청크가 문서 전체에서 어떤 맥락인지 "
    "검색 품질 향상을 위해 간결하게 설명해주세요. "
    "맥락 설명만 답변하세요."
)


class ContextGenerator:
    """
    Generates contextual prefixes for document chunks using LLM.

    Implements Anthropic's Contextual Retrieval technique:
    - Sends full document + individual chunk to LLM
    - LLM generates 2-3 sentence context describing the chunk's role
    - Context is prepended to chunk before embedding

    Features:
    - Prompt caching for cost optimization (same document = cached)
    - SQLite-based result cache to avoid re-generation
    - Domain-specific prompts for 5 SafeFactory domains
    - Long document handling (truncation for >100K tokens)
    """

    MAX_DOC_TOKENS = 100_000   # Max tokens for full document in prompt
    MAX_CHUNK_TOKENS = 2_000   # Max chunk tokens in prompt
    CACHE_DB_PATH = 'instance/context_cache.db'

    def __init__(
        self,
        api_key: str,
        model: str = 'claude-haiku-4-5-20251001',
        cache_enabled: bool = True,
        max_concurrent: int = 10,
    ):
        """
        Args:
            api_key: Anthropic API key
            model: Claude model for context generation
            cache_enabled: Whether to use SQLite result cache
            max_concurrent: Max concurrent LLM calls per document batch
        """
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.cache_enabled = cache_enabled
        self.max_concurrent = max_concurrent
        self._cache_conn = None

        # Cost tracking
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cache_read_tokens = 0
        self._total_cache_create_tokens = 0
        self._call_count = 0
        self._cache_hits = 0

        if cache_enabled:
            self._init_cache()

    def _init_cache(self):
        """Initialize SQLite cache for contextual prefixes."""
        os.makedirs(os.path.dirname(self.CACHE_DB_PATH), exist_ok=True)
        self._cache_conn = sqlite3.connect(self.CACHE_DB_PATH)
        self._cache_conn.execute("""
            CREATE TABLE IF NOT EXISTS context_cache (
                cache_key TEXT PRIMARY KEY,
                context TEXT NOT NULL,
                model TEXT NOT NULL,
                domain TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now')),
                tokens_in INTEGER DEFAULT 0,
                tokens_out INTEGER DEFAULT 0
            )
        """)
        self._cache_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_context_cache_domain ON context_cache(domain)"
        )
        self._cache_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_context_cache_created ON context_cache(created_at)"
        )
        self._cache_conn.commit()

    def _get_cache_key(self, doc_hash: str, chunk_hash: str, domain: str) -> str:
        """Generate cache key from document + chunk + domain hashes."""
        combined = f"{doc_hash}:{chunk_hash}:{domain}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _get_cached(self, cache_key: str) -> Optional[str]:
        """Look up cached contextual prefix."""
        if not self._cache_conn:
            return None
        row = self._cache_conn.execute(
            "SELECT context FROM context_cache WHERE cache_key = ?",
            (cache_key,)
        ).fetchone()
        if row:
            self._cache_hits += 1
            return row[0]
        return None

    def _set_cached(self, cache_key: str, context: str, domain: str = '',
                    tokens_in: int = 0, tokens_out: int = 0):
        """Store contextual prefix in cache."""
        if not self._cache_conn:
            return
        self._cache_conn.execute(
            """INSERT OR REPLACE INTO context_cache
               (cache_key, context, model, domain, tokens_in, tokens_out)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (cache_key, context, self.model, domain, tokens_in, tokens_out)
        )
        self._cache_conn.commit()

    def _estimate_tokens(self, text: str) -> int:
        """Approximate token count: len(text) // 3 for Korean."""
        return len(text) // 3

    def _truncate_document(self, document: str, max_tokens: int = None) -> str:
        """Truncate document to fit within token budget.
        For very long documents, keeps first 80% + last 20% of budget."""
        max_tokens = max_tokens or self.MAX_DOC_TOKENS
        est_tokens = self._estimate_tokens(document)

        if est_tokens <= max_tokens:
            return document

        # Strategy: keep first 80% + last 20% of budget
        char_budget = max_tokens * 3  # ~3 chars per token for Korean
        first_part = int(char_budget * 0.8)
        last_part = int(char_budget * 0.2)

        return (
            document[:first_part]
            + "\n\n[... 중간 내용 생략 ...]\n\n"
            + document[-last_part:]
        )

    def _call_llm(self, truncated_doc: str, chunk: str, domain_instruction: str) -> str:
        """Make a single LLM call with prompt caching."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"<document>\n{truncated_doc}\n</document>",
                        "cache_control": {"type": "ephemeral"}
                    },
                    {
                        "type": "text",
                        "text": (
                            f"다음은 위 문서에서 발췌한 청크입니다:\n"
                            f"<chunk>\n{chunk[:self.MAX_CHUNK_TOKENS * 3]}\n</chunk>\n\n"
                            f"{domain_instruction}\n"
                            f"맥락 설명만 답변하세요."
                        )
                    }
                ]
            }
        ]

        response = self.client.messages.create(
            model=self.model,
            max_tokens=200,
            messages=messages,
        )

        # Track usage
        usage = response.usage
        self._total_input_tokens += usage.input_tokens
        self._total_output_tokens += usage.output_tokens
        self._total_cache_read_tokens += getattr(usage, 'cache_read_input_tokens', 0)
        self._total_cache_create_tokens += getattr(usage, 'cache_creation_input_tokens', 0)
        self._call_count += 1

        return response.content[0].text.strip()

    def generate_context(
        self,
        document: str,
        chunk: str,
        domain: str = '',
        metadata: Optional[Dict] = None,
    ) -> str:
        """Generate contextual prefix for a single chunk.

        Args:
            document: Full document text
            chunk: Individual chunk text
            domain: Domain key for prompt selection
            metadata: Additional metadata hints

        Returns:
            Contextual prefix string (2-3 sentences)
        """
        doc_hash = hashlib.md5(document[:5000].encode()).hexdigest()
        chunk_hash = hashlib.md5(chunk.encode()).hexdigest()

        # 1. Check cache
        cache_key = self._get_cache_key(doc_hash, chunk_hash, domain)
        if self.cache_enabled:
            cached = self._get_cached(cache_key)
            if cached:
                return cached

        # 2. Prepare document (truncate if needed)
        truncated_doc = self._truncate_document(document)

        # 3. Select domain-specific prompt
        domain_instruction = DOMAIN_CONTEXT_PROMPTS.get(domain, DEFAULT_CONTEXT_PROMPT)

        # 4. Call LLM
        context = self._call_llm(truncated_doc, chunk, domain_instruction)

        # 5. Cache result
        if self.cache_enabled:
            self._set_cached(
                cache_key, context, domain=domain,
                tokens_in=self._total_input_tokens,
                tokens_out=self._total_output_tokens,
            )

        return context

    def generate_batch(
        self,
        document: str,
        chunks: List[str],
        domain: str = '',
        metadata: Optional[Dict] = None,
    ) -> List[str]:
        """Generate contextual prefixes for all chunks from one document.

        Uses prompt caching: the document is sent once via cache_control,
        then each chunk is processed with the cached document context.

        Args:
            document: Full document text
            chunks: List of chunk texts
            domain: Domain key for prompt selection
            metadata: Additional metadata hints

        Returns:
            List of contextual prefix strings, aligned with chunks
        """
        results: List[Optional[str]] = [None] * len(chunks)
        doc_hash = hashlib.md5(document[:5000].encode()).hexdigest()
        truncated_doc = self._truncate_document(document)
        domain_instruction = DOMAIN_CONTEXT_PROMPTS.get(domain, DEFAULT_CONTEXT_PROMPT)

        # Check cache for each chunk first
        uncached_indices = []
        for i, chunk in enumerate(chunks):
            chunk_hash = hashlib.md5(chunk.encode()).hexdigest()
            cache_key = self._get_cache_key(doc_hash, chunk_hash, domain)
            cached = self._get_cached(cache_key) if self.cache_enabled else None
            if cached:
                results[i] = cached
            else:
                uncached_indices.append(i)

        if not uncached_indices:
            logger.info("[ContextGenerator] All %d chunks cached", len(chunks))
            return results  # All cached

        logger.info(
            "[ContextGenerator] Processing %d/%d uncached chunks (domain=%s)",
            len(uncached_indices), len(chunks), domain or 'default'
        )

        # Process uncached chunks sequentially (prompt caching benefits sequential calls)
        for idx in uncached_indices:
            chunk = chunks[idx]
            try:
                context = self._call_llm(truncated_doc, chunk, domain_instruction)
                results[idx] = context
                # Cache it
                if self.cache_enabled:
                    chunk_hash = hashlib.md5(chunk.encode()).hexdigest()
                    cache_key = self._get_cache_key(doc_hash, chunk_hash, domain)
                    self._set_cached(cache_key, context, domain=domain)
            except Exception as e:
                logger.warning("Context generation failed for chunk %d: %s", idx, e)
                results[idx] = ''  # Empty prefix on failure

        return results

    def get_stats(self) -> Dict:
        """Return cache statistics and cost tracking."""
        # Haiku pricing (per million tokens)
        INPUT_PRICE = 0.80   # $/MTok
        OUTPUT_PRICE = 4.00  # $/MTok
        CACHE_READ_PRICE = 0.08  # $/MTok
        CACHE_CREATE_PRICE = 1.00  # $/MTok

        regular_input = self._total_input_tokens - self._total_cache_read_tokens - self._total_cache_create_tokens
        estimated_cost = (
            (max(0, regular_input) / 1_000_000) * INPUT_PRICE
            + (self._total_output_tokens / 1_000_000) * OUTPUT_PRICE
            + (self._total_cache_read_tokens / 1_000_000) * CACHE_READ_PRICE
            + (self._total_cache_create_tokens / 1_000_000) * CACHE_CREATE_PRICE
        )

        stats = {
            'llm_calls': self._call_count,
            'cache_hits': self._cache_hits,
            'total_input_tokens': self._total_input_tokens,
            'total_output_tokens': self._total_output_tokens,
            'cache_read_tokens': self._total_cache_read_tokens,
            'cache_create_tokens': self._total_cache_create_tokens,
            'estimated_cost_usd': round(estimated_cost, 4),
        }

        # Add SQLite cache stats
        if self._cache_conn:
            row = self._cache_conn.execute("SELECT COUNT(*) FROM context_cache").fetchone()
            stats['sqlite_cache_entries'] = row[0] if row else 0

        return stats

    def close(self):
        """Close SQLite cache connection."""
        if self._cache_conn:
            self._cache_conn.close()
            self._cache_conn = None
