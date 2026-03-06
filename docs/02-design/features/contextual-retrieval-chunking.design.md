# Design: Anthropic Contextual Retrieval 기반 청킹 구현

> Plan 문서: `docs/01-plan/features/contextual-retrieval-chunking.plan.md`

---

## 1. Overview

Anthropic Contextual Retrieval을 SafeFactory 인제스천 파이프라인에 통합하여, 각 청크에 LLM 기반 맥락 설명(contextual prefix)을 생성하고 이를 임베딩과 BM25 인덱스에 자동 반영한다.

### 1.1 핵심 변경 포인트

| 구분 | 현재 (As-Is) | 변경 후 (To-Be) |
|------|-------------|-----------------|
| 청크 프리픽스 | 정적 `[문서: X \| 섹션: Y]` | LLM 생성 2~3문장 맥락 설명 |
| 프리픽스 생성 | `SemanticChunker._add_contextual_prefix()` | `ContextGenerator.generate_context()` |
| 임베딩 입력 | 정적 프리픽스 + 본문 | contextual prefix + 본문 |
| BM25 corpus | Pinecone `content` 필드 (정적 프리픽스) | Pinecone `content` 필드 (contextual prefix) |
| 비용 | $0 (정적) | ~$3/10K 청크 (Haiku + Caching) |
| CLI 플래그 | 없음 | `--contextual` |

### 1.2 아키텍처 다이어그램

```
Before:
  FileLoader → SemanticChunker.chunk_text()
                    └→ _add_contextual_prefix(static)
               → EmbeddingGenerator → PineconeUploader

After:
  FileLoader → SemanticChunker.chunk_text()
                    └→ ContextGenerator.generate_context(LLM)  ← NEW
                       └→ Prompt Caching (동일 문서 청크 비용 90% 절감)
               → EmbeddingGenerator → PineconeUploader
```

---

## 2. Module Design

### 2.1 `src/context_generator.py` (NEW)

```python
"""
Context Generator Module
Generates contextual prefixes for chunks using LLM (Anthropic Contextual Retrieval).
"""

import hashlib
import json
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
    - Long document handling (sectional fallback for >128K tokens)
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

        if cache_enabled:
            self._init_cache()

    def _init_cache(self):
        """Initialize SQLite cache for contextual prefixes."""
        ...

    def _get_cache_key(self, document_hash: str, chunk_hash: str, domain: str) -> str:
        """Generate cache key from document + chunk + domain hashes."""
        ...

    def _get_cached(self, cache_key: str) -> Optional[str]:
        """Look up cached contextual prefix."""
        ...

    def _set_cached(self, cache_key: str, context: str):
        """Store contextual prefix in cache."""
        ...

    def _estimate_tokens(self, text: str) -> int:
        """Approximate token count: len(text) // 3 for Korean."""
        return len(text) // 3

    def _truncate_document(self, document: str, max_tokens: int = None) -> str:
        """Truncate document to fit within token budget.
        For very long documents, keeps first + last sections."""
        ...

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
        ...

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
        ...

    def get_stats(self) -> Dict:
        """Return cache statistics and cost tracking."""
        ...
```

### 2.2 `generate_context()` — 상세 구현 설계

#### 2.2.1 프롬프트 구조

```python
def generate_context(self, document, chunk, domain='', metadata=None):
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

    # 4. Build messages with prompt caching
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"<document>\n{truncated_doc}\n</document>",
                    "cache_control": {"type": "ephemeral"}  # Prompt Caching
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

    # 5. Call LLM
    response = self.client.messages.create(
        model=self.model,
        max_tokens=200,
        messages=messages,
    )
    context = response.content[0].text.strip()

    # 6. Cache result
    if self.cache_enabled:
        self._set_cached(cache_key, context)

    return context
```

#### 2.2.2 Prompt Caching 동작 원리

```
Document A (50K tokens) → 10 chunks

Call 1: [Document A (50K, NEW)] + [Chunk 1 (500)]     → 50.5K input tokens
Call 2: [Document A (50K, CACHED)] + [Chunk 2 (500)]   → 0.5K input + 50K cached
Call 3: [Document A (50K, CACHED)] + [Chunk 3 (500)]   → 0.5K input + 50K cached
...
Call 10: [Document A (50K, CACHED)] + [Chunk 10 (500)]  → 0.5K input + 50K cached

비용: Call 1 = $0.80/MTok × 50.5K = $0.04
      Call 2-10 = $0.08/MTok × 50K + $0.80/MTok × 0.5K = $0.004 each
      Total = $0.04 + $0.036 = $0.076 (vs $0.40 without caching = 81% 절감)
```

### 2.3 `generate_batch()` — 배치 처리 설계

```python
def generate_batch(self, document, chunks, domain='', metadata=None):
    """Process all chunks from one document with prompt caching."""
    results = []
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
            results.append(cached)
        else:
            results.append(None)  # placeholder
            uncached_indices.append(i)

    if not uncached_indices:
        return results  # All cached

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
                self._set_cached(cache_key, context)
        except Exception as e:
            logger.warning("Context generation failed for chunk %d: %s", idx, e)
            results[idx] = ''  # Empty prefix on failure

    return results
```

### 2.4 SQLite Cache Schema

```sql
CREATE TABLE IF NOT EXISTS context_cache (
    cache_key TEXT PRIMARY KEY,
    context TEXT NOT NULL,
    model TEXT NOT NULL,
    domain TEXT DEFAULT '',
    created_at TEXT DEFAULT (datetime('now')),
    tokens_in INTEGER DEFAULT 0,
    tokens_out INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_context_cache_domain ON context_cache(domain);
CREATE INDEX IF NOT EXISTS idx_context_cache_created ON context_cache(created_at);
```

**캐시 키**: `md5(doc_hash + chunk_hash + domain)`

**무효화**: 동일 문서/청크 조합은 항상 같은 맥락 → 재인제스천 시에도 캐시 유효. 문서 내용 변경 시 doc_hash 변경으로 자동 무효화.

---

## 3. Integration Design

### 3.1 SemanticChunker 통합

#### 변경 파일: `src/semantic_chunker.py`

**변경 1**: `__init__`에 `context_generator` 파라미터 추가

```python
class SemanticChunker:
    def __init__(
        self,
        openai_api_key: str,
        ...
        enable_contextual: bool = True,
        context_generator: Optional['ContextGenerator'] = None,  # NEW
    ):
        ...
        self.context_generator = context_generator
```

**변경 2**: `chunk_text()` 메서드의 Step 4에서 contextual prefix 분기

```python
# In chunk_text(), around line 982
# Add contextual prefix
enhanced_content = segment
if self.context_generator:
    # LLM-based contextual prefix (Anthropic Contextual Retrieval)
    contextual_prefix = self._contextual_prefixes[i]  # pre-generated in batch
    if contextual_prefix:
        enhanced_content = f"{contextual_prefix}\n\n{segment}"
    else:
        # Fallback to static prefix
        enhanced_content = self._add_contextual_prefix(
            segment, document_title, section_title, source_file
        )
elif self.enable_contextual:
    # Legacy static prefix
    enhanced_content = self._add_contextual_prefix(
        segment, document_title, section_title, source_file
    )
```

**변경 3**: `chunk_text()` 시작 부분에서 batch context generation

```python
def chunk_text(self, text, source_file, use_embeddings=True, metadata=None, meta_json=None):
    ...
    # Step 3: Add overlap
    overlapped_segments = self._add_overlap(merged_segments)

    # Step 3.5 (NEW): Generate contextual prefixes in batch
    self._contextual_prefixes = [None] * len(overlapped_segments)
    if self.context_generator and overlapped_segments:
        domain = self._detect_domain(source_file)
        try:
            raw_segments = [seg for seg in merged_segments]  # Pre-overlap segments
            self._contextual_prefixes = self.context_generator.generate_batch(
                document=text,
                chunks=raw_segments,
                domain=domain,
                metadata=metadata,
            )
            # Pad if lengths differ (overlap may change count)
            while len(self._contextual_prefixes) < len(overlapped_segments):
                self._contextual_prefixes.append(None)
        except Exception as e:
            logger.warning("[Contextual Retrieval] Batch generation failed: %s", e)

    # Step 4: Create Chunk objects ... (existing code with modified prefix logic)
```

**변경 4**: 도메인 감지 헬퍼

```python
def _detect_domain(self, source_file: str) -> str:
    """Detect domain from source file path."""
    normalized = unicodedata.normalize('NFC', source_file)
    if '/laborlaw/' in normalized:
        return 'laborlaw'
    if '카드북' in normalized or '/현장실습/' in normalized:
        return 'field-training'
    if '/안전보건공단/' in normalized:
        return 'safeguide'
    if re.search(r'LM\d{10}', source_file):
        return 'semiconductor'
    return ''
```

### 3.2 PineconeAgent 통합

#### 변경 파일: `src/agent.py`

**변경 1**: `__init__`에 `context_generator` 옵셔널 주입

```python
class PineconeAgent:
    def __init__(
        self,
        openai_api_key: str,
        pinecone_api_key: str,
        pinecone_index_name: str,
        ...
        context_generator: Optional['ContextGenerator'] = None,  # NEW
    ):
        ...
        # Pass context_generator to chunker
        self.chunker = SemanticChunker(
            openai_api_key=openai_api_key,
            model=embedding_model,
            max_chunk_tokens=max_chunk_tokens,
            context_generator=context_generator,  # NEW
        )
```

### 3.3 CLI 통합

#### 변경 파일: `main.py`

**변경 1**: `--contextual` 플래그 추가

```python
process_parser.add_argument(
    "--contextual", action="store_true",
    help="Anthropic Contextual Retrieval 활성화 (LLM 기반 청크 맥락 생성)"
)
process_parser.add_argument(
    "--context-model", type=str, default="claude-haiku-4-5-20251001",
    help="Contextual prefix 생성 모델 (기본: claude-haiku-4-5-20251001)"
)
```

**변경 2**: Agent 생성 시 ContextGenerator 주입

```python
if args.contextual:
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    if not anthropic_key:
        print("ERROR: --contextual 사용 시 ANTHROPIC_API_KEY 환경변수 필요")
        sys.exit(1)
    from src.context_generator import ContextGenerator
    context_gen = ContextGenerator(
        api_key=anthropic_key,
        model=args.context_model,
    )
else:
    context_gen = None

agent = PineconeAgent(
    ...,
    context_generator=context_gen,
)
```

### 3.4 환경변수 추가

#### 변경 파일: `.env.example`

```bash
# Contextual Retrieval (optional - for --contextual flag)
ANTHROPIC_API_KEY=       # Anthropic API key for contextual prefix generation
CONTEXT_MODEL=claude-haiku-4-5-20251001   # Model for context generation
CONTEXT_CACHE_ENABLED=true                # SQLite cache for generated prefixes
```

### 3.5 BM25 자동 통합 (변경 불필요)

현재 `HybridSearcher.build_index()` (hybrid_searcher.py:136-153)는:

```python
content = doc.get('metadata', {}).get('content', '')
tokens = self._tokenize(content)
self.corpus.append(tokens)
```

Pinecone의 `content` 메타데이터 필드를 읽어 BM25 corpus를 구성한다.
`PineconeUploader.prepare_vector()`가 `chunk.content`를 `content` 메타데이터에 저장하므로, contextual prefix가 포함된 텍스트가 자동으로 BM25에 반영된다.

**→ HybridSearcher 코드 변경 불필요. Contextual BM25는 재인제스천만으로 자동 적용됨.**

---

## 4. Long Document Strategy

### 4.1 문서 크기별 처리 전략

| 문서 크기 (토큰) | 전략 | 비용 효율 |
|-----------------|------|----------|
| < 100K | 전체 문서를 프롬프트에 포함 | 최적 (prompt caching) |
| 100K ~ 200K | 앞 80K + 뒤 20K 포함 (중간 생략) | 양호 |
| > 200K | 섹션 단위로 분할, 섹션 + 문서 요약 포함 | 합리적 |

### 4.2 `_truncate_document()` 구현

```python
def _truncate_document(self, document: str, max_tokens: int = None):
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
```

---

## 5. Cost Tracking Design

### 5.1 비용 로깅

`ContextGenerator`가 매 LLM 호출마다 토큰 사용량을 추적:

```python
# After each LLM call
usage = response.usage
self._total_input_tokens += usage.input_tokens
self._total_output_tokens += usage.output_tokens
self._total_cache_read_tokens += getattr(usage, 'cache_read_input_tokens', 0)
self._total_cache_create_tokens += getattr(usage, 'cache_creation_input_tokens', 0)
self._call_count += 1
```

### 5.2 비용 리포트

```python
def get_stats(self) -> Dict:
    return {
        'call_count': self._call_count,
        'total_input_tokens': self._total_input_tokens,
        'total_output_tokens': self._total_output_tokens,
        'cache_hit_tokens': self._total_cache_read_tokens,
        'cache_create_tokens': self._total_cache_create_tokens,
        'cache_hit_rate': self._total_cache_read_tokens / max(1, self._total_input_tokens + self._total_cache_read_tokens),
        'estimated_cost_usd': self._estimate_cost(),
        'sqlite_cache_hits': self._sqlite_cache_hits,
    }
```

CLI 인제스천 완료 시 비용 리포트 출력:
```
📊 Contextual Retrieval Stats:
   - LLM 호출: 142회
   - 입력 토큰: 245K (캐시: 210K, 신규: 35K)
   - 출력 토큰: 14K
   - 예상 비용: $0.12
   - 캐시 히트율: 85.7%
```

---

## 6. Implementation Order

### Phase A: Core (MVP)

| 순서 | ID | Task | 변경 파일 | 의존성 |
|:----:|:---:|------|----------|--------|
| 1 | A-1 | `ContextGenerator` 클래스 생성 | `src/context_generator.py` (NEW) | - |
| 2 | A-2 | 도메인별 프롬프트 상수 | `src/context_generator.py` | A-1 |
| 3 | A-3 | `generate_context()` 구현 (단일 청크) | `src/context_generator.py` | A-1 |
| 4 | A-4 | SQLite 캐시 구현 | `src/context_generator.py` | A-1 |
| 5 | A-5 | `generate_batch()` 구현 (배치 + prompt caching) | `src/context_generator.py` | A-3, A-4 |
| 6 | A-6 | 긴 문서 처리 (`_truncate_document`) | `src/context_generator.py` | A-1 |
| 7 | A-7 | 비용 추적 로깅 | `src/context_generator.py` | A-3 |

### Phase B: Integration

| 순서 | ID | Task | 변경 파일 | 의존성 |
|:----:|:---:|------|----------|--------|
| 8 | B-1 | SemanticChunker 통합 | `src/semantic_chunker.py` | A-5 |
| 9 | B-2 | PineconeAgent 통합 | `src/agent.py` | B-1 |
| 10 | B-3 | CLI `--contextual` 플래그 | `main.py` | B-2 |
| 11 | B-4 | `.env.example` 업데이트 | `.env.example` | - |
| 12 | B-5 | 도메인 감지 헬퍼 | `src/semantic_chunker.py` | B-1 |

### Phase C: Evaluation

| 순서 | ID | Task | 변경 파일 | 의존성 |
|:----:|:---:|------|----------|--------|
| 13 | C-1 | Baseline 측정 (기존 정적 프리픽스) | `scripts/eval/eval_pipeline.py` | - |
| 14 | C-2 | 테스트 인제스천 (1개 도메인, --contextual) | CLI | B-3 |
| 15 | C-3 | 사후 측정 및 A/B 비교 | `scripts/eval/eval_pipeline.py` | C-2 |
| 16 | C-4 | 비용-성능 ROI 분석 | 수동 분석 | C-3 |

---

## 7. Test Plan

### 7.1 Unit Tests

| Test | Target | Validation |
|------|--------|------------|
| `test_generate_context_basic` | `generate_context()` | 반환값이 비어있지 않고 200자 이하 |
| `test_generate_context_domain` | Domain prompts | 노동법 → "법률명" 포함, MSDS → "물질명" 포함 |
| `test_truncate_document` | `_truncate_document()` | 100K+ 문서가 올바르게 절단됨 |
| `test_cache_hit` | SQLite cache | 동일 doc+chunk → 캐시 히트, LLM 호출 0회 |
| `test_cache_miss_different_doc` | Cache invalidation | 다른 문서 → 캐시 미스 |
| `test_batch_with_caching` | `generate_batch()` | 10 청크 중 5개 캐시 → LLM 5회만 호출 |
| `test_fallback_on_error` | Error handling | LLM 호출 실패 시 빈 문자열 반환, 정적 프리픽스 fallback |

### 7.2 Integration Tests

| Test | Target | Validation |
|------|--------|------------|
| `test_chunker_with_context_generator` | SemanticChunker + ContextGenerator | chunk.content에 contextual prefix 포함 |
| `test_chunker_without_context_generator` | SemanticChunker (기존) | 기존 정적 프리픽스 동작 확인 (회귀 방지) |
| `test_agent_contextual_flag` | PineconeAgent | `--contextual` 시 ContextGenerator 활성화 |
| `test_cost_tracking` | Cost stats | get_stats() 반환값에 토큰 수/비용 포함 |

### 7.3 E2E Tests

| Test | Target | Validation |
|------|--------|------------|
| `test_e2e_ingest_and_search` | 전체 파이프라인 | --contextual 인제스천 후 검색 품질 향상 확인 |
| `test_e2e_bm25_contextual` | BM25 통합 | contextual prefix 키워드가 BM25 매칭에 기여 |

---

## 8. Rollback Plan

| 단계 | 조건 | 조치 |
|------|------|------|
| 1 | Contextual prefix 품질 저하 | `--contextual` 플래그 제거하고 기존 방식 사용 |
| 2 | 비용 초과 | 모델을 더 저렴한 것으로 변경 또는 도메인별 비활성화 |
| 3 | 인제스천 실패 | ContextGenerator 호출 실패 시 자동으로 정적 프리픽스 fallback |
| 4 | 검색 품질 미개선 | 기존 namespace 유지 (--contextual은 별도 namespace 옵션 제공) |

**핵심 안전장치**: `context_generator=None`이면 기존 정적 프리픽스 동작이 100% 유지됨. 기존 코드에 대한 regression 위험 없음.

---

## 9. ADR (Architecture Decision Records)

### ADR-1: LLM 모델 선택 — Claude Haiku 4.5

**결정**: 맥락 생성에 `claude-haiku-4-5-20251001` 사용
**이유**:
- Prompt Caching 지원으로 동일 문서 반복 호출 비용 90% 절감
- 한국어 이해도 우수
- 빠른 응답 속도 (맥락 생성은 간단한 작업)
- 비용: $0.80/MTok input, $4/MTok output (cached: $0.08/MTok)

**대안 검토**:
- GPT-4o-mini: 저렴하지만 prompt caching 미지원 → 대량 청크 시 비용 불리
- Claude Sonnet: 품질 우수하지만 비용 4x → 맥락 생성에는 과잉
- Gemini Flash: 저렴하지만 prompt caching 구현 방식 상이

### ADR-2: 캐시 전략 — SQLite + Prompt Caching 이중 캐시

**결정**: 2단계 캐시 적용
- L1: SQLite 로컬 캐시 (content hash 기반, 영구)
- L2: Anthropic Prompt Caching (세션 내 동일 문서, 일시적)

**이유**:
- L1은 재인제스천 시 변경되지 않은 청크를 완전 건너뜀 (LLM 호출 0)
- L2는 동일 문서 내 여러 청크 처리 시 API 비용 절감
- 두 레이어가 독립적이므로 한쪽 장애 시 다른 쪽이 동작

### ADR-3: 통합 방식 — Optional Injection (비침투적)

**결정**: `ContextGenerator`를 `SemanticChunker`에 optional parameter로 주입
**이유**:
- `context_generator=None`이면 기존 동작 100% 유지
- 기존 코드 변경 최소화 (regression 위험 최소)
- 테스트 시 mock 주입 용이
- CLI `--contextual` 플래그로 opt-in

---

## 10. Dependency Requirements

```bash
# 신규 의존성
pip install anthropic>=0.39.0    # Claude API client (prompt caching 지원)

# 기존 의존성 (변경 없음)
# openai, pinecone, tiktoken, rank-bm25, ...
```

`requirements.txt`에 `anthropic>=0.39.0` 추가.
