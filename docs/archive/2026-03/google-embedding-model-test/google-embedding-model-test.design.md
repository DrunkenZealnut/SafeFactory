# Design: Gemini Embedding 2 테스트

> **Feature**: google-embedding-model-test
> **Plan Reference**: `docs/01-plan/features/google-embedding-model-test.plan.md`
> **Created**: 2026-03-18
> **Status**: Draft
> **Level**: Dynamic

---

## 1. 설계 개요

Gemini Embedding 2 (`gemini-embedding-2-preview`)를 기존 `EmbeddingGenerator`에 통합하고,
OpenAI `text-embedding-3-small`과 동일 조건에서 A/B 비교 테스트를 수행하기 위한 상세 설계.

**핵심 설계 원칙**:
- 기존 `EmbeddingGenerator` 인터페이스(`generate`, `generate_batch`) 변경 없이 provider 분기
- `output_dimensionality=1536`으로 기존 Pinecone 인덱스 호환
- 비교 테스트는 독립 스크립트(`scripts/`)로 분리하여 프로덕션 코드 영향 최소화

---

## 2. 컴포넌트 설계

### 2.1 EmbeddingGenerator 확장

**현재 구조**: OpenAI 전용, `__init__`에서 OpenAI client 생성

**변경 설계**: provider 패턴 도입 (openai / gemini 분기)

```python
# src/embedding_generator.py

MODELS = {
    # OpenAI
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    # Gemini
    "gemini-embedding-2-preview": 3072,
    "gemini-embedding-001": 3072,
}

# Provider 판별
GEMINI_MODELS = {"gemini-embedding-2-preview", "gemini-embedding-001"}

def __init__(self, api_key, model="text-embedding-3-small", dimensions=None):
    self.model = model
    self._provider = "gemini" if model in GEMINI_MODELS else "openai"

    if self._provider == "gemini":
        from google import genai
        self._gemini_client = genai.Client(api_key=api_key)
        # Gemini: MRL로 dimension 조절 (기본 3072, 1536/768 등 설정 가능)
        self.dimensions = dimensions or self.MODELS[model]
        self._encoding = None  # tiktoken 불필요
    else:
        # 기존 OpenAI 로직 유지
        self._http_client = httpx.Client(verify=certifi.where())
        self.client = OpenAI(api_key=api_key, ...)
        self.dimensions = ...
        self._encoding = tiktoken.get_encoding("cl100k_base")
```

**메서드별 변경**:

| 메서드 | 변경 내용 |
|--------|----------|
| `__init__` | provider 분기, Gemini client 초기화 |
| `_truncate` | Gemini: 문자 수 기반 보수적 truncation (8192 토큰 ≈ 24,000자 한국어) |
| `generate` | Gemini: `embed_content()` 호출, task_type 파라미터 추가 |
| `_call_api` | Gemini: 배치 `embed_content()` 호출 |
| `_embed_batch_with_retry` | 변경 없음 (기존 재시도 로직 재활용) |
| `generate_batch` | 변경 없음 (기존 배치 로직 재활용) |
| `get_model_info` | provider 정보 추가 |

### 2.2 generate() 메서드 — Gemini 분기 상세

```python
def generate(self, text: str, task_type: str = None) -> EmbeddingResult:
    text = self._truncate(text)

    if self._provider == "gemini":
        return self._generate_gemini(text, task_type)
    else:
        return self._generate_openai(text)

def _generate_gemini(self, text: str, task_type: str = None) -> EmbeddingResult:
    from google.genai import types

    config = types.EmbedContentConfig(
        output_dimensionality=self.dimensions,
    )
    if task_type:
        config.task_type = task_type  # "RETRIEVAL_QUERY" or "RETRIEVAL_DOCUMENT"

    response = self._gemini_client.models.embed_content(
        model=self.model,
        contents=text,
        config=config,
    )

    embedding = response.embeddings[0].values

    return EmbeddingResult(
        text=text,
        embedding=embedding,
        model=self.model,
        dimensions=len(embedding),
        token_count=None,  # Gemini API는 토큰 카운트 미반환
    )
```

### 2.3 _call_api() 배치 — Gemini 분기 상세

```python
def _call_api(self, batch: List[str], task_type: str = None) -> List[EmbeddingResult]:
    batch = [self._truncate(t) for t in batch]

    if self._provider == "gemini":
        return self._call_api_gemini(batch, task_type)
    else:
        return self._call_api_openai(batch)

def _call_api_gemini(self, batch: List[str], task_type: str = None) -> List[EmbeddingResult]:
    from google.genai import types

    config = types.EmbedContentConfig(
        output_dimensionality=self.dimensions,
    )
    if task_type:
        config.task_type = task_type

    # Gemini embed_content는 contents에 리스트 전달 가능
    results = []
    for text in batch:
        response = self._gemini_client.models.embed_content(
            model=self.model,
            contents=text,
            config=config,
        )
        results.append(EmbeddingResult(
            text=text,
            embedding=response.embeddings[0].values,
            model=self.model,
            dimensions=len(response.embeddings[0].values),
            token_count=None,
        ))
    return results
```

> **Note**: Gemini `embed_content`는 단일 contents만 받을 수 있으므로 (배치 미지원 확인 필요),
> 루프로 처리. API가 리스트 배치를 지원하면 단일 호출로 최적화.

### 2.4 _truncate() — Provider별 분기

```python
# Gemini용 토큰 제한
GEMINI_MAX_INPUT_TOKENS = 8192
GEMINI_MAX_CHARS = 24000  # 보수적 추정 (한국어 기준 ~3자/토큰)

def _truncate(self, text: str) -> str:
    if self._provider == "gemini":
        # Gemini: 문자 수 기반 보수적 truncation
        if len(text) > self.GEMINI_MAX_CHARS:
            return text[:self.GEMINI_MAX_CHARS]
        return text
    else:
        # OpenAI: tiktoken 기반 정확한 truncation (기존 로직)
        tokens = self._encoding.encode(text)
        if len(tokens) <= self.MAX_INPUT_TOKENS:
            return text
        return self._encoding.decode(tokens[:self.MAX_INPUT_TOKENS])
```

### 2.5 PineconeAgent 변경

`src/agent.py`의 `PineconeAgent.__init__`에서 모델에 따라 API 키를 분기.

```python
# src/agent.py — __init__ 변경

def __init__(self, openai_api_key, pinecone_api_key, pinecone_index_name,
             embedding_model="text-embedding-3-small",
             gemini_api_key=None, ...):

    # 임베딩 모델의 provider에 따라 API 키 선택
    is_gemini_embedding = embedding_model.startswith("gemini-embedding")
    embedding_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY") if is_gemini_embedding else openai_api_key

    self.embedding_generator = EmbeddingGenerator(
        api_key=embedding_api_key,
        model=embedding_model,
    )
```

**검색 시 task_type 전달**:

```python
# src/agent.py — search() 변경

def search(self, query, top_k=5, namespace=None, filter=None):
    # Gemini 모델일 때 RETRIEVAL_QUERY task_type 전달
    query_embedding = self.embedding_generator.generate(
        query,
        task_type="RETRIEVAL_QUERY" if self.embedding_generator._provider == "gemini" else None
    )
    ...
```

### 2.6 Admin 설정 확장

**`services/settings.py`** — DEFAULTS 추가:
```python
DEFAULTS = {
    ...
    'embedding_model': 'text-embedding-3-small',
    'embedding_provider': 'openai',  # 신규: openai | gemini
    ...
}
```

**`api/v1/admin.py`** — 선택지 추가:
```python
_VALID_SETTING_VALUES = {
    ...
    'embedding_model': [
        'text-embedding-3-small', 'text-embedding-3-large',
        'text-embedding-ada-002',
        'gemini-embedding-2-preview',  # 신규
    ],
}

# admin_settings_available_models()
'embedding_models': [
    {'value': 'text-embedding-3-small', 'label': 'OpenAI Small (1536D, $0.02/1M)'},
    {'value': 'text-embedding-3-large', 'label': 'OpenAI Large (3072D, $0.13/1M)'},
    {'value': 'text-embedding-ada-002', 'label': 'OpenAI Ada-002 (1536D, 레거시)'},
    {'value': 'gemini-embedding-2-preview', 'label': 'Gemini 2 Preview (1536D MRL, $0.20/1M)'},
],
```

---

## 3. 데이터 흐름

### 3.1 인제스트 흐름 (문서 → Pinecone)

```
문서 파일
  ↓
FileLoader → SemanticChunker → 청크 리스트
  ↓
EmbeddingGenerator.generate_batch(texts, task_type="RETRIEVAL_DOCUMENT")
  ↓  ← [분기] provider == "gemini" → _call_api_gemini() → embed_content()
  ↓  ← [분기] provider == "openai" → _call_api_openai() → embeddings.create()
  ↓
PineconeUploader.upload(vectors, namespace="semiconductor-v2-gemini")
```

### 3.2 검색 흐름 (쿼리 → 결과)

```
사용자 쿼리
  ↓
EmbeddingGenerator.generate(query, task_type="RETRIEVAL_QUERY")
  ↓  ← [분기] provider == "gemini" → embed_content(task_type=RETRIEVAL_QUERY)
  ↓  ← [분기] provider == "openai" → embeddings.create()
  ↓
PineconeUploader.query(vector, namespace="semiconductor-v2-gemini")
  ↓
검색 결과 (top_k)
```

### 3.3 테스트 네임스페이스 전략

| 네임스페이스 | 임베딩 모델 | 용도 |
|-------------|-----------|------|
| `semiconductor-v2` | OpenAI text-embedding-3-small (1536D) | 기존 프로덕션 |
| `semiconductor-v2-gemini` | Gemini Embedding 2 (1536D MRL) | 테스트 비교 |

- 동일 Pinecone 인덱스 내 네임스페이스 분리 (인덱스 dimension 동일: 1536)
- 동일 소스 문서, 동일 청크로 인제스트

---

## 4. 비교 테스트 스크립트 설계

### 4.1 파일 구조

```
scripts/
├── benchmark_embeddings.py      # 메인 비교 테스트 스크립트
├── ingest_gemini_test.py        # Gemini 임베딩으로 테스트 인제스트
└── benchmark_queries.json       # 25개 비교 쿼리셋
```

### 4.2 benchmark_queries.json 스키마

```json
{
  "queries": [
    {
      "id": "semi-01",
      "domain": "semiconductor-v2",
      "query": "웨이퍼 세정 공정의 화학물질 안전 관리",
      "expected_keywords": ["세정", "화학물질", "안전"],
      "difficulty": "medium"
    }
  ]
}
```

### 4.3 benchmark_embeddings.py 핵심 로직

```python
"""A/B 비교 테스트: OpenAI vs Gemini Embedding 2"""

import json, time, os
from dotenv import load_dotenv
from src.embedding_generator import EmbeddingGenerator
from pinecone import Pinecone

def run_benchmark():
    load_dotenv()

    # 두 개의 EmbeddingGenerator 인스턴스
    openai_gen = EmbeddingGenerator(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="text-embedding-3-small"
    )
    gemini_gen = EmbeddingGenerator(
        api_key=os.getenv("GEMINI_API_KEY"),
        model="gemini-embedding-2-preview",
        dimensions=1536
    )

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

    with open("scripts/benchmark_queries.json") as f:
        queries = json.load(f)["queries"]

    results = []
    for q in queries:
        # OpenAI 검색
        t0 = time.time()
        oai_emb = openai_gen.generate(q["query"])
        oai_results = index.query(
            vector=oai_emb.embedding, top_k=5,
            namespace="semiconductor-v2", include_metadata=True
        )
        oai_latency = (time.time() - t0) * 1000

        # Gemini 검색 (RETRIEVAL_QUERY task type)
        t0 = time.time()
        gem_emb = gemini_gen.generate(q["query"], task_type="RETRIEVAL_QUERY")
        gem_results = index.query(
            vector=gem_emb.embedding, top_k=5,
            namespace="semiconductor-v2-gemini", include_metadata=True
        )
        gem_latency = (time.time() - t0) * 1000

        results.append({
            "query_id": q["id"],
            "query": q["query"],
            "openai": {
                "top5_ids": [m.id for m in oai_results.matches],
                "top5_scores": [m.score for m in oai_results.matches],
                "latency_ms": round(oai_latency, 1),
            },
            "gemini": {
                "top5_ids": [m.id for m in gem_results.matches],
                "top5_scores": [m.score for m in gem_results.matches],
                "latency_ms": round(gem_latency, 1),
            },
        })

    # 결과 출력 및 저장
    print_summary(results)
    with open("scripts/benchmark_results.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
```

### 4.4 평가 메트릭 계산

```python
def print_summary(results):
    """MRR, 평균 유사도, Latency 비교 출력"""
    oai_scores, gem_scores = [], []
    oai_latencies, gem_latencies = [], []

    for r in results:
        oai_scores.append(r["openai"]["top5_scores"][0] if r["openai"]["top5_scores"] else 0)
        gem_scores.append(r["gemini"]["top5_scores"][0] if r["gemini"]["top5_scores"] else 0)
        oai_latencies.append(r["openai"]["latency_ms"])
        gem_latencies.append(r["gemini"]["latency_ms"])

    # Top-1 평균 유사도
    print(f"Top-1 Avg Score  — OpenAI: {sum(oai_scores)/len(oai_scores):.4f}  |  Gemini: {sum(gem_scores)/len(gem_scores):.4f}")
    # 평균 Latency
    print(f"Avg Latency (ms) — OpenAI: {sum(oai_latencies)/len(oai_latencies):.1f}  |  Gemini: {sum(gem_latencies)/len(gem_latencies):.1f}")
    # 결과 중첩률 (동일 문서가 Top-5에 몇 개나 겹치는지)
    overlaps = []
    for r in results:
        oai_set = set(r["openai"]["top5_ids"])
        gem_set = set(r["gemini"]["top5_ids"])
        overlaps.append(len(oai_set & gem_set) / max(len(oai_set), 1))
    print(f"Top-5 Overlap    — {sum(overlaps)/len(overlaps)*100:.1f}%")
```

### 4.5 ingest_gemini_test.py 핵심 로직

```python
"""기존 semiconductor-v2 청크를 Gemini 임베딩으로 재인제스트"""

def ingest_gemini():
    load_dotenv()
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

    gemini_gen = EmbeddingGenerator(
        api_key=os.getenv("GEMINI_API_KEY"),
        model="gemini-embedding-2-preview",
        dimensions=1536
    )

    # 기존 semiconductor-v2 벡터에서 메타데이터 + 텍스트 추출
    # (Pinecone list → fetch로 metadata의 content_preview 사용)
    # 또는 SQLite metadata_manager에서 원본 텍스트 로드

    vectors = []
    for chunk_text, chunk_id, metadata in load_existing_chunks("semiconductor-v2"):
        emb = gemini_gen.generate(chunk_text, task_type="RETRIEVAL_DOCUMENT")
        vectors.append({
            "id": chunk_id,
            "values": emb.embedding,
            "metadata": metadata,
        })

        if len(vectors) >= 100:
            index.upsert(vectors=vectors, namespace="semiconductor-v2-gemini")
            vectors = []

    if vectors:
        index.upsert(vectors=vectors, namespace="semiconductor-v2-gemini")
```

---

## 5. 구현 순서 (Implementation Order)

### Step 1: EmbeddingGenerator Gemini 지원 추가
- **파일**: `src/embedding_generator.py`
- **변경**:
  1. `MODELS` dict에 Gemini 모델 추가
  2. `GEMINI_MODELS` set 추가
  3. `__init__`에 provider 분기 로직
  4. `_generate_gemini()` 메서드 추가
  5. `_call_api_gemini()` 메서드 추가
  6. `_truncate()` provider별 분기
  7. `generate()` 에 `task_type` 파라미터 추가
  8. `_call_api()` 에 `task_type` 파라미터 추가
- **검증**: `if __name__ == "__main__"` 블록에서 Gemini 임베딩 생성 테스트

### Step 2: PineconeAgent 호환성 업데이트
- **파일**: `src/agent.py`
- **변경**:
  1. `__init__`에 `gemini_api_key` 파라미터 추가 (optional)
  2. 임베딩 모델 provider에 따른 API 키 분기
  3. `search()`에서 Gemini일 때 `task_type="RETRIEVAL_QUERY"` 전달
  4. `MODELS` dimension 조회 시 `EmbeddingGenerator.MODELS` 참조
- **검증**: Gemini 모델로 PineconeAgent 생성 및 검색 테스트

### Step 3: 테스트 인제스트 스크립트
- **파일**: `scripts/ingest_gemini_test.py` (신규)
- **내용**: 기존 semiconductor-v2 청크 → Gemini 임베딩 → `semiconductor-v2-gemini` 네임스페이스
- **검증**: 인제스트 완료 후 `index.describe_index_stats()` 확인

### Step 4: 비교 벤치마크 스크립트
- **파일**: `scripts/benchmark_embeddings.py` (신규), `scripts/benchmark_queries.json` (신규)
- **내용**: 25개 쿼리 A/B 비교, 메트릭 계산, 결과 JSON 출력
- **검증**: 벤치마크 실행 및 결과 확인

### Step 5: Admin 설정 확장 (optional)
- **파일**: `services/settings.py`, `api/v1/admin.py`
- **변경**: Gemini 임베딩 모델 선택지 추가
- **검증**: admin UI에서 모델 변경 가능 확인

---

## 6. API 인터페이스 변경

### 6.1 EmbeddingGenerator 공개 인터페이스

```python
class EmbeddingGenerator:
    # 변경된 시그니처
    def generate(self, text: str, task_type: str = None) -> EmbeddingResult:
        """task_type: "RETRIEVAL_QUERY" | "RETRIEVAL_DOCUMENT" | None (Gemini 전용)"""

    def generate_batch(self, texts: List[str], batch_size: int = 100,
                       task_type: str = None) -> List[Optional[EmbeddingResult]]:
        """배치 생성. task_type은 Gemini 모델에만 적용."""

    def get_model_info(self) -> dict:
        """provider 정보 포함: {"model": ..., "provider": "gemini", ...}"""
```

> **하위 호환성**: `task_type=None` 기본값으로 기존 호출부 변경 불필요.
> OpenAI 모델에서는 `task_type` 무시.

### 6.2 PineconeAgent 공개 인터페이스

```python
class PineconeAgent:
    def __init__(self, openai_api_key, pinecone_api_key, pinecone_index_name,
                 embedding_model="text-embedding-3-small",
                 gemini_api_key=None,  # 신규 optional 파라미터
                 ...):
```

> **하위 호환성**: `gemini_api_key=None` 기본값으로 기존 코드 영향 없음.

---

## 7. 에러 처리

| 시나리오 | 처리 방식 |
|---------|----------|
| `GEMINI_API_KEY` 미설정 + Gemini 모델 선택 | `__init__`에서 `RuntimeError` 발생 |
| Gemini API 429 rate limit | 기존 `_embed_batch_with_retry` 재시도 로직 재활용 |
| Gemini API 응답 형식 변경 (Preview) | `response.embeddings[0].values` 접근 시 `AttributeError` catch → 로그 |
| 배치 중 일부 실패 | `_embed_individually` 폴백 (기존 로직) |
| dimension 불일치 | `__init__`에서 dimensions 검증 (1536 ≤ dimensions ≤ 3072) |

---

## 8. 테스트 검증 기준

| 항목 | 기준 |
|------|------|
| 단일 임베딩 생성 | Gemini 모델로 1536D 벡터 반환 확인 |
| 배치 임베딩 생성 | 10개 텍스트 배치 정상 처리 |
| 인제스트 | semiconductor-v2-gemini 네임스페이스에 벡터 업로드 성공 |
| 검색 | Gemini 임베딩 쿼리로 유사 문서 반환 확인 |
| task_type 효과 | RETRIEVAL_QUERY vs None 검색 결과 차이 확인 |
| 하위 호환성 | 기존 OpenAI 호출부 변경 없이 정상 동작 |
| 에러 핸들링 | API 키 미설정, rate limit 시 적절한 에러 메시지 |
