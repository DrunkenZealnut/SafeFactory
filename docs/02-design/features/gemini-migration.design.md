# Gemini Embedding Migration Design Document

> **Summary**: 검색 파이프라인의 임베딩 모델을 Gemini Embedding 2로 전환하는 상세 설계
>
> **Project**: SafeFactory
> **Author**: zealnutkim
> **Date**: 2026-03-18
> **Status**: Draft
> **Planning Doc**: [gemini-migration.plan.md](../../01-plan/features/gemini-migration.plan.md)

---

## 1. Overview

### 1.1 Design Goals

1. `get_agent()` 싱글톤이 admin `embedding_model` 설정을 동적으로 반영
2. Gemini 모델 사용 시 네임스페이스에 `-gemini` 접미사를 자동 적용
3. Admin에서 모델 변경 시 즉시 전환 (싱글톤 캐시 무효화)
4. OpenAI ↔ Gemini 무중단 롤백 보장

### 1.2 Design Principles

- **최소 변경**: 기존 코드 흐름을 최대한 유지, 네임스페이스 해석 레이어만 추가
- **단일 진입점**: `resolve_namespace()` 하나로 모든 네임스페이스 변환 처리
- **설정 기반 전환**: 코드 변경 없이 Admin 설정으로 모델 전환 가능

---

## 2. Architecture

### 2.1 현재 흐름 (Before)

```
User Request
  → resolve_search_context() → namespace = "semiconductor-v2"   (고정)
  → get_agent() → PineconeAgent(embedding_model="text-embedding-3-small")  (하드코딩)
  → agent.search(namespace="semiconductor-v2")
  → Pinecone: semiconductor-v2 (OpenAI 임베딩)
```

### 2.2 변경 후 흐름 (After)

```
User Request
  → resolve_search_context() → base_namespace = "semiconductor-v2"
  → resolve_namespace("semiconductor-v2") → "semiconductor-v2-gemini"  (NEW)
  → get_agent() → PineconeAgent(embedding_model=get_setting('embedding_model'))
  → agent.search(namespace="semiconductor-v2-gemini")
  → Pinecone: semiconductor-v2-gemini (Gemini 임베딩)
```

### 2.3 Component Diagram

```
┌──────────────────┐      ┌──────────────────────┐
│  api/v1/search   │─────▶│ services/rag_pipeline │
│   (api_ask)      │      │  (run_rag_pipeline)   │
└──────────────────┘      └──────────┬───────────┘
                                     │
                      ┌──────────────▼────────────┐
                      │ services/major_config      │
                      │  resolve_search_context()  │
                      │  → base namespace          │
                      └──────────────┬─────────────┘
                                     │
                ┌────────────────────▼──────────────────┐
                │ services/domain_config                 │
                │  resolve_namespace(base_ns)  ← NEW    │
                │  embedding_model → "-gemini" suffix    │
                └────────────────────┬──────────────────┘
                                     │
            ┌────────────────────────▼───────────────────┐
            │ services/singletons                         │
            │  get_agent() → PineconeAgent                │
            │    embedding_model = get_setting(...)  ← MOD│
            └────────────────────────┬───────────────────┘
                                     │
                      ┌──────────────▼────────────┐
                      │ src/agent PineconeAgent    │
                      │  .search(namespace=...)    │
                      │  .embedding_generator      │
                      └──────────────┬─────────────┘
                                     │
                      ┌──────────────▼────────────┐
                      │ Pinecone                   │
                      │  semiconductor-v2-gemini   │
                      └────────────────────────────┘
```

### 2.4 Dependencies

| Component | Depends On | Purpose |
|-----------|-----------|---------|
| `resolve_namespace()` | `get_setting('embedding_model')` | 현재 모델에 따라 네임스페이스 결정 |
| `get_agent()` | `get_setting('embedding_model')` | 올바른 임베딩 모델로 PineconeAgent 생성 |
| `rag_pipeline` | `resolve_namespace()` | 검색 시 올바른 네임스페이스 사용 |
| `api/v1/search` | `resolve_namespace()` | 직접 검색 시 올바른 네임스페이스 사용 |

---

## 3. Detailed Design

### 3.1 `services/domain_config.py` — `resolve_namespace()`

```python
# 네임스페이스 매핑 전략: Gemini 모델 → base_ns + "-gemini"
_GEMINI_NAMESPACE_SUFFIX = "-gemini"

def resolve_namespace(base_namespace: str) -> str:
    """Resolve actual Pinecone namespace based on current embedding model.

    When embedding model is Gemini, appends '-gemini' suffix.
    Special cases: 'all' and '' (empty) are returned as-is.

    Args:
        base_namespace: Base namespace (e.g., 'semiconductor-v2')

    Returns:
        Resolved namespace (e.g., 'semiconductor-v2-gemini')
    """
    if not base_namespace or base_namespace == 'all':
        return base_namespace

    from services.settings import get_setting
    model = get_setting('embedding_model', 'text-embedding-3-small')

    if model.startswith("gemini-embedding"):
        return f"{base_namespace}{_GEMINI_NAMESPACE_SUFFIX}"
    return base_namespace
```

**설계 결정**:
- `'all'`과 빈 문자열은 변환하지 않음 (전체 검색/기본 네임스페이스)
- `get_setting()` 호출은 캐시된 값을 반환하므로 성능 영향 없음 (TTL 60초)
- 접미사 방식으로 새 임베딩 모델 추가 시 확장 가능

### 3.2 `services/singletons.py` — `get_agent()` 수정

```python
def get_agent():
    """Get or create the PineconeAgent instance."""
    global _agent
    instance = _agent
    if instance is None:
        from src.agent import PineconeAgent
        from services.settings import get_setting
        with _lock:
            if _agent is None:
                embedding_model = get_setting('embedding_model', 'text-embedding-3-small')
                _agent = PineconeAgent(
                    openai_api_key=_require_env("OPENAI_API_KEY"),
                    pinecone_api_key=_require_env("PINECONE_API_KEY"),
                    pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "document-index"),
                    embedding_model=embedding_model,
                    create_index_if_not_exists=False
                )
            instance = _agent
    return instance
```

**변경 요점**: `embedding_model` 파라미터를 `get_setting()`에서 동적으로 가져옴.

### 3.3 `services/settings.py` — 캐시 무효화 연동

기존 `invalidate_cache()`는 설정 캐시만 무효화. `embedding_model` 변경 시 `_agent` 싱글톤도 무효화 필요.

Admin 설정 변경은 `api/v1/admin.py`의 `update_setting()` 엔드포인트에서 처리.
해당 엔드포인트에서 `embedding_model` 변경 감지 시 agent를 무효화하는 로직 추가.

```python
# api/v1/admin.py — update_setting 엔드포인트 내부
if key == 'embedding_model':
    from services.singletons import invalidate_agent
    invalidate_agent()
```

```python
# services/singletons.py — 새 함수 추가
def invalidate_agent():
    """Invalidate the PineconeAgent singleton (e.g., after embedding model change)."""
    global _agent
    with _lock:
        _agent = None
```

### 3.4 `services/rag_pipeline.py` — 네임스페이스 resolve 적용

`run_rag_pipeline()` 내에서 namespace가 결정된 직후에 `resolve_namespace()` 적용.

```python
# run_rag_pipeline() 내부, line ~766 이후
major_key, namespace = resolve_search_context(data, _user)
namespace = resolve_namespace(namespace)  # ← 추가
```

**적용 위치**: Phase 0 (Domain Classification) 이전에 적용. `classify_domain()`이 반환하는 `detected_namespace`에도 적용 필요:

```python
# line ~789 이후
if detected_namespace and detected_namespace != namespace and domain_confidence > 0:
    namespace = resolve_namespace(detected_namespace)  # ← resolve 적용
```

**Safety cross-search**: `SAFETY_CROSS_SEARCH_NAMESPACE` (kosha)에도 resolve 적용:

```python
# rag_pipeline.py 내 safety cross-search 부분
safety_ns = resolve_namespace(SAFETY_CROSS_SEARCH_NAMESPACE)
```

### 3.5 `api/v1/search.py` — `/search` 엔드포인트

```python
# api_search() 내부, namespace 결정 후
from services.domain_config import resolve_namespace
namespace = resolve_namespace(namespace)  # ← 추가
```

`/ask` 및 `/ask/stream`은 `run_rag_pipeline()`을 통하므로 3.4에서 처리됨.

### 3.6 `main.py` — CLI Gemini 모델 지원

```python
process_parser.add_argument(
    "--embedding-model", type=str, default="text-embedding-3-small",
    choices=[
        "text-embedding-3-small", "text-embedding-3-large",
        "gemini-embedding-2-preview",  # ← 추가
    ],
    help="Embedding model"
)
```

### 3.7 `services/settings.py` — 기본값 변경

```python
DEFAULTS = {
    ...
    'embedding_model': 'gemini-embedding-2-preview',  # 변경: text-embedding-3-small → gemini
    ...
}
```

---

## 4. 네임스페이스 매핑 전체 테이블

| Base Namespace | OpenAI (text-embedding-3-*) | Gemini (gemini-embedding-*) |
|----------------|:--:|:--:|
| `semiconductor-v2` | `semiconductor-v2` | `semiconductor-v2-gemini` |
| `laborlaw-v2` | `laborlaw-v2` | `laborlaw-v2-gemini` |
| `counsel` | `counsel` | `counsel-gemini` |
| `precedent` | `precedent` | `precedent-gemini` |
| `field-training` | `field-training` | `field-training-gemini` |
| `kosha` | `kosha` | `kosha-gemini` |
| `safeguide` | `safeguide` | `safeguide-gemini` |
| `all` | `all` (변환 없음) | `all` (변환 없음) |
| `""` (빈값) | `""` (변환 없음) | `""` (변환 없음) |

---

## 5. Error Handling

### 5.1 Gemini API 키 누락

`PineconeAgent.__init__`에서 이미 처리됨 (line 72-74):
```python
if is_gemini_embedding:
    embedding_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
    if not embedding_api_key:
        raise RuntimeError("GEMINI_API_KEY is required for Gemini embedding models")
```

### 5.2 Gemini 네임스페이스 미존재

`resolve_namespace()`는 네임스페이스 존재 여부를 검증하지 않음 (Pinecone이 빈 결과 반환).
이는 의도적 설계 — 존재 여부 확인은 추가 API 호출이 필요하므로 성능 영향.

### 5.3 롤백 시나리오

```
Admin → embedding_model = "text-embedding-3-small" 설정
  → invalidate_agent() 호출
  → 다음 요청에서 get_agent()가 OpenAI 모델로 새 PineconeAgent 생성
  → resolve_namespace()가 접미사 없는 원래 네임스페이스 반환
  → 즉시 OpenAI 네임스페이스로 검색
```

---

## 6. Test Plan

### 6.1 Test Cases

| # | 시나리오 | 검증 방법 | 기대 결과 |
|---|---------|-----------|-----------|
| T-01 | Gemini 모델 설정 후 검색 | `/api/v1/search` 호출 | `semiconductor-v2-gemini`에서 검색됨 |
| T-02 | OpenAI 모델로 롤백 | Admin에서 모델 변경 후 검색 | `semiconductor-v2`에서 검색됨 |
| T-03 | RAG 파이프라인 전체 동작 | `/api/v1/ask` 호출 | Gemini 임베딩으로 정상 응답 |
| T-04 | Domain auto-routing | 노동법 쿼리 반도체 페이지에서 검색 | `laborlaw-v2-gemini`로 라우팅 |
| T-05 | Safety cross-search | 반도체 검색 시 kosha 참조 | `kosha-gemini`에서 보조 검색 |
| T-06 | `all` 네임스페이스 | 전체 검색 | `all` 그대로 유지 (변환 없음) |
| T-07 | CLI Gemini 인제스트 | `main.py --embedding-model gemini-...` | 정상 인제스트 |
| T-08 | 싱글톤 무효화 | Admin에서 모델 변경 | 새 PineconeAgent 생성 확인 |

### 6.2 수동 검증 절차

```bash
# 1. 현재 설정 확인
curl -s localhost:5001/api/v1/admin/settings | python -m json.tool | grep embedding

# 2. Gemini로 변경
curl -X POST localhost:5001/api/v1/admin/settings \
  -H "Content-Type: application/json" \
  -d '{"embedding_model": "gemini-embedding-2-preview"}'

# 3. 검색 테스트
curl -X POST localhost:5001/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "웨이퍼 세정 공정", "namespace": "semiconductor-v2"}'

# 4. 롤백 테스트
curl -X POST localhost:5001/api/v1/admin/settings \
  -H "Content-Type: application/json" \
  -d '{"embedding_model": "text-embedding-3-small"}'
```

---

## 7. Implementation Order

### Step 1: `services/domain_config.py`
- `resolve_namespace()` 함수 추가
- Export에 추가

### Step 2: `services/singletons.py`
- `get_agent()`에서 `get_setting('embedding_model')` 참조
- `invalidate_agent()` 함수 추가

### Step 3: `api/v1/admin.py`
- `embedding_model` 설정 변경 시 `invalidate_agent()` 호출

### Step 4: `services/rag_pipeline.py`
- `resolve_namespace()` import
- `run_rag_pipeline()` 내 3곳에 적용:
  - 초기 namespace 결정 후
  - Domain auto-routing 후
  - Safety cross-search 네임스페이스

### Step 5: `api/v1/search.py`
- `/search` 엔드포인트에 `resolve_namespace()` 적용

### Step 6: `main.py`
- `--embedding-model` 선택지에 `gemini-embedding-2-preview` 추가

### Step 7: `services/settings.py`
- 기본값을 `gemini-embedding-2-preview`로 변경

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-03-18 | Initial draft | zealnutkim |
