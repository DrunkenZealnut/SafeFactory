# Multi-Query 파이프라인 리팩토링

## Executive Summary

| 항목 | 내용 |
|------|------|
| Feature | Multi-Query 파이프라인 리팩토링 |
| 작성일 | 2026-03-16 |
| 예상 기간 | 2-3일 |

### Value Delivered

| 관점 | 내용 |
|------|------|
| **Problem** | QueryEnhancer와 RAG 파이프라인의 쿼리 확장·검색 로직이 `run_rag_pipeline()` 함수 내에 밀결합되어, 개별 테스트·확장·디버깅이 어렵고 매 요청마다 ThreadPoolExecutor를 생성하여 불필요한 오버헤드 발생 |
| **Solution** | 쿼리 확장(Enhancement) 단계를 독립 모듈로 분리하고, 검색 결과 병합 전략을 체계화하며, 실행기(executor) 재사용과 캐싱 레이어를 도입 |
| **Function UX Effect** | 쿼리 응답 시간 단축(executor 재사용 + 캐싱), 검색 품질 향상(가중 병합), 관리자의 enhancement 전략 변경 용이 |
| **Core Value** | RAG 시스템의 검색 정확도와 응답 속도를 동시에 개선하여 사용자 신뢰도 향상 |

---

## 1. 현황 분석

### 1.1 현재 구조

```
[run_rag_pipeline()] 600줄+ 단일 함수
  ├── Phase 0: Domain Classification
  ├── Phase 1: Query Enhancement (line 647-717)
  │   ├── synonym expansion
  │   ├── multi_query (LLM 호출)
  │   ├── HyDE (LLM 호출)
  │   └── keyword extraction
  ├── Phase 2: Multi-Query Search (line 719-778)
  │   └── for eq in enhanced_queries: agent.search(eq, ...)
  ├── Phase 4: Hybrid Search (BM25)
  ├── Phase 5: Reranking
  ├── Phase 6: Context Optimization
  └── Phase 7: LLM Generation
```

### 1.2 문제점

| # | 문제 | 위치 | 영향도 |
|---|------|------|--------|
| P1 | **Phase 1 밀결합** — synonym expansion, multi-query, HyDE, keyword extraction이 `run_rag_pipeline()` 내부에 inline으로 구현됨 | `rag_pipeline.py:647-717` | 높음 |
| P2 | **매 요청 ThreadPoolExecutor 생성** — `concurrent.futures.ThreadPoolExecutor(max_workers=3)`가 요청마다 새로 생성되어 스레드 생성/소멸 오버헤드 발생 | `rag_pipeline.py:668` | 중간 |
| P3 | **HyDE 결과 혼재** — HyDE 가상 문서가 일반 쿼리 변형과 동일한 리스트에 append되어, 검색 결과 병합 시 가중치 구분 불가 | `rag_pipeline.py:694` | 중간 |
| P4 | **결과 병합 전략 부재** — 모든 쿼리 변형의 결과를 단순 dedup(sha256)으로만 병합. 원본 쿼리 vs 변형 쿼리 vs HyDE 간 품질 차이 미반영 | `rag_pipeline.py:761-767` | 높음 |
| P5 | **`_get_num_variations()` 위치** — 모듈 레벨 함수로 QueryEnhancer와 분리됨 | `rag_pipeline.py:149-157` | 낮음 |
| P6 | **`enhance_query()` 미사용** — QueryEnhancer에 `enhance_query()` 통합 메서드가 있지만 pipeline에서 직접 개별 메서드 호출 | `query_enhancer.py:366-403` | 중간 |
| P7 | **LLM 캐싱 없음** — 동일/유사 쿼리에 대한 multi-query 결과를 캐싱하지 않아 반복 비용 발생 | `query_enhancer.py:197-240` | 중간 |
| P8 | **검색 루프 내 재시도 로직 inline** — 2회 재시도(cold start 대응)가 for loop 내부에 하드코딩 | `rag_pipeline.py:738-774` | 낮음 |

### 1.3 관련 파일

| 파일 | 역할 | 변경 범위 |
|------|------|-----------|
| `src/query_enhancer.py` | 쿼리 확장 (multi-query, HyDE, synonyms, keywords) | 대폭 수정 |
| `services/rag_pipeline.py` | RAG 파이프라인 (Phase 1, 2 중심) | 대폭 수정 |
| `services/singletons.py` | QueryEnhancer 싱글턴 관리 | 소폭 수정 |
| `services/query_router.py` | 쿼리 타입별 설정 (`QUERY_TYPE_CONFIG`) | 수정 없음 (참조만) |

---

## 2. 리팩토링 목표

### 2.1 핵심 목표

| # | 목표 | 측정 기준 |
|---|------|-----------|
| G1 | Phase 1 (Query Enhancement)을 독립 함수/메서드로 분리 | `run_rag_pipeline()`에서 단일 함수 호출로 대체 |
| G2 | Phase 2 (Multi-Query Search)의 결과 병합 전략 체계화 | 쿼리 소스별 가중치 적용 가능 |
| G3 | ThreadPoolExecutor 재사용 | 모듈 레벨 executor 1회 생성 |
| G4 | 동일 쿼리 LLM 캐싱 도입 | TTL 기반 in-memory 캐시 |
| G5 | 기존 동작 100% 호환 유지 | 리팩토링 전후 동일 쿼리에 동일 결과 |

### 2.2 비목표 (Scope Out)

- 새로운 쿼리 확장 기법(step-back, sub-question) 추가 — 별도 feature로 진행
- QueryEnhancer 멀티 프로바이더 구조 개편 — 현재 동작 유지
- Phase 4-7 (Hybrid Search, Reranking, Context Optimization, LLM Generation) 변경
- 프론트엔드/API 인터페이스 변경

---

## 3. 리팩토링 계획

### 3.1 Step 1: QueryEnhancer 통합 메서드 정비

**변경 대상**: `src/query_enhancer.py`

- `enhance_query()` 메서드를 실제 pipeline 호출 패턴에 맞게 재설계
- `_get_num_variations()` 로직을 QueryEnhancer 내부 메서드로 이동
- 병렬 실행 로직을 QueryEnhancer 내부로 캡슐화
- 반환값을 구조화된 `EnhancementResult` (dataclass)로 변경

```python
@dataclass
class EnhancementResult:
    original: str
    variations: list[str]       # multi-query 변형들
    hyde_doc: str | None        # HyDE 가상 문서 (별도 보관)
    expanded_query: str | None  # synonym 확장 쿼리
    keywords: list[str]         # 추출된 키워드

    @property
    def search_queries(self) -> list[str]:
        """검색에 사용할 쿼리 목록 (가중치 정보 포함 가능)"""
        queries = list(self.variations)
        if self.expanded_query and self.expanded_query not in queries:
            queries.append(self.expanded_query)
        return queries

    @property
    def hyde_queries(self) -> list[str]:
        """HyDE 문서 기반 쿼리 (별도 검색 풀)"""
        return [self.hyde_doc] if self.hyde_doc else []
```

### 3.2 Step 2: 모듈 레벨 ThreadPoolExecutor

**변경 대상**: `src/query_enhancer.py`

```python
# 모듈 레벨 executor (앱 수명주기와 동일)
_enhancement_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=3, thread_name_prefix="query-enhance"
)
```

- `atexit` 또는 `singletons.shutdown_all()`에 등록하여 정리
- QueryEnhancer 내부에서 이 executor 사용

### 3.3 Step 3: In-Memory 캐싱 레이어

**변경 대상**: `src/query_enhancer.py`

- `multi_query()` 결과를 TTL 기반 캐시에 저장 (기본 300초)
- 캐시 키: `(query_text, num_variations)` 해시
- `functools.lru_cache` 또는 간단한 dict + timestamp 구현
- 캐시 크기 제한: 최대 200 엔트리

### 3.4 Step 4: RAG Pipeline Phase 1 분리

**변경 대상**: `services/rag_pipeline.py`

기존 Phase 1 코드 블록(line 647-717)을 `_enhance_query()` 함수로 추출:

```python
def _enhance_query(
    search_query: str,
    namespace: str,
    route_cfg: dict,
    use_enhancement: bool,
) -> EnhancementResult:
    """Phase 1: 쿼리 확장 — synonym, multi-query, HyDE, keyword 추출."""
    ...
```

`run_rag_pipeline()`에서는 한 줄로 호출:

```python
enhancement = _enhance_query(search_query, namespace, route_cfg, use_enhancement)
```

### 3.5 Step 5: Phase 2 검색 루프 분리 및 결과 병합 개선

**변경 대상**: `services/rag_pipeline.py`

기존 Phase 2 코드 블록(line 720-778)을 `_search_with_variations()` 함수로 추출:

```python
def _search_with_variations(
    agent,
    enhancement: EnhancementResult,
    namespace: str,
    domain_filter: dict,
    top_k: int,
    search_top_k: int,
) -> list[dict]:
    """Phase 2: multi-query 검색 + 결과 병합."""
    ...
```

결과 병합 개선:
- 원본 쿼리 결과에 가중치 boost (첫 번째 변형이 항상 원본)
- HyDE 결과는 별도 검색 후 병합 (일반 변형과 분리)
- dedup 로직은 유지하되 content hash 함수를 별도로 분리
- 재시도 로직을 `_search_single_query()` 헬퍼로 추출

### 3.6 Step 6: Singletons 정리

**변경 대상**: `services/singletons.py`

- executor shutdown을 `shutdown_all()`에 등록
- `invalidate_query_enhancer()`에 캐시 초기화 추가

---

## 4. 구현 순서 (의존성 기반)

```
Step 1: EnhancementResult dataclass + enhance_query() 재설계
  ↓
Step 2: 모듈 레벨 executor 도입
  ↓
Step 3: 캐싱 레이어 추가
  ↓ (Step 1-3은 query_enhancer.py 내부 변경)
Step 4: Pipeline Phase 1 분리 (_enhance_query)
  ↓
Step 5: Pipeline Phase 2 분리 (_search_with_variations)
  ↓
Step 6: Singletons 정리 + 통합 검증
```

---

## 5. 리스크 및 대응

| 리스크 | 확률 | 영향 | 대응 |
|--------|------|------|------|
| 리팩토링 후 검색 결과 품질 변화 | 중 | 높음 | 동일 쿼리 세트로 전후 비교 테스트 (A/B) |
| executor 공유로 인한 스레드 안전성 이슈 | 낮 | 중 | QueryEnhancer 메서드는 stateless, executor submit만 사용 |
| 캐시 메모리 증가 | 낮 | 낮 | 200 엔트리 제한 + TTL 300초 |
| HyDE 결과 분리로 검색 결과 수 변화 | 중 | 중 | HyDE 결과 병합 가중치를 설정 가능하게 구현 |

---

## 6. 성공 기준

- [ ] `run_rag_pipeline()` Phase 1 코드가 단일 함수 호출로 대체됨
- [ ] `run_rag_pipeline()` Phase 2 코드가 단일 함수 호출로 대체됨
- [ ] `EnhancementResult` dataclass로 쿼리 확장 결과가 구조화됨
- [ ] ThreadPoolExecutor가 모듈 레벨에서 1회 생성됨
- [ ] 동일 쿼리 캐싱이 동작함 (로그로 확인 가능)
- [ ] 기존 API 응답 형식 변경 없음
- [ ] 기존 검색 품질 유지 (동일 쿼리 → 동일 상위 결과)
