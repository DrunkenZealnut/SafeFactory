# Design: 전체 검색 관련 플로우 점검

> Plan 참조: `docs/01-plan/features/search-flow-audit.plan.md`

---

## 1. 점검 결과 요약 (Verified Issues)

Plan의 가설(H-1~H-4)과 점검 항목(DF/DC/SC/EH/PF)을 코드 레벨에서 검증한 결과:

### 1.1 심각도별 분류

| 심각도 | 건수 | 주요 이슈 |
|--------|------|-----------|
| **Critical** | 2 | `_run_multi_query_search` 미정의 버그, `domain_confidence` 미초기화 |
| **High** | 3 | `secondary_ns` dead variable, `ask`/`ask/stream` 코드 중복, Graph score 스케일 혼합 |
| **Medium** | 5 | 성능 병렬화 기회, 커뮤니티 검색 한국어 매칭, 설정 키 혼동 등 |
| **Low** | 3 | 미사용 파라미터, 캐시 적중률 로깅 부재 등 |
| **합계** | **13** | |

---

## 2. Critical Issues (즉시 수정 필요)

### BUG-1: `_run_multi_query_search` 함수 미정의 (NameError)

**위치**: `services/rag_pipeline.py:1035`

**현상**:
```python
# RB-1: Re-search with enhanced query when results are insufficient
if len(results) < 3 and use_enhancement and domain_confidence and domain_confidence < 0.5:
    _retry_enhancement = _enhance_query(...)
    _retry_results = _run_multi_query_search(  # ← 이 함수는 존재하지 않음
        search_query, _retry_enhancement, namespace, top_k * 2,
        route_cfg, data, skip_bm25,
    )
```

**근본 원인**: `search-quality` 설계 문서에서 `_run_multi_query_search`를 정의하도록 설계했으나, 실제 구현에서는 `_search_with_variations`로 리팩토링됨. re-search 호출부만 업데이트가 누락됨.

**영향**: RB-1 re-search 트리거 시 `NameError` 발생 → except로 무시 → re-search 기능이 **완전 비활성화** 상태. 결과가 3개 미만이고 confidence가 낮은 검색에서 개선 기회를 상실.

**수정 방안**:
```python
# 방안 A: _search_with_variations 사용 (권장)
_retry_results = _search_with_variations(
    agent, _retry_enhancement, namespace,
    domain_filter, top_k * 2,
)

# 방안 B: 함수 인터페이스 맞지 않으면 inline 검색
# _search_with_variations는 (agent, enhancement, namespace, domain_filter, search_top_k)
# 시그니처이므로, 기존 호출의 (search_query, enhancement, namespace, top_k, route_cfg, data, skip_bm25)과 다름
```

**선택**: 방안 A — `_search_with_variations`에 `agent`와 `domain_filter`를 전달하는 방식으로 수정. `route_cfg`, `data`, `skip_bm25` 파라미터는 `_search_with_variations`에서 불필요.

**수정 코드**:
```python
# rag_pipeline.py:1030-1045 교체
if len(results) < 3 and use_enhancement and domain_confidence and domain_confidence < 0.5:
    try:
        logging.info("[Re-Search] Triggering (results=%d, conf=%.2f)", len(results), domain_confidence)
        _retry_enhancement = _enhance_query(search_query, namespace, route_cfg, True, use_hyde=True)
        _retry_results = _search_with_variations(
            agent, _retry_enhancement, namespace,
            domain_filter, top_k * 2,
        )
        _existing_ids = {r.get('id') for r in results}
        for rr in _retry_results:
            if rr.get('id') not in _existing_ids:
                results.append(rr)
        logging.info("[Re-Search] Total results after retry: %d", len(results))
    except Exception as _rs_e:
        logging.warning("[Re-Search] Failed: %s", _rs_e)
```

---

### BUG-2: `domain_confidence` 미초기화 (`namespace == 'all'` 시 NameError)

**위치**: `services/rag_pipeline.py:791-821`

**현상**:
```python
detected_namespace = None
detected_domain_label = None
secondary_ns = None
if namespace != 'all':                                    # ← 'all'이면 이 블록 스킵
    detected_namespace, domain_confidence, ... = classify_domain(...)  # ← domain_confidence 미할당

# QW-2: Adaptive top_k based on domain confidence
if domain_confidence and domain_confidence < 0.7:         # ← NameError!
```

**영향**: `namespace='all'` (통합 검색 페이지) 사용 시 `NameError: name 'domain_confidence' is not defined` → 500 에러 반환. 다만 현재 `except Exception` 최상위 핸들러가 잡아서 사용자에게는 "답변 생성 중 오류" 메시지만 표시됨.

**수정 방안**:
```python
# rag_pipeline.py:791 영역에 초기화 추가
detected_namespace = None
detected_domain_label = None
secondary_ns = None
domain_confidence = 0.0          # ← 추가
_secondary_conf = 0.0            # ← 추가
if namespace != 'all':
    detected_namespace, domain_confidence, detected_domain_label, \
        secondary_ns, _secondary_conf = classify_domain(query, namespace)
    ...
```

---

## 3. High Issues (조기 수정 권장)

### HIGH-1: `secondary_ns` Dead Variable

**위치**: `services/rag_pipeline.py:793, 796`

**현상**: `classify_domain()`이 5-tuple을 반환하고 `secondary_ns`에 할당하지만, 이후 어디에서도 사용하지 않음. `search-quality` 설계 문서에서는 secondary namespace에 대한 추가 검색을 의도했으나 미구현.

**수정 방안**:
- **Option A (정리)**: 현재 미사용이므로 변수만 `_`로 교체하여 의도적 무시 표시
- **Option B (구현)**: secondary namespace 검색을 실제로 구현

**권장**: Option A — secondary search는 성능 비용이 크고 현재 품질에 실질적 영향이 미미. 추후 필요 시 별도 feature로 구현.

```python
# rag_pipeline.py:795-796
detected_namespace, domain_confidence, detected_domain_label, \
    _secondary_ns, _secondary_conf = classify_domain(query, namespace)
```

---

### HIGH-2: `ask` / `ask/stream` LLM 호출 코드 중복

**위치**: `api/v1/search.py:314-363` (ask) vs `api/v1/search.py:434-586` (ask/stream)

**현상**: 다음 로직이 거의 동일하게 중복:
1. `run_rag_pipeline(data)` 호출 후 결과 언패킹 (7개 변수)
2. `build_llm_messages()` 호출
3. `_resolve_llm()` → provider/model 결정
4. temperature / max_tokens 설정
5. provider별 분기 (gemini / anthropic / openai)

**위험**: 한쪽만 수정하고 다른 쪽을 누락하는 "불일치 버그"가 발생하기 쉬움.

**수정 방안**: 공통 로직을 헬퍼 함수로 추출.

```python
def _prepare_llm_call(pipeline: dict, data: dict):
    """Extract LLM call parameters from pipeline result.

    Returns: (messages, provider, model, temperature, max_tokens,
              sources, related_images, meta_data)
    """
    ...
```

**범위 제한**: 이번 audit에서는 **식별만** 하고, 리팩토링은 별도 feature로 관리. 변경 범위가 크고 regression 위험이 있기 때문.

---

### HIGH-3: Graph Enrichment 결과의 Score 스케일 혼합

**위치**: `services/rag_pipeline.py:894-900`

**현상**:
```python
results.append({
    'id': _vid,
    'score': _gr.graph_score if _gr else 0.0,   # graph_score: conf / (1.0 + hop * 0.5)
    'content': _meta.get('content', ''),
    'metadata': _meta,
    'graph_path': ...,
})
```

Graph에서 추가된 결과의 `score`는 `graph_score`(엔티티 경로 신뢰도 기반, 0~1)이고, 벡터 검색 결과의 `score`는 코사인 유사도(0~1). 스케일은 동일하지만 **의미가 다름**.

**후속 Phase 영향**:
- **Phase 4 (Hybrid)**: `search_with_keyword_boost`가 `score` 필드를 벡터 점수로 취급 → graph 결과의 순위가 왜곡될 수 있음
- **Phase 5 (Rerank)**: cross-encoder가 content 기반으로 재점수화하므로 영향 적음
- **Phase 6 (Filtering)**: `min_score` 필터에서 graph 결과가 부당하게 제외될 수 있음 (graph_score가 낮은 경우)

**수정 방안**:
```python
results.append({
    'id': _vid,
    'score': max(_gr.graph_score, MIN_RELEVANCE_SCORE + 0.1) if _gr else 0.3,
    'metadata': _meta,
    'source': 'graph',  # 출처 태그 추가
    ...
})
```

**권장**: `source` 태그를 추가하여 Phase 4/6에서 graph 결과를 구분 처리할 수 있게 하되, 당장은 `score`에 최소 보장값만 설정.

---

## 4. Medium Issues

### MED-1: Phase 7.5 + 7.6 + 7.7 직렬 실행

**위치**: `services/rag_pipeline.py:1148-1207`

**현상**: Safety Cross-Search, MSDS Cross-Search, Laborlaw Law API 호출이 순차적으로 실행됨. 각각 독립적인 외부 API 호출이므로 병렬화 가능.

**영향**: 최악 케이스에서 Safety(~2s) + MSDS(~15s) + Law API(~3s) = ~20s 추가 지연.

**수정 방안**:
```python
import concurrent.futures

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
    futures = {}

    if safety_ns and namespace in ('', primary_ns):
        futures['safety'] = ex.submit(_search_safety_context, search_query)

    if namespace in MSDS_CROSS_SEARCH_NAMESPACES:
        chemical_names = _extract_chemical_names(search_query, context)
        if chemical_names:
            futures['msds'] = ex.submit(_search_msds_context, chemical_names)

    if namespace == 'laborlaw':
        futures['law'] = ex.submit(search_labor_laws, query)

    # Collect results...
```

**비용**: 코드 복잡도 증가. 에러 핸들링을 개별 future에서 처리해야 함.

**판단**: laborlaw에서만 실제로 3개 Phase가 모두 활성화되므로, laborlaw 응답 시간 개선 효과가 가장 큼. 별도 feature로 분리 권장.

---

### MED-2: `CommunitySearcher` 한국어 매칭 비효율

**위치**: `services/community_searcher.py:54-74`

**현상**: `query.lower().split()`으로 단어를 분리하는데, 한국어는 공백 기준 분리가 형태소 단위가 아니라서 매칭 정확도가 낮음. 예: "반도체 제조 공정"은 3단어지만, "반도체제조공정"은 1단어로 매칭 실패.

**수정 방안**: 기존 `_okt` (konlpy) 모듈이 hybrid_searcher에서 사용 중이므로 동일하게 적용 가능. 단, community_searcher는 KG 데이터가 없으면 실행되지 않으므로 우선순위 낮음.

---

### MED-3: `DOMAIN_CONFIG` 키와 `DOMAIN_KEYWORDS` 키 네이밍 혼동

**현상**:
- `DOMAIN_CONFIG`: `safeguide` → `namespace: 'kosha'`
- `NAMESPACE_DOMAIN_MAP`: `'kosha'` → `'safeguide'`
- `DOMAIN_KEYWORDS` (query_router): `'kosha'` (네임스페이스 키 사용)
- `DOMAIN_RRF_CONFIG` (hybrid_searcher): `'safeguide'` (도메인 키 사용)
- `DOMAIN_RERANK_CONFIG` (reranker): `'safeguide'` (도메인 키 사용)

**결론**: 기능적으로는 문제 없음. `query_router`는 네임스페이스 키를, hybrid/reranker는 `NAMESPACE_DOMAIN_MAP`을 통해 변환된 도메인 키를 사용. 하지만 네이밍 불일치가 혼동을 유발할 수 있음.

**수정 방안**: 주석으로 키 규칙 문서화. 리팩토링은 파급 범위가 커서 비권장.

---

### MED-4: `_search_with_variations` 순차 검색

**위치**: `services/rag_pipeline.py:733-758`

**현상**: N개의 enhanced query를 순차적으로 Pinecone에 검색. `_search_single_query` 안에 이미 retry 로직이 있어서 최악 케이스에서 N × 2 × API 호출.

**수정 방안**: `concurrent.futures.ThreadPoolExecutor`로 병렬 검색. PineconeAgent가 thread-safe한지 확인 필요 (httpx.Client는 thread-safe).

---

### MED-5: Graph entity cache 만료 정책 부재

**위치**: `services/graph_searcher.py:96-117`

**현상**: `_entity_cache`가 서버 재시작 또는 `invalidate_cache()` 호출 전까지 영구 유지. KG entity가 업데이트되어도 반영 안 됨.

**수정 방안**: TTL 기반 캐시로 전환 (예: 1시간). 또는 KG 빌드 후 `invalidate_cache()`를 자동 호출하는 hook 추가.

---

## 5. Low Issues

### LOW-1: `build_llm_prompts`의 Unused 파라미터

**위치**: `services/rag_pipeline.py:1210-1217`

**현상**: `calc_result`, `labor_classification` 파라미터에 "Unused (kept for API compatibility)" 주석. 실제로 호출부에서 항상 `None` 전달됨. 하지만 `search.py`에서 `pipeline.get('labor_calc_result')` / `pipeline.get('labor_classification')` 값을 전달하므로 잠재적으로 `None`이 아닐 수 있음.

**확인 결과**: `labor_calc_result`는 `run_rag_pipeline`에서 설정되지 않음 (result dict에 없음). `labor_classification`도 동일. 따라서 호출부에서 `pipeline.get()`은 항상 `None` 반환.

**수정 방안**: 파라미터 시그니처에서 제거하거나, docstring을 현재 상태와 맞게 업데이트.

---

### LOW-2: Enhancement 캐시 적중률 로깅 부재

**위치**: `src/query_enhancer.py:56-66`

**현상**: `_cache_get`에서 hit/miss를 로깅하지 않아 캐시 효과를 측정할 수 없음.

---

### LOW-3: `USE_LOCAL_RERANKER` 환경변수 사용 확인

**위치**: `src/reranker.py:538`

**현상**: `get_reranker()` 팩토리 함수에서 `USE_LOCAL_RERANKER`를 참조하여 cross-encoder 사용 여부 결정. 정상 동작 중. `.env.example`에도 문서화됨.

**결론**: 이슈 아님.

---

## 6. 수정 구현 순서

### Phase 1: Critical 버그 수정 (즉시)

| 순서 | 이슈 | 파일 | 예상 변경량 |
|------|------|------|------------|
| 1 | BUG-2: `domain_confidence` 초기화 | `rag_pipeline.py:791` | +2줄 |
| 2 | BUG-1: `_run_multi_query_search` → `_search_with_variations` | `rag_pipeline.py:1035` | ~5줄 변경 |

### Phase 2: High 이슈 정리

| 순서 | 이슈 | 파일 | 예상 변경량 |
|------|------|------|------------|
| 3 | HIGH-1: `secondary_ns` → `_secondary_ns` | `rag_pipeline.py:796` | 1줄 |
| 4 | HIGH-3: Graph 결과에 `source` 태그 + 최소 score 보장 | `rag_pipeline.py:894` | ~3줄 |

### Phase 3: Medium 이슈 (별도 feature 권장)

| 이슈 | 제안 feature 이름 |
|------|-------------------|
| MED-1: Phase 7.5~7.7 병렬화 | `pipeline-phase7-parallel` |
| MED-2: 커뮤니티 한국어 매칭 | `community-korean-tokenize` |
| MED-4: multi-query 병렬 검색 | `parallel-multi-query-search` |

### Phase 4: Low 이슈 (기회 비용 시 처리)

- LOW-1: 사용 안 되는 파라미터 정리
- LOW-2: 캐시 hit/miss 로깅 추가

---

## 7. 변경 영향 분석

### 7.1 BUG-1 + BUG-2 수정의 영향 범위

```
rag_pipeline.py (직접 수정)
  └─ run_rag_pipeline() 함수 내부만 변경
  └─ 함수 시그니처 변경 없음
  └─ 호출부 (search.py: api_ask, api_ask_stream) 변경 불필요
```

### 7.2 회귀 위험

| 수정 | 회귀 위험 | 검증 방법 |
|------|-----------|-----------|
| BUG-2 | 매우 낮음 | `namespace='all'`로 `/ask` 호출 테스트 |
| BUG-1 | 낮음 | 결과 <3 + confidence <0.5인 쿼리로 re-search 트리거 확인 |
| HIGH-1 | 없음 | 변수명 변경만 (기능 변경 없음) |
| HIGH-3 | 낮음 | graph 결과 포함 시 최종 sources에 올바르게 포함되는지 확인 |

---

## 8. 검증 계획

### 8.1 수정 후 검증 쿼리 세트

| # | 쿼리 | 도메인 | 검증 대상 |
|---|------|--------|-----------|
| T-1 | "CVD와 PVD의 차이점" | semiconductor | 정상 검색 (regression) |
| T-2 | "최저임금 위반 여부 확인" | laborlaw | laborlaw 전체 경로 |
| T-3 | "안전보건 교육" | all (통합) | BUG-2 수정 검증 |
| T-4 | 매우 희귀한 용어 검색 | semiconductor | BUG-1 re-search 트리거 검증 |
| T-5 | "CMP 공정 위험성" | semiconductor | Graph + MSDS cross-search |
| T-6 | "직업계고 현장실습 안전" | field-training | 도메인 라우팅 |

### 8.2 debug 모드 활용

```json
POST /api/v1/ask
{
  "query": "안전보건 교육",
  "namespace": "all",
  "debug": true
}
```
→ `latencies` 필드에서 각 Phase 소요시간 확인
→ `query_type`, `detected_namespace` 필드로 라우팅 검증

---

## 9. 파일 변경 목록

| 파일 | 변경 내용 | 줄 수 |
|------|-----------|-------|
| `services/rag_pipeline.py` | BUG-1, BUG-2, HIGH-1, HIGH-3 수정 | ~15줄 |

**변경 파일 1개, 총 변경량 ~15줄** — 최소 범위 수정으로 Critical/High 이슈 해결.
