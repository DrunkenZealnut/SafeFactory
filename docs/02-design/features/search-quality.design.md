# 검색 품질 개선 Design

> Plan 참조: `docs/01-plan/features/search-quality.plan.md`

---

## 1. 수정 파일 맵

```
services/
├── rag_pipeline.py      # [MOD] QW-1,2,3 + CI-1,2,4 + RB-1
├── query_router.py      # [MOD] QW-5 + CI-4
└── (graph_config.py 변경없음)

src/
├── hybrid_searcher.py   # [MOD] QW-4
├── reranker.py          # [MOD] CI-3
└── query_enhancer.py    # [MOD] CI-1 + RB-2
```

---

## 2. Week 1 — Quick Wins 상세 설계

### QW-1: Keyword Boost 보장

**파일**: `services/rag_pipeline.py` (Phase 4 부근)

**현재 상태**: `search_with_keyword_boost()` 호출 시 `keywords` 변수가 Phase 1에서 추출된 값이지만, `use_enhancement=False`일 때 빈 리스트로 전달됨.

**수정**:
```python
# rag_pipeline.py Phase 4 직전 (line ~888 부근)
# 현재: if use_enhancement and len(results) > 3:
# 변경: keywords가 비었으면 쿼리에서 직접 추출
if not keywords:
    keywords = _extract_fallback_keywords(search_query)

# 신규 헬퍼 함수
def _extract_fallback_keywords(query: str) -> list[str]:
    """쿼리에서 2글자 이상 단어를 키워드로 추출 (enhancement 비활성화 시 fallback)."""
    import re
    tokens = re.findall(r'[가-힣a-zA-Z0-9]{2,}', query)
    return tokens[:10]
```

### QW-2: Adaptive top_k

**파일**: `services/rag_pipeline.py` (Phase 0 직후)

**현재 상태**: `top_k`가 사용자 입력값 그대로 전체 파이프라인에 사용됨.

**수정 위치**: Phase 0 완료 직후 (line ~800)
```python
# Phase 0 이후, Phase 1 이전에 삽입
# domain_confidence는 classify_domain()의 두 번째 반환값
_adaptive_mult = 1.0
if domain_confidence < 0.4:
    _adaptive_mult = 2.0
elif domain_confidence < 0.7:
    _adaptive_mult = 1.5
_original_top_k = top_k
top_k = min(int(top_k * _adaptive_mult), top_k * 3)  # cap: 3배 이내
if _adaptive_mult > 1.0:
    logging.info("[Adaptive top_k] confidence=%.2f, top_k: %d → %d",
                 domain_confidence, _original_top_k, top_k)
```

**주의**: `domain_confidence`가 0인 경우(도메인 분류 미실행)는 기본값 유지.

### QW-3: MIN_TOKEN_COUNT 상향

**파일**: `services/rag_pipeline.py` (line 41)

**현재**: `MIN_TOKEN_COUNT = int(os.environ.get('MIN_TOKEN_COUNT', '30'))`

**변경**: `MIN_TOKEN_COUNT = int(os.environ.get('MIN_TOKEN_COUNT', '80'))`

**근거**: 30토큰 ≈ 한글 90자. 목차, 제목만으로 구성된 짧은 청크가 결과에 포함되어 컨텍스트를 낭비. 80토큰(≈240자)이면 최소 1문단 이상의 의미 있는 내용 보장.

### QW-4: 동적 RRF K

**파일**: `src/hybrid_searcher.py` (line 32, 68-80)

**현재**: `RRF_K = int(os.environ.get("RRF_K", "60"))` — 고정값

**변경**: `__init__`에서 기본값을 받되, `search()` 호출 시 결과 크기에 따라 동적 조정.

```python
# hybrid_searcher.py — search() 메서드 내부, RRF 계산 직전
def _dynamic_rrf_k(self, result_count: int) -> int:
    """결과 크기에 비례한 RRF K값. 작은 결과셋에서 순위 분별력 확보."""
    if result_count <= 10:
        return max(20, self.rrf_k // 3)
    elif result_count <= 30:
        return max(30, self.rrf_k // 2)
    return self.rrf_k  # 기본값 유지

# search(), search_with_keyword_boost() 내에서:
# self.rrf_k → self._dynamic_rrf_k(len(vector_results))
```

### QW-5: 도메인 라우팅 동의어

**파일**: `services/query_router.py` (line 77 DOMAIN_KEYWORDS 위)

**추가**:
```python
# 도메인 키워드 동의어 맵 — 쿼리 전처리에서 정규화
DOMAIN_SYNONYMS = {
    '근로기준법': '노동법',
    '산업안전보건법': '안전보건',
    '산안법': '안전보건',
    '화학물질안전': 'MSDS',
    '물질안전보건자료': 'MSDS',
    'SDS': 'MSDS',
    '현장실습': '현장실습',
    '인턴': '현장실습',
    '실습생': '현장실습',
}

# classify_domain() 함수 시작 부분에 동의어 치환 추가
def classify_domain(query: str, default_namespace: str = '') -> tuple:
    # 동의어 치환
    _q = query
    for synonym, canonical in DOMAIN_SYNONYMS.items():
        if synonym in _q:
            _q = _q + ' ' + canonical  # 원본 보존 + canonical 추가
    # 이후 기존 키워드 매칭 로직에 _q 사용
    ...
```

---

## 3. Week 2 — Core Improvements 상세 설계

### CI-1: HyDE 조건부 활성화

**파일**: `services/rag_pipeline.py` (Phase 1 호출부), `src/query_enhancer.py`

**현재**: `QUERY_TYPE_CONFIG`에서 `use_hyde`가 `procedural`/`comparison`에만 `True`.

**변경**: `rag_pipeline.py`에서 confidence 기반 오버라이드 추가.

```python
# rag_pipeline.py — Phase 1 호출 직전 (line ~805 부근)
route_cfg = QUERY_TYPE_CONFIG.get(query_type, QUERY_TYPE_CONFIG['factual'])
_use_hyde = route_cfg.get('use_hyde', False)

# confidence가 낮으면 factual이라도 HyDE 활성화
if not _use_hyde and domain_confidence < 0.6:
    _use_hyde = True
    logging.info("[HyDE Override] Enabled for low-confidence query (%.2f)", domain_confidence)

# Phase 1 호출 시 _use_hyde 전달
enhancement = _run_query_enhancement(
    ...,
    use_hyde=_use_hyde,
    ...
)
```

### CI-2: MMR 재활성화

**파일**: `services/rag_pipeline.py` (Phase 5 직후)

**현재**: `reranker.py`에 `mmr()` 메서드가 구현되어 있으나, `rag_pipeline.py`에서 미호출.

**변경**: Phase 5 리랭킹 후 MMR 적용 단계 추가.

```python
# rag_pipeline.py — Phase 5 직후, Phase 6 직전에 삽입
# ========================================
# Phase 5.5: MMR Diversity (중복 제거)
# ========================================
if use_enhancement and len(results) > 5:
    try:
        # 간이 MMR: content 기반 유사도로 중복 제거
        _seen_contents = set()
        _diverse_results = []
        for r in results:
            _content_key = (r.get('content', '') or r.get('metadata', {}).get('content', ''))[:200]
            if _content_key not in _seen_contents:
                _seen_contents.add(_content_key)
                _diverse_results.append(r)
        _removed = len(results) - len(_diverse_results)
        if _removed > 0:
            results = _diverse_results
            logging.info("[MMR Diversity] Removed %d near-duplicate results", _removed)
    except Exception as e:
        logging.warning("[MMR Diversity] Failed: %s", e)
```

**설계 결정**: 임베딩 기반 full MMR 대신 content prefix 기반 간이 중복 제거 채택. 이유:
- `mmr()` 메서드는 `query_embedding`과 `doc_embeddings`를 요구하나, Phase 5 시점에서 개별 문서 임베딩이 없음
- content prefix 200자 비교가 실용적이고 latency 무시 가능 (<1ms)
- 향후 임베딩 기반 MMR 필요 시 Phase 2에서 임베딩을 캐시하여 전달

### CI-3: Robust 정규화

**파일**: `src/reranker.py`

**현재**: Min-Max 정규화 (`(x - min) / (max - min)`)

**변경**: Percentile 기반 정규화로 이상치 영향 감소.

```python
# reranker.py — hybrid_rerank() 메서드 내 정규화 로직
def _robust_normalize(scores: list[float]) -> list[float]:
    """Percentile-based normalization (5th~95th)."""
    if not scores or len(scores) < 2:
        return scores
    sorted_s = sorted(scores)
    p5 = sorted_s[max(0, len(sorted_s) // 20)]       # 5th percentile
    p95 = sorted_s[min(len(sorted_s) - 1, len(sorted_s) * 19 // 20)]  # 95th percentile
    rng = p95 - p5
    if rng < 1e-9:
        return [0.5] * len(scores)
    return [max(0.0, min(1.0, (s - p5) / rng)) for s in scores]
```

**적용 위치**: 각 Reranker 구현체의 `hybrid_rerank()`에서 기존 `_normalize()`를 `_robust_normalize()`로 교체.

### CI-4: 다중 도메인 라우팅

**파일**: `services/query_router.py`, `services/rag_pipeline.py`

**현재**: `classify_domain()` → 단일 (namespace, confidence, label) 반환.

**변경**: 상위 2개 도메인 반환 + 파이프라인에서 병합 검색.

```python
# query_router.py — classify_domain() 수정
def classify_domain(query: str, default_namespace: str = '') -> tuple:
    """Returns (primary_ns, confidence, label, secondary_ns, secondary_conf)"""
    ...
    # 기존 scoring 로직 유지
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    primary_ns, primary_score = sorted_scores[0] if sorted_scores else ('', 0)
    secondary_ns, secondary_score = sorted_scores[1] if len(sorted_scores) > 1 else ('', 0)

    # 정규화
    primary_conf = min(primary_score / 5.0, 1.0)
    secondary_conf = min(secondary_score / 5.0, 1.0)

    return (primary_ns, primary_conf, label,
            secondary_ns if secondary_conf > 0.3 else None, secondary_conf)
```

```python
# rag_pipeline.py — Phase 0 수정
detected_namespace, domain_confidence, detected_domain_label, \
    secondary_ns, secondary_conf = classify_domain(query, namespace)

# Phase 2 이후: secondary namespace도 검색
if secondary_ns and secondary_conf > 0.3 and secondary_ns != namespace:
    try:
        _secondary_results = _run_multi_query_search(
            ..., namespace=secondary_ns, top_k=top_k // 2)
        results.extend(_secondary_results)
        logging.info("[Multi-Domain] Added %d results from %s (conf=%.2f)",
                     len(_secondary_results), secondary_ns, secondary_conf)
    except Exception as e:
        logging.warning("[Multi-Domain] Secondary search failed: %s", e)
```

**하위 호환**: `classify_domain()`의 반환값이 5-tuple로 변경되므로, 기존 호출부 (`3-tuple unpacking`)를 모두 업데이트해야 함. `query_router.py` 내 다른 함수에서의 호출도 확인 필요.

---

## 4. Week 3 — Robustness 상세 설계

### RB-1: 2단계 재검색

**파일**: `services/rag_pipeline.py` (Phase 6 직후)

```python
# Phase 6 이후, Phase 7 이전
# confidence가 낮고 결과도 적으면 재검색
if len(results) < 3 and domain_confidence < 0.5 and use_enhancement:
    logging.info("[Re-Search] Triggering 2nd search (results=%d, conf=%.2f)", len(results), domain_confidence)
    try:
        # HyDE 강제 활성화 + top_k 2배
        _retry_enhancement = _run_query_enhancement(
            ..., use_hyde=True, num_variations=3)
        _retry_results = _run_multi_query_search(
            ..., top_k=top_k * 2, enhancement=_retry_enhancement)
        # 기존 결과에 병합 (중복 제거)
        _existing_ids = {r.get('id') for r in results}
        for rr in _retry_results:
            if rr.get('id') not in _existing_ids:
                results.append(rr)
        logging.info("[Re-Search] Added %d new results", len(results) - len(_existing_ids))
    except Exception as e:
        logging.warning("[Re-Search] Failed: %s", e)

_timings['phase6_5_research_ms'] = round((time.perf_counter() - _t0) * 1000)
```

### RB-2: Multi-query 캐싱 TTL 증대

**파일**: `src/query_enhancer.py` (line 48-49)

**변경**:
```python
# 현재
_CACHE_TTL = 300   # 5 minutes
_CACHE_MAX = 200   # max entries

# 변경
_CACHE_TTL = 3600   # 1 hour
_CACHE_MAX = 500    # max entries
```

**근거**: 동일 질문이 수업 시간(40~50분) 내에 반복될 확률 높음. 1시간 TTL + 500 항목으로 수업 1교시 동안 캐시 유지.

---

## 5. 구현 순서 (파일 단위)

### Week 1

| 순서 | 파일 | 항목 | 의존성 |
|------|------|------|--------|
| 1 | `rag_pipeline.py` | QW-3: MIN_TOKEN_COUNT 80 | 없음 |
| 2 | `rag_pipeline.py` | QW-1: fallback keywords + boost 보장 | 없음 |
| 3 | `rag_pipeline.py` | QW-2: Adaptive top_k | Phase 0 |
| 4 | `hybrid_searcher.py` | QW-4: 동적 RRF K | 없음 |
| 5 | `query_router.py` | QW-5: DOMAIN_SYNONYMS | 없음 |

### Week 2

| 순서 | 파일 | 항목 | 의존성 |
|------|------|------|--------|
| 6 | `rag_pipeline.py` | CI-1: HyDE 조건부 활성화 | QW-2 |
| 7 | `rag_pipeline.py` | CI-2: MMR 간이 중복 제거 | 없음 |
| 8 | `reranker.py` | CI-3: Robust 정규화 | 없음 |
| 9 | `query_router.py` + `rag_pipeline.py` | CI-4: 다중 도메인 | QW-5 |

### Week 3

| 순서 | 파일 | 항목 | 의존성 |
|------|------|------|--------|
| 10 | `rag_pipeline.py` | RB-1: 2단계 재검색 | CI-1 |
| 11 | `query_enhancer.py` | RB-2: 캐싱 TTL 증대 | 없음 |

---

## 6. 에러 핸들링 원칙

모든 개선 항목은 **try/except로 감싸서 fallback** 보장:

| 항목 | 실패 시 동작 |
|------|-------------|
| Adaptive top_k | 원래 top_k 사용 |
| 동적 RRF K | 기본 RRF_K (60) 사용 |
| 도메인 동의어 | 원래 쿼리로 매칭 |
| HyDE 오버라이드 | HyDE 없이 진행 |
| MMR 중복 제거 | 원래 결과 유지 |
| Robust 정규화 | Min-Max fallback |
| 다중 도메인 | primary 도메인만 검색 |
| 2단계 재검색 | 기존 결과만 사용 |

---

## 7. 성능 예산

| 항목 | 추가 latency | 측정 |
|------|-------------|------|
| QW-1~5 전체 | <5ms | 로직 계산만 |
| CI-1 HyDE 오버라이드 | +500ms (LLM 호출) | 캐시 히트 시 0ms |
| CI-2 MMR 간이 | <1ms | content prefix 비교 |
| CI-3 Robust 정규화 | <1ms | 수학 계산 |
| CI-4 다중 도메인 | +300~500ms (추가 검색) | 병렬화 가능 |
| RB-1 재검색 | +1~2초 (조건부) | 저신뢰도일 때만 |
| **전체 (worst case)** | **+2초** | **저신뢰도 쿼리만** |
