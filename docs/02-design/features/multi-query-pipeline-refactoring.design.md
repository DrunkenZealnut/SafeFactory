# Multi-Query 파이프라인 리팩토링 — 상세 설계

> Plan 문서: `docs/01-plan/features/multi-query-pipeline-refactoring.plan.md`

---

## 1. 설계 개요

### 1.1 변경 범위 요약

| 파일 | 변경 유형 | 핵심 변경 |
|------|-----------|-----------|
| `src/query_enhancer.py` | **대폭 수정** | `EnhancementResult` dataclass 추가, `enhance_query()` 재설계, 모듈 executor, TTL 캐시 |
| `services/rag_pipeline.py` | **대폭 수정** | Phase 1 → `_enhance_query()` 추출, Phase 2 → `_search_with_variations()` 추출, `_get_num_variations()` 제거 |
| `services/singletons.py` | **소폭 수정** | `shutdown_all()`에 executor 종료 추가, `invalidate_query_enhancer()`에 캐시 클리어 추가 |

### 1.2 변경하지 않는 것

- `services/query_router.py` — `QUERY_TYPE_CONFIG`, `classify_query_type()`, `classify_domain()` 그대로 유지
- `services/domain_config.py` — `NAMESPACE_DOMAIN_MAP` 등 그대로 유지
- `services/filters.py` — `build_domain_filter()` 그대로 유지
- `src/__init__.py` — `HttpClientMixin` 그대로 유지
- Phase 4-7 (Hybrid Search, Reranking, Context Optimization, LLM Generation) — 변경 없음
- API 응답 형식 — 변경 없음
- 프론트엔드 — 변경 없음

---

## 2. `src/query_enhancer.py` 상세 설계

### 2.1 `EnhancementResult` dataclass

파일 상단, `QueryEnhancer` 클래스 앞에 위치.

```python
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class EnhancementResult:
    """쿼리 확장 결과를 구조화하여 반환."""
    original: str
    variations: List[str] = field(default_factory=list)
    hyde_doc: Optional[str] = None
    expanded_query: Optional[str] = None
    keywords: List[str] = field(default_factory=list)

    @property
    def search_queries(self) -> List[str]:
        """일반 검색에 사용할 쿼리 목록.

        포함 순서: 원본 → multi-query 변형 → synonym 확장 쿼리.
        원본은 항상 첫 번째 (variations[0]이 원본).
        """
        queries = list(self.variations) if self.variations else [self.original]
        if self.expanded_query and self.expanded_query not in queries:
            queries.append(self.expanded_query)
        return queries

    @property
    def hyde_queries(self) -> List[str]:
        """HyDE 기반 검색 쿼리. 일반 쿼리와 분리하여 관리."""
        if self.hyde_doc and self.hyde_doc != self.original:
            return [self.hyde_doc]
        return []

    @property
    def all_queries(self) -> List[str]:
        """모든 검색 쿼리 (일반 + HyDE). 기존 동작 호환용."""
        return self.search_queries + self.hyde_queries
```

**설계 결정**:
- `variations`에는 항상 원본 쿼리가 `[0]` 위치에 포함 (기존 `multi_query()` 동작 유지)
- `hyde_doc`을 별도 필드로 분리하여 검색 시 가중치 구분 가능
- `all_queries` property로 기존 `enhanced_queries` 리스트와 1:1 호환

### 2.2 모듈 레벨 ThreadPoolExecutor

```python
import atexit
import concurrent.futures

_enhancement_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=3,
    thread_name_prefix="query-enhance",
)
atexit.register(_enhancement_executor.shutdown, wait=False)
```

**설계 결정**:
- `max_workers=3` 유지 (multi_query + hyde + keywords 3개 병렬)
- `atexit`으로 프로세스 종료 시 정리
- `singletons.shutdown_all()`에서도 `shutdown_enhancement_executor()` 호출

**외부 접근용 함수** (singletons에서 호출):

```python
def shutdown_enhancement_executor():
    """모듈 레벨 executor 종료. singletons.shutdown_all()에서 호출."""
    _enhancement_executor.shutdown(wait=False)
```

### 2.3 TTL 캐시

`QueryEnhancer` 클래스 외부에 모듈 레벨로 구현.

```python
import threading
import time as _time

_cache_lock = threading.Lock()
_multi_query_cache: dict = {}  # key → (result, timestamp)
_CACHE_TTL = 300   # 5분
_CACHE_MAX = 200   # 최대 엔트리

def _cache_get(key: str) -> Optional[List[str]]:
    """TTL 캐시에서 multi-query 결과 조회."""
    with _cache_lock:
        entry = _multi_query_cache.get(key)
        if entry is None:
            return None
        result, ts = entry
        if _time.time() - ts > _CACHE_TTL:
            del _multi_query_cache[key]
            return None
        return result

def _cache_set(key: str, value: List[str]):
    """TTL 캐시에 multi-query 결과 저장."""
    with _cache_lock:
        # 크기 제한: 오래된 엔트리 제거
        if len(_multi_query_cache) >= _CACHE_MAX:
            # 가장 오래된 20% 제거
            sorted_keys = sorted(
                _multi_query_cache,
                key=lambda k: _multi_query_cache[k][1],
            )
            for k in sorted_keys[:_CACHE_MAX // 5]:
                del _multi_query_cache[k]
        _multi_query_cache[key] = (value, _time.time())

def clear_enhancement_cache():
    """캐시 전체 초기화. invalidate_query_enhancer()에서 호출."""
    with _cache_lock:
        _multi_query_cache.clear()
```

**캐시 키 생성**:

```python
def _make_cache_key(query: str, num_variations: int) -> str:
    return hashlib.md5(f"{query}:{num_variations}".encode()).hexdigest()
```

**설계 결정**:
- `functools.lru_cache` 대신 dict 기반 구현 → TTL 지원 필요
- 캐시는 `multi_query()` 결과에만 적용 (HyDE는 창의적 생성이므로 캐싱 부적합)
- `threading.Lock` 사용 (read-heavy이므로 `RLock` 불필요)
- 크기 초과 시 오래된 20% 일괄 삭제 (개별 삭제보다 효율적)

### 2.4 `_get_num_variations()` 이동

`rag_pipeline.py`에서 `QueryEnhancer` 정적 메서드로 이동:

```python
@staticmethod
def _get_num_variations(query: str) -> int:
    """쿼리 길이에 따라 멀티쿼리 변형 수를 동적으로 결정."""
    length = len(query)
    if length < 15:
        return 1
    elif length < 40:
        return 2
    else:
        return 3
```

### 2.5 `multi_query()` 수정

캐시 적용 추가. 기존 시그니처·동작 유지.

```python
def multi_query(self, query: str, num_variations: int = 3) -> List[str]:
    # 캐시 조회
    cache_key = _make_cache_key(query, num_variations)
    cached = _cache_get(cache_key)
    if cached is not None:
        logging.info("[Multi-Query Cache] HIT for '%.30s...'", query)
        return cached

    # 기존 LLM 호출 로직 (변경 없음)
    ...

    # 캐시 저장
    _cache_set(cache_key, result)
    return result
```

### 2.6 `enhance_query()` 재설계

기존 `enhance_query()` 메서드를 pipeline 실제 호출 패턴에 맞게 교체.

```python
def enhance_query(
    self,
    query: str,
    domain: str = "general",
    use_multi_query: bool = True,
    use_hyde: bool = False,
    use_keywords: bool = True,
) -> EnhancementResult:
    """모든 확장 기법을 병렬 적용하여 EnhancementResult 반환.

    Args:
        query: 원본 사용자 쿼리.
        domain: 도메인 키 (e.g., 'semiconductor', 'laborlaw').
        use_multi_query: multi-query 변형 생성 여부.
        use_hyde: HyDE 가상 문서 생성 여부.
        use_keywords: 키워드 추출 여부.

    Returns:
        EnhancementResult with structured query expansion data.
    """
    # 1. Synonym expansion (동기, 빠름)
    expanded_query = self.expand_with_synonyms(query, domain)
    if expanded_query == query:
        expanded_query = None
    else:
        logging.info("[Synonym Expansion] '%s' → '%s'", query, expanded_query)

    # 2. 병렬 실행: multi_query + HyDE + keywords
    variations = [query]
    hyde_doc = None
    keywords = []

    futures = {}
    if use_multi_query:
        num_vars = self._get_num_variations(query)
        futures['multi'] = _enhancement_executor.submit(
            self.multi_query, query, num_vars,
        )
    if use_hyde and len(query) >= 10:
        futures['hyde'] = _enhancement_executor.submit(
            self.hyde, query, domain,
        )
    if use_keywords:
        futures['keywords'] = _enhancement_executor.submit(
            self.extract_keywords_fast, query,
        )

    # 3. 결과 수집
    if 'multi' in futures:
        try:
            variations = futures['multi'].result(timeout=10)
            logging.info(
                "[Query Enhancement] Generated %d query variations",
                len(variations),
            )
        except Exception as e:
            logging.warning("[Query Enhancement] multi_query failed: %s", e)
            variations = [query]

    if 'hyde' in futures:
        try:
            hyde_doc = futures['hyde'].result(timeout=10)
            if hyde_doc == query:
                hyde_doc = None
            elif hyde_doc:
                logging.info("[HyDE] Generated hypothetical document")
        except Exception as e:
            logging.warning("[HyDE] Failed: %s", e)
            hyde_doc = None

    if 'keywords' in futures:
        try:
            keywords = futures['keywords'].result(timeout=10)
            logging.info("[Query Enhancement] Keywords: %s", keywords)
        except Exception as e:
            logging.warning("[Query Enhancement] keywords failed: %s", e)
            keywords = []

    return EnhancementResult(
        original=query,
        variations=variations,
        hyde_doc=hyde_doc,
        expanded_query=expanded_query,
        keywords=keywords,
    )
```

**설계 결정**:
- `use_hyde` 기본값을 `False`로 변경 (pipeline 실제 동작 반영: factual 타입은 HyDE 미사용)
- 모듈 레벨 `_enhancement_executor` 사용으로 매 호출마다 executor 생성 제거
- 로깅 패턴은 기존 pipeline 로깅과 동일하게 유지 (그래야 로그 파싱에 영향 없음)
- `use_keywords=True` 시 `extract_keywords_fast` 사용 (LLM 기반 `extract_keywords` 아님 — pipeline 현행 동작)

### 2.7 기존 `enhance_query()` 하위호환

기존 dict 반환 `enhance_query()`를 사용하는 코드가 있을 수 있으므로 (`if __name__ == "__main__"` 블록), `__main__` 블록을 `EnhancementResult` 사용으로 업데이트.

### 2.8 삭제 대상

- 기존 `enhance_query()` 메서드의 dict 반환 코드 → `EnhancementResult` 반환으로 교체
- `rag_pipeline.py`의 `_get_num_variations()` 함수 → `QueryEnhancer._get_num_variations()`으로 이동

---

## 3. `services/rag_pipeline.py` 상세 설계

### 3.1 `_enhance_query()` 함수 (Phase 1 대체)

기존 line 647-717의 Phase 1 블록 전체를 대체하는 모듈 레벨 함수.

```python
from src.query_enhancer import EnhancementResult

def _enhance_query(
    search_query: str,
    namespace: str,
    route_cfg: dict,
    use_enhancement: bool,
) -> EnhancementResult:
    """Phase 1: 쿼리 확장.

    synonym expansion, multi-query, HyDE, keyword extraction을
    QueryEnhancer.enhance_query()에 위임.

    Args:
        search_query: 원본 검색 쿼리.
        namespace: Pinecone 네임스페이스.
        route_cfg: QUERY_TYPE_CONFIG에서 가져온 라우팅 설정.
        use_enhancement: enhancement 활성화 여부.

    Returns:
        EnhancementResult (enhancement 비활성화 시 원본만 포함).
    """
    if not use_enhancement:
        return EnhancementResult(
            original=search_query,
            variations=[search_query],
        )

    try:
        query_enhancer = get_query_enhancer()
        domain = NAMESPACE_DOMAIN_MAP.get(namespace, 'semiconductor')

        return query_enhancer.enhance_query(
            query=search_query,
            domain=domain,
            use_multi_query=route_cfg.get('use_multi_query', True),
            use_hyde=route_cfg.get('use_hyde', False),
            use_keywords=True,
        )
    except Exception as e:
        logging.warning("[Query Enhancement] Failed: %s, using original query", e)
        return EnhancementResult(
            original=search_query,
            variations=[search_query],
        )
```

**`run_rag_pipeline()` 내 호출**:

```python
# Phase 1: Query Enhancement
_t0 = time.perf_counter()
enhancement = _enhance_query(search_query, namespace, route_cfg, use_enhancement)
enhanced_queries = enhancement.all_queries  # 기존 변수명 호환
keywords = enhancement.keywords
_timings['phase1_enhancement_ms'] = round((time.perf_counter() - _t0) * 1000)
```

### 3.2 `_search_single_query()` 헬퍼 (재시도 로직 추출)

```python
def _search_single_query(
    agent,
    query: str,
    namespace: str,
    top_k: int,
    domain_filter: dict,
    is_all_namespace: bool,
) -> list:
    """단일 쿼리로 벡터 검색 수행. 실패 시 1회 재시도.

    Args:
        agent: PineconeAgent 인스턴스.
        query: 검색 쿼리 텍스트.
        namespace: Pinecone 네임스페이스.
        top_k: 반환할 최대 결과 수.
        domain_filter: 메타데이터 필터.
        is_all_namespace: 전체 네임스페이스 검색 여부.

    Returns:
        검색 결과 리스트 (실패 시 빈 리스트).
    """
    for attempt in range(2):
        try:
            if is_all_namespace:
                try:
                    uploader = get_uploader()
                    stats = uploader.get_stats()
                    ns_list = [ns for ns in stats.get('namespaces', {}).keys() if ns]
                except Exception:
                    ns_list = ['semiconductor', 'laborlaw', 'field-training']
                return agent.search_all_namespaces(
                    query=query,
                    namespaces=ns_list,
                    top_k=top_k,
                    filter=domain_filter,
                )
            else:
                return agent.search(
                    query=query,
                    top_k=top_k,
                    namespace=namespace,
                    filter=domain_filter,
                )
        except Exception as e:
            if attempt == 0:
                logging.warning(
                    "[Search] Attempt 1 failed for '%.30s...': %s — retrying",
                    query, e,
                )
                time.sleep(0.5)
            else:
                logging.warning(
                    "[Search] Attempt 2 failed for '%.30s...': %s",
                    query, e,
                )
    return []
```

### 3.3 `_content_hash()` 유틸리티

```python
def _content_hash(content: str) -> str:
    """검색 결과 dedup용 content hash."""
    return hashlib.sha256(content[:5000].encode()).hexdigest()
```

### 3.4 `_search_with_variations()` 함수 (Phase 2 대체)

```python
def _search_with_variations(
    agent,
    enhancement: EnhancementResult,
    namespace: str,
    domain_filter: dict,
    top_k: int,
    search_top_k: int,
) -> list:
    """Phase 2: multi-query 검색 + 결과 dedup 병합.

    일반 쿼리(search_queries)와 HyDE 쿼리(hyde_queries)를 순차 검색하고
    content hash 기반으로 중복을 제거하여 반환.

    Args:
        agent: PineconeAgent 인스턴스.
        enhancement: Phase 1의 EnhancementResult.
        namespace: Pinecone 네임스페이스.
        domain_filter: 메타데이터 필터.
        top_k: 최종 반환 top_k.
        search_top_k: Pinecone 검색 시 사용할 top_k (multiplier 적용됨).

    Returns:
        중복 제거된 검색 결과 리스트.
    """
    is_all_ns = (namespace == 'all')
    all_results = []
    seen_ids = set()

    def _collect(query: str):
        results = _search_single_query(
            agent, query, namespace, search_top_k, domain_filter, is_all_ns,
        )
        for r in results:
            content = r.get('metadata', {}).get('content', '')
            cid = _content_hash(content)
            if cid not in seen_ids:
                seen_ids.add(cid)
                all_results.append(r)

    # 일반 쿼리 검색 (원본 + multi-query 변형 + synonym 확장)
    for eq in enhancement.search_queries:
        _collect(eq)

    # HyDE 쿼리 검색 (별도 풀)
    for hq in enhancement.hyde_queries:
        _collect(hq)

    logging.info(
        "[Search] Retrieved %d unique documents from %d queries",
        len(all_results),
        len(enhancement.search_queries) + len(enhancement.hyde_queries),
    )
    return all_results
```

**설계 결정**:
- `all_queries` property를 사용하지 않고 `search_queries` + `hyde_queries`를 순서대로 검색
  → 향후 HyDE 결과에 다른 `search_top_k`를 적용하거나 가중치를 줄 수 있는 확장점 확보
- 기존 dedup 로직 (sha256 content[:5000]) 100% 유지
- 내부 `_collect()` 클로저로 반복 코드 제거

### 3.5 `run_rag_pipeline()` Phase 1-2 교체

**Before** (70줄):
```python
# Phase 1: Query Enhancement (line 647-717)
_t0 = time.perf_counter()
enhanced_queries = [search_query]
keywords = []
use_multi_query = route_cfg.get('use_multi_query', True)
use_hyde = route_cfg.get('use_hyde', False)
if use_enhancement:
    try:
        query_enhancer = get_query_enhancer()
        # ... 50줄의 inline 로직 ...
    except Exception as e:
        ...
_timings['phase1_enhancement_ms'] = ...

# Phase 2: Multi-Query Search (line 719-778)
domain_filter = build_domain_filter(search_query, namespace)
_t0 = time.perf_counter()
# ... 60줄의 inline 검색 루프 ...
results = all_results
_timings['phase2_search_ms'] = ...
```

**After** (15줄):
```python
# ========================================
# Phase 1: Query Enhancement
# ========================================
_t0 = time.perf_counter()
enhancement = _enhance_query(search_query, namespace, route_cfg, use_enhancement)
enhanced_queries = enhancement.all_queries
keywords = enhancement.keywords
_timings['phase1_enhancement_ms'] = round((time.perf_counter() - _t0) * 1000)

# ========================================
# Phase 2: Multi-Query Search (with domain metadata filtering)
# ========================================
domain_filter = build_domain_filter(search_query, namespace)
_t0 = time.perf_counter()
top_k_mult = route_cfg.get('top_k_mult', TOP_K_DEFAULT_MULT)
if skip_bm25:
    search_top_k = top_k * TOP_K_NO_BM25_MULT
else:
    search_top_k = top_k * top_k_mult
results = _search_with_variations(
    agent, enhancement, namespace, domain_filter, top_k, search_top_k,
)
_timings['phase2_search_ms'] = round((time.perf_counter() - _t0) * 1000)
```

**`enhanced_queries` 변수 유지 이유**:
- Phase 2 이후 코드에서 `enhanced_queries` 변수를 참조하는 곳이 있는지 확인 필요
- `logging.info("[Search] Retrieved %d ... from %d queries", ..., len(enhanced_queries))`
  → `_search_with_variations()` 내부로 이동했으므로 외부 참조 없음
- 만약 다른 Phase에서 참조하면 `enhancement.all_queries`로 대체

### 3.6 `_get_num_variations()` 제거

`rag_pipeline.py`에서 `_get_num_variations()` 함수(line 149-157) 삭제.
→ `QueryEnhancer._get_num_variations()` 정적 메서드로 이동 완료.

### 3.7 import 변경

```python
# 추가
from src.query_enhancer import EnhancementResult

# 제거 (더 이상 직접 사용하지 않음)
# concurrent.futures (Phase 1에서 제거)
```

`concurrent.futures`는 다른 Phase에서 사용 중인지 확인 후 제거.

---

## 4. `services/singletons.py` 상세 설계

### 4.1 `shutdown_all()` 수정

```python
def shutdown_all():
    """Close all singleton instances that hold resources."""
    from src.query_enhancer import shutdown_enhancement_executor
    with _lock:
        to_close = [_query_enhancer, _context_optimizer, _reranker]
    for inst in to_close:
        _close_if_possible(inst)
    shutdown_enhancement_executor()  # 추가
```

### 4.2 `invalidate_query_enhancer()` 수정

```python
def invalidate_query_enhancer():
    """Reset QueryEnhancer so it is re-created with the latest model setting."""
    from src.query_enhancer import clear_enhancement_cache
    global _query_enhancer
    with _lock:
        old, _query_enhancer = _query_enhancer, None
    _close_if_possible(old)
    clear_enhancement_cache()  # 추가
```

---

## 5. 구현 순서 (파일별 의존성)

```
[1] src/query_enhancer.py
    ├── EnhancementResult dataclass 추가
    ├── 모듈 레벨 executor + shutdown 함수
    ├── TTL 캐시 (모듈 레벨) + clear 함수
    ├── _get_num_variations() → 정적 메서드
    ├── multi_query() 캐시 적용
    └── enhance_query() 재설계 (EnhancementResult 반환)

[2] services/rag_pipeline.py (query_enhancer.py 완료 후)
    ├── import EnhancementResult 추가
    ├── _content_hash() 추가
    ├── _search_single_query() 추가
    ├── _search_with_variations() 추가
    ├── _enhance_query() 추가
    ├── run_rag_pipeline() Phase 1-2 교체
    └── _get_num_variations() 제거

[3] services/singletons.py (query_enhancer.py 완료 후)
    ├── shutdown_all() → executor 종료 추가
    └── invalidate_query_enhancer() → 캐시 클리어 추가
```

---

## 6. 동작 호환성 검증 체크리스트

| # | 검증 항목 | 방법 | 기대 결과 |
|---|----------|------|-----------|
| V1 | enhancement 비활성화 시 원본 쿼리만 검색 | `use_enhancement=False` 요청 | `EnhancementResult.all_queries == [원본]` |
| V2 | factual 타입 쿼리: multi-query O, HyDE X | factual 쿼리 전송 | variations 생성됨, hyde_doc is None |
| V3 | procedural 타입 쿼리: multi-query O, HyDE O | procedural 쿼리 전송 | variations + hyde_doc 모두 생성 |
| V4 | calculation 타입 쿼리: multi-query X, HyDE X | 계산 쿼리 전송 | variations == [원본], hyde_doc is None |
| V5 | synonym expansion 동작 | 반도체 도메인 "CVD" 포함 쿼리 | expanded_query에 "화학기상증착" 포함 |
| V6 | 동일 쿼리 캐시 히트 | 동일 쿼리 2회 연속 전송 | 2번째에 캐시 HIT 로그 |
| V7 | API 응답 형식 변경 없음 | `/api/v1/ask` 호출 | 기존과 동일한 JSON 구조 |
| V8 | 검색 결과 dedup 유지 | multi-query로 중복 결과 반환 시 | content hash dedup 동작 |
| V9 | all namespace 검색 | `namespace='all'` 요청 | 모든 네임스페이스에서 검색 |
| V10 | Pinecone cold start 재시도 | 첫 시도 실패 시 | 0.5초 후 재시도 로그 |

---

## 7. 로깅 호환성

기존 로그 패턴을 유지하여 운영 모니터링에 영향 없도록 함.

| 로그 메시지 | 위치 변경 | 패턴 유지 |
|------------|-----------|-----------|
| `[Synonym Expansion] '%s' → '%s'` | pipeline → QueryEnhancer.enhance_query() | ✅ 동일 |
| `[Query Enhancement] Generated %d query variations` | pipeline → QueryEnhancer.enhance_query() | ✅ 동일 |
| `[Query Enhancement] multi_query failed: %s` | pipeline → QueryEnhancer.enhance_query() | ✅ 동일 |
| `[HyDE] Added hypothetical document query` | pipeline → QueryEnhancer.enhance_query() | `[HyDE] Generated hypothetical document`로 소폭 변경 |
| `[Query Enhancement] Keywords: %s` | pipeline → QueryEnhancer.enhance_query() | ✅ 동일 |
| `[Search] Retrieved %d unique documents from %d queries` | pipeline → _search_with_variations() | ✅ 동일 |
| `[Search] Attempt N failed...` | pipeline → _search_single_query() | ✅ 동일 |
| **NEW** `[Multi-Query Cache] HIT for '%.30s...'` | QueryEnhancer.multi_query() | 신규 추가 |

---

## 8. 파일별 최종 구조 변화

### `src/query_enhancer.py`

```
[변경 전]
├── DOMAIN_SYNONYMS (dict)
├── class QueryEnhancer(HttpClientMixin)
│   ├── __init__()
│   ├── _chat_complete()
│   ├── expand_with_synonyms()
│   ├── multi_query()
│   ├── hyde()
│   ├── extract_keywords_fast()
│   ├── extract_keywords()
│   └── enhance_query()  → dict 반환
└── if __name__ == "__main__"

[변경 후]
├── DOMAIN_SYNONYMS (dict)
├── _enhancement_executor (ThreadPoolExecutor)  ← NEW
├── _multi_query_cache, _cache_get/_cache_set   ← NEW
├── shutdown_enhancement_executor()              ← NEW
├── clear_enhancement_cache()                    ← NEW
├── @dataclass EnhancementResult                 ← NEW
├── class QueryEnhancer(HttpClientMixin)
│   ├── __init__()
│   ├── _chat_complete()
│   ├── expand_with_synonyms()
│   ├── _get_num_variations()  ← NEW (정적, rag_pipeline에서 이동)
│   ├── multi_query()          ← MODIFIED (캐시 적용)
│   ├── hyde()
│   ├── extract_keywords_fast()
│   ├── extract_keywords()
│   └── enhance_query()        ← MODIFIED (EnhancementResult 반환, 병렬 실행)
└── if __name__ == "__main__"   ← MODIFIED (EnhancementResult 사용)
```

### `services/rag_pipeline.py`

```
[변경 전]
├── _get_num_variations()                    ← 삭제
├── run_rag_pipeline()
│   ├── Phase 0: Domain Classification
│   ├── Phase 1: Query Enhancement (70줄 inline)
│   ├── Phase 2: Multi-Query Search (60줄 inline)
│   ├── Phase 4-7: ...

[변경 후]
├── _content_hash()                          ← NEW
├── _search_single_query()                   ← NEW
├── _search_with_variations()                ← NEW
├── _enhance_query()                         ← NEW
├── run_rag_pipeline()
│   ├── Phase 0: Domain Classification
│   ├── Phase 1: enhancement = _enhance_query(...)  (3줄)
│   ├── Phase 2: results = _search_with_variations(...)  (8줄)
│   ├── Phase 4-7: ... (변경 없음)
```
