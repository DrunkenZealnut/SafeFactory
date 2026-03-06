# Design: Vector Database 검색 품질 리뷰 및 품질 향상

> Plan Reference: `docs/01-plan/features/vector-db-search-quality.plan.md`

---

## 1. Design Overview

### 1.1 설계 범위

Plan의 3개 Phase 중 **Phase A (품질 측정 체계)** 와 **Phase B (핵심 품질 개선)** 를 상세 설계한다.
Phase C (고급 최적화)는 Phase A/B 결과에 따라 별도 설계한다.

### 1.2 설계 원칙

1. **측정 우선 (Measure First)**: 개선 전에 반드시 기준선 측정을 완료한다
2. **점진적 적용 (Incremental)**: 각 변경은 독립적으로 ON/OFF 가능하게 설계한다
3. **후방 호환성 (Backward Compatible)**: 기존 API 응답 포맷을 깨뜨리지 않는다
4. **최소 침습 (Minimal Invasive)**: 기존 파이프라인 구조를 최대한 유지하며 개선한다

---

## 2. Phase A: 품질 측정 체계 설계

### 2.1 Golden Dataset 스키마

#### 파일 위치
```
scripts/eval/
  golden_dataset.json       # 전체 golden dataset
  eval_pipeline.py          # 자동 평가 스크립트
  eval_report.py            # 리포트 생성
  results/                  # 평가 결과 저장
    baseline_YYYYMMDD.json
```

#### golden_dataset.json 스키마
```json
{
  "version": "1.0",
  "created_at": "2026-03-06",
  "entries": [
    {
      "id": "semi-001",
      "domain": "semiconductor",
      "query": "CVD 공정에서 박막 균일도를 높이는 방법은?",
      "difficulty": "medium",
      "query_type": "procedural",
      "relevant_doc_ids": [
        "a1b2c3d4e5f6..."
      ],
      "relevant_keywords": ["CVD", "박막", "균일도", "Chemical Vapor Deposition"],
      "expected_answer_summary": "CVD 공정에서 박막 균일도 향상을 위해 가스 유량, 온도, 압력 제어가 핵심이며...",
      "metadata_filters": {
        "ncs_category": "반도체제조"
      }
    }
  ]
}
```

#### 필드 설명

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `id` | string | Y | 고유 ID (도메인 접두사 + 번호) |
| `domain` | enum | Y | `semiconductor`, `laborlaw`, `field-training`, `safeguide`, `msds` |
| `query` | string | Y | 사용자 질문 원문 |
| `difficulty` | enum | Y | `easy`, `medium`, `hard` |
| `query_type` | enum | Y | `factual`, `procedural`, `comparison`, `calculation` |
| `relevant_doc_ids` | string[] | Y | Pinecone vector ID 목록 (정답 문서) |
| `relevant_keywords` | string[] | N | 정답 문서에 포함되어야 할 키워드 |
| `expected_answer_summary` | string | N | 기대 답변 요약 (LLM 평가용) |
| `metadata_filters` | object | N | 해당 쿼리에 적용되어야 할 메타데이터 필터 |

#### 도메인별 목표 항목 수

| 도메인 | 1차 (최소) | 2차 (확장) | 비고 |
|--------|-----------|-----------|------|
| semiconductor | 20 | 50 | NCS 카테고리별 균등 |
| laborlaw | 20 | 50 | content_type별 균등 (law/case/qa) |
| field-training | 15 | 30 | equipment_type별 분산 |
| safeguide | 10 | 20 | — |
| msds | 10 | 20 | 화학물질 다양성 확보 |

### 2.2 자동 평가 스크립트 설계

#### eval_pipeline.py 인터페이스

```python
class RAGEvaluator:
    """RAG 파이프라인 품질 평가기."""

    def __init__(self, golden_dataset_path: str, config: dict = None):
        """
        Args:
            golden_dataset_path: golden_dataset.json 경로
            config: 평가 설정 오버라이드
                - top_k: 평가 대상 top_k (default: 10)
                - phases_to_test: Phase ON/OFF 조합 리스트
                - domains: 평가 대상 도메인 리스트
        """

    def evaluate_single(self, entry: dict, pipeline_config: dict = None) -> dict:
        """단일 질문에 대한 파이프라인 실행 및 메트릭 계산.

        Returns:
            {
                "entry_id": "semi-001",
                "domain": "semiconductor",
                "query_type": "procedural",
                "metrics": {
                    "recall_at_k": 0.8,        # relevant docs in top_k / total relevant
                    "precision_at_k": 0.4,      # relevant docs in top_k / k
                    "mrr": 0.5,                 # 1 / rank of first relevant doc
                    "ndcg_at_k": 0.65,          # normalized DCG
                    "hit_rate": 1.0,            # 1 if any relevant doc in top_k
                },
                "phase_results": {
                    "phase1_queries": ["원본", "변형1", "변형2"],
                    "phase2_vector_count": 30,
                    "phase4_after_hybrid": 25,
                    "phase5_after_rerank": 20,
                    "phase6_after_filter": 10,
                },
                "latencies_ms": {
                    "phase1_enhancement": 820,
                    "phase2_vector_search": 340,
                    "phase4_hybrid": 150,
                    "phase5_rerank": 280,
                    "phase6_optimize": 50,
                    "total": 1640,
                },
                "confidence": {
                    "score": 0.72,
                    "level": "high",
                },
            }
        """

    def evaluate_all(self, pipeline_config: dict = None) -> dict:
        """전체 golden dataset 평가 실행.

        Returns:
            {
                "summary": {
                    "total_entries": 75,
                    "avg_recall_at_10": 0.82,
                    "avg_mrr": 0.68,
                    "avg_ndcg_at_10": 0.71,
                    "avg_latency_ms": 1850,
                    "confidence_distribution": {"high": 45, "medium": 22, "low": 8},
                },
                "by_domain": { ... },
                "by_query_type": { ... },
                "by_difficulty": { ... },
                "entries": [ ... ],  # 개별 결과
            }
        """

    def ablation_study(self) -> dict:
        """Phase별 기여도 분석 (ablation study).

        각 Phase를 ON/OFF하며 메트릭 변화를 측정.
        조합:
        - baseline: 모든 Phase ON
        - no_enhancement: Phase 1 OFF (원본 쿼리만)
        - no_hybrid: Phase 4 OFF (BM25 스킵)
        - no_rerank: Phase 5 OFF (리랭킹 스킵)
        - no_optimize: Phase 6 OFF (컨텍스트 최적화 스킵)
        - vector_only: Phase 1,4,5,6 모두 OFF

        Returns:
            {
                "baseline": {"recall": 0.82, "mrr": 0.68, ...},
                "no_enhancement": {"recall": 0.75, "mrr": 0.62, ...},
                ...
                "phase_contribution": {
                    "phase1_enhancement": +0.07,  # recall 기여도
                    "phase4_hybrid": +0.03,
                    "phase5_rerank": +0.05,
                    "phase6_optimize": +0.02,
                }
            }
        """
```

#### 메트릭 계산 공식

```python
def recall_at_k(retrieved_ids: list, relevant_ids: list, k: int) -> float:
    """Recall@K = |relevant ∩ retrieved[:k]| / |relevant|"""
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    if not relevant_set:
        return 0.0
    return len(retrieved_set & relevant_set) / len(relevant_set)

def mrr(retrieved_ids: list, relevant_ids: list) -> float:
    """MRR = 1 / rank_of_first_relevant_doc"""
    relevant_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0

def ndcg_at_k(retrieved_ids: list, relevant_ids: list, k: int) -> float:
    """NDCG@K with binary relevance."""
    import math
    relevant_set = set(relevant_ids)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(retrieved_ids[:k], start=1)
        if doc_id in relevant_set
    )
    ideal_dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, min(len(relevant_ids), k) + 1)
    )
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0
```

### 2.3 Phase별 레이턴시 계측 설계

#### 변경 파일: `services/rag_pipeline.py`

`run_rag_pipeline()` 함수 내에 Phase별 타이머를 삽입한다.

```python
import time

def run_rag_pipeline(data):
    _timings = {}

    # Phase 1
    _t0 = time.perf_counter()
    # ... existing Phase 1 code ...
    _timings['phase1_enhancement_ms'] = round((time.perf_counter() - _t0) * 1000)

    # Phase 2
    _t0 = time.perf_counter()
    # ... existing Phase 2 code ...
    _timings['phase2_vector_search_ms'] = round((time.perf_counter() - _t0) * 1000)

    # ... 동일 패턴으로 Phase 3~7 ...

    result['latencies'] = _timings
    return result
```

#### API 응답 확장

`api/v1/search.py`의 `/ask` 응답에 선택적으로 `latencies` 필드를 포함한다.
디버그 모드(`?debug=1`)에서만 노출하여 프로덕션 응답 크기를 유지한다.

```python
# /api/v1/ask 응답 (debug=1)
{
    "answer": "...",
    "sources": [...],
    "confidence": {...},
    "latencies": {                    # debug=1일 때만 포함
        "phase1_enhancement_ms": 820,
        "phase2_vector_search_ms": 340,
        "phase4_hybrid_ms": 150,
        "phase5_rerank_ms": 280,
        "phase6_optimize_ms": 50,
        "phase7_context_build_ms": 10,
        "total_pipeline_ms": 1650,
        "llm_generation_ms": 3200,
    },
    "retrieval_stats": {              # debug=1일 때만 포함
        "candidates_fetched": 30,
        "after_dedup": 25,
        "after_min_score": 22,
        "after_min_tokens": 20,
        "final_sources": 10,
    }
}
```

---

## 3. Phase B: 핵심 품질 개선 설계

### 3.1 B1: Thread-safety 수정

#### 현재 문제 (`src/hybrid_searcher.py:345-356`)

```python
# BEFORE (race condition)
def search_with_keyword_boost(self, query, vector_results, keywords,
                              top_k=10, keyword_boost=KEYWORD_BOOST, domain=''):
    domain_cfg = DOMAIN_RRF_CONFIG.get(domain, {})
    saved_vw, saved_bw = self.vector_weight, self.bm25_weight  # 저장
    if domain_cfg:
        self.vector_weight = domain_cfg['vector_weight']       # 인스턴스 상태 변경!
        self.bm25_weight = domain_cfg['bm25_weight']           # 인스턴스 상태 변경!
    results = self.search(query, vector_results, top_k=top_k)
    self.vector_weight, self.bm25_weight = saved_vw, saved_bw  # 복원 (예외 시 미복원)
```

#### 수정 설계

**방안: 내부 메서드에 가중치 파라미터 전달**

```python
# AFTER (thread-safe)
def search_with_keyword_boost(self, query, vector_results, keywords,
                              top_k=10, keyword_boost=KEYWORD_BOOST, domain=''):
    domain_cfg = DOMAIN_RRF_CONFIG.get(domain, {})
    vw = domain_cfg.get('vector_weight', self.vector_weight)
    bw = domain_cfg.get('bm25_weight', self.bm25_weight)

    # search()를 직접 호출하는 대신 RRF를 인라인으로 수행
    results = self._search_with_weights(query, vector_results, top_k, vw, bw)

    if not keywords:
        return results

    # keyword boosting (기존 로직 동일)
    # ...

def _search_with_weights(self, query, vector_results, top_k,
                         vector_weight, bm25_weight):
    """Thread-safe 검색: 가중치를 파라미터로 받아 인스턴스 상태를 변경하지 않음."""
    if not vector_results:
        return []
    if BM25_AVAILABLE:
        self.build_index(vector_results)
    bm25_results = self.bm25_search(query, top_k=len(vector_results))
    if not bm25_results:
        return [self._add_rrf_metadata(doc, i) for i, doc in enumerate(vector_results[:top_k])]
    return self._rrf_with_weights(vector_results, bm25_results, vector_weight, bm25_weight)[:top_k]

def _rrf_with_weights(self, vector_results, bm25_results, vector_weight, bm25_weight):
    """가중치를 파라미터로 받는 RRF 계산."""
    rrf_scores = {}
    for rank, doc in enumerate(vector_results, start=1):
        content = doc.get('metadata', {}).get('content', '')
        doc_id = hashlib.sha256(content[:5000].encode()).hexdigest()
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {'doc': doc, 'rrf_score': 0}
        rrf_scores[doc_id]['rrf_score'] += vector_weight / (self.rrf_k + rank)

    for rank, (idx, score) in enumerate(bm25_results, start=1):
        doc = self.doc_map.get(idx)
        if not doc:
            continue
        content = doc.get('metadata', {}).get('content', '')
        doc_id = hashlib.sha256(content[:5000].encode()).hexdigest()
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {'doc': doc, 'rrf_score': 0}
        rrf_scores[doc_id]['rrf_score'] += bm25_weight / (self.rrf_k + rank)

    sorted_results = sorted(rrf_scores.values(), key=lambda x: x['rrf_score'], reverse=True)
    return [self._enrich_rrf_result(item) for item in sorted_results]
```

#### 영향 범위
- 파일: `src/hybrid_searcher.py`
- 호출자: `services/rag_pipeline.py:429` (Phase 4)
- 테스트: 동시 요청 10개로 결과 일관성 검증

---

### 3.2 B2: 쿼리 라우팅 고도화

#### 쿼리 유형 분류 설계

쿼리를 4개 유형으로 분류하고 유형별 파이프라인 파라미터를 차별화한다.

```python
# 새 파일: services/query_router.py

QUERY_TYPE_CONFIG = {
    'factual': {
        # "CVD란 무엇인가?", "최저임금 금액은?" 등
        'top_k_mult': 3,
        'use_hyde': False,       # 사실 질문은 HyDE 불필요
        'use_multi_query': True,
        'rerank_weight': 0.80,   # 리랭커 신뢰 높임
        'description': '단일 사실/정의 질문',
    },
    'procedural': {
        # "CVD 공정 절차는?", "퇴직금 청구 방법은?" 등
        'top_k_mult': 4,         # 절차 관련 문서가 분산되어 있으므로 더 넓게
        'use_hyde': True,
        'use_multi_query': True,
        'rerank_weight': 0.70,
        'description': '절차/방법 질문',
    },
    'comparison': {
        # "CVD와 PVD의 차이는?", "정규직과 비정규직 차이?" 등
        'top_k_mult': 5,         # 비교 대상 양쪽 문서 필요
        'use_hyde': True,
        'use_multi_query': True,  # 각 비교 대상별 쿼리 생성
        'rerank_weight': 0.65,   # 다양성 유지를 위해 원본 점수 비중 높임
        'description': '비교/차이점 질문',
    },
    'calculation': {
        # "월급 200만원의 4대보험료는?", "연차 일수 계산" 등
        'top_k_mult': 2,         # 계산 규칙 문서만 필요
        'use_hyde': False,
        'use_multi_query': False,
        'rerank_weight': 0.85,   # 정확한 규칙 문서 매칭 중시
        'description': '계산/수치 질문',
    },
}

# 분류 패턴 (LLM 미사용, 규칙 기반)
QUERY_TYPE_PATTERNS = {
    'calculation': [
        r'\d+.*(?:만원|원|시간|일|개월)',  # 숫자 + 단위
        r'(?:계산|산출|산정|몇|얼마)',
        r'(?:4대보험|보험료|세금|수당)',
    ],
    'comparison': [
        r'(?:차이|비교|다른|구분|vs)',
        r'(\S+)[와과]\s*(\S+).*(?:차이|다른|비교)',
    ],
    'procedural': [
        r'(?:방법|절차|순서|과정|단계|어떻게)',
        r'(?:하는\s*법|하려면|해야)',
    ],
    # factual: 위 패턴에 매칭되지 않으면 기본값
}

def classify_query_type(query: str) -> str:
    """규칙 기반 쿼리 유형 분류. 매칭 안 되면 'factual' 반환."""
    import re
    for qtype in ['calculation', 'comparison', 'procedural']:
        patterns = QUERY_TYPE_PATTERNS[qtype]
        for pattern in patterns:
            if re.search(pattern, query):
                return qtype
    return 'factual'
```

#### 파이프라인 통합 위치

`services/rag_pipeline.py`의 `run_rag_pipeline()` Phase 1 이전에 분류하고,
Phase별 파라미터를 `QUERY_TYPE_CONFIG`에서 가져온다.

```python
# rag_pipeline.py 변경 (Phase 1 이전)
from services.query_router import classify_query_type, QUERY_TYPE_CONFIG

def run_rag_pipeline(data):
    # ... 기존 파라미터 파싱 ...

    # Query routing
    query_type = classify_query_type(search_query)
    route_config = QUERY_TYPE_CONFIG.get(query_type, QUERY_TYPE_CONFIG['factual'])
    logging.info("[Query Router] Type: %s (%s)", query_type, route_config['description'])

    # Phase 1에서 route_config 활용
    if use_enhancement:
        use_hyde_for_this = route_config['use_hyde'] and len(search_query) >= 10
        use_multi_for_this = route_config['use_multi_query']
        # ...

    # Phase 2에서 top_k 배수 활용
    base_mult = route_config['top_k_mult']
    if mention_filters:
        search_top_k = top_k * max(base_mult, TOP_K_MENTION_MULT)
    else:
        search_top_k = top_k * base_mult

    # Phase 5에서 리랭킹 가중치 활용
    rerank_kwargs['rerank_weight'] = route_config['rerank_weight']
    rerank_kwargs['original_weight'] = 1.0 - route_config['rerank_weight']
```

---

### 3.3 B3: 중복 제거 임계값 도메인별 설정

#### 변경 파일: `src/context_optimizer.py`

```python
# 도메인별 중복 제거 임계값
DOMAIN_DEDUP_THRESHOLD = {
    'laborlaw': 0.80,       # 법조항은 구조적으로 유사 → 낮은 임계값으로 다양성 확보
    'semiconductor': 0.90,  # 기술 문서 다양 → 현행 유지
    'field-training': 0.85, # 카드북 간 유사 구조
    'safeguide': 0.85,
    'msds': 0.90,           # 화학물질별 문서 독립적
}

class ContextOptimizer:
    def deduplicate(self, docs, content_key="content", domain: str = '') -> list:
        """도메인별 임계값으로 중복 제거."""
        threshold = DOMAIN_DEDUP_THRESHOLD.get(domain, self.similarity_threshold)
        # ... 기존 로직에서 self.similarity_threshold → threshold 사용
```

#### 호출부 변경: `services/rag_pipeline.py:497`

```python
# BEFORE
results = context_optimizer.deduplicate(results)

# AFTER
domain_key = NAMESPACE_DOMAIN_MAP.get(namespace, 'semiconductor')
results = context_optimizer.deduplicate(results, domain=domain_key)
```

---

### 3.4 B4: BM25 인덱스 전략 (현행 유지 + 문서화)

**결정: 옵션 3 (현행 유지)** 을 Phase A/B에서 채택한다.

#### 근거
1. 전체 코퍼스 BM25 인덱스는 메모리 비용이 큼 (네임스페이스별 수만 문서)
2. 현행 방식도 vector search 결과 내에서 BM25 재정렬이므로 의미 있음
3. Phase A의 ablation study에서 BM25 기여도를 먼저 측정한 후, 효과가 유의미하면 Phase C에서 전체 코퍼스 인덱스 도입 검토

#### 변경 사항
- 코드 변경 없음
- `hybrid_searcher.py` 모듈 독스트링에 현행 동작 방식 명시
- Phase A ablation study에 `vector_only` (BM25 완전 OFF) 항목 포함

---

### 3.5 B5: Phase별 레이턴시 모니터링

> 2.3절 설계 참조. `run_rag_pipeline()` 내 `time.perf_counter()` 기반 계측.

---

### 3.6 B6: Pinecone 메타데이터 한국어 절단 수정

#### 현재 문제 (`src/pinecone_uploader.py:130-132`)

```python
# BEFORE: 바이트 단위 절단 → 한국어 문자 중간 잘림 가능
content_bytes = (content or "").encode("utf-8")
safe_content = content_bytes[:32768].decode("utf-8", errors="ignore")
safe_preview = content_bytes[:3000].decode("utf-8", errors="ignore")
```

#### 수정 설계

```python
# AFTER: 문자 단위로 먼저 절단, 그 뒤 바이트 안전 확인
MAX_CONTENT_CHARS = 10000    # ~30KB for Korean (3 bytes/char)
MAX_PREVIEW_CHARS = 1000     # ~3KB for Korean

def _safe_truncate(text: str, max_chars: int, max_bytes: int) -> str:
    """문자 단위 절단 후 바이트 한도 내 안전 확인."""
    truncated = text[:max_chars]
    encoded = truncated.encode('utf-8')
    if len(encoded) <= max_bytes:
        return truncated
    # 바이트 한도 초과 시 문자를 줄여가며 맞춤
    while len(encoded) > max_bytes:
        truncated = truncated[:-100]
        encoded = truncated.encode('utf-8')
    return truncated

# prepare_vector()에서:
safe_content = _safe_truncate(content or "", MAX_CONTENT_CHARS, 32768)
safe_preview = _safe_truncate(content or "", MAX_PREVIEW_CHARS, 3000)
```

#### 영향 범위
- 파일: `src/pinecone_uploader.py`
- 기존 인덱스된 데이터: 영향 없음 (신규 인제스천부터 적용)
- 재인덱싱 필요 여부: 불필요 (기존 데이터는 이미 저장됨)

---

## 4. 구현 순서 (Implementation Order)

### Phase A: 품질 측정 (Week 1)

| 순서 | 작업 | 파일 | 의존성 | 예상 시간 |
|------|------|------|--------|----------|
| A-1 | Golden dataset JSON 스키마 정의 + 도메인당 5개 샘플 | `scripts/eval/golden_dataset.json` | 없음 | 2h |
| A-2 | 메트릭 계산 함수 구현 (recall, mrr, ndcg) | `scripts/eval/eval_pipeline.py` | 없음 | 2h |
| A-3 | Phase별 레이턴시 계측 삽입 | `services/rag_pipeline.py` | 없음 | 1h |
| A-4 | RAGEvaluator 클래스 구현 + 단일 평가 | `scripts/eval/eval_pipeline.py` | A-1, A-2 | 3h |
| A-5 | 전체 평가 + ablation study 구현 | `scripts/eval/eval_pipeline.py` | A-4 | 3h |
| A-6 | Golden dataset 확장 (도메인당 20개) | `scripts/eval/golden_dataset.json` | A-4 | 4h |
| A-7 | 기준선 측정 실행 + 결과 기록 | `scripts/eval/results/` | A-5, A-6 | 2h |

### Phase B: 핵심 개선 (Week 2)

| 순서 | 작업 | 파일 | 의존성 | 예상 시간 |
|------|------|------|--------|----------|
| B-1 | Thread-safety 수정 | `src/hybrid_searcher.py` | 없음 | 2h |
| B-2 | Thread-safety 동시 요청 테스트 | 테스트 스크립트 | B-1 | 1h |
| B-3 | 쿼리 라우터 구현 | `services/query_router.py` (신규) | 없음 | 3h |
| B-4 | 파이프라인에 쿼리 라우터 통합 | `services/rag_pipeline.py` | B-3 | 2h |
| B-5 | 중복 제거 도메인별 임계값 | `src/context_optimizer.py` | 없음 | 1h |
| B-6 | Pinecone 한국어 절단 수정 | `src/pinecone_uploader.py` | 없음 | 1h |
| B-7 | debug=1 파라미터 API 확장 | `api/v1/search.py` | A-3 | 1h |
| B-8 | 개선 후 golden dataset 재평가 | `scripts/eval/` | B-1~B-7, A-7 | 2h |

---

## 5. 변경 파일 목록

| 파일 | 변경 유형 | Phase | 설명 |
|------|----------|-------|------|
| `scripts/eval/golden_dataset.json` | 신규 | A | Golden dataset |
| `scripts/eval/eval_pipeline.py` | 신규 | A | 자동 평가 스크립트 |
| `services/rag_pipeline.py` | 수정 | A+B | 레이턴시 계측 + 쿼리 라우터 통합 |
| `services/query_router.py` | 신규 | B | 쿼리 유형 분류 + 라우팅 설정 |
| `src/hybrid_searcher.py` | 수정 | B | Thread-safety 수정 |
| `src/context_optimizer.py` | 수정 | B | 도메인별 중복 제거 임계값 |
| `src/pinecone_uploader.py` | 수정 | B | 한국어 메타데이터 절단 수정 |
| `api/v1/search.py` | 수정 | B | debug 모드 레이턴시/stats 응답 |

---

## 6. 테스트 계획

### 6.1 단위 테스트

| 테스트 | 대상 | 검증 내용 |
|--------|------|----------|
| `test_metrics.py` | `eval_pipeline.py` 메트릭 함수 | recall, mrr, ndcg 계산 정확성 (known-answer) |
| `test_query_router.py` | `query_router.py` | 10+ 쿼리 유형 분류 정확성 |
| `test_thread_safety.py` | `hybrid_searcher.py` | 10 concurrent requests 결과 일관성 |
| `test_safe_truncate.py` | `pinecone_uploader.py` | 한국어/영문/혼합 텍스트 절단 정확성 |

### 6.2 통합 테스트

| 테스트 | 검증 내용 |
|--------|----------|
| Golden dataset baseline | Phase A-7에서 수행. 모든 메트릭 계산 검증 |
| Before/After 비교 | Phase B-8에서 수행. 각 개선의 메트릭 변화 확인 |
| API 응답 호환성 | debug=0 시 기존 응답과 동일, debug=1 시 추가 필드 |

### 6.3 성능 테스트

| 테스트 | 목표 |
|--------|------|
| 파이프라인 P95 레이턴시 | < 5s (LLM 생성 제외) |
| Thread-safety 동시 10 요청 | 결과 불일치 0건 |
| 메모리 사용량 | 기존 대비 10% 이내 증가 |

---

## 7. 롤백 계획

모든 변경은 독립적으로 ON/OFF 가능하도록 설계:

| 변경 | 롤백 방법 |
|------|----------|
| 레이턴시 계측 | `result['latencies']` 필드만 제거 (기능 영향 없음) |
| 쿼리 라우터 | `services/query_router.py` import 제거, 기존 상수 복원 |
| Thread-safety | `_search_with_weights()` 대신 기존 `search()` 직접 호출 복원 |
| 중복 제거 임계값 | `domain` 파라미터 무시하고 기존 `self.similarity_threshold` 사용 |
| 한국어 절단 | `_safe_truncate()` 대신 기존 바이트 절단 복원 |

---

## 8. 설계 결정 기록 (ADR)

### ADR-1: BM25 전체 코퍼스 인덱스 보류
- **결정**: Phase A/B에서는 현행 방식 유지
- **근거**: 메모리 비용 대비 효과 미검증. ablation study 결과에 따라 Phase C에서 재검토
- **대안**: Elasticsearch/OpenSearch 도입 → 과도한 인프라 복잡성

### ADR-2: 쿼리 분류를 규칙 기반으로 구현
- **결정**: 정규식 패턴 매칭 (LLM 미사용)
- **근거**: 추가 레이턴시 0ms. LLM 기반 분류는 500ms+ 추가 비용
- **대안**: LLM 기반 분류 → Phase C에서 정확도 비교 후 필요 시 전환

### ADR-3: Golden dataset 수동 라벨링
- **결정**: 도메인별 20개로 시작, 사용자 피드백으로 점진 확장
- **근거**: 자동 라벨링의 신뢰도 문제. 소규모 수동 라벨이 더 정확
- **대안**: LLM 기반 relevance 판단 → Phase C 사용자 피드백 루프에서 활용
