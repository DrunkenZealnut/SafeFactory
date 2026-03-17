# 검색 품질 개선 Plan

## Executive Summary

| 관점 | 내용 |
|------|------|
| **Feature** | search-quality |
| **시작일** | 2026-03-17 |
| **예상 기간** | 3주 (Quick Wins 1주 + Core 1주 + Robustness 1주) |

| 관점 | 설명 |
|------|------|
| **Problem** | 현재 RAG 파이프라인은 고정 top_k, 단일 도메인 라우팅, BM25 미활용, MMR 비활성화 등으로 인해 멀티도메인 질문 실패, 중복 결과 제공, 검색 누락이 발생함 |
| **Solution** | 7개 영역(라우팅, 쿼리강화, 하이브리드검색, 리랭킹, 컨텍스트, 파이프라인, 청킹)에서 Quick Win 5건 + Core 개선 4건 + Robustness 2건을 단계적으로 적용 |
| **Function UX Effect** | 크로스도메인 질문 대응, 답변 내 관련 개념 다양성 향상, 검색 정확도 향상으로 "한 번에 원하는 답변" 경험 제공 |
| **Core Value** | 산업안전 교육 플랫폼의 신뢰도 제고 — 검색 품질이 곧 교육 품질 |

---

## 1. 현재 시스템 진단

### 1.1 파이프라인 구조

```
Phase 0: Domain Classification (키워드 기반, 정확도 ~75%)
Phase 1: Query Enhancement (Multi-query + HyDE + Synonym)
Phase 2: Vector Search (Pinecone, text-embedding-3-small)
Phase 3: Graph Enrichment (GraphRAG, 신규 추가됨)
Phase 4: Hybrid Search (BM25 + Vector RRF)
Phase 5: Reranking (Cohere/Pinecone/Local Cross-encoder)
Phase 6: Context Optimization (토큰 예산, 필터링, 재정렬)
Phase 7: LLM Generation (Gemini/OpenAI/Claude)
```

### 1.2 영역별 문제점 요약

| 영역 | 핵심 문제 | 영향도 |
|------|----------|--------|
| **도메인 라우팅** | 단일 도메인만 선택, cross-domain 쿼리 실패 | 높음 |
| **쿼리 강화** | HyDE가 factual 질문에 비활성화, 동의어 수동 관리 | 중간 |
| **하이브리드 검색** | Keyword Boost 미연결, BM25 매 쿼리 재구축 | 중간 |
| **리랭킹** | MMR 미사용 → 중복 결과, Min-Max 정규화 이상치 취약 | 높음 |
| **컨텍스트** | MIN_TOKEN_COUNT=30 (너무 낮음), 노이즈 문서 포함 | 중간 |
| **파이프라인** | 고정 top_k, confidence 기반 적응 로직 부재 | 높음 |
| **청킹** | 고정 500토큰, 도메인별 최적 크기 다름 | 낮음 |

---

## 2. 개선 항목 및 우선순위

### 2.1 Week 1 — Quick Wins (즉시 효과, 낮은 비용)

| # | 개선 항목 | 수정 파일 | 변경 규모 | 기대 효과 |
|---|----------|-----------|-----------|-----------|
| **QW-1** | Keyword Boost 활성화 | `rag_pipeline.py` | 1줄 수정 | 키워드 매칭 문서 우선순위 상승 |
| **QW-2** | Adaptive top_k | `rag_pipeline.py` | 10줄 추가 | 저신뢰도 쿼리 → 더 많이 검색, 고신뢰도 → 효율적 |
| **QW-3** | MIN_TOKEN_COUNT 상향 | `rag_pipeline.py` | 1줄 수정 | 30 → 80으로 변경, 노이즈 문서 제거 |
| **QW-4** | 동적 RRF K | `hybrid_searcher.py` | 5줄 수정 | 결과 크기에 비례한 K값으로 순위 안정화 |
| **QW-5** | 도메인 라우팅 동의어 | `query_router.py` | 10줄 추가 | "노동법"↔"근로기준법" 등 동의어 인식 |

### 2.2 Week 2 — Core Improvements (중간 노력, 높은 효과)

| # | 개선 항목 | 수정 파일 | 변경 규모 | 기대 효과 |
|---|----------|-----------|-----------|-----------|
| **CI-1** | HyDE 조건부 활성화 | `query_enhancer.py`, `rag_pipeline.py` | 15줄 수정 | factual도 confidence<0.6일 때 HyDE 적용 → 검색 누락 30% 감소 |
| **CI-2** | MMR 재활성화 | `rag_pipeline.py`, `reranker.py` | 20줄 수정 | 결과 다양성 향상, 중복 제거 |
| **CI-3** | Robust 정규화 | `reranker.py` | 15줄 수정 | Min-Max → Percentile 기반, 이상치 영향 감소 |
| **CI-4** | 다중 도메인 라우팅 | `query_router.py`, `rag_pipeline.py` | 40줄 추가 | 상위 2개 도메인 검색 후 병합 |

### 2.3 Week 3 — Robustness (높은 효과, 장기 안정)

| # | 개선 항목 | 수정 파일 | 변경 규모 | 기대 효과 |
|---|----------|-----------|-----------|-----------|
| **RB-1** | 2단계 재검색 | `rag_pipeline.py` | 30줄 추가 | confidence<0.5일 때 enhanced query로 자동 재검색 |
| **RB-2** | Multi-query 캐싱 TTL 증대 | `query_enhancer.py` | 5줄 수정 | 300초 → 1시간, 캐시 히트율 60%→85% |

### 2.4 Scope Out (이번 범위 제외)

| 제외 항목 | 이유 |
|-----------|------|
| 동적 청크 크기 | 재인덱싱 필요, 비용 높음 |
| LLM 기반 Synonym 생성 | API 비용, latency 증가 우려 |
| 프롬프트 외부화 (DB 관리) | 별도 feature로 분리 |
| 다국어 지원 | 현재 한국어 전용 플랫폼 |
| LLM 쿼리타입 재분류 | 정규식으로 충분한 수준 |

---

## 3. 상세 설계 방향

### QW-1: Keyword Boost 활성화

```python
# rag_pipeline.py Phase 4에서 이미 호출 중이나,
# keywords 변수가 전달되지 않는 케이스가 있음
# → keywords를 Phase 1에서 항상 추출하도록 보장
```

### QW-2: Adaptive top_k

```python
# confidence 기반 동적 조정
# confidence > 0.7: top_k * 1.0 (기본)
# confidence 0.4~0.7: top_k * 1.5
# confidence < 0.4: top_k * 2.0
adaptive_mult = 1.0
if domain_confidence < 0.4:
    adaptive_mult = 2.0
elif domain_confidence < 0.7:
    adaptive_mult = 1.5
effective_top_k = int(top_k * adaptive_mult)
```

### CI-2: MMR 재활성화

```python
# reranker.py의 mmr_rerank() 메서드가 이미 구현되어 있으나 미사용
# Phase 5 이후 diversity 파라미터를 추가하여 MMR 적용
# lambda_param=0.7 (관련성 70% + 다양성 30%)
```

### CI-4: 다중 도메인 라우팅

```python
# classify_domain()의 반환값을 (primary, secondary) 튜플로 확장
# secondary 도메인 confidence > 0.5일 때 함께 검색
# 결과를 RRF로 병합
```

---

## 4. 성공 기준

| 지표 | 현재 (추정) | 목표 | 측정 방법 |
|------|------------|------|-----------|
| **크로스도메인 질문 답변률** | ~40% | 80%+ | 테스트 질문 10건 수동 평가 |
| **상위 5개 결과 중 관련 문서 비율** | ~60% | 80%+ | 테스트 질문 20건 Precision@5 |
| **결과 다양성 (MMR 적용 후)** | 상위 5개 중 중복 ~2건 | 0건 | 중복 content 비율 측정 |
| **검색 latency** | baseline | +100ms 이내 증가 | Phase별 타이밍 측정 |
| **캐시 히트율** | ~60% | 85%+ | Multi-query 캐시 통계 |

---

## 5. 리스크 및 대응

| 리스크 | 확률 | 영향 | 대응 |
|--------|------|------|------|
| Adaptive top_k로 검색 비용 증가 | 중 | 낮 | 상한 cap (top_k * 3 이내) |
| MMR로 관련성 높은 문서 제외 | 낮 | 중 | lambda_param 0.7~0.8로 보수적 설정 |
| 다중 도메인 검색 latency 증가 | 중 | 중 | 병렬 검색, 캐시 활용 |
| 기존 검색 품질 회귀 | 낮 | 높 | A/B 테스트, 설정 토글 |

---

## 6. 구현 순서

```
Week 1: Quick Wins (5건)
├── Day 1: QW-1 Keyword Boost + QW-3 MIN_TOKEN_COUNT
├── Day 2: QW-2 Adaptive top_k
├── Day 3: QW-4 동적 RRF K + QW-5 도메인 동의어
└── Day 4-5: 테스트 질문 세트 평가

Week 2: Core Improvements (4건)
├── Day 1: CI-1 HyDE 조건부 활성화
├── Day 2: CI-2 MMR 재활성화
├── Day 3: CI-3 Robust 정규화
├── Day 4: CI-4 다중 도메인 라우팅
└── Day 5: 통합 테스트 + 튜닝

Week 3: Robustness (2건)
├── Day 1-2: RB-1 2단계 재검색
├── Day 3: RB-2 캐싱 TTL 증대
└── Day 4-5: 최종 평가 + 문서화
```
