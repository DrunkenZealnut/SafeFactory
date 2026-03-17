# Plan: 전체 검색 관련 플로우 점검

## Executive Summary

| 항목 | 내용 |
|------|------|
| Feature | 전체 검색 관련 플로우 점검 (Search Flow Audit) |
| 작성일 | 2026-03-17 |
| 예상 기간 | 2-3일 |
| 레벨 | Dynamic |

### Value Delivered

| 관점 | 내용 |
|------|------|
| **Problem** | 검색 파이프라인이 8단계+α로 성장하면서 단계 간 데이터 흐름, 비활성 코드, 설정 불일치 등 기술 부채가 누적되어 전체 플로우의 정합성 검증이 필요하다 |
| **Solution** | 전체 검색 플로우를 API 진입점부터 LLM 응답까지 추적하여, 각 단계의 입출력 정합성·비활성 코드·설정 불일치·에러 핸들링을 체계적으로 점검한다 |
| **Function UX Effect** | 검색 정확도 향상, 불필요한 지연 제거, 에러 시 graceful fallback 강화로 사용자 체감 응답 품질 개선 |
| **Core Value** | 파이프라인 신뢰성 확보 — 모든 검색 경로가 의도대로 동작함을 검증하고, 향후 기능 확장의 안전한 기반을 마련 |

---

## 1. 배경 및 목적

SafeFactory의 RAG 검색 파이프라인은 지속적인 기능 추가(GraphRAG, Community Search, MSDS Cross-Search, 자동 도메인 라우팅, HyDE, Major Config 등)를 거치면서 **8개 이상의 Phase**로 구성된 복합 시스템으로 성장했다.

이번 점검의 목적:
1. **데이터 흐름 정합성**: 각 Phase의 입출력이 다음 Phase에 올바르게 전달되는지 확인
2. **비활성/데드 코드 식별**: 사용되지 않는 import, 미참조 함수, 조건부 비활성 경로 탐지
3. **설정 불일치 탐지**: 환경변수·admin 설정·하드코딩 값 간 충돌이나 미반영 여부
4. **에러 핸들링 완전성**: 각 Phase 실패 시 graceful fallback이 올바르게 작동하는지
5. **성능 병목 식별**: 불필요한 중복 연산, 직렬 실행 가능한 병렬화 대상

---

## 2. 현재 검색 플로우 맵

### 2.1 API 진입점 (3개 엔드포인트)

| Endpoint | 파일 | 용도 |
|----------|------|------|
| `POST /api/v1/search` | `api/v1/search.py:154` | 단순 벡터/하이브리드/키워드 검색 |
| `POST /api/v1/ask` | `api/v1/search.py:292` | RAG Q&A (동기) |
| `POST /api/v1/ask/stream` | `api/v1/search.py:409` | RAG Q&A (SSE 스트리밍) |

### 2.2 RAG 파이프라인 Phase 구조 (`run_rag_pipeline`)

```
Phase 0: Domain Classification (auto-routing)
  └─ classify_domain() → namespace override
  └─ Adaptive top_k 조정

Phase 1: Query Enhancement
  └─ _enhance_query() → multi-query, HyDE, keyword extraction
  └─ HyDE override for low-confidence queries (CI-1)
  └─ Fallback keyword extraction (QW-1)

Phase 2: Multi-Query Search
  └─ _search_with_variations() → vector search × N queries
  └─ Content hash dedup

Phase 3: Graph Enrichment (GraphRAG)
  └─ GraphSearcher.search() → entity matching + graph traversal
  └─ Chunk ID fetch from Pinecone

Phase 3.5: Global Search (Community)
  └─ CommunitySearcher.search() → community summary context
  └─ Only for 'overview' query type

Phase 4: Hybrid Search (BM25 + Vector)
  └─ HybridSearcher.search_with_keyword_boost()
  └─ Domain-specific RRF weights

Phase 5: Reranking
  └─ Reranker.hybrid_rerank() → cross-encoder + original score blend
  └─ Domain-specific rerank weights

Phase 5.5: MMR Diversity
  └─ Near-duplicate removal (first 200 chars)

Phase 6: Filtering + Context Optimization
  └─ min_score / min_token_count 필터링
  └─ Re-search trigger (RB-1: results < 3 + low confidence)
  └─ ContextOptimizer.deduplicate() + reorder_for_llm()

Phase 7: Build Context and Sources
  └─ context_parts 조립, source metadata 추출
  └─ Global context prepend

Phase 7.5: Safety Cross-Search
  └─ semiconductor → kosha namespace cross-search

Phase 7.6: MSDS Cross-Search
  └─ Chemical name extraction + MSDS API lookup

Phase 7.7: Laborlaw Enrichment
  └─ Law API search + legal analysis pass (Gemini)
```

### 2.3 보조 서비스 구성

| 모듈 | 싱글톤 | 역할 |
|------|--------|------|
| `singletons.py` | 14개 인스턴스 | 지연 초기화, 캐시 무효화 |
| `query_router.py` | - | 쿼리 타입 분류 + 도메인 라우팅 |
| `domain_config.py` | - | 도메인별 프롬프트, 네임스페이스 매핑 |
| `major_config.py` | - | 전공 기반 검색 컨텍스트 결정 |
| `filters.py` | - | 도메인별 메타데이터 필터 빌더 |
| `graph_config.py` | - | 그래프 검색 설정 |
| `graph_searcher.py` | GraphSearcher | 엔티티 매칭 + 그래프 순회 |
| `community_searcher.py` | CommunitySearcher | 커뮤니티 요약 글로벌 검색 |

---

## 3. 점검 항목 (Checklist)

### 3.1 데이터 흐름 정합성

| # | 점검 항목 | 대상 파일 | 우선순위 |
|---|-----------|-----------|----------|
| DF-1 | `/search` 엔드포인트의 `major` 파라미터 처리가 `run_rag_pipeline`과 동일한 로직인지 확인 | `search.py:169-171` vs `rag_pipeline.py:773-776` | High |
| DF-2 | `classify_domain()` 반환값 5개가 `run_rag_pipeline`에서 모두 올바르게 언패킹되는지 | `query_router.py:163` → `rag_pipeline.py:795-796` | High |
| DF-3 | `enhancement.search_queries` vs `enhancement.all_queries` 사용처 일관성 | `rag_pipeline.py:828` + `_search_with_variations` | Medium |
| DF-4 | Graph Enrichment 결과의 `score` 필드가 후속 Phase(Hybrid, Rerank)에서 올바르게 처리되는지 | `rag_pipeline.py:894-900` | High |
| DF-5 | Community context가 LLM 프롬프트에 올바르게 주입되는지 (전체 경로 추적) | `rag_pipeline.py:1124-1125` → `build_llm_messages` | Medium |
| DF-6 | `_run_multi_query_search` 참조 확인 — RB-1 re-search에서 호출하는데 정의가 존재하는지 | `rag_pipeline.py:1035` | Critical |
| DF-7 | `/search` 엔드포인트가 `run_rag_pipeline`과 다른 별도 검색 경로를 사용하는데 품질이 일관적인지 | `search.py:154-289` | Medium |
| DF-8 | `ask`와 `ask/stream`의 파이프라인 결과 활용이 동일한지 (코드 중복 및 불일치 위험) | `search.py:292-406` vs `search.py:409-601` | High |

### 3.2 비활성/데드 코드

| # | 점검 항목 | 대상 | 우선순위 |
|---|-----------|------|----------|
| DC-1 | `build_llm_prompts()` vs `build_llm_messages()` — 둘 다 존재하면 어느 것이 사용되는지 | `rag_pipeline.py` | Medium |
| DC-2 | `calc_result` 파라미터가 `build_llm_prompts`에서 "Unused" 주석이 있는데 다른 경로에서 사용되는지 | `rag_pipeline.py:1217` | Low |
| DC-3 | `labor_classification` 파라미터도 "Unused" — 실제 전달 경로 확인 | `rag_pipeline.py:1220` | Low |
| DC-4 | `SKIP_BM25_HYBRID` 환경변수 사용 시 Phase 4가 완전히 스킵되는데, Phase 2의 `search_top_k` 배수만 조정 — 의도 확인 | `rag_pipeline.py:782, 845-846` | Medium |
| DC-5 | `USE_LOCAL_RERANKER` 환경변수가 `.env.example`에 있는데 실제 참조 여부 | `.env.example` + `reranker.py` | Low |
| DC-6 | `graph_config.py`의 `community.enabled`가 `True`인데 실제 KG 데이터가 있는지 확인 필요 | `graph_config.py` + DB | Medium |

### 3.3 설정 불일치

| # | 점검 항목 | 대상 | 우선순위 |
|---|-----------|------|----------|
| SC-1 | `NAMESPACE_DOMAIN_MAP`(domain_config)과 `DOMAIN_KEYWORDS`(query_router) 키 불일치 — `kosha` vs `safeguide` 네이밍 | `domain_config.py:335` vs `query_router.py:131` | High |
| SC-2 | `DOMAIN_RRF_CONFIG`(hybrid_searcher)과 `DOMAIN_RERANK_CONFIG`(reranker)의 도메인 키 일관성 | `hybrid_searcher.py:39-45` vs `reranker.py:17-22` | Medium |
| SC-3 | `_NAMESPACE_MODEL_OVERRIDE`에서 `laborlaw` → `claude-opus-4-6` 지정 vs admin 설정의 `llm_answer_provider/model` 우선순위 | `search.py:43-45` | Medium |
| SC-4 | `MIN_RELEVANCE_SCORE=0.2`와 `SAFETY_CROSS_SEARCH_MIN_SCORE=0.3` — 기본 검색보다 cross-search가 더 엄격한 이유 확인 | `rag_pipeline.py:40, 205` | Low |
| SC-5 | `MAJOR_CONFIG`의 `routing_keywords`와 `DOMAIN_KEYWORDS`(query_router) 간 중복/불일치 | `major_config.py` vs `query_router.py` | Medium |

### 3.4 에러 핸들링 및 Fallback

| # | 점검 항목 | 대상 | 우선순위 |
|---|-----------|------|----------|
| EH-1 | Graph Enrichment 실패 시 `results` 리스트가 이전 상태 유지되는지 (mutation 위험) | `rag_pipeline.py:867-903` | High |
| EH-2 | `get_graph_searcher()` 초기화 실패 시 (KG 테이블 미존재) 에러 전파 차단 여부 | `singletons.py:279-287` + `graph_searcher.py` | High |
| EH-3 | Hybrid Search 실패 시 `results`가 원본을 유지하는지 (재할당 이전 실패 케이스) | `rag_pipeline.py:942-955` | Medium |
| EH-4 | Reranker 실패 시 원본 순서 유지 검증 | `rag_pipeline.py:963-986` | Medium |
| EH-5 | `_search_single_query` 2회 재시도 후 빈 리스트 반환이 downstream에 안전한지 | `rag_pipeline.py:650-705` | Medium |
| EH-6 | MSDS API 타임아웃 (15s) 이 전체 파이프라인 응답 시간에 미치는 영향 | `rag_pipeline.py:544` | Medium |
| EH-7 | `laborlaw` 네임스페이스의 "results 없어도 계속 진행" 특수 처리의 의도 및 부작용 | `rag_pipeline.py:854, 1048` | Medium |

### 3.5 성능 및 최적화

| # | 점검 항목 | 대상 | 우선순위 |
|---|-----------|------|----------|
| PF-1 | Phase 7.5 + 7.6 + 7.7이 직렬 실행 — 병렬화 가능 여부 | `rag_pipeline.py:1148-1207` | Medium |
| PF-2 | `_enhance_query`의 LLM 호출(multi-query, HyDE)이 캐싱되는데, 캐시 적중률 로깅 부재 | `query_enhancer.py:56-66` | Low |
| PF-3 | `_search_with_variations`의 N개 쿼리가 순차 실행 — 병렬 검색 고려 | `rag_pipeline.py:733-758` | Medium |
| PF-4 | Phase 3 GraphSearcher의 entity cache 만료 정책 부재 (서버 재시작 전까지 영구 캐시) | `graph_searcher.py:96-117` | Low |
| PF-5 | `community_searcher._select_relevant_communities`의 단어 분할 기반 매칭이 한국어에서 비효율적 | `community_searcher.py:54-74` | Medium |

---

## 4. 점검 방법론

### 4.1 정적 분석
- 각 Phase의 입출력 타입/필드를 코드 리뷰로 추적
- 미사용 import, 미참조 함수, 도달 불가 분기 탐지
- 설정 값 출처(env, admin DB, 하드코딩) 교차 확인

### 4.2 동적 검증
- `debug=true` 파라미터로 `latencies` 타이밍 확인
- 각 도메인별(semiconductor, laborlaw, kosha, field-training, msds, all) 샘플 쿼리 실행
- Graph/Community 검색 경로가 실제로 활성화되는지 로그 확인

### 4.3 결과물
- 발견된 이슈를 **Critical / High / Medium / Low**로 분류
- 각 이슈에 대한 수정 방안 제시
- 수정 후 `/pdca analyze` 실행하여 검증

---

## 5. 예상 발견 이슈 (가설)

| # | 가설 | 근거 | 심각도 |
|---|------|------|--------|
| H-1 | `_run_multi_query_search` 함수가 정의되지 않았을 가능성 (RB-1 re-search에서 참조) | `rag_pipeline.py:1035`에서 호출하지만, `_search_with_variations`으로 대체되었을 수 있음 | Critical |
| H-2 | `kosha` vs `safeguide` 키 불일치로 일부 설정이 미적용 | `NAMESPACE_DOMAIN_MAP`은 `kosha` → `safeguide`, 하지만 `DOMAIN_RRF_CONFIG`은 `safeguide` 키 사용 | High |
| H-3 | Graph Enrichment가 `score` 필드에 `graph_score`를 넣는데, 후속 Hybrid/Rerank가 `score` 키를 원본 벡터 스코어로 기대 | 스코어 스케일 불일치 (graph_score: 0-1 vs vector_score: 0-1) | Medium |
| H-4 | `ask`와 `ask/stream`의 LLM 호출 코드가 거의 동일하게 중복 — 불일치 발생 위험 | `search.py:330-363` vs `search.py:512-566` | Medium |

---

## 6. 범위 제한 (Out of Scope)

- Document Processing Pipeline (`src/` CLI 모듈) — 이번 점검 대상 아님
- 프론트엔드 JS 코드 — API 호출/응답 포맷만 확인
- 인증/인가 로직 — 검색 품질과 무관
- Pinecone 인덱스 데이터 품질 — 별도 점검 필요

---

## 7. 참고 파일 목록

| 파일 | 핵심 역할 |
|------|-----------|
| `services/rag_pipeline.py` | RAG 파이프라인 메인 (Phase 0~7.7) |
| `api/v1/search.py` | 검색 API 엔드포인트 3개 |
| `services/query_router.py` | 쿼리 타입 분류 + 도메인 라우팅 |
| `services/singletons.py` | 서비스 싱글톤 관리 |
| `services/domain_config.py` | 도메인별 프롬프트, 설정 |
| `services/major_config.py` | 전공 기반 검색 컨텍스트 |
| `services/filters.py` | 도메인별 메타데이터 필터 |
| `services/graph_config.py` | GraphRAG 설정 |
| `services/graph_searcher.py` | 그래프 검색 서비스 |
| `services/community_searcher.py` | 커뮤니티 글로벌 검색 |
| `src/query_enhancer.py` | 쿼리 확장 (multi-query, HyDE) |
| `src/hybrid_searcher.py` | BM25 + Vector RRF 융합 |
| `src/reranker.py` | 크로스인코더 리랭킹 |
| `src/context_optimizer.py` | 컨텍스트 최적화 (중복 제거, 재정렬) |
