# Plan: Vector Database 검색 품질 리뷰 및 품질 향상 방법 연구

## Executive Summary

| 항목 | 내용 |
|------|------|
| Feature | Vector DB 검색 품질 리뷰 및 향상 |
| 시작일 | 2026-03-06 |
| 예상 기간 | 2-3주 (리뷰 1주 + 개선 구현 1-2주) |

### Results Target

| 지표 | 현재 추정 | 목표 |
|------|----------|------|
| Recall@10 | 측정 미비 | 85%+ |
| MRR (Mean Reciprocal Rank) | 측정 미비 | 0.7+ |
| 평균 응답 신뢰도 | medium | high (0.7+) |
| 사용자 만족도 | 미측정 | 정성적 개선 확인 |

### 1.3 Value Delivered

| 관점 | 설명 |
|------|------|
| **Problem** | 5개 도메인에 걸친 RAG 검색에서 검색 품질이 체계적으로 측정/관리되지 않아, 사용자 질문에 대한 답변 정확도와 관련성을 객관적으로 파악할 수 없음 |
| **Solution** | 현행 7-Phase RAG 파이프라인의 각 단계별 품질 병목을 분석하고, 측정 체계 구축 + 핵심 개선안을 도출하여 검색 품질을 체계적으로 향상 |
| **Function UX Effect** | 더 정확한 검색 결과와 높은 답변 신뢰도를 통해 사용자가 원하는 정보를 빠르게 찾을 수 있게 됨 |
| **Core Value** | 데이터 기반 검색 품질 관리 체계 확립으로 지속적 품질 개선 가능한 기반 마련 |

---

## 2. 현황 분석 (As-Is)

### 2.1 현행 RAG 파이프라인 아키텍처 (7-Phase)

```
Phase 1: Query Enhancement (multi-query + HyDE + synonym expansion + keyword extraction)
    ↓
Phase 2: Multi-Query Vector Search (Pinecone, domain metadata filter)
    ↓
Phase 3: @mention Post-Filtering (source_file/filename 매칭)
    ↓
Phase 4: Hybrid Search (BM25 + Vector RRF fusion, domain-specific weights)
    ↓
Phase 5: Reranking (Pinecone bge-reranker-v2-m3 / local cross-encoder / lightweight)
    ↓
Phase 6: Context Optimization (dedup + lost-in-middle reorder + min_score/token filter)
    ↓
Phase 7: Context Build + LLM Prompt Assembly
```

### 2.2 핵심 구성 요소별 현황

#### A. 임베딩 (Embedding)
- **모델**: `text-embedding-3-small` (1536차원)
- **인제스천 단위**: SemanticChunker 기반 구조적 분할
- **토큰 카운트**: `len(text) // 3` 근사치 사용 (한영 혼합)
- **벡터 ID**: MD5 해시 (source_file + chunk_index + content_preview) — 결정적/멱등

#### B. 쿼리 확장 (Query Enhancement)
| 기법 | 구현 상태 | 조건 |
|------|----------|------|
| Multi-Query | O | 쿼리 길이별 변형 수 동적 결정 (1~3개) |
| HyDE | O | 쿼리 길이 >= 10자 시 활성화 |
| Synonym Expansion | O | 도메인별 동의어 매핑 (semiconductor, laborlaw, field-training, msds) |
| Keyword Extraction | O | 정규식 기반 fast extraction (LLM 미사용) |

#### C. 벡터 검색 (Vector Search)
| 파라미터 | 값 | 비고 |
|----------|-----|------|
| top_k 배수 | @mention: x3, no-BM25: x4, default: x3 | 리랭킹 후보 확보용 |
| 유사도 메트릭 | cosine (Pinecone default) | — |
| 네임스페이스 | 5개 (semiconductor, laborlaw, field-training, safeguide, msds) | — |
| 메타데이터 필터 | 도메인별 빌더 (NCS category/section, law_name/article, equipment_type 등) | — |
| 중복 제거 | SHA-256 content hash (쿼리 변형 간 결과 병합 시) | — |

#### D. 하이브리드 검색 (BM25 + Vector)
| 항목 | 상세 |
|------|------|
| BM25 구현 | `rank_bm25.BM25Okapi` (설치 시) 또는 SimpleHybridSearcher (폴백) |
| 토크나이저 | konlpy Okt (설치 시) 또는 정규식 + 조사 제거 |
| RRF 상수 (k) | 60 (env: `RRF_K`) |
| 도메인별 가중치 | laborlaw: V0.4/B0.6, semiconductor: V0.6/B0.4, msds: V0.4/B0.6 등 |
| 키워드 부스팅 | content: +0.1, document_title: x3.0, section_title: x2.0 |
| **Thread-safety 이슈** | `search_with_keyword_boost()`에서 `self.vector_weight/bm25_weight` 직접 변경 후 복원 — race condition 가능 |

#### E. 리랭킹 (Reranking)
| 옵션 | 모델 | 특징 |
|------|------|------|
| Pinecone Inference (기본) | `bge-reranker-v2-m3` | 다국어, 1024 토큰, API 기반 |
| Local Cross-Encoder | `mmarco-mMiniLMv2-L12-H384-v1` | 다국어, max_length=512 |
| Lightweight (폴백) | keyword overlap | 정확도 낮음 |
| 하이브리드 점수 | rerank 70-80% + original 20-30% (도메인별 가변) | min-max 정규화 |

#### F. 컨텍스트 최적화
| 기법 | 구현 |
|------|------|
| 중복 제거 | Jaccard + n-gram(3) 유사도 평균 >= 0.90 |
| LLM 어텐션 최적화 | Lost-in-Middle 전략 (고점수→시작/끝, 저점수→중간) |
| 최소 점수 필터 | `MIN_RELEVANCE_SCORE = 0.2` |
| 최소 토큰 필터 | `MIN_TOKEN_COUNT = 30` |
| 답변 신뢰도 | citation_coverage(40%) + avg_source_score(35%) + substantiveness(25%) |

### 2.3 품질 측정 현황

**현재 측정되는 것:**
- 답변 신뢰도 (compute_answer_confidence): high/medium/low 3단계
- 검색 결과 수, 리랭킹 적용 여부 로깅

**측정되지 않는 것:**
- Recall, Precision, MRR, NDCG 등 IR 표준 메트릭
- 도메인별/쿼리 유형별 검색 품질 비교
- 사용자 피드백 기반 relevance 판단
- 각 파이프라인 Phase별 기여도/성능 영향 분석
- 인제스천 품질 (chunk 크기 분포, 의미 단위 정확도)

---

## 3. 문제점 분석 (Problem Analysis)

### 3.1 Critical Issues

| # | 문제 | 위치 | 영향도 | 설명 |
|---|------|------|--------|------|
| C1 | 품질 측정 체계 부재 | 전체 | HIGH | IR 표준 메트릭 미적용. 개선 효과 정량화 불가 |
| C2 | Thread-safety 결함 | `hybrid_searcher.py:347-356` | HIGH | `search_with_keyword_boost()`에서 instance 상태 직접 변경. 동시 요청 시 race condition. 예외 발생 시 가중치 복원 실패 가능 |
| C3 | 임베딩 모델 한계 | `embedding_generator.py` | MEDIUM | `text-embedding-3-small` (1536d)은 한국어 전문 도메인에서 최적이 아닐 수 있음 |
| C4 | BM25 인덱스 비효율 | `hybrid_searcher.py:125-155` | MEDIUM | 매 쿼리마다 vector_results로 BM25 인덱스 재구축. 전체 코퍼스 미활용 → BM25가 vector search 결과 내에서만 재정렬하는 효과 |
| C5 | Pinecone 메타데이터 한국어 절단 | `pinecone_uploader.py:115-130` | MEDIUM | content를 32KB UTF-8 바이트 단위로 절단 → 한국어(3byte/char) 문장 중간 잘림 가능 |
| C6 | Laborlaw 예외 처리 일관성 | `rag_pipeline.py:410` | LOW | laborlaw만 no-results 체크 우회 → 빈 context로 LLM 호출 가능 |

### 3.2 Improvement Opportunities

| # | 기회 | 위치 | 기대 효과 |
|---|------|------|----------|
| I1 | 검색 품질 평가 프레임워크 | 신규 | 도메인별 golden dataset + 자동 평가로 개선 추적 가능 |
| I2 | 청크 품질 개선 | `semantic_chunker.py` | 의미 단위 분할 정확도 향상 → 검색 정밀도 향상 |
| I3 | 임베딩 모델 실험 | `embedding_generator.py` | `text-embedding-3-large` 또는 multilingual 모델 A/B 테스트 |
| I4 | Reranker 한국어 최적화 | `reranker.py` | 한국어 특화 cross-encoder 또는 fine-tuned 모델 검토 |
| I5 | 도메인별 가중치 자동 튜닝 | `hybrid_searcher.py`, `reranker.py` | 현재 수동 설정된 가중치를 평가 데이터 기반으로 최적화 |
| I6 | 쿼리 라우팅 고도화 | `rag_pipeline.py` | 쿼리 유형별 (사실, 절차, 비교, 계산) 최적 검색 전략 분기 |
| I7 | BM25 전체 코퍼스 인덱스 | `hybrid_searcher.py` | Pinecone 결과만이 아닌 전체 코퍼스 BM25 → RRF 결합 |
| I8 | 사용자 피드백 루프 | 신규 | 답변 유용성 피드백 수집 → relevance label로 활용 |
| I9 | Phase별 레이턴시 모니터링 | `rag_pipeline.py` | 어느 Phase가 병목인지 정량 파악 → 최적화 우선순위 결정 |
| I10 | @mention 검색 버퍼 확대 | `rag_pipeline.py:334` | Pinecone $eq 한계로 mention 필터는 Phase 3 후처리 의존 → 후보 풀 확대 필요 |
| I11 | Top-K 배수 투명성 | `rag_pipeline.py:333-338` | 숨겨진 top_k 배수로 후보 수 예측 불가 → API 응답에 candidates 메트릭 포함 |

---

## 4. 개선 계획 (Improvement Plan)

### Phase A: 품질 측정 체계 구축 (Week 1)

#### A1. Golden Dataset 구축
- 도메인별 20-50개 대표 질문-정답 쌍 수작업 라벨링
- 정답 문서(relevant documents)와 함께 기대 답변 요약 포함
- 포맷: JSON `{ query, relevant_doc_ids[], expected_answer_summary, domain, difficulty }`

#### A2. 자동 평가 스크립트
- Recall@K, MRR, NDCG@K 계산
- 파이프라인 Phase별 중간 결과 캡처 및 기여도 분석
- 도메인별/난이도별 분류 통계

#### A3. 기준선(Baseline) 측정
- 현행 파이프라인으로 golden dataset 전수 평가
- Phase별 성능 기여도 분해 (쿼리 확장 ON/OFF, BM25 ON/OFF, 리랭킹 ON/OFF)

### Phase B: 핵심 품질 개선 (Week 2)

#### B1. Thread-safety 수정 (C2)
- `search_with_keyword_boost()`에서 `self.vector_weight/bm25_weight` 직접 변경 대신 로컬 변수 또는 내부 메서드 파라미터로 전달
- 예외 발생 시에도 인스턴스 상태 오염 방지
- 예상 영향: 동시 요청 환경(gunicorn workers)에서의 결과 일관성 보장

#### B2. 쿼리 라우팅 고도화 (I6)
- 쿼리 유형 분류기 추가 (사실형, 절차형, 비교형, 계산형)
- 유형별 top_k 배수, HyDE 활성화 여부, 리랭킹 가중치 차별화

#### B3. 중복 제거 임계값 조정 (F)
- 현재 0.90 → 도메인별 최적 임계값 실험
- laborlaw (법조항 유사 구조): 더 낮은 임계값 (0.80)
- semiconductor (기술 문서 다양): 현행 유지 (0.90)

#### B4. BM25 인덱스 전략 개선 (C4, I7)
- 현행: vector_results로만 BM25 인덱스 구축 → BM25가 이미 검색된 결과만 재정렬하는 한계
- 옵션 1: 네임스페이스별 영구 BM25 인덱스 (메모리 상주) — 진정한 하이브리드 검색
- 옵션 2: 주기적 인덱스 리빌드 (인제스천 후 트리거)
- 옵션 3: 현행 유지하되 BM25 결과와 vector 결과의 독립성 문서화
- 비용-효과 분석 후 결정 (메모리 vs 품질 향상폭)

#### B5. Phase별 레이턴시 모니터링 추가 (I9)
- 각 Phase 시작/종료 시간 기록
- API 응답에 `phase_latencies` 딕셔너리 포함
- P50/P95/P99 레이턴시 대시보드용 데이터

#### B6. Pinecone 메타데이터 한국어 절단 수정 (C5)
- UTF-8 바이트 단위 대신 문자 단위로 먼저 절단 후 인코딩
- 한국어 3byte/char 고려하여 안전 마진 적용

### Phase C: 고급 최적화 (Week 3, 선택)

#### C1. 임베딩 모델 A/B 테스트 (I3)
- `text-embedding-3-large` (3072d) vs 현행 `small` (1536d)
- Golden dataset 기준 Recall/MRR 비교
- 비용 증가 대비 품질 향상폭 분석

#### C2. 리랭커 한국어 최적화 (I4)
- `bge-reranker-v2-m3` vs `BAAI/bge-reranker-v2-gemma` 비교
- 한국어 도메인별 리랭킹 정확도 벤치마크

#### C3. 사용자 피드백 루프 (I8)
- 답변 유용성 평가 UI (thumbs up/down)
- 피드백 데이터 → golden dataset 자동 확장
- 주간 품질 리포트 자동화

---

## 5. 기술적 제약 및 리스크

| 리스크 | 확률 | 영향 | 완화 방안 |
|--------|------|------|----------|
| Golden dataset 라벨링 비용/시간 | HIGH | MEDIUM | 도메인당 20개로 최소 시작, 점진 확장 |
| 임베딩 모델 변경 시 전체 re-indexing 필요 | MEDIUM | HIGH | 별도 네임스페이스에서 A/B 테스트 후 마이그레이션 |
| BM25 전체 코퍼스 인덱스의 메모리 비용 | MEDIUM | MEDIUM | 네임스페이스별 분리 + lazy loading |
| 리랭커 변경 시 레이턴시 증가 | LOW | MEDIUM | 타임아웃 및 폴백 메커니즘 기존 구축 완료 |

---

## 6. 성공 기준

| 기준 | 측정 방법 | 목표 |
|------|----------|------|
| Recall@10 | Golden dataset 자동 평가 | 85% 이상 |
| MRR | Golden dataset 자동 평가 | 0.70 이상 |
| 답변 신뢰도 | compute_answer_confidence 평균 | high 비율 60%+ |
| 레이턴시 | Phase별 소요 시간 로깅 | P95 < 5s (전체 파이프라인) |
| Thread-safety | 동시 요청 테스트 | race condition 0건 |

---

## 7. 참고 파일 목록

| 파일 | 역할 |
|------|------|
| `services/rag_pipeline.py` | 7-Phase RAG 파이프라인 오케스트레이터 |
| `src/query_enhancer.py` | 쿼리 확장 (Multi-Query, HyDE, Synonym, Keyword) |
| `src/hybrid_searcher.py` | BM25 + Vector RRF 하이브리드 검색 |
| `src/reranker.py` | Cross-encoder / Pinecone Inference 리랭킹 |
| `src/context_optimizer.py` | 중복 제거, Lost-in-Middle 리오더링 |
| `src/embedding_generator.py` | OpenAI 임베딩 생성 |
| `src/semantic_chunker.py` | 의미 기반 청킹 |
| `src/agent.py` | PineconeAgent (검색 인터페이스) |
| `services/filters.py` | 도메인별 메타데이터 필터 빌더 |
| `services/domain_config.py` | 도메인 설정 및 프롬프트 |
| `services/singletons.py` | 싱글턴 인스턴스 관리 |

---

## 8. 다음 단계

1. **Design 단계**: Phase A (품질 측정 체계) 상세 설계 → `/pdca design vector-db-search-quality`
2. **Do 단계**: Phase A 구현 → 기준선 측정 → Phase B 구현
3. **Check 단계**: Golden dataset 기반 Before/After 비교 → `/pdca analyze vector-db-search-quality`
