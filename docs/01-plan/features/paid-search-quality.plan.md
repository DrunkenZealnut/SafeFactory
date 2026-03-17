# 유료요금을 통한 추가 검색품질향상 Planning Document

> **Summary**: 유료 API 업그레이드를 통해 SafeFactory RAG 파이프라인의 검색 정확도, 리랭킹 품질, 임베딩 해상도를 한 단계 끌어올리는 제안
>
> **Project**: SafeFactory
> **Author**: Claude
> **Date**: 2026-03-17
> **Status**: Draft

---

## Executive Summary

| Perspective | Content |
|-------------|---------|
| **Problem** | 현재 무료/저가 API 조합(text-embedding-3-small, 로컬 cross-encoder)으로는 한국어 복합 쿼리의 의미 해상도와 리랭킹 정밀도에 한계가 있어, 전문 도메인(반도체/노동법) 검색 품질이 정체 구간에 진입함 |
| **Solution** | 3단계 유료 업그레이드 전략: (1) Cohere Rerank v3.5 활성화, (2) 임베딩 모델 text-embedding-3-large 전환, (3) GPT-4o 기반 지능형 쿼리 이해 도입 — 각 단계별 ROI 측정 후 점진 적용 |
| **Function/UX Effect** | 검색 결과 상위 5건의 적중률 향상(예상 +15~25%), 크로스 도메인 질문 커버리지 확대, 응답 시간 유지(< 3초), 사용자 체감 "원하는 답이 바로 나온다" |
| **Core Value** | 월 $30~80 수준의 합리적 비용으로 전문 지식 검색 품질을 프리미엄급으로 끌어올려, SafeFactory의 실질적 업무 활용도와 사용자 신뢰를 확보 |

---

## 1. Overview

### 1.1 Purpose

SafeFactory의 현재 RAG 파이프라인은 11개 검색 품질 개선 기능이 적용되어 있으나, 핵심 AI 모델들이 저가/무료 티어에 머물러 있어 품질 상한선이 존재한다. 유료 API 업그레이드를 통해 이 상한선을 돌파하고, 비용 대비 최대 품질 향상을 달성한다.

### 1.2 Background

**현재 상태 (2026-03-17 기준)**:
- 임베딩: OpenAI `text-embedding-3-small` (1536차원, ~$0.02/1M tokens) — 비용 효율적이나 의미 해상도 제한
- 리랭킹: 로컬 cross-encoder 또는 Pinecone Inference API — 한국어 특화 부족
- 쿼리 이해: `gpt-4o-mini` 기반 — 복잡한 한국어 질의 분해 능력 제한
- LLM 응답: Gemini 2.5-flash (일반), Claude Opus (노동법) — 이미 적절한 수준

**기 구현된 품질 개선 (검색 품질 개선 11건)**:
- QW-1~5: 쿼리 강화 5건 (폴백 키워드, 적응형 top_k, 토큰 필터, RRF K, 도메인 동의어)
- CI-1~4: 컨텍스트 개선 4건 (HyDE, MMR, 정규화, 다중 도메인)
- RB-1~2: 재검색/캐싱 2건

**남은 병목**:
1. 임베딩 차원 한계로 미세 의미 차이 구분 어려움
2. 리랭커의 한국어 성능 한계 (로컬 cross-encoder)
3. 쿼리 이해 단계에서 복합 질의 분해 부족

### 1.3 Related Documents

- `docs/03-analysis/search-quality.analysis.md` — 검색 품질 분석 보고서
- `docs/03-analysis/search-flow-audit.analysis.md` — 검색 흐름 감사
- `services/rag_pipeline.py` — RAG 파이프라인 구현체
- `src/reranker.py` — 리랭커 구현체

---

## 2. Scope

### 2.1 In Scope

- [x] **Tier 1**: Cohere Rerank v3.5 활성화 (최소 비용, 최대 효과)
- [ ] **Tier 2**: 임베딩 모델 text-embedding-3-large 전환 (중간 비용, 높은 효과)
- [ ] **Tier 3**: GPT-4o 기반 지능형 쿼리 이해 (선택적, 복잡 쿼리 특화)
- [ ] **Tier 0 (무료)**: CI-1 HyDE 버그 수정 + CI-4 다중 도메인 검색 활성화
- [ ] 비용 모니터링 대시보드 구축
- [ ] A/B 테스트 프레임워크 (품질 비교 측정)

### 2.2 Out of Scope

- LLM 응답 생성 모델 변경 (현재 Gemini/Claude 조합 유지)
- Pinecone 플랜 업그레이드 (현재 Serverless 유지)
- 새로운 도메인 추가
- 프론트엔드 UI 변경
- 벡터 DB 자체 마이그레이션 (Pinecone 유지)

---

## 3. Requirements

### 3.1 Functional Requirements

| ID | Requirement | Priority | Status | 예상 비용 |
|----|-------------|----------|--------|-----------|
| FR-01 | Cohere Rerank v3.5 활성화 — `COHERE_API_KEY` 설정 시 자동 적용 | **High** | Pending | ~$3-10/월 |
| FR-02 | 리랭커 A/B 비교 로깅 — Cohere vs 로컬 cross-encoder 성능 수치 기록 | **High** | Pending | $0 |
| FR-03 | text-embedding-3-large (3072차원) 전환 — 신규 문서부터 적용, 기존 벡터 재인덱싱 | **Medium** | Pending | ~$0.13/1M tokens (6.5x) |
| FR-04 | 듀얼 임베딩 지원 — 전환 기간 중 1536/3072 혼용 검색 호환 | **Medium** | Pending | $0 |
| FR-05 | GPT-4o 쿼리 이해 — 복합 질의 분해, 의도 파악, 서브쿼리 생성 | **Low** | Pending | ~$2.50/1M input |
| FR-06 | 쿼리 복잡도 기반 모델 라우팅 — 단순 쿼리는 gpt-4o-mini, 복합 쿼리만 GPT-4o | **Low** | Pending | $0 |
| FR-07 | CI-1 HyDE 조건부 활성화 버그 수정 (무료) | **High** | Pending | $0 |
| FR-08 | CI-4 Secondary namespace 검색 실행 및 결과 병합 (무료) | **Medium** | Pending | $0 |
| FR-09 | 월별 API 비용 추적 로깅 | **Medium** | Pending | $0 |

### 3.2 Non-Functional Requirements

| Category | Criteria | Measurement Method |
|----------|----------|-------------------|
| Performance | 전체 파이프라인 응답시간 < 5초 (리랭킹 포함) | `_timings` 메트릭 모니터링 |
| Performance | Cohere API 호출 < 500ms (p95) | 리랭킹 단계 타이밍 |
| Cost | 월 총 API 비용 < $100 (Tier 1+2 기준) | API 사용량 로깅 |
| Reliability | API 장애 시 로컬 리랭커 자동 폴백 | 에러 핸들링 테스트 |
| Quality | 검색 상위 5건 적중률 +15% 이상 향상 | 테스트 쿼리셋 평가 |
| Compatibility | 기존 1536차원 벡터와 3072차원 벡터 혼재 기간 검색 정상 동작 | 통합 테스트 |

---

## 4. Success Criteria

### 4.1 Definition of Done

- [ ] Cohere Rerank v3.5 통합 완료 및 프로덕션 배포
- [ ] 리랭킹 A/B 비교 데이터 최소 100건 수집
- [ ] text-embedding-3-large 전환 완료 (또는 명확한 Go/No-Go 판단)
- [ ] CI-1, CI-4 버그 수정 배포
- [ ] API 비용 모니터링 로깅 동작 확인
- [ ] 테스트 쿼리셋 30건 이상으로 품질 비교 완료

### 4.2 Quality Criteria

- [ ] 검색 적중률(Precision@5) 15% 이상 향상 확인
- [ ] API 장애 시 graceful fallback 동작 확인
- [ ] 월 비용 $100 이내 유지 확인
- [ ] 기존 기능 regression 없음

---

## 5. Risks and Mitigation

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Cohere API 지연으로 응답시간 증가 | Medium | Medium | 500ms 타임아웃 + 로컬 cross-encoder 자동 폴백 |
| text-embedding-3-large 재인덱싱 비용 증가 | High | Medium | 단계적 전환 — 신규 문서 먼저, 기존은 배치 재인덱싱 |
| 3072차원 전환 시 Pinecone 인덱스 재생성 필요 | High | High | 새 인덱스 생성 → 데이터 마이그레이션 → 무중단 전환 |
| GPT-4o 쿼리 이해 비용이 예상 초과 | Medium | Low | 쿼리 복잡도 라우팅으로 GPT-4o 호출 최소화 |
| API 키 노출 위험 | High | Low | `.env` 파일 관리 강화, gitignore 확인 |
| 유료 API 중단/요금 변경 | Medium | Low | 다중 리랭커 아키텍처 유지 (Cohere/Pinecone/Local 3중 폴백) |

---

## 6. 단계별 실행 전략

### 6.1 Tier 0: 무료 버그 수정 (즉시)

**비용: $0 | 효과: 기존 기능 정상화**

| 항목 | 설명 | 파일 |
|------|------|------|
| CI-1 수정 | HyDE 조건부 활성화 파라미터를 `_enhance_query()`에 전달 | `services/rag_pipeline.py:823` |
| CI-4 활성화 | Secondary namespace 검색 실행 + 결과 RRF 병합 | `services/rag_pipeline.py` Phase 2 |
| QW-5 보완 | `'화학물질안전': 'MSDS'` 동의어 추가 | `services/query_router.py` |

### 6.2 Tier 1: Cohere Rerank v3.5 (1주차)

**비용: ~$3-10/월 | 효과: 리랭킹 품질 대폭 향상**

| 항목 | 설명 |
|------|------|
| 이미 구현됨 | `src/reranker.py`에 Cohere 리랭커 코드 존재 |
| 필요 작업 | `.env`에 `COHERE_API_KEY` 추가, 설정에서 Cohere 우선순위 활성화 |
| 핵심 장점 | rerank-v3.5는 한국어 지원 우수, 4096 토큰 컨텍스트, 문맥 이해 탁월 |
| A/B 로깅 | 기존 cross-encoder 점수 vs Cohere 점수 병렬 기록 → 정량 비교 |
| 폴백 | API 장애 시 자동으로 로컬 cross-encoder로 전환 (기 구현) |

**예상 비용 산출**:
- 일일 검색 쿼리: ~200건 (추정)
- 쿼리당 리랭크 대상: ~20개 문서
- Cohere 요금: ~$0.10/1K 검색
- 월 비용: 200 * 30 / 1000 * $0.10 = **~$0.60/월** (매우 저렴)

### 6.3 Tier 2: 임베딩 모델 업그레이드 (2-4주차)

**비용: ~$15-40/월 (재인덱싱 일회성 + 운영) | 효과: 의미 해상도 2배 향상**

| 항목 | 설명 |
|------|------|
| 현재 | text-embedding-3-small (1536차원, $0.02/1M tokens) |
| 목표 | text-embedding-3-large (3072차원, $0.13/1M tokens) |
| 핵심 결정 | **Pinecone 인덱스 재생성 필수** (차원 변경 불가) |
| 전환 전략 | 새 인덱스 생성 → 전체 문서 재임베딩 → DNS-like 전환 → 구 인덱스 삭제 |
| 대안 검토 | `text-embedding-3-large`에 `dimensions=1536` 파라미터로 차원 축소 → 인덱스 재생성 불필요, 품질 일부 향상 |

**재인덱싱 비용 산출**:
- 현재 벡터 수: 확인 필요 (`python main.py stats`)
- 예상 10,000 chunks * 평균 300 tokens = 3M tokens
- 재인덱싱 일회성 비용: 3M * $0.13/1M = **~$0.39** (무시 가능)
- 운영 비용: 일일 신규 문서 기준 월 $1-5

### 6.4 Tier 3: 지능형 쿼리 이해 (선택적, 4주차+)

**비용: ~$10-30/월 | 효과: 복합 질의 정확도 대폭 향상**

| 항목 | 설명 |
|------|------|
| 현재 | gpt-4o-mini로 쿼리 확장/동의어 생성 |
| 목표 | 복합 질의 감지 → GPT-4o로 의도 분석 → 서브쿼리 분해 → 개별 검색 후 병합 |
| 라우팅 기준 | 쿼리 길이 > 50자 OR 접속사(그리고/또는/비교) 포함 OR 복수 도메인 키워드 |
| 예시 | "반도체 세정 공정에서 불산 취급 시 노동법상 보호 장구 기준은?" → 반도체 + MSDS + 노동법 3개 서브쿼리 |

**비용 라우팅 예상**:
- 전체 쿼리의 20%만 GPT-4o 라우팅 (복합 쿼리)
- 나머지 80%는 gpt-4o-mini 유지
- 월 비용: 200 * 30 * 0.2 * 500 tokens * $2.50/1M = **~$1.50/월**

---

## 7. 비용 종합 분석

### 7.1 현재 월 예상 비용

| API | 용도 | 월 예상 비용 |
|-----|------|-------------|
| OpenAI (embedding) | text-embedding-3-small | ~$1-3 |
| OpenAI (LLM) | gpt-4o-mini 쿼리 강화 | ~$2-5 |
| Google Gemini | 응답 생성 | ~$3-8 |
| Anthropic Claude | 노동법 응답 | ~$5-15 |
| Pinecone | 벡터 DB | ~$10-20 |
| **소계** | | **~$21-51/월** |

### 7.2 업그레이드 후 월 예상 비용 (Tier 1+2)

| API | 용도 | 월 예상 비용 | 변화 |
|-----|------|-------------|------|
| OpenAI (embedding) | text-embedding-3-large | ~$3-8 | +$2-5 |
| Cohere | Rerank v3.5 | ~$1-3 | **신규** |
| OpenAI (LLM) | gpt-4o-mini 쿼리 강화 | ~$2-5 | 동일 |
| Google Gemini | 응답 생성 | ~$3-8 | 동일 |
| Anthropic Claude | 노동법 응답 | ~$5-15 | 동일 |
| Pinecone | 벡터 DB | ~$10-20 | 동일 |
| **소계** | | **~$24-59/월** | **+$3-8/월** |

### 7.3 ROI 분석

| 항목 | 추가 비용 | 예상 품질 향상 | ROI 판단 |
|------|----------|---------------|----------|
| Tier 0 (버그 수정) | $0 | +5-10% | **즉시 실행** |
| Tier 1 (Cohere) | +$1-3/월 | +10-15% | **강력 추천** |
| Tier 2 (Large 임베딩) | +$2-5/월 | +5-10% | 추천 (재인덱싱 노력 대비) |
| Tier 3 (GPT-4o 쿼리) | +$1-3/월 | +5-10% (복합 쿼리 한정) | 선택적 |
| **합계** | **+$4-11/월** | **+20-35%** | |

---

## 8. Architecture Considerations

### 8.1 Project Level

기존 프로젝트 — Dynamic 수준 유지 (Flask + Python 서비스 레이어 패턴)

### 8.2 Key Architectural Decisions

| Decision | Options | Selected | Rationale |
|----------|---------|----------|-----------|
| 리랭커 전략 | Cohere만 / 다중 폴백 | **다중 폴백** | API 장애 대비 3중 체인 유지 |
| 임베딩 전환 | 즉시 전환 / 점진 전환 / 차원 축소 | **점진 전환** | 서비스 중단 최소화 |
| 쿼리 라우팅 | 전량 GPT-4o / 복잡도 기반 | **복잡도 기반** | 비용 80% 절감 |
| 비용 모니터링 | 수동 / 자동 로깅 | **자동 로깅** | 예산 초과 사전 감지 |
| A/B 테스트 | 없음 / 로깅 비교 / 실시간 분할 | **로깅 비교** | 구현 간단, 효과 측정 충분 |

### 8.3 리랭커 폴백 체인 (현재 아키텍처 활용)

```
Cohere Rerank v3.5 (4096 tokens, 한국어 우수)
    ↓ (API 장애/타임아웃)
Pinecone Inference (bge-reranker-v2-m3)
    ↓ (API 장애/타임아웃)
Local cross-encoder (sentence-transformers)
    ↓ (모델 로드 실패)
Keyword matching (규칙 기반 최후 폴백)
```

---

## 9. Next Steps

1. [ ] **즉시**: Tier 0 무료 버그 수정 (CI-1, CI-4, QW-5)
2. [ ] Design 문서 작성 (`/pdca design paid-search-quality`)
3. [ ] Cohere API 키 발급 및 Tier 1 적용
4. [ ] 테스트 쿼리셋 30건 준비 (도메인별 6건 x 5)
5. [ ] A/B 비교 로깅 구현
6. [ ] Tier 2 임베딩 전환 Go/No-Go 판단

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-03-17 | Initial draft | Claude |
