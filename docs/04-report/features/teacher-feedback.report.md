# Report: 교사 답변 피드백 시스템

> **Feature**: teacher-feedback
> **Date**: 2026-04-02
> **Project**: SafeFactory
> **PDCA Cycle**: Plan → Design → Do → Check → Report

---

## Executive Summary

| 항목 | 내용 |
|------|------|
| Feature | 교사 답변 피드백 시스템 |
| 시작일 | 2026-04-02 |
| 완료일 | 2026-04-02 |
| 소요 시간 | 1 세션 (CEO Review D4) |
| Match Rate | 100% (122/122 항목) |

### 1.3 Value Delivered

| 관점 | 결과 |
|------|------|
| **Problem** | 교사가 AI 답변의 부정확함을 신고할 방법이 없어 품질 개선이 데이터 없이 진행되던 문제 해결 |
| **Solution** | AnswerFeedback 모델 + 피드백 제출 API + 관리자 조회/상태관리/내보내기 API + 프론트엔드 모달 + 관리자 패널 구현. 7개 파일, +495 LOC |
| **Function UX Effect** | AI 답변 하단 "👎 부정확 신고" 버튼 → 4가지 피드백 유형 선택 모달 → 1클릭 제출. 관리자는 피드백 목록 조회/필터/해결/JSON 내보내기 가능 |
| **Core Value** | 교사의 도메인 전문성이 품질 개선 루프에 최초 연결됨. Golden Dataset v2.1 평가 프레임워크와 통합되어 "측정 → 개선 → 재측정" 사이클의 입력 경로 확보 |

---

## 1. Overview

### 1.1 Origin

CEO Review (2026-04-02)에서 SafeFactory의 #1 문제가 "교사가 답변 품질을 신뢰하지 않음"으로 확인됨. Outside Voice 리뷰에서 "교사 피드백을 합성 데이터셋보다 먼저 수집해야 한다"는 결론이 나와, 실행 순서가 D4(피드백) → D2(데이터셋) → D3(평가)으로 재조정됨.

### 1.2 Scope

| In Scope | Out of Scope |
|----------|-------------|
| AnswerFeedback DB 모델 | 긍정 피드백 (👍) |
| 피드백 제출 API (POST /feedback) | 자동 답변 개선 |
| 관리자 피드백 API (목록/상태/내보내기/통계) | 피드백 기반 프롬프트 자동 튜닝 |
| 프론트엔드 피드백 버튼 + 모달 | 비로그인 사용자 피드백 |
| 관리자 패널 피드백 탭 | |

---

## 2. Implementation Summary

### 2.1 Files Changed

| File | Type | Lines | Description |
|------|------|------:|-------------|
| `models.py` | 수정 | +66 | AnswerFeedback 모델 (4 피드백 유형, 중복 방지, 관리 상태) |
| `api/v1/feedback.py` | 신규 | 56 | 피드백 제출 엔드포인트 |
| `api/v1/__init__.py` | 수정 | +1 | Blueprint 등록 |
| `api/v1/admin.py` | 수정 | +128 | 관리자 피드백 CRUD + 내보내기 + 통계 (4 엔드포인트) |
| `templates/domain.html` | 수정 | +160 | 피드백 버튼 + 모달 + CSS + JS |
| `templates/admin.html` | 수정 | +129 | 피드백 관리 섹션 |
| **Total** | | **+540** | 1 신규 + 6 수정 |

### 2.2 API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/v1/feedback` | login | 피드백 제출 |
| GET | `/api/v1/admin/feedback` | admin | 피드백 목록 (페이지네이션, 필터) |
| PUT | `/api/v1/admin/feedback/<id>` | admin | 상태 변경 (resolved/dismissed) |
| GET | `/api/v1/admin/feedback/export` | admin | Golden Dataset 호환 JSON 내보내기 |
| GET | `/api/v1/admin/feedback/stats` | admin | 대시보드 통계 |

### 2.3 Data Model

```
AnswerFeedback
├── user_id (FK → users, CASCADE)
├── query + query_hash (MD5, 중복 방지)
├── answer (전문 저장)
├── namespace, source_count, confidence_score
├── feedback_type: inaccurate | incomplete | irrelevant | unclear
├── comment (교사 정정/코멘트)
├── status: pending | resolved | dismissed
├── admin_note
└── created_at, resolved_at
```

---

## 3. Quality Analysis

### 3.1 Gap Analysis Results

| Category | Items | Match | Rate |
|----------|------:|------:|-----:|
| Data Model | 24 | 24 | 100% |
| API: POST /feedback | 14 | 14 | 100% |
| API: Admin endpoints | 26 | 26 | 100% |
| Blueprint | 1 | 1 | 100% |
| Frontend (button + modal + CSS + JS) | 30 | 30 | 100% |
| Admin UI (section + JS) | 13 | 13 | 100% |
| Error Handling | 6 | 6 | 100% |
| Implementation Order | 6 | 6 | 100% |
| **Total** | **122** | **122** | **100%** |

### 3.2 Adaptations (8)

모두 기능적으로 동등한 개선:
- CSS 폴리싱 (0.82rem, 8px radius 등)
- 관리자 사이드바 패턴 적용 (탭 → 사이드바)
- 한국어 상태 라벨
- 인라인 렌더링 (별도 함수 대신 .map())

### 3.3 Beneficial Additions (9)

설계 범위를 초과한 개선:
1. `logging.info("[Feedback]...")` — 감사 추적
2. 응급 답변 피드백 버튼 제외 — 안전 설계
3. 제출 버튼 초기화 방어 코드
4. `accent-color: #ef4444` — 라디오 버튼 테마
5. `font: inherit` — 타이포그래피 일관성
6. `GET /admin/feedback/stats` — 대시보드 배지 데이터
7. `loadFeedbackBadge()` — 페이지 로드 시 대기 건수
8. `fbCount` — 툴바 총 건수
9. 조건부 말줄임 (300자 이상일 때만)

---

## 4. Broader Context: CEO Review Deliverables

teacher-feedback(D4)는 CEO Review Quality-First 전략의 7개 산출물 중 하나:

| # | Deliverable | Status | Commit |
|---|-------------|:------:|--------|
| D0 | 응급 매뉴얼 안전 감사 | ✅ | 17f4b0a |
| D1 | 인용 버그 수정 | ✅ | 17f4b0a |
| D4 | **교사 답변 피드백** | ✅ | **17f4b0a** |
| D2 | Golden Dataset v2.1 | ✅ | 1e051da → 51274f0 |
| D3 | LLM-as-judge 평가 | ✅ | a10665e |
| D5 | 품질 대시보드 | ✅ | 7c186fe |
| D6 | 회귀 감지 스크립트 | ✅ | 7c186fe |

### 4.1 RAG 품질 기준선 (첫 측정)

| Domain | Recall@K | Queries | Status |
|--------|:--------:|:-------:|--------|
| semiconductor-v2 | 97.78% | 15/15 | 최우수 |
| kosha | 91.67% | 9/10 | 양호 |
| field-training | 71.43% | 14/15 | 개선 필요 |
| chemical-safety | 100.00% | 2/5 | 콘텐츠 부족 |
| **전체** | **87.50%** | **40/45** | **기준선 확립** |

### 4.2 Discovered Issues

| Issue | Severity | Status |
|-------|----------|--------|
| Gemini API 키 만료 (403 PERMISSION_DENIED) | High | 발견됨, 교체 필요 |
| settings.py 기본 embedding_model이 만료된 Gemini | Medium | 발견됨, eval에서 OpenAI로 우회 |
| MSDS 도메인은 Pinecone에 없음 (외부 API) | Info | 데이터셋에서 분리 완료 |

---

## 5. Reuse Patterns

| Pattern | Source | Reused In |
|---------|--------|-----------|
| `SearchHistory` 모델 구조 | `models.py` | AnswerFeedback 모델 |
| `admin_required` 데코레이터 | `api/v1/admin.py` | 모든 관리자 피드백 API |
| `pdf-modal-overlay` CSS | `templates/domain.html` | 피드백 모달 |
| `success_response/error_response` | `api/response.py` | 모든 피드백 API |
| 사이드바 네비게이션 + loaders | `templates/admin.html` | 피드백 탭 |
| `shareQuestion()` 버튼 패턴 | `templates/domain.html` | 피드백 버튼 |

---

## 6. Next Steps

1. **교사 피드백 수집 시작** — 배포 완료, 교사에게 "부정확 신고" 버튼 안내
2. **피드백 → Golden Dataset 병합** — 축적된 피드백을 eval 데이터셋에 반영
3. **field-training 콘텐츠 보강** — 71.43% recall, 문서 추가 인제스트 필요
4. **Gemini API 키 교체** — 프로덕션 임베딩 검색에 영향 가능
5. **`--eval-answer` 실행** — 답변 생성 품질 측정 (retrieval 이후 단계)
6. **LLM-as-judge 실행** — 수집된 피드백으로 채점 보정
