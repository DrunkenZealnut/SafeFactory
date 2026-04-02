# Plan: 교사 답변 피드백 시스템

> **Feature**: teacher-feedback
> **Date**: 2026-04-02
> **Project**: SafeFactory
> **Version**: 1.0
> **Origin**: CEO Review D4 — 품질 개선을 위한 교사 피드백 수집

---

## Executive Summary

| Perspective | Description |
|-------------|-------------|
| **Problem** | 교사가 SafeFactory의 AI 답변이 부정확하다고 느껴도 신고할 방법이 없고, 개발자는 어떤 답변이 문제인지 알 수 없어 품질 개선이 데이터 없이 진행됨 |
| **Solution** | AI 답변에 "부정확 신고" 버튼을 추가하고, 신고된 답변을 DB에 저장하여 관리자 패널에서 조회·분석·평가 데이터로 활용 |
| **Function UX Effect** | 답변 하단에 👎 버튼 한 번 클릭으로 피드백 제출, 선택적 코멘트 입력, 관리자는 신고 목록에서 패턴 분석 가능 |
| **Core Value** | 교사의 도메인 전문성을 품질 개선 루프에 연결 — "측정할 수 없는 것은 개선할 수 없다"의 첫 단계 |

---

## 1. Overview

### 1.1 Purpose

AI 답변에 대한 교사 피드백을 수집·저장하여, 답변 품질 평가 데이터셋(Golden Dataset) 구축과 RAG 파이프라인 개선의 기초 데이터로 활용한다.

### 1.2 Background

- CEO Review에서 SafeFactory의 #1 문제는 "교사가 답변 품질을 신뢰하지 않음"으로 확인됨
- 현재 시스템에는 답변 품질을 측정하거나 피드백을 수집하는 경로가 전혀 없음
- `scripts/eval/eval_pipeline.py`에 평가 프레임워크가 존재하지만, `golden_dataset.json`의 `expected_sources`가 비어있어 실질적 평가 불가
- 교사 피드백을 먼저 수집해야 합성 데이터가 아닌 실제 데이터 기반으로 평가 프레임워크를 보정할 수 있음 (CEO Review Outside Voice 결론)

### 1.3 Related Documents

- `models.py` — 기존 DB 모델 (SearchHistory, SharedQuestion 등 유사 패턴 존재)
- `api/v1/search.py` — `/ask`, `/ask/stream` 엔드포인트 (답변 생성 지점)
- `api/v1/admin.py` — 관리자 API (admin_required 데코레이터 재사용)
- `templates/domain.html` — 검색/Q&A 인터페이스 (피드백 버튼 추가 지점)
- `templates/admin.html` — 관리자 패널 (신고 목록 UI 추가 지점)
- `scripts/eval/golden_dataset.json` — 향후 피드백 데이터를 평가 데이터셋으로 변환

---

## 2. Scope

### 2.1 In Scope

- **AnswerFeedback 모델**: 신고된 답변 저장 (query, answer, namespace, feedback_type, comment, user_id)
- **피드백 제출 API**: `POST /api/v1/feedback` — 로그인 사용자가 답변에 피드백 제출
- **피드백 조회 API**: `GET /api/v1/admin/feedback` — 관리자용 신고 목록 (페이지네이션, 필터)
- **피드백 내보내기 API**: `GET /api/v1/admin/feedback/export` — JSON/CSV 형식 내보내기 (Golden Dataset 변환용)
- **프론트엔드 버튼**: `domain.html`의 답변 영역에 👎 "부정확 신고" 버튼
- **관리자 UI**: `admin.html`에 신고 목록 탭 추가

### 2.2 Out of Scope

- 긍정 피드백 (👍) — 1차에서는 부정확 신고만 수집 (신호 대 잡음 비율 유지)
- 자동 답변 개선 — 피드백은 수집만, 자동 파이프라인 수정은 후속 과제
- 피드백 기반 프롬프트 자동 튜닝 — 후속 과제 (D3 평가 프레임워크 이후)
- 비로그인 사용자 피드백 — 로그인 사용자만 (스팸 방지, 추적 가능)

---

## 3. Requirements

### 3.1 Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-01 | 답변 표시 후 "부정확 신고" 버튼 노출 | Must |
| FR-02 | 버튼 클릭 시 피드백 타입 선택 모달 (부정확/불완전/관련없음/이해불가) | Must |
| FR-03 | 선택적 코멘트 입력란 (교사가 올바른 답변을 적을 수 있음) | Must |
| FR-04 | 제출 시 query, answer(전문), namespace, sources, confidence 저장 | Must |
| FR-05 | 관리자 페이지에서 신고 목록 조회 (최신순, namespace별 필터) | Must |
| FR-06 | 관리자가 신고를 resolved/dismissed 처리 | Should |
| FR-07 | 신고 데이터를 Golden Dataset JSON으로 내보내기 | Must |
| FR-08 | 동일 사용자가 같은 답변에 중복 신고 방지 | Must |

### 3.2 Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NFR-01 | 피드백 제출은 200ms 이내 응답 |
| NFR-02 | SQLite 호환 (기존 DB 인프라 유지) |
| NFR-03 | 기존 답변 렌더링 성능에 영향 없음 |

---

## 4. Data Model

### 4.1 AnswerFeedback 모델

```python
class AnswerFeedback(db.Model):
    __tablename__ = 'answer_feedbacks'
    __table_args__ = (
        db.Index('ix_feedback_created', 'created_at'),
        db.Index('ix_feedback_namespace', 'namespace'),
    )

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False)

    # 원본 질문/답변 전문 저장 (재현 가능하도록)
    query = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, nullable=False)
    namespace = db.Column(db.String(100), nullable=False, default='')
    source_count = db.Column(db.Integer, nullable=False, default=0)
    confidence_score = db.Column(db.Float, nullable=True)

    # 피드백 내용
    feedback_type = db.Column(db.String(20), nullable=False)
    # Types: 'inaccurate' | 'incomplete' | 'irrelevant' | 'unclear'
    comment = db.Column(db.Text, nullable=True)  # 교사의 정정/코멘트

    # 관리 상태
    status = db.Column(db.String(20), nullable=False, default='pending')
    # Status: 'pending' | 'resolved' | 'dismissed'
    admin_note = db.Column(db.Text, nullable=True)

    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    resolved_at = db.Column(db.DateTime, nullable=True)

    user = db.relationship('User', backref='answer_feedbacks')
```

### 4.2 피드백 타입 정의

| Type | 한국어 | 설명 |
|------|--------|------|
| `inaccurate` | 부정확함 | 사실과 다른 내용 포함 |
| `incomplete` | 불완전함 | 중요한 정보 누락 |
| `irrelevant` | 관련없음 | 질문과 무관한 답변 |
| `unclear` | 이해불가 | 전문용어 과다 또는 구조 불량 |

---

## 5. API Design

### 5.1 피드백 제출

```
POST /api/v1/feedback
Auth: login_required

Body:
{
    "query": "CVD 공정에서 사용되는 가스는?",
    "answer": "CVD 공정에서는...(전문)",
    "namespace": "semiconductor-v2",
    "source_count": 5,
    "confidence_score": 0.72,
    "feedback_type": "inaccurate",
    "comment": "PECVD에서는 SiH4 외에 NH3도 사용합니다"  // optional
}

Response: { "status": "success", "data": { "id": 123 } }
```

### 5.2 관리자 피드백 목록

```
GET /api/v1/admin/feedback?page=1&per_page=20&namespace=semiconductor-v2&status=pending
Auth: admin_required

Response: {
    "status": "success",
    "data": {
        "items": [...],
        "total": 45,
        "page": 1,
        "pages": 3
    }
}
```

### 5.3 관리자 상태 업데이트

```
PUT /api/v1/admin/feedback/<id>
Auth: admin_required

Body: { "status": "resolved", "admin_note": "프롬프트 수정 반영" }
```

### 5.4 Golden Dataset 내보내기

```
GET /api/v1/admin/feedback/export?format=json&status=resolved
Auth: admin_required

Response: Golden Dataset 호환 JSON (eval_pipeline.py에서 바로 사용 가능)
```

---

## 6. UI Design

### 6.1 답변 영역 (domain.html)

답변 렌더링 완료 후 하단에 피드백 버튼 표시:

```
┌─────────────────────────────────────┐
│  AI 답변 텍스트...                   │
│  출처: [1] 반도체 공정 [2] CVD 자료  │
│                                     │
│  ────────────────────────────────── │
│  이 답변이 도움이 되었나요?           │
│  [👎 부정확 신고]                     │
└─────────────────────────────────────┘
```

버튼 클릭 시 모달:

```
┌─────────────────────────────────────┐
│  답변 피드백                         │
│                                     │
│  어떤 문제가 있나요?                  │
│  ○ 사실과 다른 내용 (부정확)          │
│  ○ 중요한 정보 누락 (불완전)          │
│  ○ 질문과 무관한 답변 (관련없음)      │
│  ○ 이해하기 어려움 (이해불가)         │
│                                     │
│  추가 의견 (선택):                    │
│  ┌─────────────────────────────┐    │
│  │                             │    │
│  └─────────────────────────────┘    │
│                                     │
│        [취소]  [제출]                │
└─────────────────────────────────────┘
```

### 6.2 관리자 패널 (admin.html)

기존 탭(문서/커뮤니티/뉴스) 옆에 "답변 피드백" 탭 추가:

```
[문서관리] [커뮤니티] [뉴스] [답변 피드백]

┌─────────────────────────────────────┐
│ 필터: [전체 ▾] [pending ▾] [검색...]│
│                                     │
│ #45 | semiconductor-v2 | inaccurate │
│ Q: CVD 공정에서 사용되는 가스는?      │
│ 교사 코멘트: PECVD에서는 SiH4 외에.. │
│ 2026-04-03 10:30 | [해결] [무시]     │
│                                     │
│ #44 | field-training | incomplete   │
│ Q: 연삭기 안전수칙은?                │
│ 교사 코멘트: 보호구 착용 부분이 빠짐  │
│ 2026-04-03 09:15 | [해결] [무시]     │
└─────────────────────────────────────┘

[내보내기: JSON] [내보내기: CSV]
```

---

## 7. Implementation Plan

### 7.1 구현 순서

| Step | Task | File(s) | Effort |
|------|------|---------|--------|
| 1 | AnswerFeedback 모델 추가 | `models.py` | S |
| 2 | 피드백 제출 API | `api/v1/feedback.py` (신규) | S |
| 3 | 관리자 피드백 API (목록/상태변경/내보내기) | `api/v1/admin.py` | M |
| 4 | Blueprint 등록 | `api/v1/__init__.py` | S |
| 5 | 답변 영역에 피드백 버튼 + 모달 | `templates/domain.html` | M |
| 6 | 관리자 패널에 피드백 탭 | `templates/admin.html` | M |
| 7 | 중복 신고 방지 (unique constraint) | `models.py` | S |

### 7.2 파일 변경 목록

| File | Change Type | Description |
|------|-------------|-------------|
| `models.py` | 수정 | AnswerFeedback 모델 추가 |
| `api/v1/feedback.py` | 신규 | 피드백 제출 API |
| `api/v1/admin.py` | 수정 | 관리자 피드백 조회/상태변경/내보내기 API 추가 |
| `api/v1/__init__.py` | 수정 | feedback blueprint import 추가 |
| `templates/domain.html` | 수정 | 피드백 버튼 + 모달 UI |
| `templates/admin.html` | 수정 | 피드백 탭 UI |

### 7.3 재사용 패턴

- **모델 패턴**: `SearchHistory` 모델의 `user_id`, `query`, `namespace`, `created_at`, `to_dict()` 패턴 재사용
- **API 패턴**: `api/v1/questions.py`의 페이지네이션 + 필터 패턴 재사용
- **관리자 패턴**: `api/v1/admin.py`의 `admin_required` 데코레이터 재사용
- **프론트엔드 패턴**: `domain.html`의 기존 모달 (북마크, 공유 질문) 패턴 재사용
- **관리자 UI 패턴**: `admin.html`의 기존 탭/테이블 패턴 재사용

---

## 8. Downstream Integration

### 8.1 Golden Dataset 연동

피드백 내보내기 API가 생성하는 JSON 형식:

```json
{
  "version": "2.0",
  "source": "teacher-feedback",
  "domains": {
    "semiconductor": [
      {
        "id": "fb-045",
        "query": "CVD 공정에서 사용되는 가스는?",
        "namespace": "semiconductor-v2",
        "feedback_type": "inaccurate",
        "teacher_comment": "PECVD에서는 SiH4 외에 NH3도 사용합니다",
        "original_answer_excerpt": "...",
        "difficulty": "medium",
        "query_type": "factual"
      }
    ]
  }
}
```

이 형식은 `scripts/eval/eval_pipeline.py`의 `load_golden_dataset()`과 호환되도록 설계됨.

### 8.2 향후 활용 경로

```
교사 피드백 → DB 저장 → 관리자 검토 → resolved 처리
                                          ↓
                                   내보내기 (JSON)
                                          ↓
                              Golden Dataset v2에 병합
                                          ↓
                              eval_pipeline.py 실행
                                          ↓
                              RAG 파이프라인 개선 근거
```
