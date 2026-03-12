# 공유한 질문들 클라우드로 보기 Design Document

> **Summary**: SharedQuestion 데이터에서 키워드를 추출하여 워드 클라우드로 시각화하는 기능
>
> **Project**: SafeFactory
> **Author**: zealnutkim
> **Date**: 2026-03-12
> **Status**: Draft
> **Planning Doc**: [shared-questions-cloud.plan.md](../01-plan/features/shared-questions-cloud.plan.md)

---

## 1. Overview

### 1.1 Design Goals

- SharedQuestion 질문 텍스트에서 키워드를 추출하여 워드 클라우드로 시각화
- 기존 "인기 질문" 리스트 뷰와 토글 가능한 클라우드 뷰 제공
- 키워드 클릭 시 해당 키워드가 포함된 질문 목록 표시
- 추가 Python 의존성 없이 정규식 기반 키워드 추출

### 1.2 Design Principles

- 기존 코드 패턴 유지 (Flask Blueprint, api_success/api_error)
- 프론트엔드 라이브러리 최소화 (CDN 1개: wordcloud2.js)
- Graceful fallback: 데이터 부족 시 리스트 뷰 유지

---

## 2. Architecture

### 2.1 Component Diagram

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
│  domain.html │────▶│ /questions/      │────▶│ SharedQuestion│
│  (Canvas)   │     │  wordcloud API   │     │   (SQLite)    │
└─────────────┘     └──────────────────┘     └──────────────┘
      │                     │
      ▼                     ▼
 wordcloud2.js      keyword_extractor
 (CDN, Canvas)      (services/)
```

### 2.2 Data Flow

```
API Request (namespace, period)
  → SharedQuestion 조회 (is_hidden=False)
  → 질문 텍스트 수집
  → 키워드 추출 (정규식 기반 한국어/영어 토큰화)
  → 불용어 필터링
  → 빈도수 계산 + like_count 가중치
  → JSON 응답: [{text, weight}]

Frontend:
  → wordcloud2.js Canvas 렌더링
  → 클릭 이벤트 → 해당 키워드 질문 목록 표시
```

---

## 3. API Specification

### 3.1 Endpoint

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| GET | /api/v1/questions/wordcloud | 워드 클라우드 키워드 데이터 | Not required |

### 3.2 Request Parameters

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| namespace | string | '' | 도메인 필터 |
| period | string | 'all' | 기간 필터: 7d, 30d, all |
| limit | int | 80 | 최대 키워드 수 (max: 100) |

### 3.3 Response

```json
{
  "success": true,
  "data": {
    "keywords": [
      {"text": "안전교육", "weight": 45},
      {"text": "MSDS", "weight": 32},
      {"text": "보호구", "weight": 28}
    ],
    "total_questions": 150
  }
}
```

### 3.4 Keyword by Click — 기존 popular API 활용

키워드 클릭 시 기존 `/api/v1/questions/popular` 를 활용하되, 검색 필터를 추가하지 않고 프론트엔드에서 질문 텍스트를 검색란에 넣어 AI 답변을 요청하는 방식으로 구현.

---

## 4. Implementation Details

### 4.1 키워드 추출 로직 (`services/keyword_extractor.py`)

```python
def extract_keywords(questions, limit=80):
    """질문 텍스트들에서 키워드를 추출하고 빈도수를 계산"""
    # 1. 모든 질문 텍스트 합침
    # 2. 정규식으로 한국어(2글자+), 영어(3글자+) 토큰 추출
    # 3. 불용어 제거
    # 4. Counter로 빈도수 계산
    # 5. like_count 기반 가중치 적용
    # 6. 상위 N개 반환
```

**한국어 토큰화 전략**:
- 정규식: `[가-힣]{2,}` (2글자 이상 한국어)
- 영어: `[a-zA-Z]{3,}` (3글자 이상 영어)
- 불용어: 조사, 접속사, 일반 동사 등 제외

### 4.2 프론트엔드 구현 (`domain.html`)

- 인기 질문 섹션 헤더에 토글 버튼 추가 (📋 리스트 / ☁️ 클라우드)
- wordcloud2.js CDN 로드
- Canvas 엘리먼트에 워드 클라우드 렌더링
- 클릭 시 해당 키워드를 검색란에 입력

### 4.3 File Structure

```
services/
  └── keyword_extractor.py   (NEW: 키워드 추출 모듈)
api/v1/
  └── questions.py            (MODIFY: wordcloud 엔드포인트 추가)
templates/
  └── domain.html             (MODIFY: 클라우드 뷰 토글 추가)
```

---

## 5. Implementation Order

1. [x] `services/keyword_extractor.py` — 키워드 추출 서비스 모듈
2. [x] `api/v1/questions.py` — wordcloud API 엔드포인트 추가
3. [x] `templates/domain.html` — CSS + HTML + JS 추가

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-03-12 | Initial draft | zealnutkim |
