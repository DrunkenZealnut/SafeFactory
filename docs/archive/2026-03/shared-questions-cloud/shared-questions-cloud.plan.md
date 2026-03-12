# 공유한 질문들 클라우드로 보기 Planning Document

> **Summary**: 사용자들이 공유한 질문들을 워드 클라우드(태그 클라우드) 형태로 시각화하여, 인기 키워드와 트렌드를 한눈에 파악할 수 있는 기능
>
> **Project**: SafeFactory
> **Author**: zealnutkim
> **Date**: 2026-03-12
> **Status**: Draft

---

## Executive Summary

| Perspective | Content |
|-------------|---------|
| **Problem** | 공유된 질문들이 단순 리스트로만 표시되어 전체적인 관심 주제와 트렌드를 파악하기 어려움 |
| **Solution** | SharedQuestion 데이터에서 키워드를 추출하여 워드 클라우드로 시각화, 클릭 시 해당 질문 목록으로 이동 |
| **Function/UX Effect** | 도메인별 인기 키워드를 직관적으로 확인하고, 클릭하면 관련 질문을 바로 탐색할 수 있는 인터랙티브 경험 |
| **Core Value** | 커뮤니티 집단지성의 시각화를 통해 사용자 참여도 향상 및 지식 탐색 효율 극대화 |

---

## 1. Overview

### 1.1 Purpose

현재 공유된 질문들은 `domain.html`의 "인기 질문" 섹션에서 단순 리스트(최대 10개)로만 표시된다. 이 기능은 SharedQuestion 테이블의 질문 텍스트에서 핵심 키워드를 추출하여 워드 클라우드로 시각화함으로써, 사용자들이 각 도메인에서 어떤 주제가 가장 많이 논의되는지 직관적으로 파악할 수 있도록 한다.

### 1.2 Background

- **현재 상태**: `SharedQuestion` 모델에 질문 데이터(query, like_count, namespace)가 이미 저장되고 있음
- **기존 UI**: `domain.html`에 "🔥 인기 질문" 섹션이 리스트 형태로 존재 (`/api/v1/questions/popular`)
- **니즈**: 단순 리스트 대비 워드 클라우드는 전체 트렌드를 한눈에 보여주어 탐색 효율을 높임
- **데이터 기반**: `SearchHistory`(검색 이력)와 `SharedQuestion`(공유 질문) 두 테이블의 데이터 활용 가능

### 1.3 Related Documents

- 기존 코드: `api/v1/questions.py`, `models.py` (SharedQuestion)
- UI: `templates/domain.html` (인기 질문 섹션)

---

## 2. Scope

### 2.1 In Scope

- [ ] 워드 클라우드 API 엔드포인트 구현 (`GET /api/v1/questions/wordcloud`)
- [ ] 한국어/영어 텍스트에서 키워드 추출 및 빈도수 계산 로직
- [ ] 워드 클라우드 프론트엔드 시각화 (HTML Canvas 또는 SVG 기반)
- [ ] 도메인(namespace)별 필터링
- [ ] 키워드 클릭 시 해당 질문 목록 표시
- [ ] `domain.html` 인기 질문 섹션에 워드 클라우드 뷰 추가 (리스트/클라우드 토글)

### 2.2 Out of Scope

- 실시간 WebSocket 업데이트 (폴링으로 충분)
- 사용자별 개인화된 워드 클라우드
- SearchHistory 데이터 결합 (향후 확장)
- 워드 클라우드 이미지 다운로드/공유

---

## 3. Requirements

### 3.1 Functional Requirements

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| FR-01 | SharedQuestion에서 키워드 추출 및 빈도/가중치 계산 API | High | Pending |
| FR-02 | 도메인(namespace)별 워드 클라우드 데이터 필터링 | High | Pending |
| FR-03 | 워드 클라우드 시각화 렌더링 (크기=빈도, 색상=도메인) | High | Pending |
| FR-04 | 키워드 클릭 시 해당 키워드 포함 질문 리스트 표시 | Medium | Pending |
| FR-05 | 리스트/클라우드 뷰 토글 버튼 | Medium | Pending |
| FR-06 | 기간 필터 (7일/30일/전체) | Low | Pending |

### 3.2 Non-Functional Requirements

| Category | Criteria | Measurement Method |
|----------|----------|-------------------|
| Performance | 워드 클라우드 데이터 API 응답 < 500ms | API 응답 시간 측정 |
| Performance | 클라이언트 렌더링 < 1초 (100개 키워드 기준) | 브라우저 성능 프로파일링 |
| UX | 모바일에서도 가독성 유지 (반응형) | 디바이스 테스트 |
| Security | 비공개(is_hidden) 질문 데이터 제외 | API 필터 검증 |

---

## 4. Success Criteria

### 4.1 Definition of Done

- [ ] 워드 클라우드 API 구현 및 정상 동작
- [ ] 프론트엔드 워드 클라우드 시각화 완성
- [ ] 키워드 클릭 → 질문 목록 연동
- [ ] 5개 도메인 모두에서 정상 동작 확인
- [ ] 모바일/데스크톱 반응형 레이아웃 확인

### 4.2 Quality Criteria

- [ ] 한국어 형태소 분석 또는 n-gram 기반 키워드 추출 정확도
- [ ] 불용어(stopwords) 적절히 필터링
- [ ] API 에러 시 기존 리스트 뷰로 graceful fallback

---

## 5. Risks and Mitigation

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| 한국어 형태소 분석 라이브러리 의존성 | Medium | Medium | 서버사이드 간단한 n-gram 또는 정규식 기반 추출로 시작, 필요 시 konlpy 도입 |
| 공유 질문 데이터 부족 (초기) | Medium | High | 최소 데이터 임계값 설정, 부족 시 SearchHistory 보조 활용 또는 안내 메시지 표시 |
| 워드 클라우드 렌더링 성능 | Low | Low | 키워드 수 제한 (최대 100개), Canvas 기반 렌더링 |
| 불용어 필터링 부정확 | Low | Medium | 커스텀 불용어 사전 + 최소 빈도 임계값 적용 |

---

## 6. Architecture Considerations

### 6.1 Project Level Selection

| Level | Characteristics | Recommended For | Selected |
|-------|-----------------|-----------------|:--------:|
| **Starter** | Simple structure | Static sites, portfolios | ☐ |
| **Dynamic** | Feature-based modules, BaaS integration | Web apps with backend | ☑ |
| **Enterprise** | Strict layer separation, microservices | High-traffic systems | ☐ |

### 6.2 Key Architectural Decisions

| Decision | Options | Selected | Rationale |
|----------|---------|----------|-----------|
| Backend Framework | Flask (기존) | Flask | 기존 프로젝트 구조 유지 |
| 키워드 추출 | konlpy / 정규식+n-gram / 단순 split | 정규식+n-gram | 추가 의존성 최소화, 한국어 2-3글자 명사 추출에 충분 |
| 워드 클라우드 라이브러리 | D3-cloud / wordcloud2.js / 직접 구현 | wordcloud2.js | 경량(15KB), Canvas 기반, 설정 간단 |
| 데이터 가중치 | like_count / 빈도수 / 혼합 | 빈도수 + like_count 가중 | 인기도와 빈도 모두 반영 |
| UI 배치 | 기존 인기질문 대체 / 토글 / 별도 섹션 | 토글(리스트↔클라우드) | 기존 UX 유지하면서 새 뷰 제공 |

### 6.3 기술 구현 개요

```
Backend (API):
  GET /api/v1/questions/wordcloud
    ├── SharedQuestion 조회 (namespace, is_hidden=False)
    ├── 질문 텍스트 키워드 추출 (정규식 기반)
    ├── 빈도수 계산 + like_count 가중치
    ├── 불용어 필터링
    └── JSON 응답: [{text, weight, count}]

Frontend (domain.html):
  인기 질문 섹션
    ├── 토글 버튼: 리스트 ↔ 클라우드
    ├── wordcloud2.js Canvas 렌더링
    ├── 키워드 클릭 → 질문 리스트 모달/패널
    └── 기간 필터 (7d/30d/all)
```

---

## 7. Convention Prerequisites

### 7.1 Existing Project Conventions

- [x] `CLAUDE.md` has coding conventions section
- [ ] ESLint/Prettier (해당 없음 — Python/Flask 프로젝트)
- [x] 기존 API 패턴: Blueprint + `api/response.py` 헬퍼
- [x] 기존 모델 패턴: Flask-SQLAlchemy + `to_dict()` 메서드

### 7.2 Conventions to Define/Verify

| Category | Current State | To Define | Priority |
|----------|---------------|-----------|:--------:|
| **API 응답 형식** | exists (`api_success/api_error`) | 워드 클라우드 응답 스키마 | High |
| **프론트엔드 JS** | exists (inline `<script>`) | wordcloud2.js CDN 또는 로컬 포함 | Medium |
| **불용어 사전** | missing | `services/stopwords.py` 또는 상수 | Medium |

### 7.3 Environment Variables Needed

| Variable | Purpose | Scope | To Be Created |
|----------|---------|-------|:-------------:|
| (없음) | 기존 환경변수로 충분 | — | — |

---

## 8. Next Steps

1. [ ] Design 문서 작성 (`shared-questions-cloud.design.md`)
2. [ ] API 엔드포인트 상세 설계 (요청/응답 스키마)
3. [ ] 키워드 추출 알고리즘 프로토타이핑
4. [ ] 프론트엔드 워드 클라우드 PoC
5. [ ] 구현 시작

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-03-12 | Initial draft | zealnutkim |
