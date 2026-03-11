# Plan: 사용자별 검색기록 저장 및 불러오기

> **Feature**: search-history
> **Date**: 2026-03-11
> **Project**: SafeFactory
> **Version**: 1.0

---

## Executive Summary

| Perspective | Description |
|-------------|-------------|
| **Problem** | 사용자가 이전에 검색한 쿼리와 결과를 다시 찾을 방법이 없어, 동일한 검색을 반복해야 하고 학습 연속성이 끊김 |
| **Solution** | 로그인한 사용자의 검색/질문 기록을 DB에 자동 저장하고, 기록 조회·재검색·삭제 기능 제공 |
| **Function UX Effect** | 검색창에서 최근 검색어 자동 표시, 검색기록 페이지에서 과거 질문-답변 열람 및 원클릭 재검색 |
| **Core Value** | 반복 검색 제거로 업무 효율 향상, 개인화된 지식 축적 경험 제공 |

---

## 1. Overview

### 1.1 Purpose

SafeFactory 사용자가 수행한 검색(`/api/v1/search`) 및 AI 질문(`/api/v1/ask`) 기록을 사용자별로 저장하여, 이후 조회·재검색·삭제할 수 있는 기능을 구현한다.

### 1.2 Background

- 현재 검색/질문 시 어떤 기록도 저장되지 않음
- 로그인한 사용자(`User` 모델)는 이미 존재하며, Flask-Login으로 세션 관리 중
- 커뮤니티(Post, Comment) 등 사용자 연관 데이터 패턴이 이미 확립되어 있음
- SQLite + Flask-SQLAlchemy 기반으로 모델 추가가 용이

### 1.3 Related Documents

- `models.py` — 기존 DB 모델
- `api/v1/search.py` — 검색/질문 API 엔드포인트
- `services/rag_pipeline.py` — RAG 파이프라인

---

## 2. Scope

### 2.1 In Scope

- **검색기록 자동 저장**: `/search`, `/ask`, `/ask/stream` 요청 시 로그인 사용자의 기록 저장
- **검색기록 조회 API**: 사용자별 기록 목록 조회 (페이지네이션, 필터)
- **검색기록 삭제 API**: 개별 삭제, 전체 삭제
- **최근 검색어 API**: 검색창 자동완성용 최근 검색어 목록
- **검색기록 페이지 UI**: 기록 목록, 재검색 버튼, 삭제 기능

### 2.2 Out of Scope

- 비로그인 사용자 기록 (localStorage 기반 클라이언트 측 저장은 별도 기능)
- 검색 결과 전문 저장 (쿼리와 메타정보만 저장, 전체 결과 본문은 미저장)
- 검색 분석/통계 대시보드 (관리자용 검색 통계는 후속 기능)
- 북마크/즐겨찾기 (별도 기능으로 분리)

---

## 3. Requirements

### 3.1 Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-01 | `/search`, `/ask` 호출 시 로그인 사용자의 검색 쿼리, 네임스페이스, 검색모드, 결과 수를 자동 저장 | Must |
| FR-02 | `GET /api/v1/search/history` — 사용자 검색기록 목록 조회 (페이지네이션, 최신순) | Must |
| FR-03 | `DELETE /api/v1/search/history/<id>` — 개별 기록 삭제 | Must |
| FR-04 | `DELETE /api/v1/search/history` — 전체 기록 삭제 | Must |
| FR-05 | `GET /api/v1/search/history/recent` — 최근 검색어 목록 (중복 제거, 최대 10개) | Should |
| FR-06 | `/ask` 응답에 AI 답변 요약(첫 200자)도 기록에 포함 | Should |
| FR-07 | 검색기록 페이지 UI — 목록 표시, 재검색 클릭, 삭제 버튼 | Must |
| FR-08 | 검색창에 최근 검색어 드롭다운 표시 | Should |

### 3.2 Non-Functional Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-01 | 검색 기록 저장이 검색 응답 시간에 영향 없음 | 추가 지연 < 10ms |
| NFR-02 | 기록 조회 API 응답 시간 | < 200ms |
| NFR-03 | 사용자당 최대 기록 보관 수 | 500건 (초과 시 오래된 기록 자동 삭제) |
| NFR-04 | 개인정보 보호 — 다른 사용자의 기록 접근 불가 | 필수 |

---

## 4. Success Criteria

### 4.1 Definition of Done

- [ ] `SearchHistory` 모델 생성 및 마이그레이션 완료
- [ ] 검색/질문 시 기록 자동 저장 동작 확인
- [ ] 기록 조회/삭제 API 정상 동작
- [ ] 최근 검색어 API 정상 동작
- [ ] 검색기록 페이지 UI 완성
- [ ] 비로그인 사용자는 기록 저장/조회 불가 확인
- [ ] 다른 사용자 기록 접근 차단 확인

### 4.2 Quality Criteria

- API 응답 형식이 기존 `api/response.py` 패턴과 일관
- 기존 검색 성능 저하 없음 (기록 저장은 비동기 또는 경량)
- SQLite 동시성 고려 (기존 패턴과 동일한 thread-safe 방식)

---

## 5. Risks & Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| 대량 검색 기록으로 DB 비대화 | Medium | Medium | 사용자당 500건 제한 + 자동 정리 |
| 검색 기록 저장이 응답 지연 유발 | High | Low | 커밋 후 저장 또는 after_request 훅 활용 |
| SQLite 동시 쓰기 충돌 | Medium | Low | 기존 프로젝트 패턴(WAL mode) 유지 |

---

## 6. Architecture Considerations

### 6.1 Project Level

**Dynamic** — 로그인 사용자 기반 CRUD + 기존 Flask 앱 확장

### 6.2 Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| 저장 위치 | SQLite (기존 DB) | 별도 인프라 불필요, 기존 패턴 유지 |
| 저장 시점 | API 엔드포인트 내 동기 저장 | 단순, SQLite INSERT < 1ms |
| 기록 정리 | 사용자당 500건 초과 시 오래된 기록 삭제 | 자동 관리 |
| API 설계 | 기존 search blueprint 내 추가 | 도메인 일관성 |
| 스트림 응답 기록 | `/ask/stream` 완료 후 기록 저장 | 응답 완료 시점에 결과 수 확정 |

### 6.3 New Files

| File | Purpose |
|------|---------|
| `models.py` (수정) | `SearchHistory` 모델 추가 |
| `api/v1/search.py` (수정) | 기록 저장 로직 + 기록 CRUD 엔드포인트 추가 |
| `templates/` (수정) | 검색기록 페이지 UI + 검색창 최근 검색어 |

### 6.4 DB Schema (예상)

```
SearchHistory
├── id (INT, PK)
├── user_id (INT, FK → users.id, indexed)
├── query (TEXT, NOT NULL)
├── query_type (VARCHAR) — 'search' | 'ask'
├── namespace (VARCHAR) — 도메인 네임스페이스
├── search_mode (VARCHAR) — 'vector' | 'hybrid' | 'keyword'
├── result_count (INT) — 검색 결과 수
├── answer_preview (TEXT, nullable) — AI 답변 요약 (ask만)
├── created_at (DATETIME, default=utcnow, indexed)
└── UNIQUE constraint 없음 (동일 쿼리 반복 허용)
```

---

## 7. Implementation Order

1. `SearchHistory` 모델 추가 (`models.py`)
2. 검색/질문 API에 기록 저장 로직 삽입 (`api/v1/search.py`)
3. 기록 CRUD API 엔드포인트 추가
4. 최근 검색어 API 추가
5. 검색기록 페이지 UI 구현
6. 검색창 최근 검색어 드롭다운 연동

---

## 8. Next Steps

- [ ] Design 문서 작성: `/pdca design search-history`
- [ ] DB 스키마 상세 설계
- [ ] API 엔드포인트 상세 설계
- [ ] UI 와이어프레임

---

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-03-11 | — | Initial plan |
