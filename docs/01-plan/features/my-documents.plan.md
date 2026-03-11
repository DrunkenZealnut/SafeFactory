# Plan: 나의 자료 등록 (My Documents Bookmark)

## Executive Summary

| 관점 | 내용 |
|------|------|
| **Problem** | 검색 결과에서 유용한 문서를 발견해도 다시 찾으려면 동일한 검색을 반복해야 하며, 개인화된 자료 관리 기능이 없음 |
| **Solution** | 검색 결과와 Document 카탈로그에서 원하는 자료를 "나의 자료"로 북마크하여 개인 라이브러리 형태로 관리 |
| **Function UX Effect** | 검색 결과 카드에 북마크 버튼 추가, 마이페이지에서 저장된 자료를 폴더/도메인별로 조회·관리 |
| **Core Value** | 사용자 재방문율 향상 및 학습 자료 개인화를 통한 플랫폼 충성도 강화 |

---

## 1. Feature Overview

### 1.1 Feature Name
나의 자료 등록 (My Documents / Bookmark)

### 1.2 Feature ID
`my-documents`

### 1.3 Priority
Medium-High (사용자 리텐션 직결)

### 1.4 Target Users
- 로그인된 SafeFactory 사용자 전체
- 반복적으로 특정 자료를 참조하는 학습자/실무자

---

## 2. Problem Statement

### 2.1 Current Pain Points
1. **재검색 비용**: 이전에 찾았던 유용한 문서를 다시 보려면 같은 키워드로 검색을 반복해야 함
2. **자료 관리 부재**: 검색 기록은 시간순 나열만 가능하고, 특정 문서를 선택적으로 저장할 수 없음
3. **도메인 간 분산**: 반도체·노동법·안전보건 등 여러 도메인의 유용한 자료를 한 곳에서 모아볼 수 없음

### 2.2 Expected Outcome
- 사용자가 검색 결과에서 원클릭으로 자료를 저장
- "나의 자료" 전용 페이지에서 저장한 문서를 도메인별/폴더별로 관리
- 저장된 자료를 클릭하면 원래 검색 컨텍스트(도메인 페이지)로 이동하여 재검색

---

## 3. Scope

### 3.1 In Scope
- **UserBookmark 모델**: 사용자-문서 간 북마크 관계 저장 (source_file 기반)
- **API 엔드포인트**: 북마크 CRUD (추가/삭제/목록/존재 여부 확인)
- **검색 결과 UI 연동**: 검색 결과 카드에 북마크 토글 버튼 추가
- **나의 자료 페이지**: `/my-documents` 전용 페이지 (필터, 정렬, 삭제)
- **마이페이지 연동**: 마이페이지에서 나의 자료 바로가기 링크

### 3.2 Out of Scope (Phase 1)
- 폴더/태그 기반 분류 (향후 확장)
- 자료 메모/노트 기능
- 자료 공유/공개 라이브러리
- 오프라인 저장/다운로드

---

## 4. Technical Analysis

### 4.1 Data Model

**UserBookmark 테이블** (신규):

| Column | Type | Description |
|--------|------|-------------|
| id | Integer, PK | 자동 증가 ID |
| user_id | Integer, FK(users.id) | 사용자 ID |
| source_file | String(500) | Pinecone 벡터의 source_file 경로 (고유 식별) |
| namespace | String(100) | 도메인 namespace (semiconductor-v2, laborlaw 등) |
| title | String(300) | 문서 제목 (filename 또는 ncs_document_title) |
| memo | String(500), nullable | 사용자 메모 (Phase 1에서는 선택적) |
| created_at | DateTime | 북마크 생성 시각 |

- **Unique Constraint**: (user_id, source_file) — 동일 문서 중복 북마크 방지
- **Index**: (user_id, created_at) — 사용자별 목록 조회 최적화
- **MAX_PER_USER**: 200 (사용자당 최대 북마크 수)

### 4.2 Bookmark 대상 식별

검색 결과에는 `source_file` 필드가 항상 포함됨 (Pinecone metadata). 이를 북마크의 고유 식별자로 사용:
- **source_file**: 문서의 원본 파일 경로 (예: `ncs/반도체제조/CVD공정.md`)
- **namespace**: 검색 시 사용된 도메인 namespace
- **title**: `filename` 또는 `ncs_document_title` 메타데이터

같은 source_file의 여러 청크가 검색 결과에 나올 수 있으므로, source_file 단위로 북마크함.

### 4.3 API Design

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/bookmarks` | 북마크 추가 |
| DELETE | `/api/v1/bookmarks/<id>` | 북마크 삭제 |
| DELETE | `/api/v1/bookmarks` | 전체 삭제 |
| GET | `/api/v1/bookmarks` | 목록 조회 (페이지네이션, 필터) |
| GET | `/api/v1/bookmarks/check?source_file=...` | 특정 문서 북마크 여부 확인 |
| POST | `/api/v1/bookmarks/check-batch` | 여러 문서 일괄 북마크 여부 확인 |

### 4.4 Frontend Components

1. **검색 결과 카드 내 북마크 버튼**: 하트/북마크 아이콘 토글 (로그인 시만 표시)
2. **나의 자료 페이지** (`/my-documents`):
   - 도메인별 필터 드롭다운
   - 최신순/이름순 정렬
   - 개별 삭제 및 전체 삭제
   - 클릭 시 해당 도메인 페이지에서 문서명으로 검색
3. **네비게이션 연동**: base.html 마이페이지 옆에 "나의 자료" 링크 추가

### 4.5 Existing System Integration

| 컴포넌트 | 영향 |
|-----------|------|
| `models.py` | UserBookmark 모델 추가 |
| `api/v1/` | `bookmarks.py` 신규 모듈 생성 |
| `api/v1/__init__.py` | bookmarks 블루프린트 import |
| `web_app.py` | `/my-documents` 라우트 추가 |
| `templates/domain.html` | 검색 결과 카드에 북마크 버튼 추가 |
| `templates/base.html` | 네비게이션에 "나의 자료" 링크 추가 |
| `templates/my_documents.html` | 신규 페이지 생성 |

---

## 5. User Stories

### US-01: 검색 결과에서 자료 저장
> 사용자로서, 검색 결과에서 유용한 문서에 북마크 버튼을 눌러 "나의 자료"로 저장하고 싶다.

**Acceptance Criteria**:
- 로그인 상태에서 검색 결과 카드에 북마크 아이콘이 표시됨
- 이미 북마크된 문서는 아이콘이 채워진 상태로 표시됨
- 클릭 시 즉시 토글되며 서버에 반영됨
- 비로그인 시 북마크 버튼 미표시

### US-02: 나의 자료 목록 조회
> 사용자로서, 저장한 자료를 한 곳에서 모아보고 도메인별로 필터링하고 싶다.

**Acceptance Criteria**:
- `/my-documents` 페이지에서 저장한 자료 목록을 볼 수 있음
- 도메인별 필터 적용 가능
- 페이지네이션 지원 (20건/페이지)
- 각 항목 클릭 시 해당 도메인 검색 페이지로 이동

### US-03: 북마크 삭제
> 사용자로서, 더 이상 필요 없는 자료를 나의 자료에서 제거하고 싶다.

**Acceptance Criteria**:
- 개별 삭제 버튼으로 단건 삭제
- 전체 삭제 버튼으로 일괄 삭제 (확인 대화상자)
- 검색 결과 화면에서도 토글로 해제 가능

---

## 6. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| source_file 경로 변경 시 북마크 깨짐 | Low | Medium | source_file은 Pinecone 업로드 시 고정되므로 변경 가능성 낮음 |
| 사용자당 과도한 북마크 | Low | Low | MAX_PER_USER = 200 제한 |
| 검색 결과 렌더링 성능 저하 | Medium | Low | check-batch API로 한 번에 여러 문서 북마크 상태 확인 |

---

## 7. Implementation Priority

1. **P0**: UserBookmark 모델 + 마이그레이션
2. **P0**: 북마크 CRUD API (`api/v1/bookmarks.py`)
3. **P0**: 나의 자료 페이지 (`/my-documents` + 템플릿)
4. **P1**: 검색 결과 카드 북마크 버튼 (domain.html 수정)
5. **P1**: 네비게이션 연동 (base.html)
6. **P2**: check-batch API 최적화

---

## 8. Success Metrics

- 로그인 사용자 중 북마크 기능 사용률 > 30%
- 북마크한 사용자의 재방문율 기존 대비 > 20% 향상
- 나의 자료 페이지 이탈률 < 40%
