# Admin 페이지 디자인 통합 리뉴얼 Design

> Plan 참조: `docs/01-plan/features/admin-redesign.plan.md`

## 1. 변경 범위

| 파일 | 작업 | 변경 유형 |
|------|------|-----------|
| `templates/admin.html` | base.html 상속 구조 전환 + CSS 변수화 | 대규모 수정 |
| `web_app.py` (L430) | admin 라우트 확인 (변경 최소) | 확인만 |

> `api/v1/admin.py`는 변경 없음 (40+ 엔드포인트 그대로 유지)

---

## 2. HTML 구조 변환

### 2.1 현재 구조 (독립 HTML)

```html
<!DOCTYPE html>
<html lang="ko">
<head>
  <style>/* 454줄 자체 CSS */</style>
</head>
<body>
  <nav class="top-nav">← Home | Admin Dashboard | 사용자명</nav>
  <div class="admin-layout">
    <nav class="sidebar">7개 탭</nav>
    <div class="main-content">7개 section</div>
  </div>
  <script>/* ~1000줄 JS */</script>
</body>
</html>
```

### 2.2 변경 후 구조 (base.html 상속)

```html
{% extends "base.html" %}
{% block title %}Admin Dashboard - SafeFactory{% endblock %}

{% block head_extra %}
    /* admin 전용 CSS (CSS 변수 사용) */
{% endblock %}

{% block content %}
<div class="admin-wrapper">
  <div class="admin-layout">
    <nav class="admin-sidebar">7개 탭</nav>
    <div class="admin-main">7개 section</div>
  </div>
</div>
{% endblock %}

{% block scripts %}
<script>/* 기존 JS 그대로 */</script>
{% endblock %}
```

### 2.3 제거 대상

| 제거 항목 | 이유 |
|-----------|------|
| `<!DOCTYPE html>`, `<html>`, `<head>`, `<body>` | base.html에서 제공 |
| `<nav class="top-nav">` | base.html 공통 nav 사용 |
| 자체 font-family 선언 | base.html에서 Noto Sans KR 로드 |
| `<style>` 태그 자체 | `{% block head_extra %}` 블록으로 이동 |
| `<script>` 태그 자체 | `{% block scripts %}` 블록으로 이동 |

---

## 3. CSS 변수 매핑

### 3.1 색상 변환 테이블

| 하드코딩 값 | CSS 변수 | 용도 |
|-------------|----------|------|
| `#f5f7fb` | `var(--sf-bg)` | 페이지 배경 (base.html에서 적용) |
| `#ffffff` | `var(--sf-card-bg)` | 카드/사이드바 배경 |
| `#e5e7eb` | `var(--sf-border)` | 테두리 |
| `#f3f4f6` | `var(--sf-border-light)` | 테이블 행 구분선 |
| `#7c3aed` | `var(--sf-purple)` | 브랜드/액센트 색상 |
| `rgba(124,58,237,0.06)` | `rgba(30,41,59,0.04)` | 사이드바 active 배경 (네이비 계열로 통일) |
| `rgba(124,58,237,0.1)` | `rgba(30,41,59,0.06)` | 버튼 배경 |
| `#1f2937` | `var(--sf-text-1)` | 기본 텍스트 |
| `#4b5563` | `var(--sf-text-2)` | 보조 텍스트 |
| `#6b7280` | `var(--sf-text-2)` | 사이드바/라벨 텍스트 |
| `#9ca3af` | `var(--sf-text-3)` | 힌트 텍스트 |
| `rgba(0,0,0,0.03)` | `var(--sf-surface, rgba(0,0,0,0.03))` | 테이블 헤더/호버 |

### 3.2 인라인 스타일 변환

admin.html 설정 섹션에 `style="..."` 인라인 스타일이 다수 존재. 주요 패턴:

```css
/* 변환 전 (인라인) */
style="color:#7c3aed;margin-bottom:16px;"
style="padding:8px;background:#ffffff;color:#1f2937;border:1px solid #e5e7eb;border-radius:6px;"

/* 변환 후 (클래스) */
.admin-section-title { color: var(--sf-purple); margin-bottom: 16px; }
.admin-input {
    padding: 8px; background: var(--sf-card-bg); color: var(--sf-text-1);
    border: 1px solid var(--sf-border); border-radius: 6px;
}
```

인라인 스타일 중 **색상값만** CSS 변수로 전환. 레이아웃 관련 인라인(flex, gap, margin)은 유지.

---

## 4. 레이아웃 설계

### 4.1 컨테이너 구조

```
base.html nav (공통)
├─ .admin-wrapper (max-width: 960px, margin: 0 auto)
│   └─ .admin-layout (display: flex)
│       ├─ .admin-sidebar (width: 200px, flex-shrink: 0)
│       └─ .admin-main (flex: 1, overflow-y: auto)
│           ├─ #section-dashboard
│           ├─ #section-documents
│           ├─ #section-posts
│           ├─ #section-users
│           ├─ #section-stats
│           ├─ #section-logs
│           └─ #section-settings
base.html footer (공통, 자동 표시)
```

### 4.2 max-width 결정

기존 프로젝트 컨벤션 960px에 사이드바(200px)를 포함하면 메인 영역이 760px로 충분.
테이블이 넓은 경우 `.table-wrapper { overflow-x: auto; }` 기존 패턴으로 대응.

### 4.3 CSS 클래스 접두사

base.html CSS와의 충돌 방지를 위해 admin 전용 클래스에 `admin-` 접두사 사용:

| 현재 클래스 | 변경 후 |
|-------------|---------|
| `.sidebar` | `.admin-sidebar` |
| `.sidebar-item` | `.admin-sidebar-item` |
| `.main-content` | `.admin-main` |
| `.section` | `.admin-section` |
| `.section-title` | `.admin-section-title` |
| `.stat-cards` | `.admin-stat-cards` |
| `.stat-card` | `.admin-stat-card` |
| `.stat-value` | `.admin-stat-value` |
| `.stat-label` | `.admin-stat-label` |
| `.toolbar` | `.admin-toolbar` |
| `.btn` | `.admin-btn` |
| `.btn-primary` | `.admin-btn-primary` |
| `.btn-danger` | `.admin-btn-danger` |
| `.card` | `.admin-card` |

> `table`, `th`, `td`, `.table-wrapper` 등은 admin 섹션 내부에서만 사용되므로
> `.admin-main table` 셀렉터로 스코핑하여 충돌 방지.

---

## 5. 반응형 설계

### 5.1 Breakpoints

```css
/* 768px: sidebar → 상단 탭 (기존 유지) */
@media (max-width: 768px) {
    .admin-layout { flex-direction: column; }
    .admin-sidebar {
        width: 100%; display: flex; overflow-x: auto;
        border-right: none; border-bottom: 1px solid var(--sf-border);
    }
    .admin-sidebar-item { border-left: none; border-bottom: 3px solid transparent; }
    .admin-main { padding: 16px; }
    .admin-stat-cards { grid-template-columns: repeat(2, 1fr); }
}

/* 480px: 신규 추가 */
@media (max-width: 480px) {
    .admin-main { padding: 12px; }
    .admin-stat-cards { grid-template-columns: 1fr; }
    .admin-toolbar { flex-direction: column; }
    .admin-toolbar input, .admin-toolbar select { width: 100%; }
}
```

---

## 6. JS 변경 사항

### 6.1 변경 필요 항목

| 항목 | 이유 | 변경 |
|------|------|------|
| `.sidebar-item` 셀렉터 | 클래스명 변경 | `.admin-sidebar-item`으로 변경 |
| `.section` 셀렉터 | 클래스명 변경 | `.admin-section`으로 변경 |
| 나머지 JS | 변경 없음 | API 호출, DOM 조작 등 그대로 |

### 6.2 변경 불필요 항목

- `apiFetch()` 함수 — API 경로 변경 없음
- 각 섹션 로더 (loadDashboard, loadDocuments 등) — id 기반 DOM 조작이라 영향 없음
- `switchPostTab()` — 내부 id 기반이라 영향 없음
- 설정 저장 로직 — API 호출 기반이라 영향 없음

---

## 7. 구현 체크리스트

1. [ ] `feature/admin-redesign` 브랜치 생성
2. [ ] admin.html → `{% extends "base.html" %}` 구조 전환
3. [ ] 자체 `<nav class="top-nav">` 제거
4. [ ] CSS 클래스 `admin-` 접두사 추가 (충돌 방지)
5. [ ] 하드코딩 색상 → `var(--sf-*)` CSS 변수 전환
6. [ ] 인라인 스타일의 색상값 → CSS 변수 전환
7. [ ] `.admin-main table` 스코핑으로 테이블 스타일 격리
8. [ ] JS 내 셀렉터 `.sidebar-item` → `.admin-sidebar-item` 변경
9. [ ] JS 내 셀렉터 `.section` → `.admin-section` 변경
10. [ ] 768px 반응형 CSS 변수 적용
11. [ ] 480px breakpoint 추가
12. [ ] 전체 7개 섹션 기능 동작 확인
13. [ ] PR 생성

---

## 8. 영향 분석

| 항목 | 영향 |
|------|------|
| 공통 nav | 자동 표시 (base.html) — admin 링크 이미 존재 |
| 공통 footer | 자동 표시 (청년노동자인권센터 정보) |
| CSRF | base.html에 csrf_token 이미 포함 — admin API는 v1_bp(CSRF exempt) |
| 모바일 메뉴 | base.html mobile menu JS와 admin sidebar JS — 별도 셀렉터로 충돌 없음 |
| 로그인 체크 | web_app.py `admin_page()`에서 이미 처리 — 변경 없음 |
