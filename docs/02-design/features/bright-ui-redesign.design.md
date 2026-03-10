# bright-ui-redesign Design Document

> **Summary**: 청소년 대상 밝은 라이트 테마 UI 전면 재설계 — CSS 변수 전환 + 하드코딩 색상 치환
>
> **Project**: SafeFactory
> **Author**: Claude
> **Date**: 2026-03-10
> **Status**: Draft
> **Planning Doc**: [bright-ui-redesign.plan.md](../../01-plan/features/bright-ui-redesign.plan.md)

---

## 1. Overview

### 1.1 Design Goals

1. 다크 테마(#08090d) → 밝고 깨끗한 라이트 테마(#f8fafc~#ffffff) 전환
2. 청소년(직업계고 학생) 대상 — 친근하고 활기찬 톤, 읽기 쉬운 명암비
3. CSS 변수 시스템 활용하여 체계적 일괄 전환
4. 도메인별 색상 아이덴티티 유지 (밝은 톤으로 재해석)
5. 기존 레이아웃/반응형/기능 100% 유지

### 1.2 Design Principles

- **청소년 친화적**: 밝고 따뜻한 색상, 부드러운 곡선, 충분한 여백
- **가독성 우선**: 밝은 배경에 어두운 텍스트로 장시간 학습에 적합
- **일관성**: 모든 페이지에서 통일된 라이트 톤 유지
- **최소 변경**: CSS 값만 변경, HTML 구조/JS 로직 불변

---

## 2. Architecture

### 2.1 변경 영향 범위

```
변경 대상 (CSS only):
┌─────────────────────────────────────────────────────────┐
│  templates/base.html         ← :root 변수 + 네비게이션  │
│  templates/home.html         ← 히어로 + 카드 + 하단    │
│  templates/domain.html       ← 검색/결과/답변/차트      │
│  templates/community.html    ← 커뮤니티 패널/뱃지       │
│  templates/login.html        ← 로그인 폼              │
│  templates/admin.html        ← 관리자 사이드바/패널     │
│  services/domain_config.py   ← 도메인 색상 값          │
│  static/js/common.js         ← Chart.js 색상 팔레트    │
└─────────────────────────────────────────────────────────┘

변경 없음:
 - HTML 구조, Jinja2 로직
 - JavaScript 기능 코드
 - Python 백엔드
 - 반응형 브레이크포인트 (900px, 600px)
```

### 2.2 의존 관계

| Component | Depends On | Purpose |
|-----------|-----------|---------|
| home.html, domain.html, community.html | base.html `:root` 변수 | 전역 색상 변수 참조 |
| domain.html | domain_config.py 색상 | Jinja2로 도메인별 색상 주입 |
| domain.html | common.js Chart 팔레트 | 차트 색상 |

---

## 3. CSS Design Token Specification

### 3.1 `:root` 변수 전환 (base.html)

| Variable | Before (Dark) | After (Light) | Notes |
|----------|---------------|---------------|-------|
| `--sf-bg` | `#08090d` | `#f5f7fb` | 메인 배경 — 매우 연한 블루그레이 |
| `--sf-card-bg` | `#0f1017` | `#ffffff` | 카드 배경 — 순백 |
| `--sf-nav-bg` | `#0c0d12` | `#ffffff` | 네비게이션 — 흰색 |
| `--sf-border` | `#1a1b25` | `#e5e7eb` | 기본 테두리 — gray-200 |
| `--sf-border-light` | `#1f2029` | `#f3f4f6` | 연한 테두리 — gray-100 |
| `--sf-purple` | `#7c3aed` | `#7c3aed` | 브랜드 퍼플 — **유지** |
| `--sf-purple-light` | `#a78bfa` | `#a78bfa` | 라이트 퍼플 — **유지** |
| `--sf-purple-lighter` | `#c4b5fd` | `#ede9fe` | 배경용 퍼플 — 더 연하게 |
| `--sf-purple-dark` | `#6d28d9` | `#6d28d9` | 다크 퍼플 — **유지** |
| `--sf-text-1` | `#e4e4e7` | `#1f2937` | 주 텍스트 — gray-800 |
| `--sf-text-2` | `#a1a1aa` | `#4b5563` | 보조 텍스트 — gray-600 |
| `--sf-text-3` | `#71717a` | `#9ca3af` | 3차 텍스트 — gray-400 |
| `--sf-text-4` | `#52525b` | `#d1d5db` | 4차 텍스트 — gray-300 |
| `--sf-blue` | `#3b82f6` | `#3b82f6` | 유지 |
| `--sf-amber` | `#f59e0b` | `#f59e0b` | 유지 |
| `--sf-red` | `#ef4444` | `#ef4444` | 유지 |
| `--sf-violet` | `#8b5cf6` | `#8b5cf6` | 유지 |
| `--sf-green` | `#10b981` | `#10b981` | 유지 |
| `--sf-cyan` | `#06b6d4` | `#06b6d4` | 유지 |
| `--sf-community` | `#7c4dff` | `#7c4dff` | 유지 |

### 3.2 도메인 색상 (domain_config.py)

라이트 배경에서 더 선명하게 보이도록 채도 조정:

| Domain | color (Before) | color (After) | color_rgb (After) |
|--------|---------------|---------------|-------------------|
| semiconductor | `#00d4ff` | `#0891b2` | `8, 145, 178` |
| laborlaw | `#ff9800` | `#d97706` | `217, 119, 6` |
| field-training | `#e91e63` | `#db2777` | `219, 39, 119` |
| safeguide | `#2196f3` | `#2563eb` | `37, 99, 235` |
| msds | `#4caf50` | `#059669` | `5, 150, 105` |

gradient_from/gradient_to도 동일 톤으로 변경.

---

## 4. Component-Level Design Specification

### 4.1 base.html — 네비게이션 바

**Before**:
```css
.sf-nav {
    background: rgba(8, 9, 13, 0.85);
    backdrop-filter: blur(20px);
    border-bottom: 1px solid var(--sf-border);
}
```

**After**:
```css
.sf-nav {
    background: rgba(255, 255, 255, 0.9);
    backdrop-filter: blur(20px);
    border-bottom: 1px solid var(--sf-border);
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
}
```

**변경 항목**:
- `.sf-nav` — rgba 밝게 + 그림자 추가
- `.sf-nav-logo-text` — `var(--sf-purple)` (어두운 배경에서 돋보이게)
- `.sf-nav-tabs` — `background: var(--sf-card-bg)` (변수가 #fff로 바뀌므로 자동 적용)
- `.sf-nav-tab:hover` — `rgba(124, 58, 237, 0.06)` (연한 퍼플 배경)
- `.sf-nav-bell` — 변수 참조로 자동 적용
- `.sf-nav-link:hover` — `rgba(0, 0, 0, 0.04)` (밝은 배경용)
- `.sf-nav-logout-btn:hover` — `rgba(0, 0, 0, 0.04)`
- `.sf-nav-admin` — `rgba(124, 58, 237, 0.08)` 배경
- **모바일 메뉴**: `rgba(255, 255, 255, 0.97)` 배경

### 4.2 home.html — 히어로 섹션

**Before**:
```css
.hero::before {
    background: radial-gradient(ellipse, rgba(124,58,237,0.12) 0%, transparent 70%);
}
.hero-badge {
    background: rgba(167,139,250,0.1);
    border: 1px solid rgba(167,139,250,0.2);
    color: var(--sf-purple-lighter);
}
.hero-title {
    background: linear-gradient(135deg, var(--sf-purple-lighter), var(--sf-purple-light), var(--sf-purple));
}
```

**After**:
```css
.hero::before {
    background: radial-gradient(ellipse, rgba(124,58,237,0.06) 0%, transparent 70%);
}
.hero-badge {
    background: rgba(124,58,237,0.06);
    border: 1px solid rgba(124,58,237,0.15);
    color: var(--sf-purple);
}
.hero-title {
    background: linear-gradient(135deg, var(--sf-purple), var(--sf-purple-dark), #4c1d95);
    /* 밝은 배경에서는 진한 퍼플 그라데이션 */
}
.hero-subtitle {
    color: var(--sf-text-2);  /* was var(--sf-text-3) */
}
.hero-subtitle strong {
    color: var(--sf-text-1);  /* was var(--sf-text-2) */
}
```

### 4.3 home.html — 검색바

**Before**:
```css
.unified-search {
    background: var(--sf-card-bg);     /* #0f1017 */
    border: 2px solid var(--sf-border-light);
}
.unified-search:focus-within {
    box-shadow: 0 0 30px rgba(124,58,237,0.15);
}
.search-hint {
    background: rgba(255,255,255,0.03);
}
```

**After**:
```css
.unified-search {
    background: var(--sf-card-bg);     /* #ffffff (변수 변경으로 자동) */
    border: 2px solid var(--sf-border);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}
.unified-search:focus-within {
    border-color: var(--sf-purple);
    box-shadow: 0 0 0 3px rgba(124,58,237,0.1), 0 2px 8px rgba(0,0,0,0.04);
}
.search-domain-select {
    background: rgba(124,58,237,0.06);
    border: 1px solid rgba(124,58,237,0.15);
    color: var(--sf-purple);
}
.search-hint {
    background: rgba(0, 0, 0, 0.02);
    border: 1px solid var(--sf-border);
    color: var(--sf-text-2);
}
.search-hint:hover {
    border-color: rgba(124,58,237,0.3);
    color: var(--sf-purple);
    background: rgba(124,58,237,0.04);
}
```

### 4.4 home.html — 도메인 카드

**Before**:
```css
.domain-card {
    background: var(--sf-card-bg);
    border: 1px solid var(--sf-border);
}
.domain-card:hover {
    box-shadow: 0 12px 40px rgba(0,0,0,0.4);
}
.card-action {
    border-top: 1px solid var(--sf-border);
}
```

**After**:
```css
.domain-card {
    background: var(--sf-card-bg);  /* #ffffff */
    border: 1px solid var(--sf-border);
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}
.domain-card:hover {
    border-color: transparent;
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.08);
}
/* 도메인별 hover shadow도 opacity 조정 */
.dc-semi:hover { box-shadow: 0 12px 40px rgba(59,130,246,0.12); }
.dc-labor:hover { box-shadow: 0 12px 40px rgba(245,158,11,0.12); }
.dc-field:hover { box-shadow: 0 12px 40px rgba(239,68,68,0.12); }
.dc-safe:hover { box-shadow: 0 12px 40px rgba(139,92,246,0.12); }
.dc-msds:hover { box-shadow: 0 12px 40px rgba(16,185,129,0.12); }
.dc-comm:hover { box-shadow: 0 12px 40px rgba(6,182,212,0.12); }
```

**도메인 카드 타이틀 색상** — 라이트 배경에서 더 진한 톤:
```css
.dc-semi .card-title { color: #2563eb; }  /* was #60a5fa */
.dc-labor .card-title { color: #d97706; }  /* was #fbbf24 */
.dc-field .card-title { color: #db2777; }  /* was #f87171 */
.dc-safe .card-title { color: #7c3aed; }   /* was var(--sf-purple-lighter) */
.dc-msds .card-title { color: #059669; }   /* was #34d399 */
.dc-comm .card-title { color: #0891b2; }   /* was #22d3ee */
```

### 4.5 home.html — 하단 카드 / 커뮤니티·뉴스 프리뷰

**변경 항목**:
- `.bottom-card` — 변수로 자동 적용 + `box-shadow: 0 1px 3px rgba(0,0,0,0.04)`
- `.comm-text` — `#d4d4d8` → `var(--sf-text-1)` (어두운 텍스트)
- `.news-text` — `#d4d4d8` → `var(--sf-text-1)`
- `.comm-badge` 색상 — 밝은 배경용 뱃지 톤 조정:
  ```css
  .cb-q { background: rgba(34,197,94,0.08); color: #15803d; }
  .cb-i { background: rgba(234,179,8,0.08); color: #a16207; }
  .cb-f { background: rgba(59,130,246,0.08); color: #2563eb; }
  .cb-n { background: rgba(239,68,68,0.08); color: #dc2626; }
  ```
- `.news-cat` 뉴스 뱃지도 동일 패턴:
  ```css
  .nc-accident { background: rgba(239,68,68,0.08); color: #dc2626; }
  .nc-regulation { background: rgba(234,179,8,0.08); color: #a16207; }
  .nc-policy { background: rgba(59,130,246,0.08); color: #2563eb; }
  .nc-technology { background: rgba(34,197,94,0.08); color: #15803d; }
  .nc-general { background: rgba(124,58,237,0.08); color: #7c3aed; }
  ```

### 4.6 home.html — 푸터

```css
.sf-footer-inner {
    border-top: 1px solid var(--sf-border);  /* 자동 */
}
.sf-footer-links a {
    color: var(--sf-text-3);  /* was var(--sf-text-4) */
}
.sf-footer-copy {
    color: var(--sf-text-3);  /* was #3f3f46 */
}
```

### 4.7 domain.html — 본문 영역

**Body 배경**:
```css
/* Before */
body { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); }

/* After */
body { background: var(--sf-bg); }  /* #f5f7fb */
```

**입력 필드**:
```css
/* Before */
input[type="text"], select, textarea {
    border: 2px solid rgba(255, 255, 255, 0.1);
    background: rgba(0, 0, 0, 0.3);
    color: #fff;
}

/* After */
input[type="text"], select, textarea {
    border: 2px solid var(--sf-border);
    background: var(--sf-card-bg);
    color: var(--sf-text-1);
}
```

**헤더**:
```css
header h1 { color: var(--primary-color); }  /* 변경 없음, domain_config 값이 바뀜 */
header p { color: var(--sf-text-2); }        /* was #888 */
```

**AI 답변 영역**:
```css
/* Before */
.ai-answer { background: rgba(156, 39, 176, 0.1); border: 1px solid rgba(156, 39, 176, 0.3); }

/* After */
.ai-answer { background: rgba(124, 58, 237, 0.04); border: 1px solid rgba(124, 58, 237, 0.15); }
```

**검색 결과 카드**:
```css
/* Before: rgba(255,255,255,0.05) 배경 */
/* After: var(--sf-card-bg) + box-shadow */
.result-card {
    background: var(--sf-card-bg);
    border: 1px solid var(--sf-border);
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}
```

**차트 영역**:
```css
/* Before */
.chart-container { background: rgba(0, 0, 0, 0.3); border: 1px solid rgba(255,255,255,0.1); }

/* After */
.chart-container { background: var(--sf-card-bg); border: 1px solid var(--sf-border); }
```

### 4.8 community.html

```css
/* Before */
body { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: #e0e0e0; }
.main-panel { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); }
.page-header h1 { color: #7c4dff; }
.tab-btn { border: 1px solid rgba(255,255,255,0.15); background: rgba(255,255,255,0.05); color: #aaa; }

/* After */
body { background: var(--sf-bg); color: var(--sf-text-1); }
.main-panel { background: var(--sf-card-bg); border: 1px solid var(--sf-border); box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
.page-header h1 { color: var(--sf-purple); }
.tab-btn { border: 1px solid var(--sf-border); background: var(--sf-card-bg); color: var(--sf-text-2); }
```

### 4.9 login.html

```css
/* Before */
body { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: #e0e0e0; }
header h1 { color: #00d4ff; }
.login-card { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); }

/* After */
body { background: var(--sf-bg, #f5f7fb); color: var(--sf-text-1, #1f2937); }
header h1 { color: var(--sf-purple, #7c3aed); }
.login-card { background: #ffffff; border: 1px solid #e5e7eb; box-shadow: 0 4px 20px rgba(0,0,0,0.06); }
```

> Note: login.html은 base.html을 상속하지 않는 독립 페이지이므로, CSS 변수를 직접 정의하거나 하드코딩 값 사용.

### 4.10 admin.html

```css
/* Before */
body { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: #e0e0e0; }
.top-nav { background: rgba(0,0,0,0.4); border-bottom: 1px solid rgba(255,255,255,0.08); }
.top-nav h1 { color: #00d4ff; }

/* After */
body { background: #f5f7fb; color: #1f2937; }
.top-nav { background: #ffffff; border-bottom: 1px solid #e5e7eb; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
.top-nav h1 { color: #7c3aed; }
```

> Note: admin.html도 base.html을 상속하지 않는 독립 페이지.

### 4.11 common.js — Chart.js 색상

```javascript
// Before (어두운 배경에 맞는 밝은 색)
colors = [
    'rgba(0, 212, 255, 0.8)',
    'rgba(156, 39, 176, 0.8)',
    'rgba(76, 175, 80, 0.8)',
    'rgba(255, 193, 7, 0.8)',
    'rgba(244, 67, 54, 0.8)',
    'rgba(33, 150, 243, 0.8)'
]

// After (밝은 배경에서 적절한 채도)
colors = [
    'rgba(37, 99, 235, 0.75)',     // Blue
    'rgba(124, 58, 237, 0.75)',    // Purple
    'rgba(5, 150, 105, 0.75)',     // Green
    'rgba(217, 119, 6, 0.75)',     // Amber
    'rgba(220, 38, 38, 0.75)',     // Red
    'rgba(8, 145, 178, 0.75)'     // Cyan
]
```

---

## 5. 하드코딩 색상 치환 매핑

CSS 변수를 사용하지 않고 직접 하드코딩된 색상값들의 치환 맵:

| Before (Dark) | After (Light) | 사용 위치 |
|---------------|---------------|-----------|
| `rgba(8, 9, 13, 0.85)` | `rgba(255, 255, 255, 0.9)` | base.html nav 배경 |
| `rgba(8, 9, 13, 0.95)` | `rgba(255, 255, 255, 0.97)` | base.html 모바일 메뉴 |
| `rgba(167, 139, 250, 0.06)` | `rgba(124, 58, 237, 0.06)` | base.html tab hover |
| `rgba(167, 139, 250, 0.1)` | `rgba(124, 58, 237, 0.08)` | 여러 곳 (뱃지, 어드민) |
| `rgba(167, 139, 250, 0.2)` | `rgba(124, 58, 237, 0.12)` | hover states |
| `rgba(255, 255, 255, 0.05)` | 삭제 또는 `var(--sf-card-bg)` | 카드 배경 |
| `rgba(255, 255, 255, 0.1)` | `var(--sf-border)` | 테두리 |
| `rgba(255, 255, 255, 0.03)` | `rgba(0, 0, 0, 0.02)` | 미세 배경 |
| `rgba(0, 0, 0, 0.3)` | `var(--sf-card-bg)` | 입력 필드 배경 |
| `rgba(0, 0, 0, 0.4)` | `#ffffff` | 어드민 nav 배경 |
| `#d4d4d8` | `var(--sf-text-1)` | 커뮤니티/뉴스 텍스트 |
| `#888` | `var(--sf-text-2)` | 서브텍스트 |
| `#aaa` | `var(--sf-text-2)` | 라벨 텍스트 |
| `#e0e0e0` | `var(--sf-text-1)` | 본문 텍스트 |
| `#fff` (텍스트) | `var(--sf-text-1)` | 입력 필드 텍스트 |
| `#3f3f46` | `var(--sf-text-3)` | 푸터 카피라이트 |
| `linear-gradient(135deg, #1a1a2e, #16213e)` | `var(--sf-bg)` | domain/community body |
| `#00d4ff` | `var(--sf-purple)` 또는 도메인색 | login/admin 헤더 |

---

## 6. Implementation Order

### 6.1 Phase 1: 전역 변수 전환 (base.html)

```
1. [ ] :root CSS 변수 12개 값 변경
2. [ ] .sf-nav 배경 rgba 변경 + box-shadow 추가
3. [ ] .sf-nav-logo-text 색상 변경
4. [ ] .sf-nav-tab:hover 배경 rgba 변경
5. [ ] .sf-nav-link:hover, .sf-nav-logout-btn:hover rgba 변경
6. [ ] .sf-nav-admin 배경 rgba 변경
7. [ ] .sf-mobile-menu 배경 rgba 변경
8. [ ] .sf-mobile-menu-links a:hover 배경 rgba 변경
```

### 6.2 Phase 2: 홈페이지 (home.html)

```
1. [ ] .hero::before 그라데이션 opacity 조정
2. [ ] .hero-badge 색상 변경
3. [ ] .hero-title 그라데이션 진한 퍼플로 변경
4. [ ] .hero-subtitle, .hero-subtitle strong 텍스트 색상
5. [ ] .unified-search 그림자 추가, focus 스타일 변경
6. [ ] .search-domain-select 색상
7. [ ] .search-hint 배경/테두리/텍스트 색상
8. [ ] .domain-card box-shadow 추가, hover shadow opacity
9. [ ] .dc-* .card-title 색상 진하게
10. [ ] .dc-* .card-icon-wrap 배경 opacity 조정
11. [ ] .comm-text, .news-text 색상 변경
12. [ ] .cb-*, .nc-* 뱃지 색상 라이트 버전
13. [ ] .sf-footer 텍스트 색상
```

### 6.3 Phase 3: 도메인 페이지 (domain.html)

```
1. [ ] body background 변경
2. [ ] input/select/textarea 배경, 테두리, 색상
3. [ ] header p 색상
4. [ ] .ai-answer 영역 배경/테두리
5. [ ] 결과 카드 배경/테두리
6. [ ] .chart-container 배경/테두리
7. [ ] 기타 하드코딩 rgba 값 치환
```

### 6.4 Phase 4: 도메인 설정 (domain_config.py)

```
1. [ ] semiconductor: color, color_rgb, gradient_from, gradient_to
2. [ ] laborlaw: 동일
3. [ ] field-training: 동일
4. [ ] safeguide: 동일
5. [ ] msds: 동일
```

### 6.5 Phase 5: 부가 페이지 (community, login, admin)

```
1. [ ] community.html: body, main-panel, page-header, tab-btn, 뱃지
2. [ ] login.html: body, header h1, login-card (독립 페이지)
3. [ ] admin.html: body, top-nav, sidebar, 패널 (독립 페이지)
```

### 6.6 Phase 6: Chart.js (common.js)

```
1. [ ] renderChart() 색상 배열 교체
```

---

## 7. 청소년 대상 UX 고려사항

| 항목 | 적용 방안 |
|------|-----------|
| **가독성** | 밝은 배경 + 어두운 텍스트 (#1f2937 on #f5f7fb) — 명암비 12:1 이상 |
| **친근한 톤** | 순백(#fff) 대신 약간 따뜻한 블루그레이(#f5f7fb) 배경으로 눈 피로 감소 |
| **도메인 카드 색상** | 카드 타이틀에 진한 도메인색 사용하여 시각적 구분 명확 |
| **호버 피드백** | 그림자 강화 방식으로 인터랙티브 요소 강조 |
| **뱃지 색상** | 밝은 배경에서도 읽기 쉽도록 진한 텍스트 + 연한 배경 조합 |
| **이모지** | 기존 이모지 아이콘 유지 — 청소년에게 친근 |
| **보라색 브랜드** | 메인 퍼플(#7c3aed) 유지 — 트렌디하고 세련된 느낌 |

---

## 8. Test Plan

### 8.1 검증 체크리스트

| 항목 | 검증 방법 |
|------|-----------|
| 모든 페이지 밝은 배경 적용 | 브라우저에서 6개 페이지 확인 |
| 텍스트 가독성 | WCAG AA 명암비 체크 |
| 도메인별 색상 구분 | 5개 도메인 카드 시각 확인 |
| 호버/포커스 피드백 | 인터랙티브 요소 클릭/호버 테스트 |
| 반응형 레이아웃 | 900px, 600px 브레이크포인트 확인 |
| 모바일 메뉴 | 밝은 오버레이 + 메뉴 동작 확인 |
| Chart.js 차트 | 도메인 페이지에서 차트 색상 확인 |
| 다크 잔재 없음 | 어두운 배경/밝은 텍스트 잔재 검색 |

### 8.2 Gap Analysis 대상

- base.html: 12개 변수 + 8개 하드코딩 색상
- home.html: ~15개 색상 변경점
- domain.html: ~12개 색상 변경점
- domain_config.py: 5개 도메인 x 4개 색상값
- community.html: ~6개 색상 변경점
- login.html: ~4개 색상 변경점
- admin.html: ~5개 색상 변경점
- common.js: Chart.js 팔레트 1곳

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-03-10 | Initial draft | Claude |
