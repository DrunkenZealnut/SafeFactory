# 기능안내 UI 간소화 Design Document

> **Feature**: 기능안내 UI 간소화
> **Plan Reference**: `docs/01-plan/features/기능안내ui간소화.plan.md`
> **Date**: 2026-03-11
> **Status**: Draft

---

## 1. Design Overview

홈페이지의 도메인 카드 섹션을 **컴팩트 칩 형태**로 전환하고, Hero 영역을 축약하여 검색 중심 UI로 간소화한다.

**변경 요약:**
- Hero: 타이틀 1줄, 서브타이틀 1줄로 축약, 패딩 축소
- 도메인 카드: 6개 대형 카드(3열 그리드) → 컴팩트 칩(flex-wrap, 1-2줄)
- 검색 힌트: 4개 → 3개
- 하단 프리뷰: **유지** (커뮤니티/뉴스 프리뷰 + 링크프리뷰 기능 보존)
- 푸터: 유지

**유지 항목:**
- `partials/link_preview.html` 컴포넌트 (커뮤니티 링크 미리보기)
- 하단 커뮤니티 최신글/뉴스 프리뷰 영역 (bottom-row)
- 검색바 및 도메인 셀렉터
- 모든 라우팅/링크

---

## 2. UI Structure (Before → After)

### 2.1 Before (현재)

```
┌─────────────────────────────────────────┐
│ [Hero]                                  │
│   배지: "AI 실시간 검색"                  │
│   타이틀: 2줄 (42px)                     │
│   서브타이틀: 2줄 (16px)                  │
│   검색바 + 도메인셀렉터                    │
│   힌트칩 4개                             │
│                      padding: 56px 상하   │
├─────────────────────────────────────────┤
│ [도메인 카드 섹션]  "📚 전문 분야"         │
│ ┌──────┐ ┌──────┐ ┌──────┐              │
│ │ 🔬   │ │ ⚖️   │ │ 🏭   │  3열 그리드   │
│ │반도체 │ │노동법 │ │현장실습│  각 카드:    │
│ │ desc │ │ desc │ │ desc │  ~180px 높이  │
│ │ CTA  │ │ CTA  │ │ CTA  │              │
│ └──────┘ └──────┘ └──────┘              │
│ ┌──────┐ ┌──────┐ ┌──────┐              │
│ │ 🛡️   │ │ 🧪   │ │ 💬   │              │
│ │안전보건│ │MSDS  │ │커뮤니티│              │
│ └──────┘ └──────┘ └──────┘              │
├─────────────────────────────────────────┤
│ [하단 프리뷰]  2열 그리드                  │
│ 커뮤니티 최신글 | 뉴스 프리뷰              │
├─────────────────────────────────────────┤
│ [푸터]                                   │
└─────────────────────────────────────────┘
```

### 2.2 After (목표)

```
┌─────────────────────────────────────────┐
│ [Hero - 축약]                            │
│   배지: "AI 실시간 검색"                  │
│   타이틀: 1줄 (36px) "AI 산업안전 지식검색" │
│   서브타이틀: 1줄 (14px)                  │
│   검색바 + 도메인셀렉터                    │
│   힌트칩 3개                             │
│                      padding: 36px 상하   │
├─────────────────────────────────────────┤
│ [도메인 칩 섹션]  flex-wrap 한줄(~두줄)     │
│ 🔬반도체 ⚖️노동법 🏭현장실습              │
│ 🛡️안전보건 🧪MSDS 💬커뮤니티              │
│                                          │
├─────────────────────────────────────────┤
│ [하단 프리뷰]  2열 그리드 (유지)            │
│ 커뮤니티 최신글 | 뉴스 프리뷰              │
├─────────────────────────────────────────┤
│ [푸터]                                   │
└─────────────────────────────────────────┘
```

---

## 3. Component Design

### 3.1 Hero 영역 변경

| 속성 | Before | After |
|------|--------|-------|
| 패딩 | `56px 24px 48px` | `36px 24px 24px` |
| 타이틀 | 2줄, 42px `"무엇이든 물어보세요\nAI가 전문 문서에서 찾아드립니다"` | 1줄, 36px `"AI 산업안전 지식검색"` |
| 서브타이틀 | 2줄, 16px | 1줄, 14px `"반도체·노동법·산업안전 전문 문서 AI 검색"` |
| 힌트칩 | 4개 | 3개 (CVD/PVD, 주휴수당, 아세톤 MSDS) |
| 배지 | 유지 | 유지 |
| 검색바 | 유지 | 유지 |

### 3.2 도메인 칩 섹션 (NEW)

기존 6개 대형 카드를 **컴팩트 칩**으로 교체:

```html
<section class="domains-section">
    <div class="domain-chips">
        <a href="/semiconductor" class="domain-chip dc-semi">🔬 반도체</a>
        <a href="/laborlaw" class="domain-chip dc-labor">⚖️ 노동법</a>
        <a href="/field-training" class="domain-chip dc-field">🏭 현장실습</a>
        <a href="/safeguide" class="domain-chip dc-safe">🛡️ 안전보건</a>
        <a href="/msds" class="domain-chip dc-msds">🧪 MSDS</a>
        <a href="/community" class="domain-chip dc-comm">💬 커뮤니티</a>
    </div>
</section>
```

**칩 스타일:**

```css
.domain-chips {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 10px;
    max-width: 720px;
    margin: 0 auto;
    padding: 0 24px 32px;
}

.domain-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 10px 20px;
    border-radius: 12px;
    font-size: 14px;
    font-weight: 600;
    text-decoration: none;
    border: 1px solid var(--sf-border);
    background: var(--sf-card-bg);
    transition: all 0.2s;
    white-space: nowrap;
}

.domain-chip:hover {
    transform: translateY(-2px);
    box-shadow: var(--sf-shadow-md);
}
```

**도메인별 색상 (hover 시 적용):**

| 칩 | hover border-color | hover color |
|----|-------------------|-------------|
| dc-semi | `rgba(59,130,246,0.4)` | `#2563eb` |
| dc-labor | `rgba(245,158,11,0.4)` | `#d97706` |
| dc-field | `rgba(239,68,68,0.4)` | `#dc2626` |
| dc-safe | `rgba(139,92,246,0.4)` | `#7c3aed` |
| dc-msds | `rgba(16,185,129,0.4)` | `#059669` |
| dc-comm | `rgba(6,182,212,0.4)` | `#0891b2` |

### 3.3 하단 프리뷰 섹션 (유지)

- **변경 없음**: 커뮤니티 최신글 + 뉴스 프리뷰 2열 그리드 그대로 유지
- API 호출 (`/api/v1/community/posts`, `/api/v1/news/articles`) 유지
- `loadCommunityPreview()`, `loadNewsPreview()` JS 함수 유지
- `partials/link_preview.html` 컴포넌트 영향 없음

### 3.4 삭제 대상 CSS/HTML

| 삭제 | 이유 |
|------|------|
| `.domain-grid` CSS | 칩으로 대체 |
| `.domain-card` 관련 CSS 전체 | 칩으로 대체 |
| `.domain-card-bar`, `.domain-card-body` | 칩으로 대체 |
| `.card-top`, `.card-icon-wrap`, `.card-title`, `.card-desc`, `.card-action` | 칩으로 대체 |
| `.dc-*` 카드 색상 variants (hover shadow, icon-wrap bg 등) | 칩 색상으로 대체 |
| `.section-title` | 칩 섹션에서 제목 불필요 |
| 6개 `<a class="domain-card">` HTML 블록 | 6개 `<a class="domain-chip">` 로 대체 |
| 검색 힌트 4번째 (`현장실습 장비 안전수칙`) | 3개로 축소 |

### 3.5 반응형 변경

| Breakpoint | Before | After |
|------------|--------|-------|
| > 900px | 카드 3열 | 칩 flex-wrap (1줄 가능) |
| 600-900px | 카드 2열, 타이틀 30px | 칩 flex-wrap, 타이틀 28px |
| < 600px | 카드 1열, 타이틀 24px | 칩 flex-wrap (2-3줄), 타이틀 22px |

```css
@media (max-width: 900px) {
    .hero-title { font-size: 28px; }
}
@media (max-width: 600px) {
    .hero { padding: 24px 16px 16px; }
    .hero-title { font-size: 22px; }
    .hero-subtitle { font-size: 13px; }
    .domain-chip { padding: 8px 14px; font-size: 13px; }
}
```

---

## 4. Implementation Order

| # | Task | File | 난이도 |
|---|------|------|--------|
| 1 | Hero 타이틀/서브타이틀 문구 축약 + 패딩 축소 | `home.html` (CSS + HTML) | Low |
| 2 | 도메인 카드 섹션 → 칩 섹션으로 교체 (HTML) | `home.html` (HTML) | Low |
| 3 | 기존 카드 CSS 삭제 + 칩 CSS 추가 | `home.html` (CSS) | Low |
| 4 | 검색 힌트 4개 → 3개 축소 | `home.html` (HTML) | Low |
| 5 | 반응형 breakpoint 업데이트 | `home.html` (CSS) | Low |
| 6 | 테스트: 모든 도메인 링크 동작, 반응형, 하단 프리뷰 | 브라우저 | - |

**총 변경 파일: `templates/home.html` 1개**

---

## 5. 예상 화면 크기 비교

| 영역 | Before (예상 높이) | After (예상 높이) | 절감율 |
|------|-------------------|------------------|--------|
| Hero | ~280px | ~200px | -29% |
| 도메인 | ~420px (카드 2행) | ~60px (칩 1-2행) | -86% |
| 하단 프리뷰 | ~300px | ~300px (유지) | 0% |
| 푸터 | ~80px | ~80px (유지) | 0% |
| **합계** | **~1080px** | **~640px** | **-41%** |

모바일(600px)에서는 카드 6행 → 칩 2-3줄로 감소하여 **스크롤 50%+ 절감** 달성.

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-03-11 | Initial design | zealnutkim |
