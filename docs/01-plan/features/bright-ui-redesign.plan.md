# bright-ui-redesign Planning Document

> **Summary**: SafeFactory 웹 앱을 밝고 깔끔한 라이트 테마로 전면 디자인 변경
>
> **Project**: SafeFactory
> **Author**: Claude
> **Date**: 2026-03-10
> **Status**: Draft

---

## Executive Summary

| Perspective | Content |
|-------------|---------|
| **Problem** | 현재 다크 테마(#08090d 배경)는 전문적이지만, 밝고 친근한 느낌을 원하는 사용자 요구와 맞지 않음. 장시간 사용 시 눈의 피로도 증가 |
| **Solution** | CSS 변수 기반 라이트 테마 전환 — 밝은 배경(#f8fafc~#ffffff), 부드러운 그림자, 도메인별 파스텔 톤 액센트 컬러로 전면 재설계 |
| **Function/UX Effect** | 밝고 개방적인 느낌, 가독성 향상, 친근한 컬러 톤으로 접근성 개선. 도메인별 시각적 구분은 유지하되 파스텔/비비드 톤으로 전환 |
| **Core Value** | 안전 교육 플랫폼의 신뢰감 있고 밝은 이미지 구축, 사용자 체류 시간 및 만족도 향상 |

---

## 1. Overview

### 1.1 Purpose

SafeFactory 웹 앱의 전체 시각 디자인을 현재 다크 테마에서 밝고 깔끔한 라이트 테마로 변경한다. CSS 변수 시스템을 활용해 체계적으로 전환하며, 도메인별 색상 아이덴티티는 파스텔/라이트 톤으로 재해석한다.

### 1.2 Background

- 현재 디자인: 매우 어두운 배경(#08090d), 보라색 계열 액센트, 글래스모피즘 효과
- 모든 스타일이 인라인 `<style>` 블록에 정의되어 있어 외부 CSS 파일 없음
- CSS 변수(`:root` 커스텀 프로퍼티)로 색상 체계가 관리되어 변수 값만 변경하면 전역 적용 가능
- 5개 도메인(반도체, 노동법, 현장교육, 안전가이드, MSDS) + 커뮤니티 각각 고유 색상 보유

### 1.3 Related Documents

- 기존 템플릿: `templates/base.html`, `templates/home.html`, `templates/domain.html`
- 도메인 설정: `services/domain_config.py`
- 공용 JS: `static/js/common.js`

---

## 2. Scope

### 2.1 In Scope

- [x] `:root` CSS 변수 전체 라이트 테마 전환 (`base.html`)
- [x] 홈페이지 히어로 섹션, 도메인 카드 디자인 변경 (`home.html`)
- [x] 도메인 상세 페이지 색상 및 스타일 변경 (`domain.html`)
- [x] 도메인별 색상 팔레트 라이트 버전 정의 (`domain_config.py`)
- [x] 네비게이션 바 밝은 스타일 전환 (`base.html`)
- [x] 커뮤니티 페이지 스타일 변경 (`community.html`)
- [x] 로그인 페이지 스타일 변경 (`login.html`)
- [x] 관리자 페이지 스타일 변경 (`admin.html`)
- [x] Chart.js 색상 팔레트 라이트 버전 (`common.js`)

### 2.2 Out of Scope

- 다크/라이트 모드 토글 기능 (향후 고려)
- 레이아웃 구조 변경 (그리드, 반응형 브레이크포인트 유지)
- 기능적 변경 (검색, AI 답변, 커뮤니티 기능 등)
- 새로운 CSS 파일 생성 (기존 인라인 방식 유지)
- 폰트 변경 (Noto Sans KR 유지)

---

## 3. Requirements

### 3.1 Functional Requirements

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| FR-01 | `:root` CSS 변수를 라이트 테마 값으로 전환 | High | Pending |
| FR-02 | 배경색 #08090d → #f8fafc~#ffffff 계열로 변경 | High | Pending |
| FR-03 | 텍스트 색상 4단계 계층 라이트 버전 정의 | High | Pending |
| FR-04 | 네비게이션 바 밝은 스타일 (흰색 배경 + 미세 그림자) | High | Pending |
| FR-05 | 도메인 카드 밝은 배경 + 부드러운 그림자 | High | Pending |
| FR-06 | 히어로 섹션 밝은 그라데이션 배경 | Medium | Pending |
| FR-07 | 도메인별 색상 파스텔/라이트 톤 재정의 | High | Pending |
| FR-08 | 검색바 밝은 스타일 (흰색 배경, 미세 테두리) | Medium | Pending |
| FR-09 | AI 답변 영역 밝은 배경 스타일 | Medium | Pending |
| FR-10 | 커뮤니티 페이지 밝은 스타일 | Medium | Pending |
| FR-11 | 로그인 페이지 밝은 배경 | Low | Pending |
| FR-12 | 관리자 페이지 밝은 스타일 | Low | Pending |
| FR-13 | Chart.js 색상 팔레트 라이트 배경 최적화 | Low | Pending |

### 3.2 Non-Functional Requirements

| Category | Criteria | Measurement Method |
|----------|----------|-------------------|
| 가독성 | WCAG 2.1 AA 명암비 4.5:1 이상 | Chrome DevTools Contrast Checker |
| 일관성 | 모든 페이지에서 통일된 라이트 톤 유지 | 육안 검수 |
| 성능 | 스타일 변경으로 인한 렌더링 성능 저하 없음 | Lighthouse 점수 유지 |
| 호환성 | Chrome, Safari, Firefox 최신 버전 지원 | 크로스 브라우저 테스트 |

---

## 4. Design Specification

### 4.1 Core Color Palette (Light Theme)

```css
:root {
  /* 배경 계층 */
  --sf-bg:           #f8fafc;    /* 메인 배경 (매우 밝은 블루그레이) */
  --sf-card-bg:      #ffffff;    /* 카드 배경 (순백) */
  --sf-nav-bg:       #ffffff;    /* 네비게이션 배경 */
  --sf-border:       #e2e8f0;    /* 기본 테두리 */
  --sf-border-light: #f1f5f9;    /* 연한 테두리 */

  /* 브랜드 컬러 (보라색 유지, 톤 조정) */
  --sf-purple:         #7c3aed;  /* 메인 퍼플 (유지) */
  --sf-purple-light:   #a78bfa;  /* 라이트 퍼플 */
  --sf-purple-lighter: #ede9fe;  /* 배경용 매우 연한 퍼플 */
  --sf-purple-dark:    #6d28d9;  /* 다크 퍼플 (그라데이션) */

  /* 텍스트 계층 (다크 텍스트 on 라이트 배경) */
  --sf-text-1: #1e293b;    /* 주 텍스트 (슬레이트 900) */
  --sf-text-2: #475569;    /* 보조 텍스트 (슬레이트 600) */
  --sf-text-3: #94a3b8;    /* 3차 텍스트 (슬레이트 400) */
  --sf-text-4: #cbd5e1;    /* 4차 텍스트 (슬레이트 300) */

  /* 도메인 액센트 (약간 부드러운 톤) */
  --sf-blue:   #3b82f6;    /* 반도체 */
  --sf-amber:  #f59e0b;    /* 노동법 */
  --sf-red:    #ef4444;    /* 현장교육 */
  --sf-violet: #8b5cf6;    /* 안전가이드 */
  --sf-green:  #10b981;    /* MSDS */
  --sf-cyan:   #06b6d4;    /* 커뮤니티 */
}
```

### 4.2 Domain Color Palette (Light Version)

| Domain | Primary | Light BG | Card Accent | Shadow |
|--------|---------|----------|-------------|--------|
| Semiconductor | #0ea5e9 | #f0f9ff | rgba(14,165,233,0.08) | rgba(14,165,233,0.12) |
| Labor Law | #f59e0b | #fffbeb | rgba(245,158,11,0.08) | rgba(245,158,11,0.12) |
| Field Training | #ec4899 | #fdf2f8 | rgba(236,72,153,0.08) | rgba(236,72,153,0.12) |
| Safety Guide | #6366f1 | #eef2ff | rgba(99,102,241,0.08) | rgba(99,102,241,0.12) |
| MSDS | #10b981 | #ecfdf5 | rgba(16,185,129,0.08) | rgba(16,185,129,0.12) |

### 4.3 Component Style Changes

| Component | Before (Dark) | After (Light) |
|-----------|---------------|---------------|
| 배경 | #08090d (거의 검정) | #f8fafc (밝은 블루그레이) |
| 카드 | #0f1017 + 1px border | #ffffff + box-shadow: 0 1px 3px rgba(0,0,0,0.1) |
| 네비게이션 | rgba 투명 + backdrop-blur | #ffffff + box-shadow: 0 1px 2px rgba(0,0,0,0.05) |
| 버튼 | 보라 그라데이션 (변경 없음) | 보라 그라데이션 (유지) |
| 검색바 | 어두운 배경 + 어두운 테두리 | #ffffff + #e2e8f0 테두리 |
| 텍스트 | #e4e4e7 (밝은 회색) | #1e293b (어두운 슬레이트) |
| 호버 효과 | 밝아지는 방향 | 그림자 강화 방향 |
| 테두리 | #1a1b25 (어두운 보라) | #e2e8f0 (밝은 슬레이트) |

---

## 5. Success Criteria

### 5.1 Definition of Done

- [ ] 모든 페이지(홈, 도메인, 커뮤니티, 로그인, 관리자)에 라이트 테마 적용
- [ ] 도메인별 색상 아이덴티티 유지 (파스텔 톤으로 재해석)
- [ ] 텍스트 가독성 WCAG AA 기준 충족
- [ ] 반응형 레이아웃 기존과 동일하게 작동
- [ ] 기능적 동작 변화 없음 (검색, AI 답변, 커뮤니티 등)

### 5.2 Quality Criteria

- [ ] 모든 페이지에서 통일된 밝은 톤
- [ ] 기존 다크 테마 색상 잔재 없음
- [ ] 그림자와 테두리로 깊이감 표현
- [ ] 호버/포커스 상태 시각적 피드백 유지

---

## 6. Risks and Mitigation

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| 인라인 CSS가 6개 파일에 분산되어 변경 누락 가능 | High | Medium | 체크리스트 기반 파일별 순차 작업 |
| 도메인별 동적 컬러(Jinja2 변수)가 라이트 테마와 충돌 | Medium | Medium | domain_config.py의 color_rgb 값도 함께 업데이트 |
| Chart.js 등 JS 라이브러리 색상이 라이트 배경에서 가시성 저하 | Medium | Low | common.js 색상 팔레트 별도 조정 |
| 이미지 라이트박스 오버레이가 라이트 배경과 어색 | Low | Low | 라이트박스는 어두운 오버레이 유지 (사진 감상 목적) |

---

## 7. Implementation Plan

### 7.1 수정 대상 파일 및 순서

| Order | File | Changes | Estimated Lines |
|:-----:|------|---------|:---------------:|
| 1 | `templates/base.html` | `:root` 변수 전환, 네비게이션 스타일 | ~50 lines |
| 2 | `templates/home.html` | 히어로, 도메인 카드, 하단 카드 스타일 | ~80 lines |
| 3 | `templates/domain.html` | 검색, 결과, AI 답변, 차트 영역 | ~100 lines |
| 4 | `services/domain_config.py` | 도메인별 색상 라이트 버전 | ~20 lines |
| 5 | `templates/community.html` | 커뮤니티 카드, 뱃지 스타일 | ~30 lines |
| 6 | `templates/login.html` | 로그인 폼 배경 | ~15 lines |
| 7 | `templates/admin.html` | 사이드바, 패널 스타일 | ~20 lines |
| 8 | `static/js/common.js` | Chart.js 색상 팔레트 | ~10 lines |

### 7.2 핵심 전환 전략

1. **`:root` 변수 우선 변경** — 전역 색상을 한번에 전환 (base.html)
2. **하드코딩된 색상 치환** — rgba(), hex 직접 지정된 곳 개별 수정
3. **그림자 체계 추가** — 다크 테마의 테두리 구분 → 라이트 테마의 그림자 구분으로 전환
4. **도메인 config 동기화** — Python 쪽 색상 값도 라이트 버전으로 업데이트

---

## 8. Next Steps

1. [ ] Design 문서 작성 (`/pdca design bright-ui-redesign`)
2. [ ] 파일별 구현 작업 시작
3. [ ] 구현 후 Gap Analysis (`/pdca analyze bright-ui-redesign`)

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-03-10 | Initial draft | Claude |
