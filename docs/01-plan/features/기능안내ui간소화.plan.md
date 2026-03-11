# 기능안내 UI 간소화 Planning Document

> **Summary**: SafeFactory 홈페이지 및 도메인 페이지의 기능안내 UI를 간결하게 재구성하여 사용자 인지부하를 줄이고 핵심 기능 접근성을 높인다.
>
> **Project**: SafeFactory
> **Author**: zealnutkim
> **Date**: 2026-03-11
> **Status**: Draft

---

## Executive Summary

| 항목 | 내용 |
|------|------|
| **Feature** | 기능안내 UI 간소화 |
| **시작일** | 2026-03-11 |
| **예상 기간** | 1-2일 |

| 관점 | 설명 |
|------|------|
| **Problem** | 홈페이지에 6개 도메인 카드 + 커뮤니티/뉴스 프리뷰 + 검색바 + 힌트칩 등 정보가 과다하여 사용자가 핵심 기능을 직관적으로 파악하기 어렵다 |
| **Solution** | 도메인 카드를 축약형 리스트로, 부가 안내 요소를 축소/제거하여 검색 중심의 간결한 UI로 재구성 |
| **Function UX Effect** | 첫 방문 사용자가 3초 내에 "검색 → AI 답변" 핵심 플로우를 인지하고 실행 가능 |
| **Core Value** | 인지부하 감소로 이탈률 저감, 검색 전환율 향상 |

---

## 1. Overview

### 1.1 Purpose

SafeFactory 사이트의 홈페이지와 도메인 진입 과정에서 기능 안내 UI 요소가 많아 사용자 인지부하가 높은 문제를 해결한다. 핵심 기능(검색 → AI 답변)에 빠르게 접근할 수 있도록 UI를 간소화한다.

### 1.2 Background

현재 홈페이지 구조:
- **Hero 영역**: 배지 + 타이틀 + 서브타이틀 + 검색바 + 4개 힌트칩
- **도메인 카드 섹션**: 6개의 큰 카드 (3열 그리드) — 반도체, 노동법, 현장실습, 안전보건, MSDS, 커뮤니티
- **하단 영역**: 커뮤니티 최신글 프리뷰 + 뉴스 프리뷰 (2열 그리드)
- **푸터**: 이용약관, 개인정보, 고객센터

문제점:
1. 도메인 카드가 화면을 많이 차지하여 스크롤이 길어짐
2. 이미 상단 네비게이션에 모든 도메인 링크가 있어 카드와 중복
3. 각 카드의 설명 텍스트가 비슷한 패턴 반복 ("AI 질문하기")
4. 커뮤니티/뉴스 프리뷰가 홈 진입 시 매번 API 호출 → 로딩 지연
5. 모바일에서 6개 카드가 1열로 쌓여 스크롤이 과도하게 길어짐

### 1.3 Related Documents

- `templates/home.html` — 현재 홈페이지 템플릿
- `templates/base.html` — 네비게이션 및 레이아웃
- `templates/domain.html` — 도메인 상세 페이지

---

## 2. Scope

### 2.1 In Scope

- [x] 홈페이지 도메인 카드 섹션 간소화 (큰 카드 → 컴팩트 리스트/칩 형태)
- [x] Hero 영역 문구 간결화
- [x] 하단 커뮤니티/뉴스 프리뷰 영역 축소 또는 접기 처리
- [x] 모바일 반응형 최적화
- [x] 검색 힌트칩 개수 조정

### 2.2 Out of Scope

- 도메인 상세 페이지(domain.html) 내부 UI 변경
- 네비게이션 바 구조 변경
- 백엔드 API 변경
- 새 기능 추가

---

## 3. Requirements

### 3.1 Functional Requirements

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| FR-01 | 도메인 카드를 컴팩트 형태(아이콘+제목 한 줄)로 변경 | High | Pending |
| FR-02 | Hero 타이틀/서브타이틀 문구를 1-2줄로 축약 | Medium | Pending |
| FR-03 | 커뮤니티/뉴스 프리뷰를 접이식(collapse) 또는 제거 | Medium | Pending |
| FR-04 | 검색 힌트칩을 2-3개로 축소 | Low | Pending |
| FR-05 | 모바일에서 도메인 선택이 1스크린 내 완료 가능 | High | Pending |

### 3.2 Non-Functional Requirements

| Category | Criteria | Measurement Method |
|----------|----------|-------------------|
| Performance | 홈페이지 초기 로딩 시 불필요한 API 호출 제거 | Network 탭 확인 |
| UX | 핵심 기능(검색) 도달 시간 < 3초 | 사용자 테스트 |
| Accessibility | 키보드 네비게이션 유지 | 수동 테스트 |
| Responsive | 모바일 홈 스크롤 50% 이상 감소 | 비교 측정 |

---

## 4. Success Criteria

### 4.1 Definition of Done

- [x] 홈페이지 도메인 안내가 컴팩트 형태로 전환됨
- [x] 모바일에서 홈 스크롤 길이 50% 이상 감소
- [x] 기존 모든 도메인 링크가 정상 동작
- [x] 검색 기능 정상 작동

### 4.2 Quality Criteria

- [x] 기존 라우팅/링크 깨지지 않음
- [x] 반응형 3개 breakpoint(1280px/900px/600px) 정상 동작
- [x] Lighthouse 접근성 점수 유지

---

## 5. Risks and Mitigation

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| 도메인 카드 축소로 각 분야 인지도 저하 | Medium | Medium | 아이콘+색상으로 시각적 구분 유지 |
| 커뮤니티/뉴스 프리뷰 제거 시 해당 섹션 트래픽 감소 | Low | Medium | 네비게이션 탭으로 접근 보장 |
| 기존 사용자의 UI 변경 혼란 | Low | Low | 점진적 변경, 핵심 플로우 유지 |

---

## 6. Architecture Considerations

### 6.1 Project Level

기존 Flask + Jinja2 템플릿 기반 프로젝트. 프론트엔드 프레임워크 없이 순수 HTML/CSS/JS로 구현.

### 6.2 Key Decisions

| Decision | Options | Selected | Rationale |
|----------|---------|----------|-----------|
| 도메인 표시 형태 | A) 컴팩트 칩 B) 축소 카드 C) 아이콘 리스트 | A) 컴팩트 칩 | 1줄로 표시 가능, 모바일 최적화 |
| 커뮤니티/뉴스 프리뷰 | A) 제거 B) 접이식 C) 탭 | A) 제거 | 네비게이션에서 접근 가능, API 호출 제거 |
| 검색 힌트 | A) 2개로 축소 B) 3개 유지 C) 제거 | B) 3개 유지 | 도메인별 예시 최소 유지 |

### 6.3 변경 대상 파일

```
templates/home.html     ← 주요 변경 (도메인 카드, 하단 프리뷰, 힌트칩)
templates/base.html     ← 변경 없음
templates/domain.html   ← 변경 없음 (out of scope)
```

---

## 7. Convention Prerequisites

### 7.1 Existing Project Conventions

- [x] `CLAUDE.md` — 프로젝트 구조 및 명령어 정리됨
- [x] CSS 변수 시스템 (`theme.css`) 사용 중
- [x] BEM 미사용, 클래스 네이밍은 케밥케이스
- [x] Jinja2 템플릿 상속 (`base.html` → `home.html`)

### 7.2 Conventions to Follow

| Category | Rule |
|----------|------|
| CSS | 기존 `--sf-*` CSS 변수 시스템 유지 |
| HTML | 시맨틱 태그 사용 (`section`, `nav`, `main`) |
| JS | 인라인 `<script>` 패턴 유지 (번들러 미사용) |
| 반응형 | 기존 breakpoint 900px / 600px 유지 |

---

## 8. Next Steps

1. [ ] Design 문서 작성 (`기능안내ui간소화.design.md`)
2. [ ] 구현 (home.html 수정)
3. [ ] 테스트 (반응형 + 링크 동작)

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-03-11 | Initial draft | zealnutkim |
