# Industrial Precision 디자인 적용 Planning Document

> **Summary**: SafeFactory 전체 UI를 "Industrial Precision" 컨셉으로 재설계 — 보라색 AI SaaS 클리셰에서 네이비+앰버 기반 산업안전 전문 플랫폼 디자인으로 전환
>
> **Project**: SafeFactory
> **Author**: Claude
> **Date**: 2026-03-14
> **Status**: Draft

---

## Executive Summary

| Perspective | Content |
|-------------|---------|
| **Problem** | 현재 보라색 그라디언트 + 흰 배경 디자인은 전형적인 AI SaaS 템플릿 느낌으로, "산업안전 전문 플랫폼"이라는 정체성이 시각적으로 전달되지 않음. 타이포그래피 단조로움, 모션 부재, 다크 테마 잔재 코드 충돌 문제도 존재 |
| **Solution** | 네이비(#1e293b) + 앰버(#f59e0b) 컬러 시스템으로 전환, Outfit 디스플레이 폰트 도입, 산업 도면 격자 패턴 배경, 페이지 로드 stagger 애니메이션 추가. CSS 변수 체계 유지하며 theme.css 중심 전환 |
| **Function/UX Effect** | 산업 전문 플랫폼다운 신뢰감 + 정보 밀도 향상. 앰버 포인트로 주의를 끄는 안전 경고색 활용, 모노스페이스 메타 정보로 데이터 가독성 개선 |
| **Core Value** | "AI 스타트업"에서 "산업안전 전문 플랫폼"으로 브랜드 정체성 확립. 타 AI 서비스와 시각적 차별화 |

---

## 1. Overview

### 1.1 Purpose

SafeFactory 웹 앱의 전체 디자인을 "Industrial Precision" 컨셉으로 전환한다. 기존 CSS 변수 시스템(`theme.css`)을 활용하여 최소한의 코드 변경으로 전체 톤앤매너를 교체하고, 산업안전 도메인에 어울리는 전문적인 시각 아이덴티티를 구축한다.

### 1.2 Background

- 현재 디자인: 보라색(`#7c3aed`) 브랜드 컬러, Noto Sans KR 단일 폰트, 동일한 border-radius 반복
- `static/css/theme.css`에 CSS 변수 체계가 잘 구축되어 있어 변수 값 교체로 전역 적용 가능
- `static/design-proposal.html`에 프로토타입이 이미 작성됨
- 이전 bright-ui-redesign에서 다크→라이트 전환 성공 경험 있음

### 1.3 Related Documents

- 프로토타입: `static/design-proposal.html`
- 테마 변수: `static/css/theme.css`
- 이전 디자인 계획: `docs/01-plan/features/bright-ui-redesign.plan.md`
- 대상 템플릿: `templates/base.html`, `templates/home.html`, `templates/domain.html`
- 도메인 설정: `services/domain_config.py`

---

## 2. Requirements

### 2.1 Functional Requirements

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| FR-01 | `theme.css` 컬러 시스템 교체 (보라→네이비+앰버) | High | Pending |
| FR-02 | `base.html` 네비게이션 재설계 (앰버 언더라인 활성 탭) | High | Pending |
| FR-03 | `home.html` 히어로 섹션 재설계 (좌측 정렬 + 격자 패턴) | High | Pending |
| FR-04 | `home.html` 도메인 칩 → 카드형 전환 | Medium | Pending |
| FR-05 | `domain.html` 패널/탭/폼 스타일 업데이트 | High | Pending |
| FR-06 | Outfit + Space Mono 폰트 도입 (제목/메타) | Medium | Pending |
| FR-07 | 페이지 로드 stagger 애니메이션 추가 | Low | Pending |
| FR-08 | `domain.html` 다크 테마 잔재 코드 정리 (recent-queries-dropdown) | Medium | Pending |
| FR-09 | `domain_config.py` 도메인별 컬러 값 업데이트 | Medium | Pending |
| FR-10 | 모바일 반응형 확인 및 조정 | High | Pending |
| FR-11 | PR 생성 (feature/industrial-precision-redesign 브랜치) | High | Pending |

### 2.2 Non-Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| NFR-01 | 기존 기능(검색, AI 질문, 커뮤니티 등) 동작에 영향 없음 | Critical |
| NFR-02 | 모바일 600px/900px 브레이크포인트 반응형 유지 | High |
| NFR-03 | 페이지 로드 성능 유지 (폰트 추가로 인한 지연 최소화) | Medium |
| NFR-04 | 접근성(a11y) 기존 수준 유지 — 색상 대비 4.5:1 이상 | Medium |

---

## 3. Design Decisions

### 3.1 컬러 시스템 전환

**기존 (보라 중심)**:
```css
--sf-purple:       #7c3aed;
--sf-purple-light:  #a78bfa;
--sf-purple-dark:   #6d28d9;
```

**변경 (네이비+앰버)**:
```css
--sf-navy:          #1e293b;
--sf-navy-light:    #334155;
--sf-amber:         #f59e0b;
--sf-amber-dark:    #d97706;
```

**이유**: 보라색은 AI/테크 스타트업에서 과도하게 사용되어 차별화 불가. 네이비는 신뢰감/전문성, 앰버는 산업안전 경고색으로 도메인 적합도가 높음.

### 3.2 폰트 전략

| 용도 | 기존 | 변경 |
|------|------|------|
| 제목/네비 | Noto Sans KR 700/900 | **Outfit** 700/800 |
| 본문 | Noto Sans KR 400/500 | Noto Sans KR 400/500 (유지) |
| 메타/코드 | 없음 | **Space Mono** 400 |

**이유**: Outfit은 기하학적 산세리프로 산업적 정밀함을 표현. Space Mono는 날짜/점수 등 데이터 가독성 향상.

### 3.3 접근 방식: 점진적 CSS 변수 교체

1. `theme.css` 변수 값 교체 → 전체 톤 한 번에 전환
2. `base.html` 네비게이션 구조 수정
3. `home.html` 히어로 + 도메인 칩 재설계
4. `domain.html` 패널 스타일 업데이트
5. 다크 테마 잔재 정리
6. 모바일 확인 및 미세 조정

---

## 4. Scope

### 4.1 In Scope

- `static/css/theme.css` — 변수 값 전면 교체
- `templates/base.html` — 네비게이션 구조 + 스타일 교체
- `templates/home.html` — 히어로, 검색바, 도메인 카드, 하단 카드
- `templates/domain.html` — 탭, 폼, AI 답변, 검색 결과 스타일
- `services/domain_config.py` — 도메인별 컬러/그라디언트 값
- PR 생성: `feature/industrial-precision-redesign` 브랜치

### 4.2 Out of Scope

- `templates/index.html` (관리자 페이지 — 별도 작업)
- `templates/login.html`, `templates/admin.html` (별도 작업)
- `templates/community.html`, `templates/news.html` (base.html 변경으로 자동 적용되는 부분만)
- JavaScript 로직 변경 (스타일만 변경)
- 백엔드 코드 변경

### 4.3 Assumptions

- CSS 변수 기반이므로 변수 교체만으로 대부분 자동 적용
- domain_config.py의 color/gradient 값은 domain.html에서 인라인 CSS 변수로 주입됨
- 기존 이모지 아이콘은 유지 (SVG 아이콘 전환은 별도 작업)

---

## 5. Implementation Order

```
Phase 1: Foundation (theme.css + base.html)
  ├── theme.css 변수 전면 교체
  ├── base.html 폰트 import 추가 (Outfit, Space Mono)
  └── base.html 네비게이션 스타일 교체

Phase 2: Home Page (home.html)
  ├── 히어로 섹션 재설계
  ├── 검색바 스타일 업데이트
  ├── 도메인 칩 → 카드형 전환
  └── 하단 커뮤니티/뉴스 카드 업데이트

Phase 3: Domain Page (domain.html)
  ├── 탭 스타일 (앰버 언더라인)
  ├── AI 질문 폼 + 버튼 스타일
  ├── AI 답변 컨테이너 스타일
  ├── 검색 결과 카드 스타일
  └── 다크 테마 잔재 코드 정리

Phase 4: Domain Config + 미세 조정
  ├── domain_config.py 컬러 값 업데이트
  ├── 모바일 반응형 확인
  └── 페이지 로드 애니메이션 추가

Phase 5: PR 생성
  ├── feature 브랜치 생성
  ├── 변경 사항 커밋
  └── PR 생성 및 설명 작성
```

---

## 6. Risk Assessment

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| 도메인별 인라인 컬러가 theme.css와 충돌 | Medium | Medium | domain_config.py의 color_rgb 값도 함께 업데이트 |
| 앰버 컬러의 흰 배경 위 가독성 부족 | Medium | Low | amber-dark(#d97706) 사용, WCAG 대비 확인 |
| 폰트 로딩 지연으로 FOUT 발생 | Low | Medium | font-display: swap + preconnect 적용 |
| 커뮤니티/뉴스 등 다른 페이지 스타일 깨짐 | Medium | Low | base.html의 공통 변수로 자동 적용, 개별 확인 |

---

## 7. Success Criteria

- [ ] 전체 페이지에서 보라색(#7c3aed) 계열이 네이비+앰버로 교체됨
- [ ] 히어로 섹션이 좌측 정렬 + 격자 패턴 배경으로 변경됨
- [ ] 네비게이션 활성 탭에 앰버 언더라인 적용
- [ ] 모바일(600px, 900px) 반응형 정상 동작
- [ ] 기존 기능(AI 질문, 검색, 커뮤니티) 정상 동작
- [ ] PR 생성 완료
