# Footer 제작 Plan

## Executive Summary

| 항목 | 내용 |
|------|------|
| Feature | Footer 제작 (단체명, 이메일, 주소) |
| 작성일 | 2026-03-17 |
| 예상 소요 | 1시간 이내 |
| 난이도 | Low |

### Value Delivered

| 관점 | 설명 |
|------|------|
| **Problem** | Footer가 home.html에만 존재하고 다른 페이지에는 없음. 단체 연락처 정보도 표시되지 않아 신뢰성 부족 |
| **Solution** | base.html에 공통 footer를 추가하여 전 페이지에 단체명/이메일/주소를 표시 |
| **Function UX Effect** | 모든 페이지 하단에서 즉시 단체 정보 확인 가능, 이메일 클릭 시 메일 앱 연동 |
| **Core Value** | 플랫폼 신뢰도 향상 및 법적 요건(사업자 정보 표시) 충족 |

---

## 1. 현재 상태 분석

### 1.1 기존 Footer 위치
- `templates/home.html` (line 677-686): home 페이지에만 간단한 footer 존재
- `templates/base.html`: footer 없음 → 다른 페이지(학습, MSDS, 커뮤니티 등)에 footer 미표시

### 1.2 기존 Footer 내용
```html
<footer class="sf-footer">
  <div class="sf-footer-inner">
    <div class="sf-footer-links">
      <a href="#">이용약관</a>
      <a href="#">개인정보처리방침</a>
      <a href="#">고객센터</a>
    </div>
    <p class="sf-footer-copy">&copy; {{ now().year }} SafeFactory. AI 기반 산업안전 지식 플랫폼</p>
  </div>
</footer>
```

### 1.3 문제점
1. Footer가 home 페이지에만 존재
2. 단체명, 이메일, 주소 등 연락처 정보 없음
3. 다른 페이지에서 사이트 정보를 확인할 수 없음

---

## 2. 구현 범위

### 2.1 필수 요구사항
- [ ] **단체명** 표시 (예: SafeFactory 운영단체명)
- [ ] **이메일** 표시 + `mailto:` 링크
- [ ] **주소** 표시
- [ ] **base.html로 이동**: 모든 페이지에서 공통 표시
- [ ] **home.html 중복 footer 제거**

### 2.2 디자인 방향
- 기존 디자인 시스템(CSS 변수) 활용
- max-width: 960px 통일 (프로젝트 컨벤션)
- 반응형 대응 (모바일/데스크톱)
- 기존 링크(이용약관, 개인정보처리방침, 고객센터) 유지

### 2.3 Footer 구성 요소
```
┌─────────────────────────────────────────────┐
│  청년노동자인권센터                            │
│  서울시 종로구 성균관로12 5층                   │
│  admin@younglabor.kr                        │
│                                             │
│  이용약관 | 개인정보처리방침 | 고객센터        │
│  © 2026 SafeFactory. AI 기반 산업안전 지식 플랫폼 │
└─────────────────────────────────────────────┘
```

---

## 3. 수정 대상 파일

| 파일 | 작업 |
|------|------|
| `templates/base.html` | footer HTML + CSS 추가 (`</main>` 뒤) |
| `templates/home.html` | 기존 footer 제거 (중복 방지), footer CSS 제거 |

---

## 4. 구현 순서

1. `base.html`에 footer CSS 스타일 추가
2. `base.html`의 `</main>` 태그 뒤에 footer HTML 추가
3. `home.html`에서 기존 footer HTML 제거
4. `home.html`에서 기존 footer CSS 제거
5. 전체 페이지 확인 (home, learn, msds, community, questions, news)

---

## 5. 위험 요소

| 위험 | 영향 | 대응 |
|------|------|------|
| home.html footer CSS가 다른 페이지 레이아웃과 충돌 | Low | base.html CSS 변수 활용으로 일관성 보장 |
| `sf-page-content` min-height 계산 변경 | Low | footer 높이 고려하여 조정 |

---

## 6. 사용자 확인 필요 사항

### 확정 정보
- **단체명**: 청년노동자인권센터
- **이메일**: admin@younglabor.kr
- **주소**: 서울시 종로구 성균관로12 5층
