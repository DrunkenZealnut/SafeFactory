# Admin 페이지 디자인 통합 리뉴얼 Plan

## Executive Summary

| 항목 | 내용 |
|------|------|
| Feature | Admin 페이지 디자인 시스템 통합 리뉴얼 |
| 작성일 | 2026-03-18 |
| 예상 소요 | 3-4시간 |
| 난이도 | Medium |
| 브랜치 | `feature/admin-redesign` → PR 생성 |

### Value Delivered

| 관점 | 설명 |
|------|------|
| **Problem** | admin.html이 독립 페이지로 base.html/theme.css 미사용. 하드코딩 색상(#7c3aed, #f5f7fb 등)으로 디자인 불일치, 공통 nav/footer 부재 |
| **Solution** | base.html 상속 + theme.css CSS 변수 적용으로 전체 사이트와 디자인 통합. 공통 네비게이션/footer 자동 적용 |
| **Function UX Effect** | 관리자가 동일한 UI 경험으로 사이트 관리. 다크모드 대응 가능성 확보 |
| **Core Value** | 유지보수 비용 절감 (CSS 변수 1곳 수정으로 전체 반영) + 디자인 일관성 |

---

## 1. 현재 상태 분석

### 1.1 admin.html 현황
- **파일 크기**: 2,043줄 (독립 HTML, base.html 미상속)
- **자체 스타일**: 하드코딩 색상 (#7c3aed, #f5f7fb, #e5e7eb 등)
- **자체 네비게이션**: top-nav + sidebar (공통 nav 미사용)
- **공통 footer 없음**: 최근 추가한 청년노동자인권센터 footer 미표시
- **반응형**: 768px breakpoint 1개만 존재

### 1.2 현재 admin 섹션 (7개)
| 섹션 | ID | 기능 |
|------|-----|------|
| 대시보드 | `section-dashboard` | 통계 카드, 활동 로그 |
| 학습자료 | `section-documents` | 파일 트리, 검색, 동기화 |
| 게시판 | `section-posts` | 게시글/댓글/카테고리/뉴스 관리 |
| 사용자 | `section-users` | 사용자 목록, 역할 변경 |
| 통계 | `section-stats` | 벡터/커뮤니티 통계 |
| 활동 로그 | `section-logs` | 관리자 행동 감사 로그 |
| 시스템 설정 | `section-settings` | LLM/검색 파라미터 설정 |

### 1.3 API 현황 (정상 동작 중)
- `api/v1/admin.py`: 40+ 엔드포인트 (변경 불필요)
- admin_required 데코레이터로 권한 제어

---

## 2. 구현 범위

### 2.1 필수 요구사항
- [ ] `base.html` 상속 (`{% extends "base.html" %}`)
- [ ] 하드코딩 색상 → CSS 변수(`--sf-*`) 전환
- [ ] 공통 nav/footer 자동 적용
- [ ] 기존 7개 섹션 기능 100% 유지
- [ ] 기존 JS 로직 100% 유지 (~1000줄)

### 2.2 디자인 변경 사항

| 요소 | 현재 | 변경 후 |
|------|------|---------|
| 레이아웃 | 독립 HTML | `base.html` 상속 |
| 배경색 | `#f5f7fb` | `var(--sf-bg)` |
| 카드 배경 | `#ffffff` | `var(--sf-card-bg)` |
| 테두리 | `#e5e7eb` | `var(--sf-border)` |
| 브랜드 색 | `#7c3aed` | `var(--sf-purple)` |
| 텍스트 1 | `#1f2937` | `var(--sf-text-1)` |
| 텍스트 2 | `#6b7280` | `var(--sf-text-2)` |
| 폰트 | 시스템 폰트 | Noto Sans KR (base.html에서 로드) |
| 최대 너비 | 제한 없음 | max-width: 960px (프로젝트 컨벤션) |
| top-nav | 자체 nav | base.html 공통 nav 사용 |
| footer | 없음 | base.html 공통 footer (청년노동자인권센터) |

### 2.3 유지 사항 (변경 금지)
- sidebar 탭 전환 JS 로직
- 40+ admin API 엔드포인트
- 각 섹션의 데이터 로딩/CRUD 기능
- 파일 트리 렌더링 로직
- 설정 저장 로직

---

## 3. 수정 대상 파일

| 파일 | 작업 | 영향도 |
|------|------|--------|
| `templates/admin.html` | base.html 상속 + CSS 변수 전환 | High |
| `web_app.py` (L430) | admin 라우트 변수 전달 확인 | Low |

> API 파일(`api/v1/admin.py`)은 변경 없음

---

## 4. 구현 전략

### 4.1 접근 방식
**점진적 마이그레이션**: 기존 2043줄을 한 번에 변경하지 않고, 구조(base.html 상속) → 색상(CSS 변수) → 레이아웃(max-width) 순서로 단계적 전환

### 4.2 구현 순서
1. `feature/admin-redesign` 브랜치 생성
2. admin.html을 `{% extends "base.html" %}` 구조로 변환
3. 자체 top-nav 제거 → base.html 공통 nav 활용
4. CSS 하드코딩 색상 → `var(--sf-*)` 전환
5. sidebar + main-content를 max-width: 960px 컨테이너 안에 배치
6. 반응형 보완 (480px breakpoint 추가)
7. 전체 기능 동작 확인
8. PR 생성

---

## 5. 위험 요소

| 위험 | 영향 | 대응 |
|------|------|------|
| JS 스코프 충돌 | High | base.html의 mobile menu JS와 admin sidebar JS 이름 충돌 확인 |
| CSS 우선순위 충돌 | Medium | base.html CSS와 admin CSS 셀렉터 충돌 시 `head_extra` 블록에서 오버라이드 |
| max-width 960px로 sidebar 공간 부족 | Medium | sidebar를 상단 탭으로 변환하거나 960px 제한 완화 검토 |
| 기존 기능 깨짐 | High | 변경 전 모든 섹션 수동 테스트 필수 |

---

## 6. PR 전략

```
Branch: feature/admin-redesign
Base: main
Title: feat: admin 페이지 디자인 시스템 통합 리뉴얼
```
