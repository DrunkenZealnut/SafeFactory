# keyword-graph-view Completion Report

> **Feature**: 인기질문 키워드 그래프뷰 (옵시디언 스타일)
> **Project**: SafeFactory
> **Author**: Claude Code
> **Date**: 2026-03-18
> **Status**: Completed
> **PR**: https://github.com/DrunkenZealnut/SafeFactory/pull/9

---

## Executive Summary

| Item | Detail |
|------|--------|
| **Feature** | 인기질문 키워드 그래프뷰 |
| **Started** | 2026-03-18 |
| **Completed** | 2026-03-18 |
| **Duration** | 1 session |

| Metric | Value |
|--------|-------|
| **Match Rate** | 99% |
| **Items Checked** | 90 |
| **Files Changed** | 3 |
| **Lines Added** | +397 |
| **Iterations** | 0 (first pass) |

### 1.3 Value Delivered

| Perspective | Content |
|-------------|---------|
| **Problem** | 기존 워드클라우드는 키워드를 랜덤 배치하여 키워드 간 관계(동시출현, 주제 클러스터)를 파악할 수 없었다 |
| **Solution** | D3.js force-directed 그래프로 키워드 노드와 동시출현 엣지를 시각화하고, 옵시디언 그래프뷰 스타일의 호버 하이라이트와 줌/팬/드래그 인터랙션을 구현했다 |
| **Function/UX Effect** | 키워드 클릭 시 연결된 키워드가 하이라이트되고 관련 질문 패널이 열리며, 노드 크기(빈도)와 엣지 두께(동시출현)로 키워드 생태계를 직관적으로 탐색할 수 있다. 워드클라우드 ↔ 그래프 토글로 기존 뷰도 완전 보존 |
| **Core Value** | 질문 간 숨겨진 주제 연결고리를 시각적으로 발견하여, 사용자의 지식 탐색 깊이와 플랫폼 만족도를 높인다 |

---

## 2. PDCA Cycle Summary

```
[Plan] ✅ → [Design] ✅ → [Do] ✅ → [Check] ✅ (99%) → [Report] ✅
```

| Phase | Output | Status |
|-------|--------|:------:|
| Plan | `docs/01-plan/features/keyword-graph-view.plan.md` | ✅ |
| Design | `docs/02-design/features/keyword-graph-view.design.md` | ✅ |
| Do | 3 files modified, PR #9 created | ✅ |
| Check | `docs/03-analysis/keyword-graph-view.analysis.md` (99%) | ✅ |
| Act | Skip (99% ≥ 90%) | N/A |
| Report | This document | ✅ |

---

## 3. Implementation Summary

### 3.1 Changed Files

| File | Change | Lines |
|------|--------|:-----:|
| `services/keyword_extractor.py` | `extract_keyword_graph()` + `_extract_tokens()` 추가 | +75 |
| `api/v1/questions.py` | `GET /api/v1/questions/graph` 엔드포인트 추가 | +47 |
| `templates/wordcloud.html` | D3.js 그래프뷰 + 토글 UI 전면 재작성 | +275 |
| **Total** | | **+397** |

### 3.2 Key Implementation Details

**Backend**:
- `extract_keyword_graph()`: 질문별 키워드 동시출현 쌍을 `itertools.combinations`로 집계
- `_extract_tokens()`: DRY 헬퍼로 기존 regex 로직 재사용
- 상위 80개 노드, min_edge_weight=2 필터링으로 성능 제한

**API**:
- `GET /api/v1/questions/graph?namespace=&period=all&node_limit=80&min_edge_weight=2`
- 응답: `{nodes: [{id, text, weight, domains}], edges: [{source, target, weight}]}`
- Rate limit: 30/min, 에러 처리: logging + 500 응답

**Frontend**:
- D3.js v7 force-directed SVG 그래프
- 5개 force: link, charge(-120), center, collide, (implicit) collision
- 옵시디언 스타일 호버: 연결 노드 highlight, 비연결 opacity 0.08
- 줌(0.3x~5x)/팬/노드 드래그 + simulation reheat
- 도메인별 색상: blue(반도체), amber(현장실습), green(안전보건), violet(MSDS)
- 워드클라우드 ↔ 그래프 토글 (기존 완전 보존)
- D3 CDN 실패 시 워드클라우드 자동 폴백

### 3.3 DB Changes

없음. 기존 `shared_questions` 테이블 그대로 사용.

---

## 4. Gap Analysis Results

### 4.1 Overall

| Metric | Value |
|--------|:-----:|
| Match Rate | **99%** |
| Items Checked | 90 |
| Full Match | 88 (97.8%) |
| Partial Match | 2 (2.2%) |
| Missing/Gap | 0 (0%) |
| Beneficial Additions | 10 |

### 4.2 Category Breakdown

| Category | Score |
|----------|:-----:|
| Data Model | 100% |
| API Specification | 100% |
| UI/UX Design | 95% |
| Backend Implementation | 100% |
| Frontend Implementation | 100% |
| Mobile Responsive | 100% |
| Error Handling | 100% |
| Security | 100% |

### 4.3 Partial Matches (Minor Improvements)

1. **노드 최소 반지름**: 설계 4px → 구현 5px (가시성 개선)
2. **라벨 최소 폰트**: 설계 10px → 구현 9px (밀집 그래프 대응)

### 4.4 Beneficial Additions (10)

DRY 헬퍼(`_extract_tokens`), SVG `preserveAspectRatio`, 더블클릭 줌 비활성화, 동적 라벨 폰트, `searchInDomain()` 폴백, `escapeHtml()` XSS 방어, Edge group 컨테이너, Simulation 정리(메모리 누수 방지), SVG 복구 가드, `DOMAIN_COLORS/LABELS` 상수 중앙화

---

## 5. Risks Addressed

| Plan Risk | Mitigation Applied | Result |
|-----------|-------------------|--------|
| D3 성능 저하 (200+ 노드) | node_limit=80 API 상한 | ✅ 해결 |
| 동시출현 O(n·k²) 복잡도 | 질문별 키워드 ≤10개, min_edge_weight 필터 | ✅ 해결 |
| 모바일 터치 충돌 | D3 zoom 자동 터치 지원, 반응형 CSS | ✅ 해결 |
| D3.js 번들 크기 | CDN 로드 + 폴백 | ✅ 해결 |
| 기존 사용자 혼란 | 토글 버튼으로 양쪽 뷰 유지 | ✅ 해결 |

---

## 6. Deliverables

| Deliverable | Location |
|-------------|----------|
| Plan Document | `docs/01-plan/features/keyword-graph-view.plan.md` |
| Design Document | `docs/02-design/features/keyword-graph-view.design.md` |
| Analysis Report | `docs/03-analysis/keyword-graph-view.analysis.md` |
| Completion Report | `docs/04-report/features/keyword-graph-view.report.md` |
| Pull Request | https://github.com/DrunkenZealnut/SafeFactory/pull/9 |
| Branch | `feat/keyword-graph-view` |

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2026-03-18 | Initial completion report | Claude Code |
