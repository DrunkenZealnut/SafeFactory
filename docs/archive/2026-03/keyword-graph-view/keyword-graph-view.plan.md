# 인기질문 키워드 그래프뷰 Planning Document

> **Summary**: 현재 워드클라우드(wordcloud2.js) 기반 키워드 시각화를 옵시디언 그래프뷰 스타일의 인터랙티브 노드-엣지 그래프로 전환
>
> **Project**: SafeFactory
> **Author**: Claude Code
> **Date**: 2026-03-18
> **Status**: Draft

---

## Executive Summary

| Perspective | Content |
|-------------|---------|
| **Problem** | 현재 워드클라우드는 키워드 간 관계를 보여주지 못하며, 사용자가 키워드 생태계를 탐색하기 어렵다 |
| **Solution** | D3.js force-directed 그래프로 키워드 노드와 동시출현(co-occurrence) 엣지를 시각화하여 옵시디언 그래프뷰와 유사한 탐색 경험 제공 |
| **Function/UX Effect** | 키워드 클릭 시 연결된 키워드가 하이라이트되고, 관련 질문 패널이 열리며, 줌/팬/드래그로 자유 탐색 가능 |
| **Core Value** | 질문 간 숨겨진 주제 연결고리를 시각적으로 발견하여 지식 탐색의 깊이와 만족도를 높인다 |

---

## 1. Overview

### 1.1 Purpose

현재 `/wordcloud` 페이지는 wordcloud2.js를 사용해 키워드를 캔버스에 랜덤 배치한다. 키워드 크기로 빈도는 알 수 있지만, **키워드 간 관계(어떤 키워드가 같은 질문에서 함께 등장하는지)**를 파악할 수 없다.

옵시디언 그래프뷰처럼 키워드를 노드로, 동시출현 관계를 엣지로 표현하면:
- 키워드 클러스터(주제 그룹)를 직관적으로 파악
- 연결 허브 키워드(여러 주제를 잇는 핵심 개념) 발견
- 관심 키워드에서 관련 키워드로 자연스럽게 탐색 확장

### 1.2 Background

- 현재 `services/keyword_extractor.py`는 질문별 키워드를 추출하지만 키워드 간 관계 데이터는 반환하지 않음
- `api/v1/questions.py`의 `/wordcloud` 엔드포인트는 `[{text, weight}]` 플랫 리스트만 제공
- 프로젝트에 이미 KGEntity/KGRelation 기반 GraphRAG가 있어 그래프 시각화 패턴 참고 가능
- 옵시디언 그래프뷰의 핵심 UX: force-directed layout, 줌/팬, 노드 호버 하이라이트, 연결 노드 강조

### 1.3 Related Documents

- 현재 구현: `templates/wordcloud.html`, `services/keyword_extractor.py`, `api/v1/questions.py`
- 참고: Obsidian Graph View (https://obsidian.md)

---

## 2. Scope

### 2.1 In Scope

- [ ] 키워드 동시출현(co-occurrence) 데이터 추출 로직 추가
- [ ] 그래프 데이터 API 엔드포인트 신규 생성 (`/api/v1/questions/graph`)
- [ ] D3.js force-directed 그래프 프론트엔드 구현
- [ ] 노드 인터랙션: 클릭(관련 질문 표시), 호버(연결 하이라이트), 드래그(위치 이동)
- [ ] 줌/팬 네비게이션
- [ ] 도메인(namespace) 필터, 기간 필터 유지
- [ ] 모바일 반응형 지원
- [ ] 기존 `/wordcloud` 경로에서 그래프뷰로 전환 (또는 토글)

### 2.2 Out of Scope

- 3D 그래프 시각화 (WebGL/Three.js)
- 실시간 질문 스트리밍 반영
- KGEntity/KGRelation 기반 지식그래프와의 통합 (별도 기능)
- 그래프 데이터의 서버측 캐싱/사전 계산 (초기 버전에서는 요청 시 계산)

---

## 3. Requirements

### 3.1 Functional Requirements

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| FR-01 | 키워드 동시출현 관계를 추출하는 백엔드 로직 구현 (같은 질문에서 추출된 키워드 쌍 → 엣지) | High | Pending |
| FR-02 | `/api/v1/questions/graph` 엔드포인트: `{nodes: [{id, text, weight}], edges: [{source, target, weight}]}` 반환 | High | Pending |
| FR-03 | D3.js force-directed 그래프 렌더링 (노드 크기 = 빈도, 엣지 두께 = 동시출현 횟수) | High | Pending |
| FR-04 | 노드 클릭 시 해당 키워드 포함 질문 목록 패널 표시 (기존 `showQuestionsForKeyword` 로직 재활용) | High | Pending |
| FR-05 | 노드 호버 시 해당 노드 + 직접 연결 노드 하이라이트, 나머지 fade out (옵시디언 스타일) | High | Pending |
| FR-06 | 줌(마우스 휠/핀치), 팬(드래그 배경), 노드 드래그(위치 이동) 인터랙션 | Medium | Pending |
| FR-07 | 도메인 필터(전체/반도체/현장실습/안전보건/MSDS) 및 기간 필터(7일/30일/전체) | Medium | Pending |
| FR-08 | 노드 색상을 도메인별 또는 클러스터별로 구분 | Medium | Pending |
| FR-09 | 그래프 중앙에 가장 큰 허브 노드 배치 (force simulation으로 자연 수렴) | Low | Pending |
| FR-10 | 워드클라우드 ↔ 그래프뷰 토글 버튼 (사용자 선택 가능) | Low | Pending |

### 3.2 Non-Functional Requirements

| Category | Criteria | Measurement Method |
|----------|----------|-------------------|
| Performance | 키워드 100개 + 엣지 300개 이하에서 초기 렌더링 < 1초 | Chrome DevTools 측정 |
| Performance | 그래프 API 응답 < 500ms | Flask 로그 |
| Responsiveness | 모바일(360px~)에서 터치 줌/팬 동작 | 실기기 테스트 |
| Accessibility | 키보드 탐색 가능, 노드 tooltip에 키워드 텍스트 | 수동 검증 |
| UX | 그래프 안정화까지 자연스러운 애니메이션 (force simulation) | 시각 확인 |

---

## 4. Success Criteria

### 4.1 Definition of Done

- [ ] 그래프 API가 노드/엣지 데이터를 정상 반환
- [ ] D3.js 그래프가 옵시디언 그래프뷰와 유사한 시각적 결과 제공
- [ ] 노드 클릭 → 관련 질문 패널 동작
- [ ] 호버 하이라이트 동작
- [ ] 줌/팬/드래그 동작
- [ ] 도메인/기간 필터 동작
- [ ] 모바일 반응형 정상

### 4.2 Quality Criteria

- [ ] 기존 `/wordcloud` 경로 하위 호환성 유지
- [ ] 빈 데이터(질문 없음) 시 graceful empty state
- [ ] JavaScript 콘솔 에러 없음
- [ ] `--sf-*` CSS 변수 기반 다크/라이트 테마 호환

---

## 5. Risks and Mitigation

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| 키워드 수가 많을 때(200+) D3 force simulation 성능 저하 | Medium | Medium | 노드 수 상한 제한 (API에서 top 80), 엣지 weight 임계값 필터링 |
| 동시출현 계산이 질문 수에 따라 O(n·k²) 복잡도 | Medium | Low | 질문 수 제한(최근 1000개), 결과 캐싱 검토 |
| 모바일에서 터치 인터랙션 충돌 (스크롤 vs 팬) | Medium | Medium | 그래프 영역 touch-action: none, 외부 스크롤 분리 |
| D3.js 번들 크기 (약 90KB gzipped) | Low | Low | CDN 로드, 필요한 모듈만 import (d3-force, d3-selection, d3-zoom) |
| 워드클라우드 기존 사용자의 혼란 | Low | Low | 토글 버튼으로 양쪽 뷰 모두 유지 |

---

## 6. Architecture Considerations

### 6.1 Project Level Selection

| Level | Characteristics | Recommended For | Selected |
|-------|-----------------|-----------------|:--------:|
| **Starter** | Simple structure | Static sites | ☐ |
| **Dynamic** | Feature-based modules, BaaS | Web apps with backend | ☑ |
| **Enterprise** | Strict layer separation, DI | High-traffic systems | ☐ |

### 6.2 Key Architectural Decisions

| Decision | Options | Selected | Rationale |
|----------|---------|----------|-----------|
| 그래프 라이브러리 | D3.js / vis.js / Cytoscape.js / Sigma.js | **D3.js** | 가장 유연한 커스터마이징, CDN 사용 가능, 옵시디언 스타일 구현에 적합 |
| 렌더링 방식 | SVG / Canvas | **SVG** | 노드 100개 이하에서 SVG가 인터랙션(호버, 클릭) 처리 용이 |
| 동시출현 계산 위치 | Backend (Python) / Frontend (JS) | **Backend** | keyword_extractor.py 확장, 질문 데이터 접근이 서버에 있음 |
| 페이지 구조 | 새 페이지 / 기존 워드클라우드 페이지 수정 | **기존 페이지 수정 + 토글** | URL 유지, 사용자 선택권 보장 |
| 데이터 형식 | adjacency matrix / edge list | **edge list** | D3 force graph 네이티브 포맷, 전송 효율 |

### 6.3 변경 대상 파일

```
수정:
├── services/keyword_extractor.py     # 동시출현 추출 함수 추가
├── api/v1/questions.py               # /graph 엔드포인트 추가
├── templates/wordcloud.html          # 그래프뷰 UI 추가 (토글)

참조 (변경 없음):
├── models.py                         # SharedQuestion 모델
├── web_app.py                        # /wordcloud 라우트
├── static/css/theme.css              # --sf-* CSS 변수
```

---

## 7. Convention Prerequisites

### 7.1 Existing Project Conventions

- [x] `CLAUDE.md` has coding conventions section
- [x] `--sf-*` CSS 변수 기반 디자인 시스템 (`static/css/theme.css`)
- [x] API 응답 형식: `api/response.py`의 `success_response`/`error_response`
- [x] Blueprint 기반 API 구조 (`api/v1/`)

### 7.2 Conventions to Follow

| Category | Convention | Applied |
|----------|-----------|:-------:|
| **CSS 클래스** | `wc-` 접두사 유지 (word cloud 페이지) + `kg-` 접두사 추가 (keyword graph) | ☐ |
| **API 응답** | `success_response(data={...})` 패턴 사용 | ☐ |
| **CDN 라이브러리** | `<script>` 태그로 CDN 로드 (기존 패턴: marked.js, DOMPurify, wordcloud2.js) | ☐ |
| **인라인 스크립트** | `{% block scripts %}` 내 `<script>` 블록 (기존 패턴) | ☐ |
| **max-width** | 960px (기존 `.wc-container` 패턴) | ☐ |

### 7.3 Environment Variables Needed

추가 환경변수 불필요 (기존 인프라 활용)

---

## 8. Technical Design Sketch

### 8.1 Backend: 동시출현 추출

```python
# services/keyword_extractor.py에 추가
def extract_keyword_graph(questions, node_limit=80, min_edge_weight=2):
    """
    Returns:
        {
            "nodes": [{"id": "키워드", "weight": 10, "domains": ["semiconductor-v2"]}],
            "edges": [{"source": "키워드A", "target": "키워드B", "weight": 3}]
        }
    """
    # 1. 질문별 키워드 집합 추출
    # 2. 상위 node_limit개 키워드 선택
    # 3. 같은 질문에서 나온 키워드 쌍 → 엣지 weight 누적
    # 4. min_edge_weight 미만 엣지 제거
```

### 8.2 API 엔드포인트

```
GET /api/v1/questions/graph?namespace=&period=all&node_limit=80
Response: { success: true, data: { nodes: [...], edges: [...], total_questions: N } }
```

### 8.3 Frontend: D3 Force Graph

```
D3 Force Simulation:
- forceLink: 엣지 (distance 반비례 weight)
- forceManyBody: 노드 간 반발력 (charge -100)
- forceCenter: 캔버스 중심 배치
- forceCollide: 노드 반지름 기반 충돌 방지

인터랙션:
- zoom: d3.zoom() → SVG transform
- drag: d3.drag() → 노드 위치 변경 + simulation reheat
- hover: 연결 노드 opacity 1.0, 비연결 opacity 0.1
- click: 질문 패널 표시
```

---

## 9. Next Steps

1. [ ] Design 문서 작성 (`keyword-graph-view.design.md`)
2. [ ] 구현 시작: `keyword_extractor.py` → API → Frontend 순서
3. [ ] 구현 후 Gap 분석

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-03-18 | Initial draft | Claude Code |
