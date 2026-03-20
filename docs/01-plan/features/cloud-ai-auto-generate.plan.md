# Plan: AI 자동 클라우드 생성

## Executive Summary

| 항목 | 내용 |
|------|------|
| Feature | cloud-ai-auto-generate |
| 작성일 | 2026-03-20 |
| 예상 기간 | 1~2일 |
| 난이도 | Medium |

### Value Delivered

| 관점 | 내용 |
|------|------|
| **Problem** | 현재 그래프 노드가 JS에 하드코딩되어 있어 관리자가 내용을 변경할 수 없고, 질문 키워드 기반 인기도만 반영 |
| **Solution** | 관리자가 시드 단어를 입력하면 Gemini가 관련 용어·계층·연결 관계를 자동 생성하여 그래프 구성 |
| **Function UX Effect** | 어떤 주제든 시드 단어만 입력하면 즉시 지식 그래프가 만들어지는 동적 경험 |
| **Core Value** | 반도체 외 다른 도메인에도 적용 가능한 범용 지식 그래프 플랫폼화 |

---

## 1. 현재 구조 (AS-IS)

```
home.html JS → 80+ 노드 하드코딩 (ROADMAP 데이터)
             → /api/v1/questions/wordcloud에서 인기도만 fetch
```

- 노드 추가/삭제 시 코드 수정 필요
- 도메인 변경 불가 (반도체 전용)

## 2. 목표 구조 (TO-BE)

```
Admin 페이지 → 시드 단어 입력 (예: "반도체 8대 공정")
            → Gemini API 호출 → 관련 용어 + 계층 + 관계 자동 생성
            → DB 저장 (graph_nodes, graph_edges 테이블)
            → home.html JS → /api/v1/graph/data에서 노드/엣지 fetch
```

---

## 3. 구현 항목

### 3.1 [P1] DB 모델 — GraphNode, GraphEdge

```python
# models.py
class GraphNode(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    node_id = db.Column(db.String(50), unique=True)     # 'w', 'ox', 'cvd' 등
    label = db.Column(db.String(100))                     # '웨이퍼 제조'
    description = db.Column(db.String(500))               # 설명
    node_type = db.Column(db.String(10))                  # 'm' (주요), 's' (세부), 'd' (상세), 'sh' (공유)
    color = db.Column(db.String(10))                      # '#cba6f7'
    radius = db.Column(db.Integer, default=14)
    order_num = db.Column(db.Integer)                     # 주요 공정 순서
    namespace = db.Column(db.String(50), default='default')  # 도메인별 그래프 분리
    created_at = db.Column(db.DateTime, default=...)

class GraphEdge(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    source_id = db.Column(db.String(50))                  # node_id 참조
    target_id = db.Column(db.String(50))
    is_flow = db.Column(db.Boolean, default=False)        # 공정 흐름 화살표
    namespace = db.Column(db.String(50), default='default')
```

### 3.2 [P1] API — 그래프 데이터 CRUD + AI 생성

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/api/v1/graph/data` | GET | 그래프 노드/엣지 JSON 반환 (홈페이지용) |
| `/api/v1/admin/graph/generate` | POST | 시드 단어 → Gemini로 그래프 자동 생성 |
| `/api/v1/admin/graph/nodes` | GET/POST/DELETE | 노드 CRUD |
| `/api/v1/admin/graph/edges` | GET/POST/DELETE | 엣지 CRUD |

### 3.3 [P1] Gemini 프롬프트 — 구조화된 그래프 데이터 생성

```
입력: "반도체 8대 공정"
출력 JSON: {
  "nodes": [
    {"id": "w", "label": "웨이퍼 제조", "type": "m", "desc": "...", "order": 1},
    {"id": "ingot", "label": "잉곳 성장", "type": "s", "parent": "w", "desc": "..."},
    ...
  ],
  "edges": [
    {"source": "w", "target": "ox", "flow": true},
    {"source": "w", "target": "ingot"},
    ...
  ]
}
```

Gemini에 구조화된 JSON 응답 요청:
- 주요 노드 (type: 'm') — 시드에서 파생된 핵심 카테고리
- 세부 노드 (type: 's') — 각 주요 노드의 하위 주제
- 상세 노드 (type: 'd') — 구체적 기술/방법론
- 공유 노드 (type: 'sh') — 여러 주요 노드에 걸치는 공통 기술
- 엣지 — 부모-자식 관계 + 공정 흐름(flow)

### 3.4 [P2] Admin 페이지 — 그래프 관리 UI

- 시드 단어 입력 필드 + "AI 생성" 버튼
- 생성된 노드/엣지 미리보기
- 노드 개별 편집/삭제
- "적용" 버튼으로 DB 저장

### 3.5 [P1] 홈페이지 JS — DB에서 노드/엣지 fetch

- 하드코딩 `var nodes=[...]` 제거
- `fetch('/api/v1/graph/data')` → 동적 노드/엣지 로드
- 나머지 렌더링/인터랙션 로직은 유지

---

## 4. 영향 범위

| 파일 | 변경 내용 |
|------|-----------|
| `models.py` | GraphNode, GraphEdge 모델 추가 |
| `api/v1/graph.py` | 새 API 모듈 (데이터 조회 + AI 생성) |
| `api/v1/__init__.py` | graph 모듈 등록 |
| `api/v1/admin.py` | 그래프 관리 엔드포인트 추가 |
| `templates/admin.html` | 그래프 관리 섹션 추가 |
| `templates/home.html` | 하드코딩 노드 → API fetch로 전환 |

---

## 5. 제외 사항 (YAGNI)
- 사용자별 개인 그래프 — 관리자 전용으로 충분
- 실시간 협업 편집 — 단일 관리자 사용
- 그래프 버전 히스토리 — 첫 버전에서 불필요
- 노드 드래그로 위치 저장 — 물리 시뮬레이션이 자동 배치
