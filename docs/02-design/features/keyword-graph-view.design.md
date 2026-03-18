# 인기질문 키워드 그래프뷰 Design Document

> **Summary**: 워드클라우드를 옵시디언 그래프뷰 스타일 D3.js force-directed 그래프로 전환하는 상세 설계
>
> **Project**: SafeFactory
> **Author**: Claude Code
> **Date**: 2026-03-18
> **Status**: Draft
> **Planning Doc**: [keyword-graph-view.plan.md](../../01-plan/features/keyword-graph-view.plan.md)

---

## 1. Overview

### 1.1 Design Goals

1. **관계 시각화**: 키워드 간 동시출현(co-occurrence) 관계를 노드-엣지 그래프로 표현
2. **옵시디언 UX**: force-directed layout, 호버 하이라이트, 줌/팬/드래그 인터랙션
3. **기존 패턴 준수**: Flask Blueprint API + Jinja2 인라인 스크립트 + `--sf-*` CSS 변수
4. **최소 변경**: 파일 3개 수정으로 기능 완성 (keyword_extractor.py, questions.py, wordcloud.html)

### 1.2 Design Principles

- **확장 최소화**: 기존 `extract_keywords()` 로직을 재사용하여 그래프 데이터 생성
- **점진적 전환**: 워드클라우드 ↔ 그래프뷰 토글로 양쪽 모두 유지
- **성능 제한**: 노드 80개 상한, 엣지 weight 임계값으로 렌더링 부하 제어

---

## 2. Architecture

### 2.1 Component Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                     Browser (wordcloud.html)                  │
│  ┌─────────────────┐    ┌──────────────────────────────────┐ │
│  │  Toggle Buttons  │    │  Questions Panel (기존 재활용)    │ │
│  │ [☁️ 클라우드]     │    │  - showQuestionsForKeyword()    │ │
│  │ [🔗 그래프뷰]    │    │  - navigateToQuestion()         │ │
│  └────────┬────────┘    └──────────────┬───────────────────┘ │
│           │                            │                      │
│  ┌────────▼────────────────────────────▼───────────────────┐ │
│  │  View Container                                          │ │
│  │  ┌─────────────────────┐  ┌────────────────────────────┐│ │
│  │  │ wordcloud2.js Canvas │  │  D3.js SVG Force Graph    ││ │
│  │  │ (기존, display:none) │  │  - 노드: circle + text    ││ │
│  │  │                      │  │  - 엣지: line             ││ │
│  │  │                      │  │  - zoom: d3.zoom()        ││ │
│  │  └─────────────────────┘  └────────────────────────────┘│ │
│  └──────────────────────────────────────────────────────────┘ │
│           │ fetch                                              │
└───────────┼──────────────────────────────────────────────────┘
            │
┌───────────▼──────────────────────────────────────────────────┐
│                    Flask API (api/v1/questions.py)             │
│  ┌────────────────────────┐  ┌─────────────────────────────┐ │
│  │ GET /questions/wordcloud│  │ GET /questions/graph (신규) │ │
│  │ → [{text, weight}]     │  │ → {nodes[], edges[]}        │ │
│  └───────────┬────────────┘  └──────────────┬──────────────┘ │
│              │                               │                │
│  ┌───────────▼───────────────────────────────▼──────────────┐│
│  │            services/keyword_extractor.py                  ││
│  │  extract_keywords()          extract_keyword_graph() (신규)│
│  └───────────────────────────────┬──────────────────────────┘│
│                                  │ query                      │
│  ┌───────────────────────────────▼──────────────────────────┐│
│  │            SQLite: shared_questions table                  ││
│  └───────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

```
사용자 → 뷰 토글(그래프) → fetch /api/v1/questions/graph
  → Flask: SharedQuestion 쿼리 (namespace, period 필터)
  → keyword_extractor.extract_keyword_graph()
    → 질문별 키워드 추출 (기존 regex 재사용)
    → 상위 N개 노드 선택
    → 동시출현 쌍 집계 → 엣지 생성
  → JSON 응답 {nodes, edges, total_questions}
  → D3 force simulation 초기화
  → SVG 렌더링 → 인터랙션 활성화
```

### 2.3 Dependencies

| Component | Depends On | Purpose |
|-----------|-----------|---------|
| `extract_keyword_graph()` | `_RE_KO`, `_RE_EN`, `STOPWORDS_KO/EN` (기존) | 키워드 추출 regex 재사용 |
| `/api/v1/questions/graph` | `SharedQuestion` 모델, `extract_keyword_graph()` | DB 쿼리 + 그래프 데이터 생성 |
| D3.js force graph (frontend) | d3 v7 CDN | SVG 렌더링, force simulation |
| Questions panel (frontend) | `/api/v1/questions/popular` (기존) | 키워드 클릭 시 질문 목록 |

---

## 3. Data Model

### 3.1 Graph Data Structure (API 응답용, DB 변경 없음)

```python
# 노드 (키워드)
@dataclass
class GraphNode:
    id: str          # 키워드 텍스트 (고유 식별자)
    text: str        # 표시용 텍스트 (= id)
    weight: int      # 출현 빈도 (like 가중치 포함)
    domains: list    # 출현 namespace 목록 (색상 분류용)

# 엣지 (동시출현 관계)
@dataclass
class GraphEdge:
    source: str      # 노드 A id
    target: str      # 노드 B id
    weight: int      # 동시출현 횟수
```

### 3.2 데이터 관계

```
[SharedQuestion] 1 ──── N [Keywords (추출)]
    │
    └── keywords per question → co-occurrence pairs → edges

[GraphNode] N ──── M [GraphNode]  (via GraphEdge)
```

### 3.3 Database Schema

**DB 스키마 변경 없음** — 기존 `shared_questions` 테이블을 그대로 사용. 그래프 데이터는 요청 시 메모리에서 계산.

---

## 4. API Specification

### 4.1 Endpoint List

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| GET | `/api/v1/questions/graph` | 키워드 그래프 데이터 | Not Required |
| GET | `/api/v1/questions/wordcloud` | 키워드 워드클라우드 데이터 (기존 유지) | Not Required |
| GET | `/api/v1/questions/popular` | 인기 질문 목록 (기존 유지) | Not Required |

### 4.2 Detailed Specification

#### `GET /api/v1/questions/graph`

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `namespace` | string | `""` (전체) | 도메인 필터 (semiconductor-v2, field-training, kosha, msds) |
| `period` | string | `"all"` | 기간 필터 (7d, 30d, all) |
| `node_limit` | int | `80` | 최대 노드 수 (1~100) |
| `min_edge_weight` | int | `2` | 최소 엣지 가중치 (이 미만 엣지 제거) |

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "nodes": [
      {
        "id": "반도체",
        "text": "반도체",
        "weight": 25,
        "domains": ["semiconductor-v2"]
      },
      {
        "id": "공정",
        "text": "공정",
        "weight": 18,
        "domains": ["semiconductor-v2", "field-training"]
      }
    ],
    "edges": [
      {
        "source": "반도체",
        "target": "공정",
        "weight": 12
      },
      {
        "source": "반도체",
        "target": "웨이퍼",
        "weight": 8
      }
    ],
    "total_questions": 150,
    "node_count": 45,
    "edge_count": 87
  },
  "meta": {
    "timestamp": "2026-03-18T12:00:00+00:00",
    "version": "v1",
    "request_id": null
  }
}
```

**Error Responses:**
- `500 Internal Server Error`: 그래프 데이터 생성 실패

**Rate Limit:** 30 per minute

---

## 5. UI/UX Design

### 5.1 Screen Layout

```
┌──────────────────────────────────────────────────┐
│  Header (base.html nav)                           │
├──────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────┐  │
│  │  ☁️ 질문 클라우드                           │  │
│  │  공유된 질문에서 추출한 핵심 키워드를 ...    │  │
│  └────────────────────────────────────────────┘  │
│                                                   │
│  ┌─────────────────┐  ┌────────────────────────┐ │
│  │ 도메인 필터 버튼  │  │ 기간 필터 버튼         │ │
│  └─────────────────┘  └────────────────────────┘ │
│                                                   │
│  ┌──────────────────┐                             │
│  │ [☁️ 클라우드] [🔗 그래프] ← 뷰 토글          │ │
│  └──────────────────┘                             │
│                                                   │
│  ┌────────────────────────────────────────────┐  │
│  │                                            │  │
│  │   ○───○     ○                              │  │
│  │  /   │    / \                              │  │
│  │ ○    ○───○   ○    D3.js Force Graph        │  │
│  │  \  /     \                                │  │
│  │   ○        ○                               │  │
│  │           SVG (min-height: 500px)          │  │
│  └────────────────────────────────────────────┘  │
│  총 150개 질문에서 45개 키워드, 87개 연결 추출    │
│                                                   │
│  ┌────────────────────────────────────────────┐  │
│  │  "반도체" 관련 질문        [✕]             │  │
│  │  ┌──────────────────────────────────────┐  │  │
│  │  │ 반도체 공정에서 웨이퍼 세정 방법은? ♥3│  │  │
│  │  │ 반도체 식각 공정 온도 관리법은?    ♥2│  │  │
│  │  └──────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────┘  │
│                                                   │
│  Footer (base.html)                               │
└──────────────────────────────────────────────────┘
```

### 5.2 User Flow

```
페이지 로드 → 기본 뷰(그래프뷰) 렌더링
  ├─ [도메인 필터 클릭] → 그래프 리로드 (해당 namespace)
  ├─ [기간 필터 클릭] → 그래프 리로드 (해당 period)
  ├─ [뷰 토글 클릭] → 워드클라우드 ↔ 그래프 전환
  ├─ [노드 호버] → 해당 노드 + 연결 노드 하이라이트, 나머지 fade
  ├─ [노드 클릭] → 하단 질문 패널 표시
  ├─ [질문 항목 클릭] → 해당 도메인 페이지로 이동
  ├─ [마우스 휠] → 줌 인/아웃
  ├─ [배경 드래그] → 팬 이동
  └─ [노드 드래그] → 노드 위치 변경 (simulation reheat)
```

### 5.3 노드 시각 디자인 (옵시디언 스타일)

| 속성 | 값 | 비고 |
|------|-----|------|
| 노드 형태 | 원형 (circle) | |
| 노드 크기 | `radius = 4 + (weight / maxWeight) * 20` | 최소 4px, 최대 24px |
| 노드 색상 | 도메인별 `--sf-*` 변수 매핑 | 아래 매핑 테이블 참조 |
| 노드 라벨 | 키워드 텍스트 (weight 상위 30개만) | fontSize 10~14px |
| 엣지 형태 | 직선 (line) | |
| 엣지 두께 | `0.5 + (weight / maxEdgeWeight) * 3` | 최소 0.5px, 최대 3.5px |
| 엣지 색상 | `var(--sf-border)` (기본), 하이라이트 시 `var(--sf-text-3)` | |
| 배경 | 투명 (`.wc-canvas-wrap` 카드 배경 활용) | |

**도메인 → 노드 색상 매핑:**

| 도메인 | CSS 변수 | 색상 |
|--------|----------|------|
| semiconductor-v2 | `--sf-blue` | #3b82f6 |
| field-training | `--sf-amber` | #f59e0b |
| kosha | `--sf-green` | #10b981 |
| msds | `--sf-violet` | #8b5cf6 |
| 복수 도메인 | `--sf-text-2` | #475569 |

**호버 하이라이트 동작:**
- 호버된 노드: opacity 1.0, stroke 2px `--sf-text-1`
- 직접 연결 노드: opacity 1.0
- 연결 엣지: opacity 1.0, 색상 강조
- 비연결 노드/엣지: opacity 0.08

---

## 6. Backend Implementation Detail

### 6.1 `extract_keyword_graph()` 함수 설계

**파일**: `services/keyword_extractor.py`

```python
from itertools import combinations

def extract_keyword_graph(questions, node_limit=80, min_edge_weight=2):
    """Extract keyword co-occurrence graph from shared questions.

    Args:
        questions: list of (query: str, like_count: int, namespace: str) tuples
        node_limit: max number of keyword nodes
        min_edge_weight: minimum co-occurrence count to include an edge

    Returns:
        dict with 'nodes' and 'edges' lists
    """
    counter = Counter()         # keyword → total weight
    domain_map = {}             # keyword → set of namespaces
    per_question_keywords = []  # list of (keyword_set, weight)

    for query, like_count, namespace in questions:
        weight = 1 + (like_count or 0)
        tokens = set()

        for match in _RE_KO.findall(query):
            if match not in STOPWORDS_KO and len(match) >= 2:
                tokens.add(match)

        for match in _RE_EN.findall(query):
            upper = match.upper()
            if match.lower() not in STOPWORDS_EN and len(match) >= 3:
                tokens.add(upper)

        for token in tokens:
            counter[token] += weight
            if token not in domain_map:
                domain_map[token] = set()
            if namespace:
                domain_map[token].add(namespace)

        if len(tokens) >= 2:
            per_question_keywords.append((tokens, weight))

    # Top N nodes by weight
    top_keywords = {kw for kw, _ in counter.most_common(node_limit) if counter[kw] >= 2}

    # Build nodes
    nodes = [
        {
            "id": kw,
            "text": kw,
            "weight": counter[kw],
            "domains": sorted(domain_map.get(kw, set())),
        }
        for kw in top_keywords
    ]
    nodes.sort(key=lambda n: n["weight"], reverse=True)

    # Build edges from co-occurrence
    edge_counter = Counter()
    for tokens, weight in per_question_keywords:
        relevant = tokens & top_keywords
        for a, b in combinations(sorted(relevant), 2):
            edge_counter[(a, b)] += weight

    edges = [
        {"source": a, "target": b, "weight": w}
        for (a, b), w in edge_counter.most_common()
        if w >= min_edge_weight
    ]

    return {"nodes": nodes, "edges": edges}
```

**복잡도 분석:**
- 질문 N개, 질문당 키워드 평균 k개
- 키워드 추출: O(N)
- 엣지 계산: O(N · k²) — k ≤ 10이므로 실질적 O(N)
- 질문 1000개 기준 < 50ms 예상

### 6.2 API 엔드포인트 구현

**파일**: `api/v1/questions.py`에 추가

```python
@v1_bp.route('/questions/graph', methods=['GET'])
@rate_limit("30 per minute")
def api_question_graph():
    """Get keyword co-occurrence graph data for graph view."""
    try:
        from services.keyword_extractor import extract_keyword_graph

        namespace = request.args.get('namespace', '').strip()
        period = request.args.get('period', 'all').strip()
        node_limit = min(max(1, request.args.get('node_limit', 80, type=int)), 100)
        min_edge_weight = max(1, request.args.get('min_edge_weight', 2, type=int))

        q = db.session.query(
            SharedQuestion.query,
            SharedQuestion.like_count,
            SharedQuestion.namespace,
        ).filter_by(is_hidden=False)

        if namespace:
            q = q.filter(SharedQuestion.namespace == namespace)

        if period == '7d':
            since = datetime.now(timezone.utc) - timedelta(days=7)
            q = q.filter(SharedQuestion.created_at >= since)
        elif period == '30d':
            since = datetime.now(timezone.utc) - timedelta(days=30)
            q = q.filter(SharedQuestion.created_at >= since)

        rows = q.all()
        graph = extract_keyword_graph(
            [(row.query, row.like_count, row.namespace) for row in rows],
            node_limit=node_limit,
            min_edge_weight=min_edge_weight,
        )

        return success_response(data={
            'nodes': graph['nodes'],
            'edges': graph['edges'],
            'total_questions': len(rows),
            'node_count': len(graph['nodes']),
            'edge_count': len(graph['edges']),
        })
    except Exception:
        logging.exception('[Question] Graph data failed')
        return error_response('그래프 데이터 조회 중 오류가 발생했습니다.', 500)
```

---

## 7. Frontend Implementation Detail

### 7.1 CDN 의존성

```html
<!-- D3.js v7 (전체 번들, minified ~90KB gzipped) -->
<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
```

`{% block head_scripts %}`에 기존 wordcloud2.js와 함께 추가.

### 7.2 HTML 구조 변경 (wordcloud.html)

```html
<!-- 뷰 토글 버튼 (필터 아래에 추가) -->
<div class="wc-view-toggle">
    <div class="wc-filter-group">
        <button class="wc-filter-btn" data-view="cloud" onclick="setView(this)">☁️ 클라우드</button>
        <button class="wc-filter-btn active" data-view="graph" onclick="setView(this)">🔗 그래프</button>
    </div>
</div>

<!-- 기존 워드클라우드 캔버스 (토글 대상) -->
<div class="wc-canvas-wrap" id="cloudWrap" style="display:none">
    <canvas id="wordcloudCanvas"></canvas>
</div>

<!-- 그래프뷰 SVG (신규) -->
<div class="wc-canvas-wrap" id="graphWrap">
    <svg id="graphSvg" style="width:100%;height:500px;"></svg>
</div>
```

### 7.3 CSS 추가 (kg- 접두사)

```css
/* Graph view styles */
.wc-view-toggle {
    display: flex;
    justify-content: center;
    margin: 10px 0;
}

#graphSvg {
    cursor: grab;
}
#graphSvg:active {
    cursor: grabbing;
}

.kg-node {
    cursor: pointer;
    transition: opacity 0.2s;
}
.kg-node circle {
    stroke: var(--sf-card-bg);
    stroke-width: 1.5px;
}
.kg-node text {
    font-size: 11px;
    fill: var(--sf-text-2);
    pointer-events: none;
    user-select: none;
}
.kg-edge {
    stroke: var(--sf-border);
    stroke-opacity: 0.6;
    transition: opacity 0.2s, stroke 0.2s;
}

/* Hover state: faded */
.kg-faded {
    opacity: 0.08;
}
.kg-highlighted circle {
    stroke: var(--sf-text-1);
    stroke-width: 2.5px;
}
.kg-highlighted-edge {
    stroke: var(--sf-text-3);
    stroke-opacity: 1;
}
```

### 7.4 D3.js Force Graph 핵심 로직

```javascript
let _simulation = null;
let _graphData = null;

async function loadGraphView() {
    const wrap = document.getElementById('graphWrap');
    const svg = d3.select('#graphSvg');
    svg.selectAll('*').remove();

    const params = new URLSearchParams({ period: _currentPeriod, node_limit: '80' });
    if (_currentNs) params.set('namespace', _currentNs);

    const res = await fetch(`/api/v1/questions/graph?${params}`);
    const json = await res.json();

    if (!json.success || !json.data.nodes.length) {
        wrap.innerHTML = '<div class="wc-empty">...</div>';
        return;
    }

    _graphData = json.data;
    const { nodes, edges } = _graphData;
    const width = wrap.clientWidth;
    const height = 500;

    // Scale: node radius
    const maxWeight = d3.max(nodes, d => d.weight);
    const rScale = d3.scaleLinear()
        .domain([1, maxWeight])
        .range([4, 24]);

    // Scale: edge width
    const maxEdgeW = d3.max(edges, d => d.weight) || 1;
    const ewScale = d3.scaleLinear()
        .domain([1, maxEdgeW])
        .range([0.5, 3.5]);

    // Domain color map
    const domainColor = {
        'semiconductor-v2': 'var(--sf-blue)',
        'field-training':   'var(--sf-amber)',
        'kosha':            'var(--sf-green)',
        'msds':             'var(--sf-violet)',
    };
    function nodeColor(d) {
        if (d.domains.length === 1) return domainColor[d.domains[0]] || 'var(--sf-text-3)';
        return 'var(--sf-text-2)';
    }

    // Zoom
    const g = svg.append('g');
    svg.call(d3.zoom()
        .scaleExtent([0.3, 5])
        .on('zoom', (event) => g.attr('transform', event.transform))
    );

    svg.attr('viewBox', `0 0 ${width} ${height}`);

    // Edges
    const edgeEls = g.append('g')
        .selectAll('line')
        .data(edges)
        .join('line')
        .attr('class', 'kg-edge')
        .attr('stroke-width', d => ewScale(d.weight));

    // Nodes
    const nodeEls = g.append('g')
        .selectAll('g')
        .data(nodes)
        .join('g')
        .attr('class', 'kg-node');

    nodeEls.append('circle')
        .attr('r', d => rScale(d.weight))
        .attr('fill', nodeColor);

    // Labels (top 30 by weight only)
    const labelThreshold = nodes.length > 30
        ? nodes[29].weight
        : 0;

    nodeEls.filter(d => d.weight >= labelThreshold)
        .append('text')
        .text(d => d.text)
        .attr('dy', d => rScale(d.weight) + 12)
        .attr('text-anchor', 'middle');

    // Tooltip for all nodes
    nodeEls.append('title').text(d => `${d.text} (${d.weight})`);

    // Build adjacency for hover highlight
    const adjacency = new Map();
    nodes.forEach(n => adjacency.set(n.id, new Set()));
    edges.forEach(e => {
        adjacency.get(e.source.id || e.source)?.add(e.target.id || e.target);
        adjacency.get(e.target.id || e.target)?.add(e.source.id || e.source);
    });

    // Hover: Obsidian-style highlight
    nodeEls.on('mouseenter', (event, d) => {
        const connected = adjacency.get(d.id);
        nodeEls.classed('kg-faded', o => o.id !== d.id && !connected.has(o.id));
        nodeEls.filter(o => o.id === d.id || connected.has(o.id))
            .classed('kg-highlighted', true);
        edgeEls.classed('kg-faded', e => {
            const s = e.source.id || e.source;
            const t = e.target.id || e.target;
            return s !== d.id && t !== d.id;
        });
        edgeEls.filter(e => {
            const s = e.source.id || e.source;
            const t = e.target.id || e.target;
            return s === d.id || t === d.id;
        }).classed('kg-highlighted-edge', true);
    });

    nodeEls.on('mouseleave', () => {
        nodeEls.classed('kg-faded', false).classed('kg-highlighted', false);
        edgeEls.classed('kg-faded', false).classed('kg-highlighted-edge', false);
    });

    // Click: show questions
    nodeEls.on('click', (event, d) => {
        event.stopPropagation();
        showQuestionsForKeyword(d.text);
    });

    // Drag
    nodeEls.call(d3.drag()
        .on('start', (event, d) => {
            if (!event.active) _simulation.alphaTarget(0.3).restart();
            d.fx = d.x; d.fy = d.y;
        })
        .on('drag', (event, d) => {
            d.fx = event.x; d.fy = event.y;
        })
        .on('end', (event, d) => {
            if (!event.active) _simulation.alphaTarget(0);
            d.fx = null; d.fy = null;
        })
    );

    // Force simulation
    _simulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(edges)
            .id(d => d.id)
            .distance(d => 80 / Math.sqrt(d.weight))
        )
        .force('charge', d3.forceManyBody().strength(-120))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collide', d3.forceCollide()
            .radius(d => rScale(d.weight) + 5)
        )
        .on('tick', () => {
            edgeEls
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
            nodeEls.attr('transform', d => `translate(${d.x},${d.y})`);
        });

    // Stats
    document.getElementById('wcStats').textContent =
        `총 ${json.data.total_questions}개 질문에서 ${json.data.node_count}개 키워드, ${json.data.edge_count}개 연결 추출`;
}
```

### 7.5 뷰 토글 로직

```javascript
let _currentView = 'graph';  // 기본 뷰: 그래프

function setView(btn) {
    document.querySelectorAll('.wc-view-toggle .wc-filter-btn')
        .forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    _currentView = btn.dataset.view;

    const cloudWrap = document.getElementById('cloudWrap');
    const graphWrap = document.getElementById('graphWrap');

    if (_currentView === 'cloud') {
        cloudWrap.style.display = '';
        graphWrap.style.display = 'none';
        loadWordCloud();  // 기존 함수
    } else {
        cloudWrap.style.display = 'none';
        graphWrap.style.display = '';
        loadGraphView();
    }
}

// 기존 loadWordCloud에서 호출하는 setDomain/setPeriod에도 _currentView 체크 추가
function reloadCurrentView() {
    closeQuestionsPanel();
    if (_currentView === 'graph') {
        loadGraphView();
    } else {
        loadWordCloud();
    }
}
```

### 7.6 모바일 반응형

```css
@media (max-width: 768px) {
    #graphSvg { height: 350px; }
    .kg-node text { font-size: 9px; }
    .wc-view-toggle { margin: 6px 0; }
}
```

터치 인터랙션은 D3.js zoom이 자동 지원 (pinch-to-zoom, touch-drag).
그래프 영역에 `touch-action: none`을 설정하여 스크롤과 분리.

---

## 8. Error Handling

### 8.1 Error Cases

| Code | Situation | Handling |
|------|-----------|----------|
| 500 | `extract_keyword_graph()` 예외 | 로그 + `error_response` 반환 |
| - | 질문 0개 (빈 데이터) | API: 빈 nodes/edges 반환 → 프론트: empty state 표시 |
| - | D3.js CDN 로드 실패 | `typeof d3 === 'undefined'` 체크 → 워드클라우드 폴백 |
| - | 네트워크 에러 (fetch 실패) | try/catch → empty state 메시지 |

### 8.2 Error Response Format

기존 `api/response.py`의 `error_response()` 패턴 준수:
```json
{
  "success": false,
  "error": "그래프 데이터 조회 중 오류가 발생했습니다.",
  "meta": { "timestamp": "...", "version": "v1" }
}
```

---

## 9. Security Considerations

- [x] Input validation: `node_limit`, `min_edge_weight` 범위 제한 (서버측)
- [x] XSS 방지: D3.js `.text()` 메서드는 자동 이스케이프 (`.html()` 미사용)
- [x] Rate limiting: `@rate_limit("30 per minute")`
- [x] SQL Injection 방지: SQLAlchemy ORM 사용 (raw query 없음)
- [ ] 민감 데이터 없음: 공개 질문 텍스트만 처리

---

## 10. Test Plan

### 10.1 Test Scope

| Type | Target | Method |
|------|--------|--------|
| 수동 테스트 | API 응답 검증 | curl / 브라우저 DevTools |
| 수동 테스트 | 그래프 렌더링 + 인터랙션 | 브라우저 시각 확인 |
| 수동 테스트 | 모바일 반응형 | Chrome DevTools 모바일 모드 |
| 수동 테스트 | 빈 데이터 처리 | namespace 필터로 질문 0개 상태 유도 |

### 10.2 Test Cases

- [ ] Happy path: 질문 50개+ → 그래프 노드/엣지 정상 렌더링
- [ ] 빈 데이터: 질문 0개 → "공유된 질문이 부족하여..." empty state
- [ ] 필터 전환: 도메인/기간 변경 → 그래프 리로드
- [ ] 뷰 토글: 클라우드 ↔ 그래프 전환 → 각 뷰 정상 동작
- [ ] 노드 호버: 연결 노드 하이라이트, 나머지 fade
- [ ] 노드 클릭: 질문 패널 표시
- [ ] 줌/팬: 마우스 휠 줌, 배경 드래그 팬
- [ ] 노드 드래그: 노드 위치 이동, simulation 재시작
- [ ] 모바일: 핀치 줌, 터치 드래그 동작

---

## 11. Implementation Guide

### 11.1 File Structure (변경 대상)

```
services/
└── keyword_extractor.py    # extract_keyword_graph() 함수 추가

api/v1/
└── questions.py            # api_question_graph() 엔드포인트 추가

templates/
└── wordcloud.html          # D3 CDN + SVG + 토글 UI + 그래프 JS 추가
```

### 11.2 Implementation Order

1. [ ] **Backend 함수**: `keyword_extractor.py`에 `extract_keyword_graph()` 추가
2. [ ] **API 엔드포인트**: `questions.py`에 `/questions/graph` 추가
3. [ ] **API 검증**: curl로 응답 확인
4. [ ] **Frontend HTML**: 토글 버튼 + SVG 컨테이너 추가
5. [ ] **Frontend CSS**: `kg-*` 클래스 스타일 추가
6. [ ] **Frontend JS**: D3 force graph 렌더링 + 인터랙션
7. [ ] **뷰 토글**: 워드클라우드 ↔ 그래프 전환 로직
8. [ ] **모바일 대응**: 반응형 CSS + touch-action
9. [ ] **통합 테스트**: 전체 플로우 수동 확인

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-03-18 | Initial draft | Claude Code |
