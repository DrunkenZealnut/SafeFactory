# GraphRAG Community Layer 설계서

## 참조 문서
- Plan: `docs/01-plan/features/graphrag-community.plan.md`
- 선행 기능: `docs/02-design/features/graphrag.design.md` (GraphRAG v1.0)

---

## 1. 데이터 모델

### 1.1 신규 테이블

```python
# models.py 추가 (기존 KGEntityChunk 아래)

class KGCommunity(db.Model):
    """Knowledge Graph community cluster."""

    __tablename__ = 'kg_communities'
    __table_args__ = (
        db.Index('ix_kg_comm_ns', 'namespace'),
        db.Index('ix_kg_comm_ns_level', 'namespace', 'level'),
    )

    id = db.Column(db.Integer, primary_key=True)
    namespace = db.Column(db.String(100), nullable=False)
    community_id = db.Column(db.Integer, nullable=False)  # Leiden cluster number
    level = db.Column(db.Integer, nullable=False, default=0)  # hierarchy level (v1: 0 only)
    title = db.Column(db.String(300))  # LLM-generated title
    summary = db.Column(db.Text)  # LLM-generated summary
    member_count = db.Column(db.Integer, default=0)
    created_at = db.Column(
        db.DateTime, nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    members = db.relationship(
        'KGCommunityMember', backref='community',
        lazy='dynamic', cascade='all, delete-orphan',
    )


class KGCommunityMember(db.Model):
    """Community-to-entity membership mapping."""

    __tablename__ = 'kg_community_members'
    __table_args__ = (
        db.Index('ix_kg_cm_community', 'community_id'),
        db.Index('ix_kg_cm_entity', 'entity_id'),
    )

    id = db.Column(db.Integer, primary_key=True)
    community_id = db.Column(
        db.Integer, db.ForeignKey('kg_communities.id', ondelete='CASCADE'),
        nullable=False,
    )
    entity_id = db.Column(
        db.Integer, db.ForeignKey('kg_entities.id', ondelete='CASCADE'),
        nullable=False,
    )
    namespace = db.Column(db.String(100), nullable=False)
```

### 1.2 기존 테이블 변경

변경 없음. KGEntity, KGRelation, KGEntityChunk는 그대로 유지.

---

## 2. 설정 확장

### 2.1 graph_config.py 확장

```python
# services/graph_config.py — 각 도메인 설정에 community 블록 추가

GRAPH_CONFIG = {
    'semiconductor-v2': {
        'enabled': True,
        'hop_depth': 2,
        'graph_weight': 0.3,
        'max_graph_results': 10,
        'entity_types': ['공정', '장비', '물질', '안전규정'],
        # 신규: 커뮤니티 설정
        'community': {
            'enabled': True,
            'resolution': 1.0,          # Leiden resolution (높을수록 작은 커뮤니티)
            'min_community_size': 3,     # 최소 멤버 수
            'max_summary_tokens': 500,   # 요약 최대 토큰
        },
    },
    'laborlaw': {
        # ... 기존 설정 유지 ...
        'community': {
            'enabled': True,
            'resolution': 0.8,
            'min_community_size': 2,
            'max_summary_tokens': 400,
        },
    },
    'kosha': {
        # ... 기존 설정 유지 ...
        'community': {
            'enabled': True,
            'resolution': 1.0,
            'min_community_size': 3,
            'max_summary_tokens': 500,
        },
    },
    'msds': {
        'enabled': False,
        # community도 disabled
    },
    'field-training': {
        # ... 기존 설정 유지 ...
        'community': {
            'enabled': True,
            'resolution': 0.8,
            'min_community_size': 2,
            'max_summary_tokens': 400,
        },
    },
}
```

---

## 3. 오프라인 파이프라인: CommunityBuilder

### 3.1 파일: `src/community_builder.py`

**역할**: 기존 KG 데이터로부터 커뮤니티 감지 + 요약 생성

```python
class CommunityBuilder:
    """Builds community clusters from existing KG entities/relations."""

    def __init__(self, namespace: str, gemini_client=None):
        self.namespace = namespace
        self.config = get_graph_config(namespace).get('community', {})
        self.gemini = gemini_client or get_gemini_client()

    def build(self) -> dict:
        """Full pipeline: detect → summarize → save."""
        # Step 1: Load KG as networkx graph
        G = self._load_kg_graph()
        if G.number_of_nodes() < self.config.get('min_community_size', 3):
            return {'communities': 0, 'skipped': 'insufficient_nodes'}

        # Step 2: Leiden community detection
        communities = self._detect_communities(G)

        # Step 3: Filter small communities
        communities = self._filter_communities(communities)

        # Step 4: Save to DB
        saved = self._save_communities(communities, G)

        # Step 5: Generate summaries (LLM)
        summarized = self._generate_summaries()

        return {
            'communities': saved,
            'summarized': summarized,
            'total_nodes': G.number_of_nodes(),
        }

    def reset(self):
        """Delete all community data for this namespace."""
        KGCommunityMember.query.filter_by(namespace=self.namespace).delete()
        KGCommunity.query.filter_by(namespace=self.namespace).delete()
        db.session.commit()
```

### 3.2 Step 1: KG → NetworkX 그래프 로드

```python
def _load_kg_graph(self) -> nx.Graph:
    """Load KGEntity+KGRelation into a networkx graph."""
    G = nx.Graph()

    entities = KGEntity.query.filter_by(namespace=self.namespace).all()
    for e in entities:
        G.add_node(e.id, name=e.name, entity_type=e.entity_type,
                   description=e.description or '')

    relations = KGRelation.query.filter_by(namespace=self.namespace).all()
    for r in relations:
        if G.has_node(r.source_id) and G.has_node(r.target_id):
            G.add_edge(r.source_id, r.target_id,
                       relation_type=r.relation_type,
                       confidence=r.confidence)

    return G
```

### 3.3 Step 2: Leiden 커뮤니티 감지

```python
def _detect_communities(self, G: nx.Graph) -> dict[int, list[int]]:
    """Run Leiden algorithm. Returns {community_id: [entity_ids]}."""
    resolution = self.config.get('resolution', 1.0)

    try:
        import igraph as ig
        import leidenalg

        # NetworkX → igraph 변환
        ig_graph = ig.Graph.from_networkx(G)
        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.ModularityVertexPartition,
            resolution_parameter=resolution,
        )
        # igraph vertex index → original entity_id 매핑
        node_list = list(G.nodes())
        communities = {}
        for comm_id, members in enumerate(partition):
            entity_ids = [node_list[idx] for idx in members]
            communities[comm_id] = entity_ids
        return communities

    except ImportError:
        # Fallback: NetworkX 내장 Louvain
        logger.warning("[CommunityBuilder] leidenalg not installed, using Louvain fallback")
        from networkx.algorithms.community import louvain_communities
        partitions = louvain_communities(G, resolution=resolution)
        return {i: list(members) for i, members in enumerate(partitions)}
```

### 3.4 Step 3: 소규모 커뮤니티 필터

```python
def _filter_communities(self, communities: dict[int, list[int]]) -> dict[int, list[int]]:
    """Filter out communities smaller than min_community_size."""
    min_size = self.config.get('min_community_size', 3)
    return {cid: members for cid, members in communities.items()
            if len(members) >= min_size}
```

### 3.5 Step 4: DB 저장

```python
def _save_communities(self, communities: dict[int, list[int]], G: nx.Graph) -> int:
    """Persist communities to KGCommunity + KGCommunityMember."""
    # Clear existing
    self.reset()

    count = 0
    for comm_id, entity_ids in communities.items():
        community = KGCommunity(
            namespace=self.namespace,
            community_id=comm_id,
            level=0,
            member_count=len(entity_ids),
        )
        db.session.add(community)
        db.session.flush()  # get id

        for eid in entity_ids:
            db.session.add(KGCommunityMember(
                community_id=community.id,
                entity_id=eid,
                namespace=self.namespace,
            ))
        count += 1

    db.session.commit()
    return count
```

### 3.6 Step 5: LLM 요약 생성

```python
_SUMMARY_PROMPT = """다음은 Knowledge Graph의 한 커뮤니티(클러스터)에 속하는 엔티티와 관계 정보야.
이 커뮤니티를 대표하는 제목(20자 이내)과 요약(3-5문장)을 생성해줘.

## 엔티티
{entities}

## 관계
{relations}

## JSON 출력 형식
{{"title": "커뮤니티 제목", "summary": "커뮤니티 요약..."}}"""


def _generate_summaries(self) -> int:
    """Generate LLM summaries for all communities in this namespace."""
    communities = KGCommunity.query.filter_by(
        namespace=self.namespace, summary=None,
    ).all()

    summarized = 0
    for comm in communities:
        try:
            member_entities = (
                db.session.query(KGEntity)
                .join(KGCommunityMember, KGCommunityMember.entity_id == KGEntity.id)
                .filter(KGCommunityMember.community_id == comm.id)
                .all()
            )
            entity_ids = [e.id for e in member_entities]

            # Build entity text
            entity_text = '\n'.join(
                f"- {e.name} ({e.entity_type}): {(e.description or '')[:200]}"
                for e in member_entities
            )

            # Build relation text (relations between community members)
            relations = (
                db.session.query(KGRelation)
                .filter(
                    KGRelation.namespace == self.namespace,
                    KGRelation.source_id.in_(entity_ids),
                    KGRelation.target_id.in_(entity_ids),
                )
                .all()
            )
            entity_name_map = {e.id: e.name for e in member_entities}
            relation_text = '\n'.join(
                f"- {entity_name_map.get(r.source_id, '?')} --[{r.relation_type}]--> "
                f"{entity_name_map.get(r.target_id, '?')}"
                for r in relations
            )

            prompt = _SUMMARY_PROMPT.format(
                entities=entity_text or '(없음)',
                relations=relation_text or '(없음)',
            )

            response = self.gemini.models.generate_content(
                model='gemini-2.0-flash',
                contents=[{'role': 'user', 'parts': [{'text': prompt}]}],
                config={'response_mime_type': 'application/json', 'temperature': 0.3},
            )

            result = json.loads(response.text.strip().strip('`').lstrip('json\n'))
            comm.title = result.get('title', '')[:300]
            comm.summary = result.get('summary', '')
            db.session.commit()
            summarized += 1

        except Exception:
            logger.exception("[CommunityBuilder] Summary failed for community %d", comm.id)

    return summarized
```

---

## 4. 온라인 검색: CommunitySearcher

### 4.1 파일: `services/community_searcher.py`

**역할**: Global Search — 커뮤니티 요약 대상 map-reduce 검색

```python
class CommunitySearcher:
    """Global search using community summaries with map-reduce."""

    def __init__(self):
        self._summary_cache: dict[str, list[dict]] = {}
        self._cache_lock = threading.Lock()

    def search(self, query: str, namespace: str,
               max_communities: int = 10) -> dict:
        """Run global search on community summaries.

        Returns:
            {
                'answer_context': str,    # LLM에 전달할 종합 컨텍스트
                'communities_used': int,  # 사용된 커뮤니티 수
                'community_titles': list, # 사용된 커뮤니티 제목 목록
            }
        """
        # Step 1: 관련 커뮤니티 선택 (키워드 매칭)
        relevant = self._select_relevant_communities(query, namespace, max_communities)
        if not relevant:
            return {'answer_context': '', 'communities_used': 0, 'community_titles': []}

        # Step 2: Map — 각 커뮤니티 요약에서 관련 정보 추출
        mapped = self._map_communities(query, relevant)

        # Step 3: Reduce — 매핑 결과 종합
        context = self._reduce_results(query, mapped)

        return {
            'answer_context': context,
            'communities_used': len(relevant),
            'community_titles': [c['title'] for c in relevant],
        }
```

### 4.2 커뮤니티 선택 (키워드 기반)

```python
def _select_relevant_communities(self, query: str, namespace: str,
                                  max_count: int) -> list[dict]:
    """Select communities relevant to the query using keyword overlap."""
    summaries = self._load_summaries(namespace)
    if not summaries:
        return []

    query_lower = query.lower()
    scored = []
    for comm in summaries:
        # Score = keyword overlap between query and (title + summary)
        text = f"{comm['title']} {comm['summary']}".lower()
        # Simple word overlap scoring
        query_words = set(query_lower.split())
        text_words = set(text.split())
        overlap = len(query_words & text_words)
        if overlap > 0 or len(query_words) <= 2:
            # Short queries (1-2 words) → include all communities
            scored.append((overlap, comm))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [comm for _, comm in scored[:max_count]]


def _load_summaries(self, namespace: str) -> list[dict]:
    """Load community summaries with caching."""
    with self._cache_lock:
        if namespace in self._summary_cache:
            return self._summary_cache[namespace]

    communities = (
        KGCommunity.query
        .filter_by(namespace=namespace)
        .filter(KGCommunity.summary.isnot(None))
        .all()
    )
    result = [
        {
            'id': c.id,
            'title': c.title or '',
            'summary': c.summary or '',
            'member_count': c.member_count,
        }
        for c in communities
    ]

    with self._cache_lock:
        self._summary_cache[namespace] = result
    return result
```

### 4.3 Map 단계

```python
def _map_communities(self, query: str, communities: list[dict]) -> list[str]:
    """Extract relevant information from each community summary."""
    # 커뮤니티 수가 적으면 (≤5) 요약을 직접 사용 (LLM 호출 불필요)
    if len(communities) <= 5:
        return [
            f"[{c['title']}] {c['summary']}"
            for c in communities
        ]

    # 커뮤니티 수가 많으면 LLM으로 관련 부분만 추출
    mapped = []
    for c in communities:
        mapped.append(f"[{c['title']}] {c['summary']}")
    return mapped
```

### 4.4 Reduce 단계

```python
def _reduce_results(self, query: str, mapped: list[str]) -> str:
    """Combine mapped results into a unified context string."""
    # 단순 결합 (LLM 답변 생성은 rag_pipeline이 담당)
    context_parts = []
    for i, text in enumerate(mapped, 1):
        context_parts.append(f"### 커뮤니티 {i}\n{text}")
    return '\n\n'.join(context_parts)
```

### 4.5 캐시 무효화

```python
def invalidate_cache(self, namespace: str | None = None):
    """Clear summary cache."""
    with self._cache_lock:
        if namespace:
            self._summary_cache.pop(namespace, None)
        else:
            self._summary_cache.clear()
```

---

## 5. 쿼리 라우터 확장

### 5.1 query_router.py — overview 유형 추가

```python
# 기존 4개 유형에 'overview' 추가
QUERY_TYPE_CONFIG = {
    'factual': { ... },       # 기존 유지
    'procedural': { ... },    # 기존 유지
    'comparison': { ... },    # 기존 유지
    'calculation': { ... },   # 기존 유지
    'overview': {
        'top_k_mult': 3,
        'use_hyde': False,
        'use_multi_query': False,
        'rerank_weight': 0.80,
        'use_global_search': True,  # 신규: Global Search 활성화 플래그
    },
}

# 신규 패턴 (기존 패턴 위에 추가, 우선순위 최상)
_OVERVIEW_PATTERNS = [
    re.compile(r'(?:전반|개요|전체|총정리|요약|핵심|주요|개괄)'),
    re.compile(r'(?:알려줘|설명해|정리해).*(?:전반|전체|종합)'),
    re.compile(r'(?:가장|중요한|핵심).*(?:\d+가지|\d+개)'),
]


def classify_query_type(query: str) -> str:
    # 신규: overview 패턴을 calculation 앞에 체크
    for pattern in _OVERVIEW_PATTERNS:
        if pattern.search(query):
            return 'overview'
    # 기존 로직 유지
    for pattern in _CALCULATION_PATTERNS:
        if pattern.search(query):
            return 'calculation'
    for pattern in _COMPARISON_PATTERNS:
        if pattern.search(query):
            return 'comparison'
    for pattern in _PROCEDURAL_PATTERNS:
        if pattern.search(query):
            return 'procedural'
    return 'factual'
```

### 5.2 패턴 매칭 예시

| 쿼리 | 분류 |
|------|------|
| "반도체 공정 전반을 설명해줘" | `overview` |
| "산업안전의 핵심 5가지" | `overview` |
| "CVD 공정에서 안전 주의사항은?" | `factual` (기존) |
| "퇴직금 계산해줘" | `calculation` (기존) |

---

## 6. RAG 파이프라인 통합

### 6.1 rag_pipeline.py — Global/Local 분기

기존 Phase 3 (Graph Enrichment) 이후, Phase 3.5로 Global Search를 삽입합니다.

```python
# Phase 3: Graph Enrichment (기존 — 변경 없음)
# ...

# ========================================
# Phase 3.5: Global Search (Community)
# ========================================
_t0 = time.perf_counter()
_global_context = ''
if route_cfg.get('use_global_search') and use_enhancement:
    try:
        from services.graph_config import get_graph_config
        _comm_cfg = get_graph_config(namespace).get('community', {})
        if _comm_cfg.get('enabled'):
            from services.singletons import get_community_searcher
            _cs = get_community_searcher()
            _global_result = _cs.search(
                query=search_query,
                namespace=namespace,
                max_communities=10,
            )
            if _global_result['communities_used'] > 0:
                _global_context = _global_result['answer_context']
                logging.info(
                    "[Global Search] Used %d communities: %s",
                    _global_result['communities_used'],
                    _global_result['community_titles'],
                )
    except Exception as _ce:
        logging.warning("[Global Search] Failed (fallback to local): %s", _ce)

_timings['phase3_5_global_ms'] = round((time.perf_counter() - _t0) * 1000)

# Phase 4: Hybrid Search (기존 — 변경 없음)
# ...
```

### 6.2 컨텍스트 빌딩 수정

Phase 7 (Context Building) 에서 `_global_context`가 존재하면 일반 검색 결과 앞에 추가:

```python
# Phase 7: Context Building
if _global_context:
    # Global Search 결과를 컨텍스트 상단에 추가
    context_parts.insert(0, "## 도메인 개요 (커뮤니티 기반)\n\n" + _global_context)
```

### 6.3 Fallback 보장

- `use_global_search`가 False이면 Phase 3.5 전체 스킵
- 커뮤니티가 없거나 검색 실패 시 기존 Local Search 결과만 사용
- **기존 Phase 3 (Graph Enrichment)는 항상 실행** — Global Search는 추가 보강

---

## 7. 싱글톤 추가

### 7.1 singletons.py

```python
_community_searcher = None


def get_community_searcher():
    """Return (or lazily create) the singleton CommunitySearcher instance."""
    global _community_searcher
    if _community_searcher is None:
        with _lock:
            if _community_searcher is None:
                from services.community_searcher import CommunitySearcher
                _community_searcher = CommunitySearcher()
    return _community_searcher


def invalidate_community_searcher():
    """Reset the CommunitySearcher singleton (clears summary cache)."""
    global _community_searcher
    with _lock:
        if _community_searcher is not None:
            _community_searcher.invalidate_cache()
        _community_searcher = None
```

---

## 8. CLI 명령어

### 8.1 main.py — build-community

```python
# Subparser 추가
bc_parser = subparsers.add_parser("build-community", help="커뮤니티 감지 + 요약 생성")
bc_parser.add_argument("--namespace", required=True)
bc_parser.add_argument("--resolution", type=float, default=None, help="Leiden resolution override")
bc_parser.add_argument("--reset", action="store_true", help="기존 커뮤니티 삭제 후 재구축")
bc_parser.add_argument("--skip-summary", action="store_true", help="LLM 요약 생성 스킵")

# 실행 로직
elif args.command == "build-community":
    from src.community_builder import CommunityBuilder
    with app.app_context():
        db.create_all()
        builder = CommunityBuilder(namespace=args.namespace)
        if args.reset:
            builder.reset()
        if args.resolution:
            builder.config['resolution'] = args.resolution
        stats = builder.build(skip_summary=args.skip_summary)
        print(f"커뮤니티: {stats['communities']}개, 요약: {stats.get('summarized', 0)}개")
```

### 8.2 main.py — community-stats

```python
cs_parser = subparsers.add_parser("community-stats", help="커뮤니티 통계")
cs_parser.add_argument("--namespace", default=None)

elif args.command == "community-stats":
    with app.app_context():
        for ns in namespaces:
            comms = KGCommunity.query.filter_by(namespace=ns).all()
            print(f"[{ns}] 커뮤니티: {len(comms)}개")
            for c in comms:
                has_summary = "✅" if c.summary else "❌"
                print(f"  #{c.community_id}: {c.title or '(제목없음)'} "
                      f"({c.member_count}명) {has_summary}")
```

---

## 9. 의존성 관리

### 9.1 requirements.txt 추가

```
# GraphRAG Community (optional)
networkx>=3.0
# leidenalg는 igraph 필요 — 설치 실패 시 Louvain fallback 사용
# pip install leidenalg python-igraph
```

### 9.2 Fallback 전략

| 라이브러리 | 필수? | Fallback |
|-----------|------|----------|
| `networkx` | 예 | 없음 (필수 의존성) |
| `leidenalg` + `python-igraph` | 아니오 | NetworkX 내장 `louvain_communities` |

---

## 10. 파일 변경 요약

| 파일 | 작업 | 예상 라인 수 |
|------|------|-------------|
| `models.py` | KGCommunity, KGCommunityMember 추가 | +45줄 |
| `services/graph_config.py` | community 설정 블록 추가 | +25줄 |
| `src/community_builder.py` | **신규** — 커뮤니티 감지 + 요약 | ~250줄 |
| `services/community_searcher.py` | **신규** — Global Search | ~120줄 |
| `services/singletons.py` | `get_community_searcher()` 추가 | +20줄 |
| `services/query_router.py` | overview 유형 + 패턴 추가 | +15줄 |
| `services/rag_pipeline.py` | Phase 3.5 Global Search 분기 | +25줄 |
| `main.py` | build-community, community-stats CLI | +60줄 |
| `requirements.txt` | networkx 추가 | +2줄 |
| **합계** | | **~560줄** |

---

## 11. 데이터 흐름 다이어그램

### 오프라인 파이프라인

```
python main.py build-community --namespace semiconductor-v2
    │
    ├── KGEntity + KGRelation (기존 데이터)
    │       │
    │       ▼
    │   networkx.Graph 로드
    │       │
    │       ▼
    │   Leiden / Louvain 커뮤니티 감지
    │       │
    │       ▼
    │   min_community_size 필터링
    │       │
    │       ▼
    │   KGCommunity + KGCommunityMember 저장
    │       │
    │       ▼
    │   Gemini Flash 요약 생성 (커뮤니티당 1회)
    │       │
    │       ▼
    └── KGCommunity.title + summary 업데이트
```

### 온라인 검색 흐름

```
사용자 질문: "반도체 공정 전반의 안전 요점을 알려줘"
    │
    ▼
classify_query_type() → "overview"
    │
    ▼
route_cfg['use_global_search'] = True
    │
    ├── Phase 1-2: 기존 Query Enhancement (변경 없음)
    ├── Phase 3: Graph Enrichment (변경 없음)
    │
    ├── Phase 3.5: Global Search ★ 신규
    │       │
    │       ├── community_searcher.search()
    │       │       ├── 커뮤니티 요약 로드 (캐시)
    │       │       ├── 키워드 매칭으로 관련 커뮤니티 선택
    │       │       └── map-reduce → _global_context
    │       │
    │       └── _global_context를 컨텍스트 상단에 추가
    │
    ├── Phase 4-6: 기존 Hybrid/Rerank/Optimize (변경 없음)
    │
    └── Phase 7: Context Building (global_context 포함)
            │
            ▼
        Gemini LLM 답변 생성
```

---

## 12. 성능 예산

| 메트릭 | 목표 | 비고 |
|--------|------|------|
| 오프라인: Leiden 감지 | < 5초 | 엔티티 ~2,000개 기준 |
| 오프라인: 요약 생성 | ~1초 × 커뮤니티 수 | Gemini Flash, 배치 실행 |
| 온라인: 커뮤니티 캐시 로드 | < 10ms | 앱 시작 후 첫 요청 시 |
| 온라인: Global Search 전체 | < 200ms | 캐시 hit 시 (LLM 미호출) |
| 기존 Local Search 영향 | 0ms | Phase 3 미수정, 별도 경로 |

---

## 13. 테스트 계획

### 13.1 오프라인 검증

| # | 테스트 | 기대 결과 |
|---|--------|----------|
| T1 | `build-community --namespace semiconductor-v2` | 커뮤니티 N개 생성, 각각 title/summary 존재 |
| T2 | `community-stats` | 도메인별 커뮤니티 수, 멤버 수 출력 |
| T3 | Leiden 없이 실행 | Louvain fallback 동작, 경고 로그 |
| T4 | 엔티티 <3개 도메인 | 커뮤니티 0개, `skipped: insufficient_nodes` |
| T5 | `--reset` 후 재구축 | 기존 데이터 삭제 → 재생성 |

### 13.2 온라인 검증

| # | 쿼리 | 기대 동작 |
|---|------|----------|
| T6 | "반도체 공정 전반을 설명해줘" | overview → Global Search 활성화, 커뮤니티 컨텍스트 포함 답변 |
| T7 | "CVD 공정의 온도는?" | factual → Global Search 미활성화, 기존 Local Search만 사용 |
| T8 | "산업안전 핵심 5가지" | overview → Global Search, community_titles 로그 출력 |
| T9 | 커뮤니티 없는 도메인 쿼리 | Global Search 스킵, fallback to Local |
| T10 | 커뮤니티 서비스 예외 발생 | try/except → warning 로그, Local Search fallback |
