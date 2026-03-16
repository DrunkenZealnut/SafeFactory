# GraphRAG 도입 Design

> Plan 참조: `docs/01-plan/features/graphrag.plan.md`

---

## 1. 시스템 아키텍처

### 1.1 전체 파이프라인 변경

```
[기존]
Phase 0 → Phase 1 → Phase 2 → Phase 4 → Phase 5 → Phase 6 → Phase 7
Domain    Query      Vector     Hybrid    Rerank    Context   LLM
Classify  Enhance    Search     BM25+RRF  CrossEnc  Optimize  Generate

[변경 후]
Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6 → Phase 7
Domain    Query      Vector     Graph      Hybrid    Rerank    Context   LLM
Classify  Enhance    Search     Enrich     BM25+RRF  CrossEnc  Optimize  Generate
                                ↑ NEW
```

### 1.2 신규 컴포넌트 맵

```
src/
├── graph_builder.py        # [NEW] 오프라인 엔티티/관계 추출 + DB 저장
└── (기존 모듈 변경 없음)

services/
├── graph_searcher.py       # [NEW] 온라인 그래프 탐색 서비스
├── graph_config.py         # [NEW] 도메인별 그래프 설정
├── singletons.py           # [MOD] get_graph_searcher() 추가
├── rag_pipeline.py         # [MOD] Phase 3 삽입
└── domain_config.py        # (변경 없음 — graph_config.py로 분리)

models.py                   # [MOD] Entity, Relation, EntityChunk 모델 추가
main.py                     # [MOD] build-graph CLI 명령어 추가
```

---

## 2. 데이터 모델

### 2.1 새 SQLAlchemy 모델 (`models.py`)

```python
# ---------------------------------------------------------------------------
# Knowledge Graph
# ---------------------------------------------------------------------------

class Entity(db.Model):
    """Knowledge Graph 엔티티 노드."""
    __tablename__ = 'kg_entities'
    __table_args__ = (
        db.Index('ix_kg_entity_ns_type', 'namespace', 'entity_type'),
        db.Index('ix_kg_entity_name', 'name'),
    )

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    name_normalized = db.Column(db.String(200), nullable=False)  # 소문자, 공백제거
    entity_type = db.Column(db.String(50), nullable=False)       # 공정|장비|물질|안전규정|증상위험|법률조항
    namespace = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)                              # LLM 생성 1-2문장 설명
    aliases_json = db.Column(db.Text, default='[]')              # JSON 배열
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    # relationships
    source_relations = db.relationship('Relation', foreign_keys='Relation.source_id', backref='source_entity', lazy='dynamic')
    target_relations = db.relationship('Relation', foreign_keys='Relation.target_id', backref='target_entity', lazy='dynamic')
    chunks = db.relationship('EntityChunk', backref='entity', lazy='dynamic')

    @property
    def aliases(self):
        import json
        return json.loads(self.aliases_json) if self.aliases_json else []

    @aliases.setter
    def aliases(self, value):
        import json
        self.aliases_json = json.dumps(value, ensure_ascii=False)


class Relation(db.Model):
    """Knowledge Graph 관계 엣지."""
    __tablename__ = 'kg_relations'
    __table_args__ = (
        db.Index('ix_kg_rel_source', 'source_id'),
        db.Index('ix_kg_rel_target', 'target_id'),
        db.Index('ix_kg_rel_type', 'relation_type'),
    )

    id = db.Column(db.Integer, primary_key=True)
    source_id = db.Column(db.Integer, db.ForeignKey('kg_entities.id', ondelete='CASCADE'), nullable=False)
    target_id = db.Column(db.Integer, db.ForeignKey('kg_entities.id', ondelete='CASCADE'), nullable=False)
    relation_type = db.Column(db.String(50), nullable=False)     # uses|part_of|causes|prevents|related_to|requires
    confidence = db.Column(db.Float, default=0.8)
    evidence_chunk_id = db.Column(db.String(200))                 # Pinecone vector ID (출처 근거)
    namespace = db.Column(db.String(100), nullable=False)


class EntityChunk(db.Model):
    """엔티티-청크 매핑 (어떤 청크에서 어떤 엔티티가 언급되는지)."""
    __tablename__ = 'kg_entity_chunks'
    __table_args__ = (
        db.Index('ix_kg_ec_entity', 'entity_id'),
        db.Index('ix_kg_ec_chunk', 'chunk_vector_id'),
    )

    id = db.Column(db.Integer, primary_key=True)
    entity_id = db.Column(db.Integer, db.ForeignKey('kg_entities.id', ondelete='CASCADE'), nullable=False)
    chunk_vector_id = db.Column(db.String(200), nullable=False)   # Pinecone vector ID
    relevance_score = db.Column(db.Float, default=1.0)            # 해당 청크에서 엔티티 관련도
    namespace = db.Column(db.String(100), nullable=False)
```

### 2.2 테이블명 접두사 `kg_` 사용 이유

기존 `models.py`에 커뮤니티/인증 등 다양한 모델이 혼재. `kg_` 접두사로 Knowledge Graph 관련 테이블을 명확히 구분.

---

## 3. 오프라인 그래프 구축 (`src/graph_builder.py`)

### 3.1 클래스 설계

```python
class GraphBuilder:
    """Pinecone 청크에서 엔티티/관계를 추출하여 SQLite에 저장."""

    # 도메인별 엔티티 타입 정의
    ENTITY_TYPES = {
        'semiconductor-v2': ['공정', '장비', '물질', '안전규정'],
        'laborlaw':         ['법률조항', '권리의무', '절차', '기간금액'],
        'kosha':            ['안전규정', '위험요인', '보호장비', '증상'],
        'field-training':   ['안전규정', '장비', '절차'],
        'msds':             [],  # MSDS는 그래프 구축 대상 아님
    }

    RELATION_TYPES = ['uses', 'part_of', 'causes', 'prevents', 'related_to', 'requires']

    def __init__(self, namespace: str, gemini_client=None):
        self.namespace = namespace
        self.gemini = gemini_client or get_gemini_client()
        self.entity_types = self.ENTITY_TYPES.get(namespace, [])

    def build(self, batch_size: int = 20, max_chunks: int = None):
        """전체 그래프 구축 파이프라인."""
        # 1. Pinecone에서 네임스페이스 전체 청크 조회
        chunks = self._fetch_all_chunks(max_chunks)
        # 2. 배치 단위 엔티티/관계 추출
        for batch in self._batched(chunks, batch_size):
            extracted = self._extract_entities_batch(batch)
            self._save_to_db(extracted)
        # 3. 엔티티 정규화 (중복 병합)
        self._normalize_entities()
        # 4. 통계 출력
        return self._stats()

    def _fetch_all_chunks(self, max_chunks=None) -> List[dict]:
        """Pinecone에서 네임스페이스 전체 벡터 메타데이터 조회."""
        # pinecone list_paginated → fetch로 metadata 획득
        ...

    def _extract_entities_batch(self, chunks: List[dict]) -> List[dict]:
        """Gemini Flash로 배치 엔티티/관계 추출."""
        prompt = self._build_extraction_prompt(chunks)
        response = self.gemini.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
            config={'response_mime_type': 'application/json'}
        )
        return json.loads(response.text)

    def _build_extraction_prompt(self, chunks: List[dict]) -> str:
        """도메인 특화 엔티티 추출 프롬프트."""
        ...  # 섹션 3.2 참조

    def _save_to_db(self, extracted: List[dict]):
        """추출 결과를 kg_entities, kg_relations, kg_entity_chunks에 저장."""
        # upsert: name_normalized + namespace로 기존 엔티티 찾아 merge
        ...

    def _normalize_entities(self):
        """동일 엔티티 병합 (alias 기반 + 유사도)."""
        # name_normalized 동일 → merge
        # alias 교차 → merge
        ...

    def _stats(self) -> dict:
        """구축 통계: 엔티티 수, 관계 수, 네임스페이스별 분포."""
        ...
```

### 3.2 엔티티 추출 프롬프트 설계

```
시스템: 너는 산업안전 도메인의 Knowledge Graph 구축 전문가야.
주어진 텍스트 청크에서 엔티티(개체)와 관계를 추출해.

## 엔티티 타입
{entity_types를 도메인에 맞게 나열}

## 관계 타입
uses, part_of, causes, prevents, related_to, requires

## 규칙
1. 엔티티명은 한국어 기준으로 정규화 (예: "CVD" → "CVD 공정")
2. 동일 개념의 영어/한국어 표현은 aliases에 포함
3. 관계는 (source, relation_type, target) 트리플로 추출
4. 확실하지 않은 관계는 confidence를 낮게 설정
5. 각 엔티티에 1-2문장 설명 생성

## 입력 청크
{chunk_contents}

## JSON 출력 형식
{
  "entities": [
    {"name": "CVD 공정", "type": "공정", "description": "...", "aliases": ["Chemical Vapor Deposition", "화학기상증착"]}
  ],
  "relations": [
    {"source": "CVD 공정", "type": "uses", "target": "실란(SiH4)", "confidence": 0.9}
  ],
  "entity_chunks": [
    {"entity_name": "CVD 공정", "chunk_id": "{vector_id}", "relevance": 0.95}
  ]
}
```

### 3.3 CLI 명령어 (`main.py` 확장)

```python
# main.py에 추가
@click.command('build-graph')
@click.option('--namespace', '-n', required=True, help='대상 Pinecone 네임스페이스')
@click.option('--batch-size', default=20, help='LLM 배치 크기')
@click.option('--max-chunks', default=None, type=int, help='최대 처리 청크 수 (테스트용)')
@click.option('--reset', is_flag=True, help='기존 그래프 삭제 후 재구축')
def build_graph(namespace, batch_size, max_chunks, reset):
    """Pinecone 청크에서 Knowledge Graph를 구축합니다."""
    ...

@click.command('graph-stats')
@click.option('--namespace', '-n', default=None, help='특정 네임스페이스 통계')
def graph_stats(namespace):
    """Knowledge Graph 통계를 출력합니다."""
    ...
```

---

## 4. 온라인 그래프 검색 (`services/graph_searcher.py`)

### 4.1 클래스 설계

```python
class GraphResult:
    """그래프 탐색으로 발견된 청크 결과."""
    chunk_vector_id: str
    entity_path: List[str]        # 탐색 경로: ["CVD 공정", "실란(SiH4)", "MSDS"]
    relation_path: List[str]      # 관계 경로: ["uses", "related_to"]
    hop_distance: int             # 원래 쿼리 엔티티에서의 거리
    graph_score: float            # 거리/confidence 기반 점수


class GraphSearcher:
    """쿼리에서 엔티티를 인식하고 그래프 탐색으로 관련 청크를 찾는 서비스."""

    def __init__(self, config: dict = None):
        self._config = config or {}
        self._entity_cache = {}   # namespace → {name_normalized: entity_id}

    def search(self, query: str, namespace: str,
               hop_depth: int = 2, max_results: int = 10) -> List[GraphResult]:
        """
        1. 쿼리에서 엔티티 매칭
        2. 매칭 엔티티에서 N-hop 그래프 탐색
        3. 탐색된 엔티티의 청크 조회
        4. GraphResult 리스트 반환
        """
        config = self._get_namespace_config(namespace)
        if not config.get('enabled', False):
            return []

        # Step 1: 쿼리 엔티티 매칭
        matched_entities = self._match_query_entities(query, namespace)
        if not matched_entities:
            return []

        # Step 2: N-hop 그래프 탐색 (Recursive CTE)
        related_entities = self._traverse_graph(
            entity_ids=[e.id for e in matched_entities],
            namespace=namespace,
            max_hops=config.get('hop_depth', hop_depth),
        )

        # Step 3: 관련 청크 조회
        chunk_results = self._get_entity_chunks(related_entities, namespace, max_results)

        return chunk_results

    def _match_query_entities(self, query: str, namespace: str) -> List[Entity]:
        """쿼리 텍스트에서 엔티티 매칭 (O(1) 캐시 + substring 매칭)."""
        # 1. 캐시에서 해당 namespace 엔티티 목록 로드
        cache = self._load_entity_cache(namespace)
        # 2. query를 정규화 후 캐시 키와 substring 매칭
        # 3. alias도 함께 검색
        matched = []
        query_norm = self._normalize(query)
        for name_norm, entity_id in cache.items():
            if name_norm in query_norm:
                matched.append(entity_id)
        # 4. Entity 객체 조회
        if matched:
            return db.session.query(Entity).filter(Entity.id.in_(matched)).all()
        return []

    def _traverse_graph(self, entity_ids: List[int], namespace: str,
                        max_hops: int = 2) -> List[dict]:
        """Recursive CTE로 N-hop 그래프 탐색."""
        sql = text("""
            WITH RECURSIVE graph_walk AS (
                -- 시작 노드
                SELECT
                    e.id, e.name, 0 AS hop, e.name AS path,
                    '' AS relation_path, 1.0 AS path_confidence
                FROM kg_entities e
                WHERE e.id IN :start_ids AND e.namespace = :ns

                UNION ALL

                -- N-hop 탐색
                SELECT
                    e2.id, e2.name, gw.hop + 1,
                    gw.path || ' → ' || e2.name,
                    gw.relation_path || CASE WHEN gw.relation_path = '' THEN '' ELSE ',' END || r.relation_type,
                    gw.path_confidence * r.confidence
                FROM graph_walk gw
                JOIN kg_relations r ON r.source_id = gw.id AND r.namespace = :ns
                JOIN kg_entities e2 ON e2.id = r.target_id
                WHERE gw.hop < :max_hops
            )
            SELECT DISTINCT id, name, hop, path, relation_path, path_confidence
            FROM graph_walk
            ORDER BY hop ASC, path_confidence DESC
        """)
        rows = db.session.execute(sql, {
            'start_ids': tuple(entity_ids),
            'ns': namespace,
            'max_hops': max_hops,
        }).fetchall()
        return [dict(row._mapping) for row in rows]

    def _get_entity_chunks(self, entities: List[dict], namespace: str,
                           max_results: int) -> List[GraphResult]:
        """탐색된 엔티티의 청크를 조회하고 점수를 계산."""
        entity_ids = [e['id'] for e in entities]
        chunks = (
            db.session.query(EntityChunk)
            .filter(EntityChunk.entity_id.in_(entity_ids), EntityChunk.namespace == namespace)
            .all()
        )
        # entity hop 정보를 기반으로 graph_score 계산
        entity_hop_map = {e['id']: e for e in entities}
        results = []
        seen_chunks = set()
        for ec in chunks:
            if ec.chunk_vector_id in seen_chunks:
                continue
            seen_chunks.add(ec.chunk_vector_id)
            e_info = entity_hop_map.get(ec.entity_id, {})
            hop = e_info.get('hop', 0)
            conf = e_info.get('path_confidence', 1.0)
            # 점수: 가까울수록 + confidence 높을수록 좋음
            score = conf / (1.0 + hop * 0.5)
            results.append(GraphResult(
                chunk_vector_id=ec.chunk_vector_id,
                entity_path=e_info.get('path', '').split(' → '),
                relation_path=e_info.get('relation_path', '').split(','),
                hop_distance=hop,
                graph_score=score,
            ))
        # 점수순 정렬 + 상위 N개
        results.sort(key=lambda r: r.graph_score, reverse=True)
        return results[:max_results]

    def _load_entity_cache(self, namespace: str) -> dict:
        """네임스페이스별 엔티티 캐시 (name_normalized → entity_id)."""
        if namespace not in self._entity_cache:
            entities = db.session.query(Entity.id, Entity.name_normalized, Entity.aliases_json)\
                .filter_by(namespace=namespace).all()
            cache = {}
            for eid, name_norm, aliases_json in entities:
                cache[name_norm] = eid
                # alias도 캐시에 추가
                for alias in json.loads(aliases_json or '[]'):
                    cache[self._normalize(alias)] = eid
            self._entity_cache[namespace] = cache
        return self._entity_cache[namespace]

    def invalidate_cache(self, namespace: str = None):
        """캐시 무효화."""
        if namespace:
            self._entity_cache.pop(namespace, None)
        else:
            self._entity_cache.clear()

    @staticmethod
    def _normalize(text: str) -> str:
        """엔티티 이름 정규화: 소문자 + 공백 제거."""
        return text.lower().replace(' ', '').strip()
```

### 4.2 도메인별 그래프 설정 (`services/graph_config.py`)

```python
GRAPH_CONFIG = {
    'semiconductor-v2': {
        'enabled': True,
        'hop_depth': 2,
        'graph_weight': 0.3,
        'max_graph_results': 10,
        'entity_types': ['공정', '장비', '물질', '안전규정'],
    },
    'laborlaw': {
        'enabled': True,
        'hop_depth': 1,
        'graph_weight': 0.2,
        'max_graph_results': 8,
        'entity_types': ['법률조항', '권리의무', '절차'],
    },
    'kosha': {
        'enabled': True,
        'hop_depth': 2,
        'graph_weight': 0.3,
        'max_graph_results': 10,
        'entity_types': ['안전규정', '위험요인', '보호장비'],
    },
    'msds': {
        'enabled': False,
    },
    'field-training': {
        'enabled': True,
        'hop_depth': 1,
        'graph_weight': 0.2,
        'max_graph_results': 8,
        'entity_types': ['안전규정', '장비', '절차'],
    },
}

def get_graph_config(namespace: str) -> dict:
    return GRAPH_CONFIG.get(namespace, {'enabled': False})
```

---

## 5. RAG 파이프라인 통합

### 5.1 Phase 3 삽입 위치 (`services/rag_pipeline.py`)

Phase 2 (검색 결과) 직후, Phase 4 (Hybrid BM25) 직전에 삽입:

```python
# ========================================
# Phase 3: Graph Enrichment (NEW)
# ========================================
_t0 = time.perf_counter()
try:
    from services.graph_config import get_graph_config
    graph_cfg = get_graph_config(namespace)
    if graph_cfg.get('enabled') and use_enhancement:
        graph_searcher = get_graph_searcher()
        graph_results = graph_searcher.search(
            query=search_query,
            namespace=namespace,
            hop_depth=graph_cfg.get('hop_depth', 2),
            max_results=graph_cfg.get('max_graph_results', 10),
        )
        if graph_results:
            # 그래프 결과의 청크를 Pinecone에서 fetch
            graph_chunk_ids = [gr.chunk_vector_id for gr in graph_results]
            # 기존 results에 이미 포함된 청크 제외
            existing_ids = {r.get('id') for r in results}
            new_ids = [cid for cid in graph_chunk_ids if cid not in existing_ids]

            if new_ids:
                agent = get_agent()
                graph_chunks = agent.fetch_vectors(new_ids, namespace=namespace)
                # graph_score를 메타데이터에 추가
                graph_score_map = {gr.chunk_vector_id: gr for gr in graph_results}
                for chunk in graph_chunks:
                    gr = graph_score_map.get(chunk['id'])
                    if gr:
                        chunk['graph_score'] = gr.graph_score
                        chunk['graph_path'] = ' → '.join(gr.entity_path)
                results.extend(graph_chunks)
                logging.info("[Graph Enrichment] Added %d graph-discovered chunks", len(graph_chunks))

            # 파이프라인 메타데이터에 그래프 정보 추가
            pipeline_meta['graph_entities'] = [
                gr.entity_path[0] for gr in graph_results if gr.entity_path
            ]
except Exception as e:
    logging.warning("[Graph Enrichment] Failed (fallback to vector-only): %s", e)

_timings['phase3_graph_ms'] = round((time.perf_counter() - _t0) * 1000)
```

### 5.2 Singleton 추가 (`services/singletons.py`)

```python
_graph_searcher = None
_graph_searcher_lock = threading.RLock()

def get_graph_searcher():
    global _graph_searcher
    if _graph_searcher is None:
        with _graph_searcher_lock:
            if _graph_searcher is None:
                from services.graph_searcher import GraphSearcher
                _graph_searcher = GraphSearcher()
    return _graph_searcher

def invalidate_graph_searcher():
    global _graph_searcher
    with _graph_searcher_lock:
        _graph_searcher = None
```

### 5.3 PineconeAgent fetch 메서드

기존 `PineconeAgent`에 벡터 ID로 직접 fetch하는 메서드가 필요. 없으면 추가:

```python
# src/agent.py (PineconeAgent)
def fetch_vectors(self, ids: List[str], namespace: str) -> List[dict]:
    """벡터 ID 목록으로 Pinecone에서 메타데이터를 직접 조회."""
    result = self.index.fetch(ids=ids, namespace=namespace)
    chunks = []
    for vid, vec in result.vectors.items():
        meta = vec.metadata or {}
        chunks.append({
            'id': vid,
            'score': 0.0,  # fetch는 score 없음 → graph_score로 대체
            'content': meta.get('content', ''),
            'metadata': meta,
        })
    return chunks
```

---

## 6. 데이터 흐름 상세

### 6.1 오프라인 (그래프 구축)

```
[CLI] python main.py build-graph --namespace semiconductor-v2
  │
  ├─ 1. Pinecone에서 namespace 전체 벡터 메타데이터 fetch
  │     (list → fetch, 배치 1000개씩)
  │
  ├─ 2. 청크 20개씩 배치로 Gemini Flash 호출
  │     입력: chunk content + metadata
  │     출력: JSON (entities, relations, entity_chunks)
  │
  ├─ 3. SQLite 저장 (upsert)
  │     kg_entities: name_normalized + namespace 기준 중복 체크
  │     kg_relations: source_id + target_id + relation_type 기준
  │     kg_entity_chunks: entity_id + chunk_vector_id 기준
  │
  ├─ 4. 엔티티 정규화
  │     동일 name_normalized 병합
  │     alias 교차 병합
  │
  └─ 5. 통계 출력
        Entities: 1,234 | Relations: 3,456 | Chunks mapped: 5,678
```

### 6.2 온라인 (검색 시)

```
[User Query] "CVD 공정에서 사용하는 가스의 안전 주의사항은?"
  │
  ├─ Phase 0-2: (기존과 동일) → vector_results: 15건
  │
  ├─ Phase 3: Graph Enrichment
  │   ├─ entity_cache에서 "CVD" 매칭 → Entity(id=42, name="CVD 공정")
  │   ├─ 2-hop 탐색:
  │   │   hop 0: CVD 공정
  │   │   hop 1: 실란(SiH4) [uses], TEOS [uses], 플라즈마 [requires]
  │   │   hop 2: MSDS-실란 [related_to], 가스누출 위험 [causes]
  │   ├─ 관련 청크 10건 조회 (기존 15건과 중복 제거)
  │   └─ 신규 5건 추가 → 총 20건
  │
  ├─ Phase 4-5: (기존과 동일) Hybrid + Rerank
  │
  ├─ Phase 6: Context Optimization (개선)
  │   ├─ 엔티티 그룹핑: "CVD 공정" 관련 3청크 → 대표 1청크 선택
  │   ├─ 그래프 경로 컨텍스트 추가:
  │   │   "관련 지식: CVD 공정 → [uses] → 실란(SiH4) → [related_to] → 실란 MSDS"
  │   └─ 토큰 최적화
  │
  └─ Phase 7: LLM 생성 (그래프 경로 포함)
```

---

## 7. 구현 순서 (파일 단위)

### Week 1: 오프라인 그래프 구축

| 순서 | 파일 | 작업 | 의존성 |
|------|------|------|--------|
| 1 | `models.py` | Entity, Relation, EntityChunk 모델 추가 | 없음 |
| 2 | `services/graph_config.py` | 도메인별 그래프 설정 | 없음 |
| 3 | `src/graph_builder.py` | GraphBuilder 클래스 전체 구현 | 1, 2 |
| 4 | `main.py` | `build-graph`, `graph-stats` CLI 명령어 | 3 |
| 5 | (실행) | semiconductor-v2 그래프 구축 + 검증 | 4 |

### Week 2: 온라인 검색 통합

| 순서 | 파일 | 작업 | 의존성 |
|------|------|------|--------|
| 6 | `services/graph_searcher.py` | GraphSearcher 전체 구현 | 1, 2 |
| 7 | `services/singletons.py` | `get_graph_searcher()` 싱글톤 추가 | 6 |
| 8 | `src/agent.py` | `fetch_vectors()` 메서드 추가 (없는 경우) | 없음 |
| 9 | `services/rag_pipeline.py` | Phase 3 삽입 | 6, 7, 8 |
| 10 | (테스트) | 멀티홉 질문 세트 20건 평가 + Fallback 검증 | 9 |

---

## 8. 에러 핸들링 및 Fallback

### 8.1 Fallback 전략

| 실패 지점 | Fallback | 사용자 영향 |
|-----------|----------|------------|
| 엔티티 매칭 0건 | Phase 3 스킵, 기존 결과만 사용 | 없음 |
| CTE 쿼리 타임아웃 | 500ms 타임아웃 → Phase 3 스킵 | 없음 |
| Pinecone fetch 실패 | graph_chunks 없이 진행 | 없음 |
| 전체 Phase 3 예외 | try/except 로 감싸서 warning 로그만 | 없음 |

### 8.2 로깅

모든 Phase 3 동작은 `[Graph Enrichment]` 접두사로 로깅:
```
[Graph Enrichment] Matched 3 entities: CVD 공정, 실란, TEOS
[Graph Enrichment] Traversed 2 hops: 12 related entities found
[Graph Enrichment] Added 5 graph-discovered chunks (3 duplicates skipped)
[Graph Enrichment] Phase time: 89ms
```

---

## 9. 성능 예산

| 항목 | 목표 | 측정 방법 |
|------|------|-----------|
| 엔티티 캐시 로드 | <50ms (최초 1회) | `_load_entity_cache` 시간 |
| 쿼리 엔티티 매칭 | <5ms | substring 매칭 시간 |
| CTE 그래프 탐색 (2홉) | <100ms | SQLite 쿼리 시간 |
| Pinecone fetch (10건) | <100ms | API 호출 시간 |
| **Phase 3 전체** | **<200ms** | `phase3_graph_ms` 타이밍 |

---

## 10. DB 마이그레이션

기존 SQLite에 3개 테이블을 추가하는 방식. Flask 앱 시작 시 `db.create_all()`로 자동 생성되므로 별도 마이그레이션 스크립트 불필요. 단, 프로덕션에서는:

```python
# 앱 시작 시 또는 CLI로 실행
with app.app_context():
    from sqlalchemy import inspect
    inspector = inspect(db.engine)
    existing = inspector.get_table_names()
    if 'kg_entities' not in existing:
        db.create_all()
        logging.info("[GraphRAG] Created knowledge graph tables")
```
