# GraphRAG 도입 Plan

## Executive Summary

| 관점 | 내용 |
|------|------|
| **Feature** | graphrag |
| **시작일** | 2026-03-17 |
| **예상 기간** | 2주 (Phase 1: 1주, Phase 2: 1주) |

| 관점 | 설명 |
|------|------|
| **Problem** | 현재 벡터 유사도 기반 검색은 개념 간 관계를 이해하지 못하여, "CVD 공정에 사용되는 가스는?" 같은 멀티홉 질문에서 관련 문맥을 놓치고 단편적 답변을 생성함 |
| **Solution** | 청킹 단계에서 엔티티/관계를 추출하고 Knowledge Graph를 구축하여, 벡터 검색 결과에 그래프 탐색 결과를 RRF 방식으로 융합하는 GraphRAG 파이프라인 도입 |
| **Function UX Effect** | 사용자 질문에 대해 연관 개념까지 포함한 포괄적 답변 제공, 관련 용어/공정 자동 연결, 답변 내 개념 네트워크 시각화 |
| **Core Value** | 산업안전 교육에서 개별 지식이 아닌 구조화된 지식 체계 제공 — "개별 팩트 검색" → "지식 그래프 탐색" 전환 |

---

## 1. 배경 및 문제 정의

### 1.1 현재 RAG 파이프라인 구조

```
Query → DomainClassifier → QueryEnhancer(multi-query)
      → Vector+BM25 Hybrid Search → RRF Fusion
      → Reranker(cross-encoder) → ContextOptimizer
      → SafetyCrossSearch → Gemini LLM → Response
```

### 1.2 현재 시스템의 한계

| 문제 | 상세 | 영향 |
|------|------|------|
| **멀티홉 질문 실패** | "CVD 공정에서 사용하는 가스의 안전 주의사항"은 CVD→가스→MSDS 3홉 필요하지만 벡터 검색은 1홉만 가능 | 관련 안전정보 누락 |
| **개념 관계 부재** | "에칭"과 "포토레지스트"가 동일 공정 흐름이라는 관계가 검색에 반영 안 됨 | 맥락 없는 단편적 답변 |
| **동의어/상하위어 한계** | QueryEnhancer의 하드코딩 동의어 사전으로는 도메인 용어 커버 불가 | 검색 누락 |
| **크로스도메인 수동 연결** | semiconductor→kosha, 화학물질→msds 연결이 하드코딩 | 확장 불가, 유지보수 부담 |
| **중복 컨텍스트** | 동일 엔티티를 다루는 여러 청크가 중복 포함되어 토큰 낭비 | 답변 품질 저하 |

### 1.3 도입 목적

1. **멀티홉 추론**: 엔티티 관계 그래프를 통한 다단계 질문 응답
2. **크로스도메인 자동 연결**: 하드코딩 대신 그래프 기반 도메인 간 연결
3. **컨텍스트 최적화**: 엔티티 중심 그룹핑으로 중복 제거 및 토큰 효율화
4. **지식 구조 시각화**: 사용자에게 관련 개념 네트워크를 제공

---

## 2. 기술 조사 및 접근 방식

### 2.1 GraphRAG 접근 방식 비교

| 접근법 | 설명 | 장점 | 단점 | 적합도 |
|--------|------|------|------|--------|
| **A. Microsoft GraphRAG (Full)** | LLM으로 전체 문서 엔티티/커뮤니티 추출 → Leiden 클러스터링 → 커뮤니티 요약 | 가장 높은 답변 품질 | 인덱싱 비용 극대 (LLM 호출 수천회), 무거움 | &#x2718; 비용/규모 부적합 |
| **B. LightRAG / nano-graphrag** | 경량화된 그래프 구축, 하이브리드 검색 | 간결, 빠름 | 관계 추출 정확도 낮음 | &#x25B3; 검토 가능 |
| **C. 하이브리드 그래프 레이어 (자체 구현)** | 기존 파이프라인에 그래프 탐색 Phase 삽입, 그래프 DB는 SQLite+JSON | 기존 아키텍처 활용, 점진적 도입, 비용 최소 | 자체 NER/관계추출 필요 | &#x2714; **채택** |

### 2.2 채택 방식: 하이브리드 그래프 레이어

**핵심 원칙**: 기존 7-Phase 파이프라인을 **비파괴적으로 확장** — 그래프가 실패해도 기존 벡터 검색으로 fallback

```
기존 Pipeline:
Phase 0 → Phase 1 → Phase 2 → Phase 4 → Phase 5 → Phase 6 → Phase 7

GraphRAG Pipeline:
Phase 0 → Phase 1 → Phase 2 → [Phase 3: Graph Search] → Phase 4 → Phase 5 → Phase 6+ → Phase 7
                                    ↑ NEW
```

---

## 3. 구현 범위

### 3.1 Phase 1 — 오프라인 그래프 구축 (1주)

#### 3.1.1 엔티티 추출 파이프라인

| 항목 | 내용 |
|------|------|
| **입력** | 기존 Pinecone 청크 (content + metadata) |
| **추출 방법** | LLM 기반 추출 (Gemini Flash) — 도메인 특화 프롬프트 |
| **엔티티 타입** | `공정`, `장비`, `물질`, `안전규정`, `증상/위험`, `법률조항` |
| **관계 타입** | `uses`, `part_of`, `causes`, `prevents`, `related_to`, `requires` |
| **저장소** | SQLite 테이블 (entities, relations, entity_chunks) |
| **배치 처리** | CLI 명령어 (`python main.py build-graph --namespace semiconductor-v2`) |

#### 3.1.2 새 DB 테이블

```python
class Entity(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)        # "CVD 공정"
    entity_type = db.Column(db.String(50), nullable=False)  # "공정"
    namespace = db.Column(db.String(100), nullable=False)    # "semiconductor-v2"
    description = db.Column(db.Text)                         # 엔티티 설명 (LLM 생성)
    aliases = db.Column(db.Text)                             # JSON: ["Chemical Vapor Deposition", "화학기상증착"]

class Relation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    source_id = db.Column(db.Integer, db.ForeignKey('entities.id'))
    target_id = db.Column(db.Integer, db.ForeignKey('entities.id'))
    relation_type = db.Column(db.String(50))                 # "uses", "part_of"
    confidence = db.Column(db.Float, default=0.0)
    evidence_chunk_id = db.Column(db.String(200))            # Pinecone vector ID

class EntityChunk(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    entity_id = db.Column(db.Integer, db.ForeignKey('entities.id'))
    chunk_vector_id = db.Column(db.String(200))              # Pinecone vector ID
    relevance_score = db.Column(db.Float, default=0.0)
```

#### 3.1.3 엔티티 추출 프롬프트 (도메인별)

- 반도체: 공정명, 장비명, 재료/가스, 물성 파라미터
- 안전보건: 위험 요인, 보호 장비, 법적 기준, 증상
- 노동법: 법률 조항, 권리/의무, 절차, 기간/금액
- MSDS: 화학물질명, CAS번호, GHS 분류, 노출한계

### 3.2 Phase 2 — 온라인 그래프 검색 통합 (1주)

#### 3.2.1 새 RAG Phase: Graph Search (Phase 3)

```python
# services/graph_searcher.py (NEW)
class GraphSearcher:
    def search(self, query_entities: List[str], namespace: str,
               hop_depth: int = 2, max_results: int = 10) -> List[GraphResult]:
        """
        1. query에서 엔티티 매칭 (fuzzy match + alias)
        2. 매칭된 엔티티에서 hop_depth만큼 그래프 탐색
        3. 탐색된 엔티티들의 연결 청크를 EntityChunk에서 조회
        4. 관계 경로를 컨텍스트로 포함
        """
```

#### 3.2.2 RAG 파이프라인 수정 (`rag_pipeline.py`)

```
Phase 2 결과 (vector+BM25 results)
    ↓
Phase 3: Graph Enrichment (NEW)
    ├─ query에서 엔티티 추출 (lightweight: 키워드 매칭 + Entity 테이블 lookup)
    ├─ 매칭 엔티티 → 1~2홉 그래프 탐색
    ├─ 관련 엔티티의 청크 조회 (EntityChunk → Pinecone fetch)
    ├─ RRF 병합: vector_results + graph_results
    └─ 엔티티 메타데이터를 컨텍스트에 추가
    ↓
Phase 4 (기존 Hybrid/Rerank 계속)
```

#### 3.2.3 컨텍스트 최적화 개선 (Phase 6)

- 동일 엔티티를 다루는 청크 그룹핑 → 대표 청크만 선택
- 엔티티 정의를 컨텍스트 상단에 삽입
- 관계 경로를 "지식 맵" 형태로 프롬프트에 포함

#### 3.2.4 도메인별 그래프 활용 설정

```python
# services/domain_config.py 확장
GRAPH_CONFIG = {
    'semiconductor-v2': {
        'enabled': True,
        'hop_depth': 2,
        'graph_weight': 0.3,      # RRF에서 그래프 비중
        'entity_types': ['공정', '장비', '물질', '안전규정'],
    },
    'laborlaw': {
        'enabled': True,
        'hop_depth': 1,
        'graph_weight': 0.2,
        'entity_types': ['법률조항', '권리의무', '절차'],
    },
    'kosha': {
        'enabled': True,
        'hop_depth': 2,
        'graph_weight': 0.3,
        'entity_types': ['안전규정', '위험요인', '보호장비'],
    },
    'msds': {
        'enabled': False,         # MSDS는 키워드 기반이 더 적합
        'hop_depth': 0,
        'graph_weight': 0.0,
    },
    'field-training': {
        'enabled': True,
        'hop_depth': 1,
        'graph_weight': 0.2,
        'entity_types': ['안전규정', '장비', '절차'],
    },
}
```

### 3.3 Scope Out (이번 범위 제외)

| 제외 항목 | 이유 |
|-----------|------|
| Neo4j/전용 그래프 DB 도입 | SQLite로 충분한 규모 (수천~수만 엔티티), 인프라 복잡도 회피 |
| 실시간 그래프 업데이트 | 오프라인 배치로 충분, 문서 업데이트 빈도 낮음 |
| 프론트엔드 그래프 시각화 | Phase 2 이후 별도 기능으로 추진 |
| Microsoft GraphRAG 커뮤니티 요약 | 비용 대비 효과 불분명, LLM 호출 과다 |
| 전 도메인 동시 그래프 구축 | semiconductor-v2 먼저 구축 후 순차 확장 |

---

## 4. 기술 스택

| 구성요소 | 기술 | 이유 |
|----------|------|------|
| **엔티티 추출** | Gemini Flash 2.0 | 이미 사용 중, 비용 낮음, 한국어 성능 양호 |
| **그래프 저장소** | SQLite (기존 app.db) | 별도 인프라 불필요, 수만 노드 규모 충분 |
| **엔티티 매칭** | FTS5 + Fuzzy match | SQLite 내장 전문검색 + 유사도 매칭 |
| **관계 탐색** | Recursive CTE (SQL) | SQLite 지원, 2~3홉 탐색에 충분 |
| **RRF 융합** | 기존 RRF 로직 확장 | 코드 재사용, 검증된 방식 |
| **Fallback** | 기존 벡터 파이프라인 | 그래프 실패 시 무중단 서비스 |

---

## 5. 성공 기준

| 지표 | 현재 | 목표 | 측정 방법 |
|------|------|------|-----------|
| **멀티홉 질문 답변 품질** | 부분 답변 (1홉 제한) | 관련 엔티티 포함 완전 답변 | 테스트 질문 세트 20건 수동 평가 |
| **크로스도메인 자동 연결** | 하드코딩 2개 (kosha, msds) | 그래프 기반 자동 연결 | 하드코딩 제거 후 동일 결과 확인 |
| **답변 내 관련 개념 수** | 평균 1~2개 | 평균 3~5개 | 답변 내 엔티티 참조 카운트 |
| **검색 latency 증가** | baseline | +200ms 이내 | Phase 3 소요시간 측정 |
| **Fallback 정상 동작** | N/A | 그래프 장애 시 100% 기존 품질 유지 | 그래프 비활성화 테스트 |

---

## 6. 리스크 및 대응

| 리스크 | 확률 | 영향 | 대응 |
|--------|------|------|------|
| LLM 엔티티 추출 품질 낮음 | 중 | 높 | 도메인별 Few-shot 프롬프트 튜닝, 수동 검증 샘플 |
| 그래프 탐색 latency 초과 | 낮 | 중 | SQLite 인덱스 최적화, hop_depth 제한, 캐싱 |
| 엔티티 중복/불일치 | 중 | 중 | alias 매칭 + 정규화 파이프라인 |
| 기존 파이프라인 회귀 | 낮 | 높 | Phase 3 on/off 토글, A/B 테스트 |

---

## 7. 구현 순서

```
Week 1: 오프라인 그래프 구축
├── Day 1-2: DB 모델 + 엔티티 추출 프롬프트 개발
├── Day 3-4: CLI build-graph 명령어 + 배치 파이프라인
└── Day 5: semiconductor-v2 네임스페이스 그래프 구축 + 검증

Week 2: 온라인 검색 통합
├── Day 1-2: GraphSearcher 서비스 + RAG Phase 3 삽입
├── Day 3: 도메인별 설정 + RRF 융합 가중치 튜닝
├── Day 4: ContextOptimizer 엔티티 그룹핑 개선
└── Day 5: 테스트 질문 세트 평가 + Fallback 검증
```

---

## 8. 참고 자료

- Microsoft GraphRAG: https://github.com/microsoft/graphrag
- LightRAG: https://github.com/HKUDS/LightRAG
- nano-graphrag: https://github.com/gusye1234/nano-graphrag
- 현재 SafeFactory RAG Pipeline: `services/rag_pipeline.py`
