# GraphRAG Community Layer 계획서

## Executive Summary

| 항목 | 내용 |
|------|------|
| **Feature** | GraphRAG Community Layer — 커뮤니티 감지 + 요약 + Global Search |
| **Started** | 2026-03-17 |
| **Estimated Duration** | 1주 |
| **Owner** | SafeFactory Development Team |
| **Predecessor** | `graphrag` (완료, Match Rate 95%) |

### Value Delivered (4-Perspective)

| 관점 | 설명 |
|------|------|
| **Problem** | 현재 KG는 개별 엔티티-관계의 플랫한 구조로, "반도체 공정 전반의 안전 요점을 알려줘" 같은 광범위/요약형 질문에 대해 관련 청크를 산발적으로 수집할 뿐 구조화된 개요를 제공하지 못함 |
| **Solution** | 기존 KGEntity/KGRelation 테이블 위에 Leiden 커뮤니티 감지를 적용하여 엔티티 클러스터를 생성하고, 각 커뮤니티에 LLM 요약을 미리 생성한 후, Global Search 모드에서 map-reduce 방식으로 종합 답변 제공 |
| **Function UX Effect** | 사용자가 광범위한 질문을 하면 커뮤니티 요약 기반의 체계적인 개요 답변을 받을 수 있고, 기존 Local Search(상세 검색)와 자동 전환되어 질문 유형에 따른 최적 응답 |
| **Core Value** | 산업안전 지식 체계의 "개별 팩트 검색" → "구조화된 지식 개요 제공"으로 진화, 신규 작업자 교육 및 관리자 의사결정 지원 |

---

## 1. 배경

### 1.1 선행 작업 (graphrag v1.0)

2026-03-01~17 기간에 완료된 GraphRAG 기능:
- **오프라인**: Gemini Flash 기반 엔티티/관계 추출 → SQLite KG 저장
- **온라인**: Phase 3에서 N-hop CTE 탐색 → 벡터 결과와 RRF 융합
- **결과**: 멀티홉 질문 20건 100% Pass, Phase 3 latency 89ms

### 1.2 현재 한계

| 한계 | 상세 |
|------|------|
| **플랫 그래프** | 엔티티-관계만 존재, 상위 구조(클러스터/커뮤니티) 없음 |
| **Local Search만 지원** | 쿼리에 매칭되는 엔티티 주변 탐색만 가능 |
| **광범위 질문 취약** | "전체 개요", "주요 N가지" 유형 질문에 산발적 답변 |
| **도메인 내 지식 계층 없음** | 어떤 엔티티 그룹이 함께 동작하는지 파악 불가 |

### 1.3 MS GraphRAG에서 차용하는 아이디어

Microsoft GraphRAG의 핵심 중 SafeFactory에 적용할 3가지:

1. **Leiden Community Detection** — 그래프 알고리즘으로 엔티티 클러스터 자동 생성
2. **Community Summarization** — 각 커뮤니티에 대한 LLM 요약 사전 생성
3. **Global Search (map-reduce)** — 커뮤니티 요약 대상 map-reduce로 종합 답변

MS GraphRAG를 통째로 도입하지 않는 이유:
- 기존 Pinecone + SQLite 인프라와 호환 불가
- 전체 재인덱싱 비용 ($5-30+ vs $0.15 이하)
- 한국어 도메인 프롬프트 재작성 필요

---

## 2. 목표

### 2.1 핵심 목표

| # | 목표 | 측정 기준 |
|---|------|----------|
| G1 | 기존 KG 위에 커뮤니티 레이어 구축 | 4개 활성 도메인에서 커뮤니티 감지 성공 |
| G2 | 커뮤니티별 LLM 요약 생성 | 모든 커뮤니티에 요약 텍스트 존재 |
| G3 | Global Search 모드 추가 | 광범위 질문에서 map-reduce 기반 종합 답변 제공 |
| G4 | 기존 Local Search 무영향 | Phase 3 기존 동작 및 성능 유지 |

### 2.2 비목표 (Scope Out)

- 프론트엔드 커뮤니티 시각화 (별도 Phase)
- 도메인 간 cross-namespace 커뮤니티 연결
- 계층적(multi-level) 커뮤니티 (1-level만 구현)
- 실시간 커뮤니티 업데이트 (오프라인 배치만)

---

## 3. 기술 설계 개요

### 3.1 아키텍처 (3단계)

```
[기존 KG]                    [신규 커뮤니티 레이어]
KGEntity ─── KGRelation      KGCommunity ─── KGCommunityMember
    │                              │
    └── KGEntityChunk              └── community_summary (LLM 생성)
```

**단계 1: 커뮤니티 감지 (오프라인, LLM 불필요)**
```
KGEntity + KGRelation → networkx Graph → Leiden 알고리즘 → KGCommunity 저장
```

**단계 2: 커뮤니티 요약 (오프라인, LLM 1회)**
```
각 KGCommunity의 멤버 엔티티 + 설명 + 관계 → Gemini Flash → community_summary 저장
```

**단계 3: Global Search (온라인)**
```
쿼리 → 질문 유형 분류 (광범위?) → YES → 커뮤니티 요약 검색 → map-reduce → 종합 답변
                                  → NO  → 기존 Phase 3 Local Search (변경 없음)
```

### 3.2 데이터 모델 (신규 테이블)

```python
# models.py 추가
class KGCommunity(db.Model):
    id            = Integer, PK
    namespace     = String(100)       # 도메인
    community_id  = Integer           # Leiden 클러스터 번호
    title         = String(200)       # 커뮤니티 제목 (LLM 생성)
    summary       = Text              # 커뮤니티 요약 (LLM 생성)
    member_count  = Integer           # 멤버 엔티티 수
    level         = Integer, default=0  # 계층 레벨 (v1은 0만 사용)
    created_at    = DateTime

class KGCommunityMember(db.Model):
    id            = Integer, PK
    community_id  = FK → KGCommunity.id
    entity_id     = FK → KGEntity.id
    namespace     = String(100)
```

### 3.3 비용 예측

| 항목 | 비용 | 비고 |
|------|------|------|
| Leiden 감지 | **$0** | networkx + leidenalg, 순수 알고리즘 |
| 커뮤니티 요약 (Gemini Flash) | **$0.01~0.15** | 커뮤니티 수에 비례, 1회성 |
| Global Search 런타임 | **+$0.001/쿼리** | map-reduce LLM 호출 추가 |
| 신규 라이브러리 | **$0** | networkx(이미 존재 가능), leidenalg, graspologic |

### 3.4 성능 예산

| 메트릭 | 목표 |
|--------|------|
| 커뮤니티 감지 (오프라인) | < 5초 (엔티티 ~2,000개 기준) |
| 커뮤니티 요약 (오프라인) | LLM 호출 수 × ~1초 |
| Global Search 런타임 | < 3초 (map-reduce 포함) |
| 기존 Local Search 영향 | 0ms (변경 없음) |

---

## 4. 구현 범위

### 4.1 신규 파일

| 파일 | 역할 |
|------|------|
| `src/community_builder.py` | Leiden 감지 + 커뮤니티 요약 생성 (오프라인) |
| `services/community_searcher.py` | Global Search map-reduce 로직 (온라인) |

### 4.2 수정 파일

| 파일 | 변경 내용 |
|------|----------|
| `models.py` | KGCommunity, KGCommunityMember 모델 추가 |
| `services/graph_config.py` | 커뮤니티 설정 추가 (resolution, min_community_size) |
| `services/singletons.py` | `get_community_searcher()` 싱글톤 추가 |
| `services/rag_pipeline.py` | 쿼리 유형에 따른 Global/Local Search 분기 |
| `services/query_router.py` | `classify_query_type()`에 "overview" 유형 추가 |
| `main.py` | `build-community`, `community-stats` CLI 명령어 추가 |

### 4.3 구현 순서

```
Phase A: 모델 + 감지 (Day 1-2)
  ├── models.py: KGCommunity, KGCommunityMember
  ├── graph_config.py: 커뮤니티 설정 추가
  └── src/community_builder.py: Leiden 감지 + DB 저장

Phase B: 요약 생성 (Day 2-3)
  ├── community_builder.py: Gemini Flash 요약 생성
  └── main.py: build-community CLI 명령어

Phase C: Global Search (Day 3-5)
  ├── services/community_searcher.py: map-reduce 검색
  ├── services/query_router.py: overview 유형 분류
  ├── services/rag_pipeline.py: Global/Local 분기
  └── main.py: community-stats CLI 명령어

Phase D: 테스트 + 튜닝 (Day 5-7)
  ├── 4개 도메인 커뮤니티 구축
  ├── 광범위 질문 10건 수동 평가
  └── resolution 파라미터 튜닝
```

---

## 5. 리스크 및 대응

| 리스크 | 확률 | 영향 | 대응 |
|--------|------|------|------|
| KG 데이터 부족으로 커뮤니티 의미 없음 | 중 | 중 | 엔티티 <20개 도메인은 커뮤니티 비활성화 |
| Leiden resolution 튜닝 어려움 | 낮 | 낮 | 도메인별 기본값 제공, CLI에서 조정 가능 |
| Global Search 응답 시간 초과 | 낮 | 중 | 커뮤니티 요약 사전 생성으로 런타임 LLM 최소화 |
| leidenalg 설치 실패 (C 의존성) | 중 | 중 | fallback으로 networkx 내장 Louvain 사용 |
| 기존 Local Search 회귀 | 낮 | 높 | Global Search를 별도 경로로 구현, Phase 3 미수정 |

---

## 6. 성공 기준

| # | 기준 | 측정 방법 |
|---|------|----------|
| S1 | 4개 활성 도메인에서 커뮤니티 감지 성공 | `community-stats` CLI로 확인 |
| S2 | 모든 커뮤니티에 의미 있는 요약 존재 | 요약 텍스트 길이 > 50자 |
| S3 | 광범위 질문 10건에서 Global Search 활성화 | 로그에서 "Global Search" 확인 |
| S4 | Global Search 응답 시간 < 3초 | 타이밍 로그 측정 |
| S5 | 기존 Local Search 성능 0 영향 | A/B 비교 (latency 차이 < 5ms) |
| S6 | Fallback 정상 동작 | 커뮤니티 실패 시 Local Search로 전환 |

---

## 7. 의존성

| 의존성 | 상태 | 비고 |
|--------|------|------|
| GraphRAG v1.0 (기존 KG) | ✅ 완료 | KGEntity, KGRelation, KGEntityChunk |
| Gemini Flash API | ✅ 사용 중 | 요약 생성용 |
| networkx | 확인 필요 | pip install |
| leidenalg | 신규 | pip install leidenalg (igraph 의존) |
| 기존 KG 데이터 | ✅ 구축됨 | semiconductor-v2 등 |
