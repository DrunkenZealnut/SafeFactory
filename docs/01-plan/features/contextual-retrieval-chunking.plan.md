# Plan: Anthropic Contextual Retrieval 기반 청킹 구현

## Executive Summary

| 관점 | 내용 |
|------|------|
| **Problem** | 현재 청킹 방식은 `[문서: X \| 섹션: Y]` 형태의 정적 프리픽스만 추가하여, 청크가 문서 전체 맥락에서 분리되면 검색 시 의미를 잃는 "contextual isolation" 문제 발생 |
| **Solution** | Anthropic Contextual Retrieval 기법을 적용하여 LLM이 각 청크에 문서 맥락을 설명하는 contextual prefix를 생성, 이를 임베딩과 BM25 인덱스에 모두 반영 |
| **Function UX Effect** | 검색 실패율(top-20 retrieval failure rate) 49~67% 감소 예상, 특히 노동법/MSDS 등 전문용어 밀도 높은 도메인에서 정확도 대폭 향상 |
| **Core Value** | 문서-청크 간 의미적 연결 보존으로 RAG 파이프라인 전체 품질 향상, 사용자 질문에 대한 답변 정확도 및 신뢰도 향상 |

---

## 1. Background & Motivation

### 1.1 Anthropic Contextual Retrieval이란?

Anthropic이 2024년 발표한 검색 품질 향상 기법으로, 핵심 아이디어는 단순합니다:

> **각 청크를 임베딩하기 전에, LLM이 해당 청크의 문서 내 맥락을 설명하는 짧은 문장(contextual prefix)을 생성하여 청크 앞에 추가한다.**

#### 성능 개선 수치 (Anthropic 벤치마크)
| 기법 | Top-20 검색 실패율 | 감소율 |
|------|-------------------|--------|
| 기존 임베딩 | 5.7% | baseline |
| Contextual Embeddings | 3.7% | -35% |
| Contextual Embeddings + Contextual BM25 | 2.9% | -49% |
| + Reranking | 1.9% | **-67%** |

#### 핵심 프롬프트 (Anthropic 원본)
```
<document>
{{WHOLE_DOCUMENT}}
</document>
Here is the chunk we want to situate within the whole document:
<chunk>
{{CHUNK_CONTENT}}
</chunk>
Please give a short succinct context to situate this chunk within the overall
document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else.
```

### 1.2 왜 SafeFactory에 필요한가?

SafeFactory는 5개 도메인의 한국어 전문 문서를 다룹니다. 현재 청킹 시스템의 한계:

1. **정적 프리픽스의 한계**: `[문서: CVD공정 | 섹션: 학습1]` 형태는 메타데이터만 전달하고, 청크 내용이 문서 흐름에서 어떤 위치/역할인지 설명하지 못함
2. **맥락 단절 사례**: "3%의 수익 성장" → 어떤 회사? 어떤 분기? (SafeFactory 예: "노출기준 TWA 50ppm" → 어떤 물질?)
3. **도메인별 전문성**: 노동법의 "제23조" 청크는 "근로기준법 해고 등의 제한" 맥락이 없으면 검색 매칭 실패
4. **BM25 한계**: 현재 BM25는 원본 청크 텍스트만 사용하므로 contextual prefix의 키워드 부스트 미적용

---

## 2. As-Is 분석 (현재 청킹 시스템)

### 2.1 현재 아키텍처

```
FileLoader → SemanticChunker.chunk_text() → EmbeddingGenerator → PineconeUploader
                    │
                    ├── _split_by_structure() / _split_by_ncs_structure() / ...
                    ├── _merge_small_segments()
                    ├── _add_overlap()
                    └── _add_contextual_prefix()  ← 정적 프리픽스만
```

### 2.2 현재 `_add_contextual_prefix()` (semantic_chunker.py:809-828)

```python
def _add_contextual_prefix(self, content, document_title, section_title, source_file):
    prefix_parts = []
    if document_title:
        prefix_parts.append(f"문서: {document_title}")
    if section_title:
        prefix_parts.append(f"섹션: {section_title}")
    if prefix_parts:
        prefix = " | ".join(prefix_parts)
        return f"[{prefix}]\n\n{content}"
    return content
```

**문제점**:
- LLM 기반 맥락 설명 없음 (메타데이터 라벨만 부착)
- 문서 전체 내용을 고려하지 않음
- 도메인 특화 맥락 부재 (법률명, 물질명, 공정명 등)

### 2.3 현재 Chunk dataclass 필드

```python
@dataclass
class Chunk:
    content: str                              # 프리픽스 + 본문
    document_title: Optional[str] = None      # 문서 제목
    document_summary: Optional[str] = None    # 문서 요약 (첫 번째 청크만)
    section_title: Optional[str] = None       # 섹션 제목
```

`document_summary`는 첫 번째 청크에만 저장되고, LLM이 생성한 것이 아닌 첫 문단 추출 방식.

### 2.4 임베딩 파이프라인 (agent.py:350)

```python
embeddings = self.embedding_generator.generate_batch(batch_texts)
vector = self.pinecone_uploader.prepare_vector(
    embedding=embedding.embedding,
    content=chunk.content,     # ← 프리픽스 포함된 텍스트
    source_file=chunk.source_file,
    chunk_index=chunk.index,
    metadata=chunk.metadata,
)
```

임베딩과 Pinecone 메타데이터에 동일 텍스트 사용 → Contextual Retrieval은 이 텍스트에 LLM 맥락을 추가하는 것.

---

## 3. Critical Issues

### C1: 정적 프리픽스의 검색 기여도 낮음
- 현재 `[문서: X | 섹션: Y]` 형태는 임베딩 벡터에 미미한 영향
- 실제 의미적 맥락(이 청크가 무엇에 대해 말하고 있는지)을 전달하지 못함
- **영향**: 맥락 의존적 청크의 검색 정확도 저하

### C2: LLM 호출 비용 및 지연시간
- Contextual Retrieval은 **모든 청크마다** LLM 호출 필요
- SafeFactory의 총 청크 수 ~수만 개 → 재인제스천 시 상당한 비용/시간
- **영향**: 인제스천 파이프라인 속도 및 비용

### C3: 프롬프트 캐싱 미활용
- Anthropic은 Prompt Caching으로 비용 90% 절감 가능하다고 제시
- 현재 SafeFactory는 임베딩에 OpenAI, 답변에 Gemini/Claude 사용 → 멀티 프로바이더 환경
- **영향**: 비용 최적화 경로 설계 필요

### C4: BM25 인덱스가 contextual prefix 미포함
- 현재 `HybridSearcher`는 검색 시점에 BM25 corpus를 구축
- 원본 content 필드만 사용하므로 contextual prefix가 BM25 매칭에 기여하지 않음
- **영향**: Contextual BM25의 효과를 살리려면 Pinecone 메타데이터에 contextual content 저장 필요

### C5: 전체 문서 길이 제한
- Contextual Retrieval 프롬프트에 `{{WHOLE_DOCUMENT}}`를 포함해야 함
- 일부 NCS/노동법 문서는 수만 토큰 → LLM 컨텍스트 윈도우 초과 가능
- **영향**: 큰 문서는 요약본 또는 섹션 단위로 맥락 제공 필요

---

## 4. Improvement Opportunities

### I1: LLM 기반 Contextual Prefix 생성
- 각 청크에 대해 LLM이 2~3문장의 맥락 설명 생성
- 도메인별 프롬프트 커스터마이징 (법률명, 물질명, 공정명 포함 유도)

### I2: Prompt Caching을 통한 비용 최적화
- 동일 문서의 청크들은 `{{WHOLE_DOCUMENT}}` 부분이 동일
- Anthropic Claude의 Prompt Caching 또는 배치 처리로 비용 90% 절감 가능

### I3: Contextual BM25 적용
- contextual prefix가 포함된 텍스트를 BM25 corpus로 사용
- "근로기준법 해고 제한"이 contextual prefix에 포함되면 BM25 키워드 매칭 향상

### I4: 비동기/병렬 처리
- 문서 내 청크들을 비동기로 병렬 처리하여 인제스천 속도 개선
- 배치 크기 및 동시성 제어

### I5: 선택적 Contextual Retrieval
- 모든 청크에 적용하면 비용 과다 → 도메인/문서 크기별 선택 적용
- 예: 노동법/MSDS 도메인 우선 적용, NCS는 기존 방식 유지 옵션

### I6: 캐시 저장소
- 생성된 contextual prefix를 로컬 캐시(SQLite/JSON)에 저장
- 재인제스천 시 변경된 청크만 새로 생성

### I7: A/B 테스트 프레임워크
- 기존 정적 프리픽스 vs Contextual Retrieval 성능 비교
- Golden Dataset 기반 Recall@K, MRR, NDCG@K 측정 (기존 eval 파이프라인 활용)

---

## 5. Implementation Plan

### Phase A: Core — LLM Contextual Prefix Generator (MVP)

| ID | Task | Description | Priority |
|----|------|-------------|----------|
| A-1 | ContextGenerator 클래스 생성 | `src/context_generator.py` — LLM 호출로 청크별 맥락 생성 | P0 |
| A-2 | 도메인별 프롬프트 템플릿 | 5개 도메인 각각에 최적화된 contextual prompt | P0 |
| A-3 | SemanticChunker 통합 | `_add_contextual_prefix()` 대체 또는 확장 | P0 |
| A-4 | 긴 문서 처리 전략 | 문서 > 100K 토큰 시 섹션 요약 기반 맥락 생성 | P0 |
| A-5 | 비용 추적 로깅 | 토큰 사용량, 비용, 처리 시간 로깅 | P1 |

### Phase B: Optimization — 비용/속도 최적화

| ID | Task | Description | Priority |
|----|------|-------------|----------|
| B-1 | Prompt Caching 적용 | Claude API의 prompt caching으로 동일 문서 청크 비용 절감 | P0 |
| B-2 | 비동기 병렬 처리 | asyncio 기반 동시 청크 처리 (configurable concurrency) | P1 |
| B-3 | Contextual prefix 캐시 | SQLite 기반 캐시 — content hash → contextual prefix 매핑 | P1 |
| B-4 | 배치 모드 인제스천 | `main.py`에 `--contextual` 플래그 추가 | P1 |
| B-5 | 선택적 적용 설정 | `.env` 또는 admin 패널에서 도메인별 on/off | P2 |

### Phase C: Evaluation — 품질 측정 및 비교

| ID | Task | Description | Priority |
|----|------|-------------|----------|
| C-1 | A/B 비교 평가 | 기존 정적 프리픽스 vs Contextual Retrieval, Golden Dataset 기반 | P0 |
| C-2 | 도메인별 효과 분석 | 5개 도메인 각각의 검색 품질 변화 측정 | P0 |
| C-3 | 비용-성능 ROI 분석 | 인제스천 비용 증가 대비 검색 품질 향상 수치화 | P1 |
| C-4 | Contextual BM25 효과 측정 | BM25에 contextual text 사용 시 하이브리드 검색 개선 측정 | P1 |

---

## 6. Technical Design Sketch

### 6.1 ContextGenerator 인터페이스

```python
class ContextGenerator:
    def __init__(self, provider='anthropic', model='claude-haiku-4-5-20251001',
                 cache_enabled=True):
        ...

    def generate_context(self, document: str, chunk: str,
                         domain: str = '', metadata: dict = None) -> str:
        """Generate contextual prefix for a chunk within its document."""
        ...

    def generate_batch(self, document: str, chunks: List[str],
                       domain: str = '') -> List[str]:
        """Generate contextual prefixes for all chunks in a document.
        Uses prompt caching — document sent once, chunks iterated."""
        ...
```

### 6.2 도메인별 프롬프트 예시

**노동법 (laborlaw)**:
```
<document>
{{WHOLE_DOCUMENT}}
</document>
이 청크는 위 노동법 문서에서 발췌한 것입니다:
<chunk>
{{CHUNK_CONTENT}}
</chunk>
이 청크가 어떤 법률의 어떤 조항에 해당하며, 문서 전체에서 어떤 맥락인지
검색 품질 향상을 위해 간결하게 설명해주세요. 법률명과 조항 번호를 반드시 포함하세요.
맥락 설명만 답변하세요.
```

**MSDS (msds)**:
```
<document>
{{WHOLE_DOCUMENT}}
</document>
이 청크는 위 화학물질 안전 문서에서 발췌한 것입니다:
<chunk>
{{CHUNK_CONTENT}}
</chunk>
이 청크가 어떤 화학물질의 어떤 안전 정보(유해성/취급/응급조치 등)에 대한 것인지
검색 품질 향상을 위해 간결하게 설명해주세요. 물질명과 CAS 번호를 포함하세요.
맥락 설명만 답변하세요.
```

### 6.3 데이터 흐름 변경

```
Before:
  chunk_text → _add_contextual_prefix(static) → embed → upload

After:
  chunk_text → ContextGenerator.generate_context(LLM) → prepend to chunk
             → embed(contextual chunk) → upload(contextual chunk as content)
```

### 6.4 비용 추정

| 모델 | 입력 단가 | 출력 단가 | 청크당 예상 비용 | 10,000청크 총비용 |
|------|----------|----------|----------------|-----------------|
| Claude Haiku 4.5 | $0.80/MTok | $4/MTok | ~$0.002 | ~$20 |
| Claude Haiku 4.5 + Caching | $0.08/MTok (cached) | $4/MTok | ~$0.0003 | ~$3 |
| GPT-4o-mini | $0.15/MTok | $0.60/MTok | ~$0.0004 | ~$4 |

> Prompt Caching이 가능한 Claude Haiku를 사용하면 동일 문서 내 여러 청크에서 90% 비용 절감.

---

## 7. Risk Matrix

| Risk | Probability | Impact | Mitigation |
|------|:-----------:|:------:|------------|
| LLM 비용 초과 | Medium | High | Prompt Caching + Haiku 사용, 도메인별 선택 적용 |
| 인제스천 속도 저하 | High | Medium | 비동기 병렬 처리, 캐시, 배치 모드 |
| LLM 환각 (잘못된 맥락 생성) | Low | Medium | 도메인 특화 프롬프트, 생성 결과 검증 로직 |
| 긴 문서 컨텍스트 초과 | Medium | Medium | 섹션 단위 요약 기반 맥락 생성 |
| 기존 인덱스 호환성 | Low | Low | 기존 namespace 유지, 새 namespace로 A/B 비교 |
| BM25 인덱스 불일치 | Low | Medium | HybridSearcher가 Pinecone content 필드 사용 → 자동 반영 |

---

## 8. Success Criteria

| Metric | Baseline | Target | Method |
|--------|----------|--------|--------|
| Keyword Hit Rate | TBD (현재 baseline) | +15% | Golden Dataset eval |
| Recall@5 | TBD | +20% | Golden Dataset eval |
| MRR | TBD | +0.05 | Golden Dataset eval |
| NDCG@5 | TBD | +0.05 | Golden Dataset eval |
| 인제스천 비용 증가 | $0 | < $5/10K청크 | LLM 비용 로깅 |
| 인제스천 속도 | TBD | < 2x slowdown | 타이밍 로깅 |

> Baseline은 `scripts/eval/eval_pipeline.py`로 측정 (이전 PDCA에서 구축 완료)

---

## 9. Dependencies

- **기존 평가 인프라**: `scripts/eval/eval_pipeline.py`, `scripts/eval/golden_dataset.json` (구축 완료)
- **기존 RAG 파이프라인**: Phase-by-phase 레이턴시 계측 (구축 완료)
- **Claude API 접근**: Anthropic API key (`ANTHROPIC_API_KEY` 환경변수)
- **Prompt Caching**: Claude API beta feature — cache_control 파라미터 사용

---

## 10. References

- [Anthropic Contextual Retrieval Blog Post](https://www.anthropic.com/news/contextual-retrieval)
- [Anthropic Contextual Retrieval Appendix II](https://assets.anthropic.com/m/1632cded0a125333/original/Contextual-Retrieval-Appendix-2.pdf)
- [DataCamp Implementation Guide](https://www.datacamp.com/tutorial/contextual-retrieval-anthropic)
- [Anthropic Cookbook - Contextual Embeddings](https://deepwiki.com/anthropics/anthropic-cookbook/3.2-contextual-embeddings)
