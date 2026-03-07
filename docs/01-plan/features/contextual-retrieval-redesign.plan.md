# Plan: Anthropic Contextual Retrieval 원문 기반 RAG 파이프라인 재설계

## Executive Summary

| 관점 | 내용 |
|------|------|
| **Problem** | 현재 구현은 Contextual Embeddings만 적용. Anthropic 원문이 권장하는 최적 파이프라인(Contextual Embeddings + Contextual BM25 + Reranking + Top-20 Retrieval)과 청크 크기(800토큰), 영문 프롬프트 등 핵심 설계가 누락됨 |
| **Solution** | 원문 벤치마크 데이터를 기준으로 인제스천/검색 파이프라인 전면 재설계: 청크 크기 800토큰, top_k=20, Contextual BM25 자동 통합 검증, 프롬프트 최적화, eval 파이프라인 Recall@20 기준 추가 |
| **Function UX Effect** | 검색 실패율 67% 감소 달성 (현재 ~35% 수준 → 목표 67%), 특히 전문용어 도메인에서 답변 정확도 대폭 향상 |
| **Core Value** | Anthropic 연구 결과를 100% 충실히 반영하여 SafeFactory RAG 품질을 연구 수준으로 끌어올림 |

---

## 1. Anthropic 원문 분석 vs 현재 구현 Gap

### 1.1 원문 핵심 권장사항 (https://www.anthropic.com/engineering/contextual-retrieval)

| # | Anthropic 권장 | 현재 SafeFactory 상태 | Gap 수준 |
|---|---------------|---------------------|---------|
| 1 | **Contextual Embeddings**: 각 청크에 LLM 맥락 프리픽스 생성 | ✅ `ContextGenerator` 구현 완료 (PR #1) | 낮음 |
| 2 | **Contextual BM25**: 맥락 프리픽스 포함된 텍스트로 BM25 인덱스 구성 | ⚠️ 자동 통합 (Pinecone content 필드 경유), 검증 미완 | 중간 |
| 3 | **Reranking**: 검색 후 Reranker로 재정렬 | ✅ 구현 완료 (Pinecone Inference + local cross-encoder) | 낮음 |
| 4 | **Top-20 Retrieval**: recall@20 기준 평가, 20개 청크 검색 | ❌ 기본 top_k=10, eval은 recall@5 기준 | **높음** |
| 5 | **청크 크기 ~800토큰** | ❌ 기본 500토큰 (`max_chunk_tokens=500`) | **높음** |
| 6 | **영문 프롬프트 템플릿** (Anthropic 원본) | ❌ 한국어 도메인 프롬프트만 존재, 원문 템플릿 미반영 | 중간 |
| 7 | **Prompt Caching** ($1.02/M tokens) | ✅ `cache_control: ephemeral` 적용 | 낮음 |
| 8 | **200K 토큰 미만 → 전체 문서 프롬프트에 포함** | ⚠️ 100K 제한 (`MAX_DOC_TOKENS`), SafeFactory 문서 대부분 이 범위 내이므로 실질 영향 낮음 | 낮음 |
| 9 | **50-100토큰 컨텍스트** 생성 | ⚠️ `max_tokens=200`으로 설정, 길이 제어 없음 | 중간 |

### 1.2 성능 벤치마크 (Anthropic 발표)

| 기법 조합 | Top-20 검색 실패율 | 감소율 |
|-----------|-------------------|--------|
| 기존 임베딩 (baseline) | 5.7% | - |
| Contextual Embeddings 단독 | 3.7% | -35% |
| Contextual Embeddings + Contextual BM25 | 2.9% | **-49%** |
| Contextual Embeddings + Contextual BM25 + Reranking | 1.9% | **-67%** |

**핵심**: SafeFactory는 이미 Reranking을 갖고 있으므로, Contextual Embeddings + BM25 통합만 올바르게 작동하면 **-67% 달성 가능**.

---

## 2. 재설계 항목 (6개 Gap 해소)

### Gap 1: 청크 크기 500 → 800토큰

**현재**: `max_chunk_tokens=500` (CLI 기본값, SemanticChunker 기본값)
**변경**: 기본값을 `800`으로 변경

**근거**: Anthropic 원문에서 800토큰 청크가 최적 성능. 500토큰은 맥락이 부족하여 contextual prefix의 효과가 반감됨.

**영향 범위**:
- `src/semantic_chunker.py`: `max_chunk_tokens` 기본값 500 → 800
- `main.py`: `--max-chunk-tokens` 기본값 500 → 800
- 기존 데이터: 재인제스천 필요 (청크 크기 변경)

**위험**: 청크가 커지면 총 청크 수 감소 → Pinecone 벡터 수 감소 → 비용 절감. 단, LLM 컨텍스트 윈도우 내 청크 수는 줄어듦.

### Gap 2: Top-K 기본값 10 → 20

**현재**: `top_k=10` (RAG 파이프라인 기본값)
**변경**: `top_k=20` (Anthropic 원문 권장)

**근거**: Anthropic은 recall@20 기준으로 평가하며, top-20이 top-5/10보다 일관적으로 우수. Reranker가 있으므로 20개 중 최상위를 선별 가능.

**영향 범위**:
- `services/rag_pipeline.py`: top_k 기본값 10 → 20
- `api/v1/search.py`: `/ask` 엔드포인트 기본 top_k
- Reranker: 20개 입력 → 상위 N개 선별 (현재 로직 호환)

**위험**: LLM 컨텍스트에 더 많은 청크 포함 → 토큰 비용 증가. ContextOptimizer가 중복 제거하므로 실질 증가는 제한적.

### Gap 3: Contextual BM25 통합 검증

**현재**: "자동 통합" 가정 (Pinecone content 필드 → HybridSearcher)
**필요**: 실제로 contextual prefix가 BM25 corpus에 반영되는지 **end-to-end 검증**

**검증 항목**:
1. `PineconeUploader.prepare_vector()`가 `chunk.content`를 Pinecone `content` 메타데이터에 저장하는지 확인
2. `HybridSearcher.build_index()`가 해당 content를 읽어 BM25 corpus를 구성하는지 확인
3. contextual prefix에 포함된 키워드가 BM25 검색에서 실제 매칭되는지 테스트

**결과**: 검증 통과 시 코드 변경 불필요. 실패 시 HybridSearcher 수정.

### Gap 4: 프롬프트 최적화

**현재**: 한국어 도메인 프롬프트만 존재, Anthropic 원문 영문 템플릿 미반영
**변경**: Anthropic 원문 템플릿을 기반으로 한국어 최적화

**Anthropic 원문 프롬프트**:
```
<document>
{{WHOLE_DOCUMENT}}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{{CHUNK_CONTENT}}
</chunk>
Please give a short succinct context to situate this chunk within the overall
document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else.
```

**SafeFactory 최적화 프롬프트** (제안):
```
<document>
{document}
</document>
다음은 위 문서에서 발췌한 청크입니다:
<chunk>
{chunk}
</chunk>
이 청크를 문서 전체 맥락에서 위치시키기 위한 간결한 맥락을 작성해주세요.
검색 품질 향상이 목적이며, 50~100 토큰 이내로 핵심 맥락만 답변하세요.
{domain_instruction}
```

**변경**: `max_tokens=200` → `max_tokens=150` + 프롬프트에 "50~100 토큰" 명시

### Gap 5: Eval 파이프라인 Recall@20 추가

**현재**: `scripts/eval/eval_pipeline.py`에서 recall@K (K=5 기본)
**변경**: Recall@20을 primary metric으로 추가

**근거**: Anthropic 원문의 핵심 메트릭이 "top-20 retrieval failure rate" (= 1 - recall@20).

**영향 범위**:
- `scripts/eval/eval_pipeline.py`: `--top-k 20` 기본값 또는 recall@20 별도 계산
- Golden dataset: expected_sources 보강 (현재 빈 배열)

### Gap 6: 재인제스천 전략 및 A/B 비교

**현재**: 재인제스천 방법은 `--contextual --force`로 수동
**필요**: 재인제스천 전후 A/B 비교 자동화

**계획**:
1. 1개 네임스페이스(laborlaw)에 대해 `--contextual --force --max-chunk-tokens 800`으로 재인제스천
2. eval 파이프라인으로 before/after recall@20 비교
3. 유의미한 개선 확인 후 나머지 네임스페이스 진행

---

## 3. 구현 단계

### Phase A: 파라미터 최적화 (청크 크기 + Top-K)

| ID | 항목 | 파일 | 난이도 |
|----|------|------|--------|
| A-1 | 청크 기본값 500 → 800 | `semantic_chunker.py`, `main.py` | 낮음 |
| A-2 | RAG top_k 기본값 10 → 20 | `rag_pipeline.py` | 낮음 |
| A-3 | Reranker 입력 20개 호환 검증 | `reranker.py` | 낮음 |

### Phase B: 프롬프트 + 컨텍스트 최적화

| ID | 항목 | 파일 | 난이도 |
|----|------|------|--------|
| B-1 | 프롬프트 한국어 최적화 (원문 구조 반영) | `context_generator.py` | 중간 |
| B-2 | max_tokens 200 → 150 + 길이 제어 | `context_generator.py` | 낮음 |
| B-3 | Contextual BM25 e2e 검증 | `hybrid_searcher.py` 검증 | 중간 |

### Phase C: 평가 + 재인제스천

| ID | 항목 | 파일 | 난이도 |
|----|------|------|--------|
| C-1 | Eval recall@20 메트릭 추가 | `eval_pipeline.py` | 낮음 |
| C-2 | Golden dataset expected_sources 보강 | `golden_dataset.json` | 중간 |
| C-3 | laborlaw 네임스페이스 A/B 재인제스천 | CLI 실행 | 실행 |
| C-4 | 전체 네임스페이스 재인제스천 | CLI 실행 | 실행 |

---

## 4. 성공 기준

| 메트릭 | 현재 (추정) | 목표 | 측정 방법 |
|--------|-----------|------|----------|
| Recall@20 | TBD (baseline 측정 필요) | +49% 이상 개선 | Golden Dataset eval |
| Top-20 검색 실패율 | TBD | < 3% | 1 - Recall@20 |
| Keyword Hit Rate | TBD | +30% 이상 | Contextual BM25 효과 |
| 청크 크기 | 500 토큰 | 800 토큰 | 설정 확인 |
| 컨텍스트 길이 | 미제어 | 50-100 토큰 | 출력 분석 |

---

## 5. 위험 및 완화

| 위험 | 영향 | 완화 |
|------|------|------|
| 800토큰 청크로 LLM 컨텍스트 초과 | 답변 품질 저하 | ContextOptimizer 중복제거 + 800*20=16K는 대부분 모델 범위 내 |
| 재인제스천 비용 (Haiku API) | ~$3-5/10K 청크 | SQLite 캐시로 재실행 시 비용 0 |
| top_k=20으로 응답 지연 | 사용자 경험 | Reranker가 상위 선별, ContextOptimizer가 압축 |
| 기존 데이터와 새 데이터 혼재 | 검색 품질 불균일 | 네임스페이스 단위 일괄 재인제스천 |

---

## 6. 일정

| Phase | 내용 | 예상 |
|-------|------|------|
| A | 파라미터 최적화 | 코드 변경 소량 |
| B | 프롬프트 + BM25 검증 | 코드 + 테스트 |
| C | 평가 + 재인제스천 | 실행 + 비교 |
