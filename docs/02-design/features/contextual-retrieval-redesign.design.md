# Design: Anthropic Contextual Retrieval 원문 기반 RAG 파이프라인 재설계

> Plan 문서: `docs/01-plan/features/contextual-retrieval-redesign.plan.md`

---

## 1. Overview

Anthropic 원문(https://www.anthropic.com/engineering/contextual-retrieval)의 권장 파이프라인을 SafeFactory에 100% 반영하기 위한 6개 Gap 해소 설계.

### 1.1 변경 매트릭스

| Gap # | 변경 대상 | 파일 | 변경 유형 |
|-------|----------|------|----------|
| Gap 1 | 청크 크기 500→800 | `semantic_chunker.py`, `main.py` | 기본값 변경 |
| Gap 2 | top_k 10→20 | `rag_pipeline.py` | 기본값 변경 |
| Gap 3 | Contextual BM25 e2e 검증 | `hybrid_searcher.py` (검증만) | 검증/무변경 |
| Gap 4 | 프롬프트 최적화 | `context_generator.py` | 프롬프트 수정 |
| Gap 5 | Eval recall@20 추가 | `eval_pipeline.py` | 기능 추가 |
| Gap 6 | 재인제스천 A/B 비교 | `eval_pipeline.py` | 기능 추가 |

---

## 2. Gap 1: 청크 크기 500 → 800 토큰

### 2.1 변경 지점

**파일 1**: `src/semantic_chunker.py:306`
```python
# AS-IS
max_chunk_tokens: int = 500,

# TO-BE
max_chunk_tokens: int = 800,
```

**파일 2**: `main.py:43`
```python
# AS-IS
process_parser.add_argument("--max-chunk-tokens", type=int, default=500, help="청크당 최대 토큰 (기본: 500)")

# TO-BE
process_parser.add_argument("--max-chunk-tokens", type=int, default=800, help="청크당 최대 토큰 (기본: 800)")
```

### 2.2 연쇄 영향 분석

| 구성 요소 | 영향 | 조치 |
|----------|------|------|
| `min_chunk_tokens=100` | 800토큰 기준으로 비율 유지 | 변경 불필요 (100은 최소 임계값) |
| `overlap_tokens=50` | 800토큰 기준 6.25% 오버랩 → 적절 | 변경 불필요 |
| `_merge_small_segments()` | min 기준으로 병합, max로 분할 | 자동 호환 |
| `PineconeUploader.safe_content` | `max_chars=10000` → 800토큰 ≈ 2400자(한국어), 충분 | 변경 불필요 |
| Pinecone 메타데이터 40KB 제한 | 800토큰 + contextual prefix ≈ 3000자 < 10000자 한도 | 변경 불필요 |

### 2.3 근거

Anthropic 원문: "We used ~800-token chunks" — 모든 벤치마크가 이 크기 기준. 500토큰은 맥락이 부족하여 contextual prefix 효과 반감.

---

## 3. Gap 2: RAG Pipeline top_k 10 → 20

### 3.1 변경 지점

**파일**: `services/rag_pipeline.py:241,244`
```python
# AS-IS
top_k = max(1, min(int(data.get('top_k', 10)), 100))
...
top_k = 10

# TO-BE
top_k = max(1, min(int(data.get('top_k', 20)), 100))
...
top_k = 20
```

### 3.2 Reranker 호환 검증

현재 Reranker(`src/reranker.py:211-218`)는 `top_k` 파라미터를 받으며, 내부에서 `len(docs)`로 전체 입력을 처리 후 top_k개를 반환:

```python
def hybrid_rerank(self, query, docs, top_k=10, ...):
    reranked = self.rerank(query, docs, top_k=len(docs))  # 전체 재정렬
    ...
    return sorted(...)[:top_k]  # top_k개 반환
```

**결론**: Reranker는 입력 크기에 무관하게 동작. top_k=20은 자동 호환됨.

### 3.3 연쇄 영향 분석

| 구성 요소 | 영향 | 조치 |
|----------|------|------|
| `search_top_k = top_k * top_k_mult` | 20 * 3 = 60개 후보 검색 | Pinecone 호출 시간 약간 증가, 허용 범위 |
| `ContextOptimizer.deduplicate()` | 20개 입력 → 중복 제거 → 실제 10-15개 | 자동 호환 |
| LLM 컨텍스트 비용 | 800토큰 * 20개 = 16K 토큰 → Gemini 기준 적정 | 허용 범위 |
| `api/v1/search.py` | `/ask` 엔드포인트에서 `top_k` 전달 | 프론트엔드 기본값 확인 필요 |

### 3.4 프론트엔드 기본값 확인

```python
# api/v1/search.py에서 top_k는 클라이언트가 전달하거나 rag_pipeline 기본값 사용
# 클라이언트가 top_k를 명시하지 않으면 자동으로 20 적용
```

**주의**: 프론트엔드가 `top_k=5` 같은 값을 하드코딩하고 있다면 변경 불필요 (서버 기본값만 변경).

---

## 4. Gap 3: Contextual BM25 End-to-End 검증

### 4.1 데이터 흐름 추적

```
[인제스천 시]
SemanticChunker.chunk_text()
  → context_generator.generate_batch() → contextual prefix 생성
  → chunk.content = "{prefix}\n\n{segment}"    ← ✅ prefix 포함

PineconeAgent._process_text_file()
  → chunk.content → EmbeddingGenerator.generate() → embedding
  → PineconeUploader.prepare_vector(content=chunk.content)
    → metadata['content'] = safe_content(chunk.content)  ← ✅ prefix 포함

[검색 시]
HybridSearcher.build_index(documents)
  → content = doc.get('metadata', {}).get('content', '')  ← ✅ Pinecone content 필드
  → tokens = self._tokenize(content)                       ← ✅ prefix 키워드 포함
  → BM25Okapi(corpus)                                      ← ✅ Contextual BM25 자동 구성
```

### 4.2 검증 항목 체크리스트

| # | 검증 항목 | 방법 | 예상 결과 |
|---|----------|------|----------|
| V-1 | `chunk.content`에 prefix 포함 확인 | `--contextual`로 소규모 인제스천 후 chunk 출력 | prefix + segment |
| V-2 | Pinecone `content` 메타데이터에 prefix 저장 확인 | Pinecone Console 또는 `agent.search()` 결과 확인 | prefix 포함된 content |
| V-3 | BM25 corpus에 prefix 키워드 반영 확인 | `HybridSearcher._tokenize(content)` 출력 로그 | 도메인 키워드 토큰 존재 |
| V-4 | prefix 키워드로 BM25 검색 시 매칭 확인 | prefix에만 있는 키워드(예: "근로기준법")로 검색 | 해당 문서 상위 반환 |

### 4.3 검증 스크립트 (코드 변경 불필요 — 수동 검증)

```bash
# 1. 소규모 인제스천
python main.py process ./test_docs --namespace test-contextual --contextual --force --max-chunk-tokens 800

# 2. 검색으로 content 확인
python main.py search "근로기준법" --namespace test-contextual --top-k 3
# → content에 contextual prefix가 포함되어 있는지 육안 확인

# 3. BM25 매칭 확인: prefix에만 존재하는 키워드로 검색
# (원본 청크에는 없지만 prefix에 추가된 키워드)
```

**결론**: 코드 변경 불필요. 재인제스천 후 검증만 수행.

---

## 5. Gap 4: 프롬프트 최적화

### 5.1 현재 프롬프트 (AS-IS)

`src/context_generator.py:188-205`
```python
def _call_llm(self, truncated_doc, chunk, domain_instruction):
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"<document>\n{truncated_doc}\n</document>",
                "cache_control": {"type": "ephemeral"}
            },
            {
                "type": "text",
                "text": (
                    f"다음은 위 문서에서 발췌한 청크입니다:\n"
                    f"<chunk>\n{chunk[:self.MAX_CHUNK_TOKENS * 3]}\n</chunk>\n\n"
                    f"{domain_instruction}\n"
                    f"맥락 설명만 답변하세요."
                )
            }
        ]
    }]
```

### 5.2 Anthropic 원문 프롬프트 (참조)

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

**핵심 차이점**:
1. 원문은 "short succinct context" — 50~100 토큰을 명시
2. 원문은 "for the purposes of improving search retrieval" — 검색 목적 명확화
3. 현재 구현은 `max_tokens=200` → 너무 넓음, 길이 제어 없음

### 5.3 TO-BE 프롬프트

**변경 파일**: `src/context_generator.py`

**변경 1**: `_call_llm()` 프롬프트 텍스트 수정 (라인 ~201-205)

```python
# TO-BE
"text": (
    f"다음은 위 문서에서 발췌한 청크입니다:\n"
    f"<chunk>\n{chunk[:self.MAX_CHUNK_TOKENS * 3]}\n</chunk>\n\n"
    f"이 청크를 문서 전체 맥락에서 위치시키기 위한 간결한 맥락을 작성해주세요.\n"
    f"검색 품질 향상이 목적이며, 50~100 토큰 이내로 핵심 맥락만 답변하세요.\n"
    f"{domain_instruction}"
)
```

**변경 2**: `max_tokens` 200 → 150 (라인 ~208)

```python
# AS-IS
max_tokens=200,

# TO-BE
max_tokens=150,
```

### 5.4 도메인 프롬프트 개선

현재 도메인 프롬프트는 그대로 유지하되, 기본 프롬프트(`DEFAULT_CONTEXT_PROMPT`)를 Anthropic 원문에 가깝게 변경:

```python
# AS-IS
DEFAULT_CONTEXT_PROMPT = (
    "이 청크가 문서 전체에서 어떤 맥락인지 "
    "검색 품질 향상을 위해 간결하게 설명해주세요. "
    "맥락 설명만 답변하세요."
)

# TO-BE
DEFAULT_CONTEXT_PROMPT = (
    "이 청크가 문서 전체에서 어떤 위치와 역할을 하는지, "
    "검색 검색 시 이 청크를 정확히 찾을 수 있도록 간결한 맥락을 작성해주세요. "
    "50~100 토큰 이내의 맥락 설명만 답변하세요."
)
```

---

## 6. Gap 5: Eval 파이프라인 Recall@20 추가

### 6.1 변경 지점

**파일**: `scripts/eval/eval_pipeline.py`

**변경 1**: CLI 기본값 top_k=5 → 20 (라인 277)

```python
# AS-IS
parser.add_argument('--top-k', type=int, default=5, help='Top K for retrieval')

# TO-BE
parser.add_argument('--top-k', type=int, default=20, help='Top K for retrieval (Anthropic recommends 20)')
```

**변경 2**: `run_evaluation()`에서 recall@20 및 failure rate 추가 계산

```python
# overall 딕셔너리에 추가 (라인 ~211-218)
overall = {
    ...기존 필드...
    'retrieval_failure_rate': round(1.0 - overall['avg_recall_at_k'], 4),  # Anthropic 메트릭
}
```

**변경 3**: `print_report()`에 failure rate 출력 (라인 ~249-254)

```python
# 추가 출력
print(f"  Retrieval Failure Rate: {1.0 - overall['avg_recall_at_k']:.2%}  (Anthropic baseline: 5.7%)")
```

### 6.2 골든 데이터셋 보강 설계

현재 `golden_dataset.json`의 `expected_sources`가 모두 빈 배열:
```json
"expected_sources": []
```

→ recall@K 계산 시 keyword_hit_rate를 proxy로 사용 중 (sentinel -1.0).

**Phase C-2에서**: 실제 Pinecone에 인제스천된 소스 파일명으로 `expected_sources` 채우기.
방법: 각 쿼리를 수동 실행하여 정답 소스를 확인 후 golden dataset에 기록.

---

## 7. Gap 6: 재인제스천 A/B 비교

### 7.1 비교 스크립트 설계

**파일**: `scripts/eval/eval_pipeline.py`에 `--output` 플래그 활용

```bash
# Step 1: Baseline 측정 (현재 상태, 재인제스천 전)
python -m scripts.eval.eval_pipeline --top-k 20 --output results_before.json

# Step 2: laborlaw 네임스페이스 재인제스천
python main.py process ./documents/laborlaw --namespace laborlaw \
  --contextual --force --max-chunk-tokens 800

# Step 3: After 측정
python -m scripts.eval.eval_pipeline --top-k 20 --output results_after.json

# Step 4: 비교
python -m scripts.eval.compare_results results_before.json results_after.json
```

### 7.2 비교 스크립트 (NEW)

**파일**: `scripts/eval/compare_results.py` (신규)

```python
"""Compare two eval results JSON files and report improvements."""

import json
import sys

def compare(before_path, after_path):
    with open(before_path) as f:
        before = json.load(f)
    with open(after_path) as f:
        after = json.load(f)

    b = before.get('overall', {})
    a = after.get('overall', {})

    metrics = ['avg_keyword_hit', 'avg_recall_at_k', 'avg_mrr', 'avg_ndcg_at_k']
    print(f"{'Metric':<25} {'Before':>10} {'After':>10} {'Change':>10}")
    print("-" * 60)
    for m in metrics:
        bv = b.get(m, 0)
        av = a.get(m, 0)
        change = av - bv
        pct = (change / bv * 100) if bv > 0 else 0
        print(f"{m:<25} {bv:>10.4f} {av:>10.4f} {change:>+10.4f} ({pct:>+.1f}%)")

    # Failure rate comparison (Anthropic metric)
    b_fail = 1.0 - b.get('avg_recall_at_k', 0)
    a_fail = 1.0 - a.get('avg_recall_at_k', 0)
    reduction = ((b_fail - a_fail) / b_fail * 100) if b_fail > 0 else 0
    print(f"\nRetrieval Failure Rate: {b_fail:.2%} → {a_fail:.2%} ({reduction:+.1f}% reduction)")
    print(f"Anthropic target: -67% reduction")

if __name__ == '__main__':
    compare(sys.argv[1], sys.argv[2])
```

---

## 8. 구현 순서 (Implementation Order)

### Phase A: 파라미터 최적화

| 순서 | Task ID | 파일 | 변경 내용 | 난이도 |
|------|---------|------|----------|--------|
| 1 | A-1a | `src/semantic_chunker.py:306` | `max_chunk_tokens` 기본값 500→800 | 1줄 |
| 2 | A-1b | `main.py:43` | `--max-chunk-tokens` 기본값 500→800 | 1줄 |
| 3 | A-2a | `services/rag_pipeline.py:241` | `top_k` 기본값 10→20 | 1줄 |
| 4 | A-2b | `services/rag_pipeline.py:244` | fallback `top_k` 10→20 | 1줄 |

### Phase B: 프롬프트 + BM25 검증

| 순서 | Task ID | 파일 | 변경 내용 | 난이도 |
|------|---------|------|----------|--------|
| 5 | B-1a | `src/context_generator.py:201-205` | 프롬프트 텍스트 Anthropic 원문 기반 최적화 | 중간 |
| 6 | B-1b | `src/context_generator.py:56-60` | `DEFAULT_CONTEXT_PROMPT` 개선 | 중간 |
| 7 | B-2 | `src/context_generator.py:208` | `max_tokens` 200→150 | 1줄 |
| 8 | B-3 | (검증만) | Contextual BM25 e2e 검증 (코드 변경 불필요) | 수동 |

### Phase C: 평가 + 재인제스천

| 순서 | Task ID | 파일 | 변경 내용 | 난이도 |
|------|---------|------|----------|--------|
| 9 | C-1a | `scripts/eval/eval_pipeline.py:277` | `--top-k` 기본값 5→20 | 1줄 |
| 10 | C-1b | `scripts/eval/eval_pipeline.py` | failure rate 메트릭 추가 | 낮음 |
| 11 | C-1c | `scripts/eval/compare_results.py` | A/B 비교 스크립트 신규 | 낮음 |
| 12 | C-2 | `scripts/eval/golden_dataset.json` | expected_sources 보강 | 수동 |
| 13 | C-3 | CLI 실행 | laborlaw 재인제스천 + eval A/B | 실행 |
| 14 | C-4 | CLI 실행 | 전체 네임스페이스 재인제스천 | 실행 |

---

## 9. ADR (Architecture Decision Records)

### ADR-1: 청크 크기 800토큰 고정

**결정**: `max_chunk_tokens` 기본값을 800으로 변경
**근거**: Anthropic 벤치마크가 800토큰 기준. CLI에서 `--max-chunk-tokens`로 오버라이드 가능
**트레이드오프**: 총 청크 수 ~37% 감소 → Pinecone 벡터 비용 절감, 단 세밀한 검색은 약간 저하 가능

### ADR-2: top_k=20을 서버 기본값으로

**결정**: `rag_pipeline.py` 기본값 20으로 변경
**근거**: Anthropic 원문 "top-20 most effective". Reranker가 20개 중 최적을 선별
**트레이드오프**: search_top_k = 20 * 3 = 60 → Pinecone 검색 비용 약간 증가. ContextOptimizer가 압축하므로 LLM 비용은 미미

### ADR-3: 프롬프트에 토큰 길이 명시

**결정**: "50~100 토큰 이내" 프롬프트에 명시 + `max_tokens=150`
**근거**: Anthropic 원문 "50-100 tokens of context". 현재 200토큰은 과다
**트레이드오프**: 길이 제한으로 도메인 특화 정보가 잘릴 수 있으나, 검색 목적상 간결함이 더 효과적

---

## 10. 테스트 전략

### 10.1 단위 검증

| 검증 항목 | 방법 | 통과 기준 |
|----------|------|----------|
| 청크 크기 변경 | `python main.py process --help` | 기본값 800 표시 |
| top_k 변경 | `rag_pipeline.py` 코드 확인 | 기본값 20 |
| 프롬프트 출력 길이 | `--contextual`로 인제스천 후 prefix 길이 확인 | 50-150 토큰 |
| BM25 통합 | 수동 검증 (Gap 3 체크리스트) | prefix 키워드 매칭 |

### 10.2 통합 평가

| 검증 항목 | 방법 | 통과 기준 |
|----------|------|----------|
| Recall@20 baseline | `--top-k 20 --output before.json` | 측정 완료 |
| Recall@20 after | 재인제스천 후 `--top-k 20 --output after.json` | 개선 확인 |
| Failure rate 감소 | `compare_results.py` | -49% 이상 (최적 -67%) |
| 응답 지연 | 벽시계 시간 비교 | < 5초 (기존 대비 +1초 이내) |
