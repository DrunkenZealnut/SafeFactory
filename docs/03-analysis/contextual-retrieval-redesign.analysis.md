# Gap Analysis: Anthropic Contextual Retrieval 원문 기반 RAG 파이프라인 재설계

> Design 문서: `docs/02-design/features/contextual-retrieval-redesign.design.md`
> 분석일: 2026-03-07

---

## 1. Overall Match Rate

**Match Rate: 100% (14/14 items)**

---

## 2. Gap 항목별 검증 결과

### Gap 1: 청크 크기 500 → 800 토큰

| Task ID | Design 스펙 | 구현 상태 | Match |
|---------|------------|----------|-------|
| A-1a | `semantic_chunker.py:306` → `max_chunk_tokens: int = 800` | `306: max_chunk_tokens: int = 800,` | ✅ |
| A-1b | `main.py:43` → `default=800, help="...기본: 800"` | `43: default=800, help="청크당 최대 토큰 (기본: 800)"` | ✅ |

### Gap 2: RAG Pipeline top_k 10 → 20

| Task ID | Design 스펙 | 구현 상태 | Match |
|---------|------------|----------|-------|
| A-2a | `rag_pipeline.py:241` → `top_k', 20` | `241: top_k = max(1, min(int(data.get('top_k', 20)), 100))` | ✅ |
| A-2b | `rag_pipeline.py:244` → fallback `top_k = 20` | `244: top_k = 20` | ✅ |

### Gap 3: Contextual BM25 End-to-End 검증

| Task ID | Design 스펙 | 구현 상태 | Match |
|---------|------------|----------|-------|
| B-3 | 코드 변경 불필요 — 데이터 흐름 추적으로 자동 통합 확인 | 변경 없음. 데이터 흐름 검증: `chunk.content` → `prepare_vector(content)` → `metadata['content']` → `HybridSearcher.build_index()` → BM25 corpus | ✅ |

검증된 데이터 흐름:
1. `SemanticChunker.chunk_text()` → `enhanced_content = f"{prefix}\n\n{segment}"` ✅
2. `PineconeUploader.prepare_vector()` → `metadata['content'] = safe_content` (= chunk.content) ✅
3. `HybridSearcher.build_index()` → `content = doc.get('metadata', {}).get('content', '')` ✅

### Gap 4: 프롬프트 최적화

| Task ID | Design 스펙 | 구현 상태 | Match |
|---------|------------|----------|-------|
| B-1a | `_call_llm()` 프롬프트에 "50~100 토큰 이내" 명시 | `205: "검색 품질 향상이 목적이며, 50~100 토큰 이내로 핵심 맥락만 답변하세요."` | ✅ |
| B-1b | `DEFAULT_CONTEXT_PROMPT` 개선 — 검색 목적 명확화 | `51-55: "이 청크가 문서 전체에서 어떤 위치와 역할을 하는지..."` | ✅ |
| B-2 | `max_tokens` 200→150 | `215: max_tokens=150,` | ✅ |

### Gap 5: Eval 파이프라인 Recall@20 추가

| Task ID | Design 스펙 | 구현 상태 | Match |
|---------|------------|----------|-------|
| C-1a | `eval_pipeline.py` `--top-k` 기본값 5→20 | `280: default=20, help='Top K for retrieval (Anthropic recommends 20)'` | ✅ |
| C-1b | `retrieval_failure_rate` 메트릭 추가 | `218: 'retrieval_failure_rate': round(1.0 - ...)` | ✅ |
| C-1b+ | `print_report()` failure rate 출력 | `256-257: Failure Rate ... (Anthropic baseline: 5.7%)` | ✅ |

### Gap 6: 재인제스천 A/B 비교

| Task ID | Design 스펙 | 구현 상태 | Match |
|---------|------------|----------|-------|
| C-1c | `scripts/eval/compare_results.py` 신규 생성 | 파일 존재 ✅, `compare()` 함수 구현, before/after 비교, per-domain 비교, failure rate reduction 계산 | ✅ |

---

## 3. 검증 매트릭스 요약

| Gap # | 설명 | Task 수 | 완료 | Match Rate |
|-------|------|---------|------|-----------|
| Gap 1 | 청크 크기 800 | 2 | 2 | 100% |
| Gap 2 | top_k=20 | 2 | 2 | 100% |
| Gap 3 | BM25 e2e 검증 | 1 | 1 | 100% |
| Gap 4 | 프롬프트 최적화 | 3 | 3 | 100% |
| Gap 5 | Eval recall@20 | 3 | 3 | 100% |
| Gap 6 | A/B 비교 스크립트 | 1 | 1 | 100% |
| **Total** | | **14** | **14** | **100%** |

---

## 4. 잔여 실행 항목 (코드 외)

Design 문서의 Phase C에는 코드 변경 외에 **실행 항목**이 있음:

| Task ID | 항목 | 상태 | 비고 |
|---------|------|------|------|
| C-2 | Golden dataset `expected_sources` 보강 | ⏳ 미실행 | 재인제스천 후 소스 확인 필요 |
| C-3 | laborlaw 네임스페이스 A/B 재인제스천 | ⏳ 미실행 | 데이터 경로 확인 후 실행 |
| C-4 | 전체 네임스페이스 재인제스천 | ⏳ 미실행 | C-3 결과 확인 후 진행 |

이들은 **운영 실행 항목**으로 코드 Match Rate에는 포함하지 않음.

---

## 5. 결론

**Match Rate: 100%** — Design 문서의 모든 코드 변경 스펙이 정확하게 구현됨.

모든 6개 Gap이 코드 레벨에서 해소되었으며, 남은 작업은 실제 데이터를 사용한 재인제스천 및 A/B 비교 실행뿐임.
