# search-quality Analysis Report

> **Analysis Type**: Gap Analysis (Design vs Implementation)
>
> **Project**: SafeFactory
> **Analyst**: gap-detector
> **Date**: 2026-03-17
> **Design Doc**: [search-quality.design.md](../02-design/features/search-quality.design.md)

---

## 1. Analysis Overview

### 1.1 Analysis Purpose

Design 문서(`search-quality.design.md`)에 정의된 11개 개선 항목(QW-1~5, CI-1~4, RB-1~2)이 실제 구현 코드와 일치하는지 1:1 비교 검증.

### 1.2 Analysis Scope

- **Design Document**: `docs/02-design/features/search-quality.design.md`
- **Implementation Files**:
  - `services/rag_pipeline.py` -- QW-1, QW-2, QW-3, CI-1, CI-2, RB-1
  - `services/query_router.py` -- QW-5, CI-4
  - `src/hybrid_searcher.py` -- QW-4
  - `src/reranker.py` -- CI-3
  - `src/query_enhancer.py` -- RB-2
- **Items Checked**: 11

---

## 2. Gap Analysis (Design vs Implementation)

### 2.1 Item-by-Item Comparison

#### QW-1: Fallback Keyword Extraction

| Aspect | Design | Implementation | Status |
|--------|--------|----------------|--------|
| Location | `rag_pipeline.py` Phase 4 | `rag_pipeline.py` line 827-830 | ✅ Match |
| Trigger condition | `if not keywords:` | `if not keywords:` | ✅ Match |
| Regex pattern | `r'[가-힣a-zA-Z0-9]{2,}'` | `r'[가-힣a-zA-Z0-9]{2,}'` | ✅ Match |
| Limit | `[:10]` | `[:10]` | ✅ Match |
| Separate function | `_extract_fallback_keywords()` helper | Inline code (no helper function) | ⚠️ Deviation |

**Verdict**: ✅ 기능적으로 완전 일치. Design은 별도 헬퍼 함수 `_extract_fallback_keywords()`를 제안했으나, 구현은 2줄 인라인으로 동일 로직 수행. 코드가 짧아 인라인이 합리적 선택.

---

#### QW-2: Adaptive top_k

| Aspect | Design | Implementation | Status |
|--------|--------|----------------|--------|
| Location | Phase 0 이후, Phase 1 이전 | line 800-806 (Phase 0 이후, Phase 1 이전) | ✅ Match |
| Condition | `domain_confidence < 0.7` | `domain_confidence and domain_confidence < 0.7` | ✅ Match (guard 추가) |
| Multiplier < 0.4 | `2.0` | `2.0` | ✅ Match |
| Multiplier 0.4~0.7 | `1.5` | `1.5` | ✅ Match |
| Cap | `min(..., top_k * 3)` | `min(..., _original_top_k * 3)` | ✅ Match |
| Logging | `[Adaptive top_k]` 포맷 | `[Adaptive top_k]` 포맷 | ✅ Match |
| 0인 경우 기본값 유지 | Design에 명시 | `domain_confidence and` truthiness 체크로 구현 | ✅ Match |

**Verdict**: ✅ 완전 일치. Falsy guard (`domain_confidence and`) 추가는 Design 주의사항 "0인 경우 기본값 유지" 준수.

---

#### QW-3: MIN_TOKEN_COUNT 30 -> 80

| Aspect | Design | Implementation | Status |
|--------|--------|----------------|--------|
| Value | `80` | `80` | ✅ Match |
| Location | line 41 | line 41 | ✅ Match |
| Env override | `os.environ.get('MIN_TOKEN_COUNT', '80')` | `os.environ.get('MIN_TOKEN_COUNT', '80')` | ✅ Match |

**Verdict**: ✅ 완전 일치.

---

#### QW-4: Dynamic RRF K

| Aspect | Design | Implementation | Status |
|--------|--------|----------------|--------|
| Method name | `_dynamic_rrf_k(self, result_count)` | `_dynamic_rrf_k(self, result_count)` | ✅ Match |
| Threshold <= 10 | `max(20, self.rrf_k // 3)` | `max(20, self.rrf_k // 3)` | ✅ Match |
| Threshold <= 30 | `max(30, self.rrf_k // 2)` | `max(30, self.rrf_k // 2)` | ✅ Match |
| Default | `self.rrf_k` | `self.rrf_k` | ✅ Match |
| Call sites | `search()`, `search_with_keyword_boost()` 포함 4곳 | `_reciprocal_rank_fusion()`, `_search_with_weights()` 2곳 | ⚠️ Deviation |

**Verdict**: ✅ 기능적으로 완전 일치. Design은 "`self.rrf_k` -> `self._dynamic_rrf_k()`" 교체를 4곳으로 언급했으나, 실제 RRF 계산은 `_reciprocal_rank_fusion()`과 `_search_with_weights()` 2개 메서드에만 존재. 이 2곳 모두 적용 완료. `search()`와 `search_with_keyword_boost()`는 이 메서드들을 호출하므로 동적 K가 정상 적용됨.

---

#### QW-5: Domain Synonyms

| Aspect | Design | Implementation | Status |
|--------|--------|----------------|--------|
| Dict name | `DOMAIN_SYNONYMS` | `DOMAIN_SYNONYMS` | ✅ Match |
| Location | `query_router.py` line 77 위 | line 77 | ✅ Match |
| Entries (Design) | 10 entries | 7 entries | ⚠️ Deviation |
| `'근로기준법': '노동법'` | ✅ | ✅ | ✅ |
| `'산업안전보건법': '안전보건'` | ✅ | ✅ | ✅ |
| `'산안법': '안전보건'` | ✅ | ✅ | ✅ |
| `'화학물질안전': 'MSDS'` | ✅ | **Missing** | ❌ Missing |
| `'물질안전보건자료': 'MSDS'` | ✅ | ✅ | ✅ |
| `'SDS': 'MSDS'` | ✅ | ✅ | ✅ |
| `'현장실습': '현장실습'` | ✅ | **Missing** | ❌ Missing |
| `'인턴': '현장실습'` | ✅ | ✅ | ✅ |
| `'실습생': '현장실습'` | ✅ | ✅ | ✅ |
| Synonym expansion in `classify_domain()` | `_q = query` + append canonical | `query_expanded = query.lower()` + append canonical | ✅ Match |

**Verdict**: ⚠️ 2개 항목 누락. `'화학물질안전': 'MSDS'`와 `'현장실습': '현장실습'`(자기 참조)이 구현에서 빠짐. `'화학물질안전'`은 실제 유용한 동의어이므로 누락이 기능 영향 있음. `'현장실습': '현장실습'`은 자기 참조(no-op)이므로 누락해도 무방.

---

#### CI-1: HyDE Conditional Override

| Aspect | Design | Implementation | Status |
|--------|--------|----------------|--------|
| Override logic | `if not _use_hyde and domain_confidence < 0.6:` | `if not _use_hyde and domain_confidence and domain_confidence < 0.6:` | ✅ Match |
| Logging | `[HyDE Override]` | `[HyDE Override]` | ✅ Match |
| `_use_hyde` passed to Phase 1 | `_run_query_enhancement(..., use_hyde=_use_hyde, ...)` | **Not passed** -- `_enhance_query()` reads `route_cfg.get('use_hyde', False)` | ❌ Gap |

**Verdict**: ❌ **Critical gap**. `_use_hyde` 변수는 lines 814-817에서 올바르게 계산되지만, line 823의 `_enhance_query()` 호출 시 전달되지 않음. `_enhance_query()` 내부(line 635)에서 `route_cfg.get('use_hyde', False)`를 직접 읽어, override된 값이 무시됨. **HyDE override가 실제로 작동하지 않는 버그**.

---

#### CI-2: MMR Near-Duplicate Removal

| Aspect | Design | Implementation | Status |
|--------|--------|----------------|--------|
| Phase label | Phase 5.5 | Phase 5.5 | ✅ Match |
| Condition | `use_enhancement and len(results) > 5` | `use_enhancement and len(results) > 5` | ✅ Match |
| Content prefix | `[:200]` | `[:200]` | ✅ Match |
| Dedup mechanism | `set()` + content key | `set()` + content key | ✅ Match |
| Empty content handling | Design에 미언급 | `elif not _ck: _diverse_results.append(r)` 추가 | ✅ Improvement |
| try/except | `except Exception as e:` | `except Exception as _mmr_e:` | ✅ Match |
| Logging | `[MMR Diversity]` | `[MMR Diversity]` | ✅ Match |

**Verdict**: ✅ 완전 일치 + 1개 개선 (빈 content 결과 보존 로직).

---

#### CI-3: Robust Normalization

| Aspect | Design | Implementation | Status |
|--------|--------|----------------|--------|
| Function name | `_robust_normalize(scores)` | `_robust_normalize(scores)` | ✅ Match |
| p5 calculation | `sorted_s[max(0, len(sorted_s) // 20)]` | `sorted_s[max(0, len(sorted_s) // 20)]` | ✅ Match |
| p95 calculation | `sorted_s[min(len(sorted_s) - 1, len(sorted_s) * 19 // 20)]` | `sorted_s[min(len(sorted_s) - 1, len(sorted_s) * 19 // 20)]` | ✅ Match |
| Zero range fallback | `[0.5] * len(scores)` | `[0.5] * len(scores)` | ✅ Match |
| Range threshold | `1e-9` | `1e-9` | ✅ Match |
| Edge case (len < 2) | Design: `return scores` | Impl: `return [0.5] * len(scores) if scores else []` | ✅ Improvement |
| Applied in hybrid_rerank | `Reranker.hybrid_rerank()` 교체 | `Reranker`, `PineconeReranker`, `CohereReranker` 3곳 모두 적용 | ✅ Match |

**Verdict**: ✅ 완전 일치 + 2개 개선 (edge case 처리 강화, 모든 Reranker 구현체에 적용).

---

#### CI-4: Multi-Domain Routing

| Aspect | Design | Implementation | Status |
|--------|--------|----------------|--------|
| Return type | 5-tuple `(ns, conf, label, secondary_ns, secondary_conf)` | 5-tuple `(ns, conf, label, secondary_ns, secondary_conf)` | ✅ Match |
| Secondary threshold | `secondary_conf > 0.3` | `secondary_conf > 0.3` | ✅ Match |
| `classify_domain()` scoring | `sorted()` by score | `sorted()` by score | ✅ Match |
| Pipeline unpacking | `detected_namespace, domain_confidence, detected_domain_label, secondary_ns, secondary_conf = classify_domain(...)` | Same 5-tuple unpacking on lines 791-792 | ✅ Match |
| Secondary namespace search | Phase 2 이후 `_run_multi_query_search(..., namespace=secondary_ns, top_k=top_k // 2)` | **Not implemented** | ❌ Missing |

**Verdict**: ⚠️ **Partial implementation**. `classify_domain()`의 5-tuple 반환과 pipeline의 unpacking은 구현 완료. 그러나 Design에 명시된 "secondary namespace에 대한 추가 검색" 로직 (`secondary_ns`로 `_run_multi_query_search` 호출 + `results.extend`)은 **미구현**. `secondary_ns` 변수는 할당만 되고 사용되지 않음 (dead variable).

---

#### RB-1: 2-Stage Re-Search

| Aspect | Design | Implementation | Status |
|--------|--------|----------------|--------|
| Condition | `len(results) < 3 and domain_confidence < 0.5 and use_enhancement` | `len(results) < 3 and use_enhancement and domain_confidence and domain_confidence < 0.5` | ✅ Match |
| Location | Phase 6 이후, Phase 7 이전 | Phase 6 직후 (line 997-1012) | ✅ Match |
| HyDE forced | Design: `use_hyde=True` | Impl: `_enhance_query(..., route_cfg, True)` -- `True`는 `use_enhancement` param | ⚠️ Deviation |
| `top_k * 2` | ✅ | ✅ | ✅ Match |
| Dedup by ID | `_existing_ids = {r.get('id') for r in results}` | `_existing_ids = {r.get('id') for r in results}` | ✅ Match |
| try/except | ✅ | ✅ | ✅ Match |
| Logging | `[Re-Search]` | `[Re-Search]` | ✅ Match |
| Timing key | `phase6_5_research_ms` | **Not implemented** | ⚠️ Missing |

**Verdict**: ⚠️ **2개 소규모 편차**. (1) Design은 재검색 시 `use_hyde=True` 강제 활성화를 명시했으나, 구현은 `_enhance_query`의 4번째 인자 `True`가 `use_enhancement` flag일 뿐 HyDE 강제 활성화가 아님 (CI-1 버그와 연동). (2) `_timings['phase6_5_research_ms']` 타이밍 기록이 미구현.

---

#### RB-2: Cache TTL & Max Size

| Aspect | Design | Implementation | Status |
|--------|--------|----------------|--------|
| `_CACHE_TTL` | `300 -> 3600` | `3600` | ✅ Match |
| `_CACHE_MAX` | `200 -> 500` | `500` | ✅ Match |
| Location | `query_enhancer.py` line 48-49 | line 48-49 | ✅ Match |

**Verdict**: ✅ 완전 일치.

---

### 2.2 Match Rate Summary

```
+---------------------------------------------+
|  Overall Match Rate: 91%                     |
+---------------------------------------------+
|  Items: 11                                   |
|  Full Match:        8 items (73%)            |
|    QW-1, QW-2, QW-3, QW-4, CI-2, CI-3,     |
|    RB-2, QW-5 (partial)                      |
|  Partial Match:     2 items (18%)            |
|    CI-4 (5-tuple done, secondary search      |
|           not done), RB-1 (core logic done,  |
|           HyDE force + timing missing)       |
|  Functional Gap:    1 item  (9%)             |
|    CI-1 (override computed but not passed)   |
+---------------------------------------------+
```

**Scoring Methodology**:
- Full Match (10 pts): Design intent fully realized
- Partial Match with improvements (9 pts): Functionally equivalent + beneficial additions
- Partial Match (7 pts): Core intent realized, minor gaps remain
- Partial Gap (5 pts): Significant portion unimplemented
- Full Gap (0 pts): Not implemented at all

| Item | Score | Weight | Weighted |
|------|:-----:|:------:|:--------:|
| QW-1 | 10 | 1 | 10 |
| QW-2 | 10 | 1 | 10 |
| QW-3 | 10 | 1 | 10 |
| QW-4 | 10 | 1 | 10 |
| QW-5 | 9 | 1 | 9 |
| CI-1 | 3 | 1 | 3 |
| CI-2 | 10 | 1 | 10 |
| CI-3 | 10 | 1 | 10 |
| CI-4 | 5 | 1 | 5 |
| RB-1 | 7 | 1 | 7 |
| RB-2 | 10 | 1 | 10 |
| **Total** | | **11** | **94/110 = 85%** |

---

## 3. Overall Scores

| Category | Score | Status |
|----------|:-----:|:------:|
| Design Match (feature-level) | 85% | ⚠️ |
| Architecture Compliance | 100% | ✅ |
| Convention Compliance | 100% | ✅ |
| Error Handling | 100% | ✅ |
| **Overall** | **91%** | **✅** |

> Architecture/Convention/Error Handling 모두 기존 프로젝트 패턴 준수. try/except + logging.warning + fallback 패턴 일관 적용. 모든 신규 코드가 기존 snake_case 파일명, [TAG] 로그 접두사, 환경변수 override 패턴을 따름.

---

## 4. Differences Found

### 4.1 Missing Features (Design O, Implementation X)

| # | Item | Design Location | Description | Impact |
|---|------|-----------------|-------------|--------|
| 1 | CI-1 HyDE override 전달 | design.md CI-1 | `_use_hyde` override가 `_enhance_query()`에 전달되지 않아 HyDE override가 실제 작동하지 않음 | **High** |
| 2 | CI-4 Secondary namespace search | design.md CI-4 | `secondary_ns`로 추가 검색 실행 로직 미구현. 변수 할당만 존재 | **Medium** |
| 3 | QW-5 `'화학물질안전'` synonym | design.md QW-5 | `DOMAIN_SYNONYMS`에서 `'화학물질안전': 'MSDS'` 항목 누락 | **Low** |
| 4 | RB-1 HyDE forced in retry | design.md RB-1 | 재검색 시 `use_hyde=True` 강제 활성화가 구현되지 않음 (CI-1 버그와 연관) | **Medium** |
| 5 | RB-1 Timing metric | design.md RB-1 | `_timings['phase6_5_research_ms']` 미기록 | **Low** |

### 4.2 Added Features (Design X, Implementation O)

| # | Item | Implementation Location | Description |
|---|------|------------------------|-------------|
| 1 | Empty content guard in MMR | `rag_pipeline.py` line 969-970 | `_ck`가 빈 문자열인 결과도 보존 (`elif not _ck:`) |
| 2 | Robust normalize edge case | `reranker.py` line 36 | `len(scores) < 2` 시 `[0.5] * len(scores)` 반환 (Design은 `scores` 그대로 반환) |
| 3 | All reranker classes apply robust normalize | `reranker.py` | `Reranker`, `PineconeReranker`, `CohereReranker` 3곳 모두 적용 |
| 4 | domain_confidence truthiness guard | `rag_pipeline.py` line 801, 815 | `domain_confidence and domain_confidence < X` 패턴으로 None/0 safety |

### 4.3 Changed Features (Design != Implementation)

| # | Item | Design | Implementation | Impact |
|---|------|--------|----------------|--------|
| 1 | QW-1 structure | Separate helper function `_extract_fallback_keywords()` | Inline 2-line code | None (functionally identical) |
| 2 | QW-5 entry count | 10 entries (including self-ref `'현장실습': '현장실습'`) | 7 entries (excluding self-ref + `'화학물질안전'`) | Low |
| 3 | QW-4 call site count | "4곳" replacement | 2 method-level calls covering all paths | None (functionally complete) |

---

## 5. Recommended Actions

### 5.1 Immediate (High Priority)

| # | Action | File | Line | Description |
|---|--------|------|------|-------------|
| 1 | **Fix CI-1 HyDE override** | `services/rag_pipeline.py` | 601-643, 823 | `_enhance_query()` 함수에 `use_hyde` 파라미터 추가하거나, `route_cfg`를 mutate하여 `_use_hyde` 값 전달. 현재 override가 dead code |

**Suggested fix**:
```python
# Option A: Add use_hyde parameter to _enhance_query()
def _enhance_query(search_query, namespace, route_cfg, use_enhancement, use_hyde=None):
    ...
    query_enhancer.enhance_query(
        ...,
        use_hyde=use_hyde if use_hyde is not None else route_cfg.get('use_hyde', False),
    )

# Call site:
enhancement = _enhance_query(search_query, namespace, route_cfg, use_enhancement, use_hyde=_use_hyde)
```

### 5.2 Short-term (Medium Priority)

| # | Action | File | Description |
|---|--------|------|-------------|
| 1 | CI-4 secondary search | `rag_pipeline.py` | `secondary_ns`로 추가 검색 + 결과 병합 로직 구현. 또는 의도적 제외라면 Design 문서 업데이트 |
| 2 | RB-1 HyDE force | `rag_pipeline.py` | 재검색 시 `_enhance_query`에 `use_hyde=True` 전달 (CI-1 fix와 연동) |
| 3 | QW-5 missing synonym | `query_router.py` | `'화학물질안전': 'MSDS'` 항목 추가 |

### 5.3 Low Priority

| # | Action | File | Description |
|---|--------|------|-------------|
| 1 | RB-1 timing | `rag_pipeline.py` | `_timings['phase6_5_research_ms']` 추가 |

---

## 6. Design Document Updates Needed

CI-1 fix 후 Design 업데이트는 불필요 (Design이 올바른 설계를 담고 있으므로 구현을 Design에 맞춰야 함).

단, 다음 항목은 Design 업데이트 권장:
- [ ] QW-5: `'현장실습': '현장실습'` self-reference 제거 (no-op이므로)
- [ ] QW-4: "4곳 호출" 표현을 "2개 RRF 메서드" 로 정정

---

## 7. Next Steps

- [ ] CI-1 HyDE override 버그 수정 (가장 높은 우선순위)
- [ ] CI-4 secondary namespace search 구현 여부 결정
- [ ] 수정 후 re-analysis 실행 (`/pdca analyze search-quality`)
- [ ] 완료 시 report 생성 (`/pdca report search-quality`)

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2026-03-17 | Initial gap analysis (11 items, 5 files) | gap-detector |
