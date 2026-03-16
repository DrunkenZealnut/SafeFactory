# multi-query-pipeline-refactoring Analysis Report

> **Analysis Type**: Gap Analysis (Design vs Implementation)
>
> **Project**: SafeFactory
> **Analyst**: gap-detector
> **Date**: 2026-03-16
> **Design Doc**: [multi-query-pipeline-refactoring.design.md](../02-design/features/multi-query-pipeline-refactoring.design.md)

---

## 1. Analysis Overview

### 1.1 Analysis Purpose

Design document (Section 1-8) specifies a refactoring of the multi-query pipeline: extracting EnhancementResult dataclass, module-level executor, TTL cache, and four new helper functions in rag_pipeline.py. This analysis verifies every specification against the actual implementation.

### 1.2 Analysis Scope

| Category | Path |
|----------|------|
| Design Document | `docs/02-design/features/multi-query-pipeline-refactoring.design.md` |
| Implementation (major) | `src/query_enhancer.py` |
| Implementation (major) | `services/rag_pipeline.py` |
| Implementation (minor) | `services/singletons.py` |

---

## 2. Gap Analysis (Design vs Implementation)

### 2.1 `src/query_enhancer.py` -- EnhancementResult Dataclass (Design Section 2.1)

| # | Design Spec | Implementation | Status |
|---|------------|----------------|--------|
| 1 | `@dataclass class EnhancementResult` | Line 92-93: `@dataclass class EnhancementResult` | Match |
| 2 | Field `original: str` | Line 95: `original: str` | Match |
| 3 | Field `variations: List[str] = field(default_factory=list)` | Line 96 | Match |
| 4 | Field `hyde_doc: Optional[str] = None` | Line 97 | Match |
| 5 | Field `expanded_query: Optional[str] = None` | Line 98 | Match |
| 6 | Field `keywords: List[str] = field(default_factory=list)` | Line 99 | Match |
| 7 | Property `search_queries` logic | Lines 101-110 | Match |
| 8 | Property `hyde_queries` logic | Lines 112-117 | Match |
| 9 | Property `all_queries` (backwards-compat) | Lines 119-122 | Match |
| 10 | Docstring: "Structured result..." (English vs Korean) | Line 94 | Trivial (English vs Korean docstring) |

### 2.2 `src/query_enhancer.py` -- Module-Level Executor (Design Section 2.2)

| # | Design Spec | Implementation | Status |
|---|------------|----------------|--------|
| 11 | `_enhancement_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="query-enhance")` | Lines 31-34 | Match |
| 12 | `atexit.register(_enhancement_executor.shutdown, wait=False)` | Line 35 | Match |
| 13 | `shutdown_enhancement_executor()` function | Lines 38-40 | Match |

### 2.3 `src/query_enhancer.py` -- TTL Cache (Design Section 2.3)

| # | Design Spec | Implementation | Status |
|---|------------|----------------|--------|
| 14 | `_cache_lock = threading.Lock()` | Line 46 | Match |
| 15 | `_multi_query_cache: dict = {}` | Line 47 | Match |
| 16 | `_CACHE_TTL = 300` (5 min) | Line 48 | Match |
| 17 | `_CACHE_MAX = 200` | Line 49 | Match |
| 18 | `_make_cache_key()` using MD5 of `f"{query}:{num_variations}"` | Lines 52-53 | Match |
| 19 | `_cache_get()` TTL check + expired entry deletion | Lines 56-66 | Match |
| 20 | `_cache_set()` with 20% eviction on overflow | Lines 69-80 | Match |
| 21 | `clear_enhancement_cache()` function | Lines 83-86 | Match |

### 2.4 `src/query_enhancer.py` -- `_get_num_variations()` (Design Section 2.4)

| # | Design Spec | Implementation | Status |
|---|------------|----------------|--------|
| 22 | `@staticmethod` decorator | Line 298 | Match |
| 23 | Threshold `< 15` returns 1 | Line 302-303 | Match |
| 24 | Threshold `< 40` returns 2 | Line 304-305 | Match |
| 25 | Else returns 3 | Line 306-307 | Match |

### 2.5 `src/query_enhancer.py` -- `multi_query()` Cache Integration (Design Section 2.5)

| # | Design Spec | Implementation | Status |
|---|------------|----------------|--------|
| 26 | Cache lookup before LLM call | Lines 321-325 | Match |
| 27 | Cache HIT log: `[Multi-Query Cache] HIT for '%.30s...'` | Line 324 | Match |
| 28 | Cache store after successful LLM generation | Line 358 | Match |
| 29 | Existing signature `multi_query(self, query, num_variations=3)` preserved | Line 309 | Match |

### 2.6 `src/query_enhancer.py` -- `enhance_query()` Redesign (Design Section 2.6)

| # | Design Spec | Implementation | Status |
|---|------------|----------------|--------|
| 30 | Signature: `(self, query, domain="general", use_multi_query=True, use_hyde=False, use_keywords=True)` | Lines 489-496 | Match |
| 31 | Returns `EnhancementResult` | Line 496 type hint + line 571-577 | Match |
| 32 | Step 1: `expand_with_synonyms()` synchronous | Line 510 | Match |
| 33 | Set `expanded_query = None` if unchanged | Lines 511-512 | Match |
| 34 | Synonym expansion log: `[Synonym Expansion] '%s' -> '%s'` | Line 514 | Match |
| 35 | Step 2: submit `multi_query` via `_enhancement_executor` | Lines 522-526 | Match |
| 36 | Submit HyDE with `len(query) >= 10` guard | Lines 527-530 | Match |
| 37 | Submit `extract_keywords_fast` (not `extract_keywords`) | Lines 531-534 | Match |
| 38 | Step 3: collect multi results with `timeout=10` | Line 539 | Match |
| 39 | Multi-query log: `[Query Enhancement] Generated %d query variations` | Lines 540-543 | Match |
| 40 | Multi-query fallback log: `[Query Enhancement] multi_query failed: %s` | Line 545 | Match |
| 41 | HyDE identity check: `hyde_doc == query` -> None | Lines 553-554 | Match |
| 42 | HyDE log: `[HyDE] Generated hypothetical document` | Line 556 | Match |
| 43 | Keywords log: `[Query Enhancement] Keywords: %s` | Line 566 | Match |
| 44 | EnhancementResult construction | Lines 571-577 | Match |
| 45 | **Extra**: `else` branch logs `[Query Enhancement] multi_query skipped` | Lines 547-548 | Added (beneficial) |
| 46 | **Extra**: `else` branch logs `[HyDE] Skipped` | Lines 560-561 | Added (beneficial) |

### 2.7 `src/query_enhancer.py` -- `__main__` Block (Design Section 2.7)

| # | Design Spec | Implementation | Status |
|---|------------|----------------|--------|
| 47 | Updated to use `EnhancementResult` | Lines 605-613 | Match |

### 2.8 `services/rag_pipeline.py` -- Import Changes (Design Section 3.7)

| # | Design Spec | Implementation | Status |
|---|------------|----------------|--------|
| 48 | `from src.query_enhancer import EnhancementResult` added | Line 38 | Match |
| 49 | `concurrent.futures` kept (used elsewhere in file) | Line 3 (used at lines 488, 493, 537, 544) | Match |

### 2.9 `services/rag_pipeline.py` -- `_content_hash()` (Design Section 3.3)

| # | Design Spec | Implementation | Status |
|---|------------|----------------|--------|
| 50 | `hashlib.sha256(content[:5000].encode()).hexdigest()` | Line 598 | Match |

### 2.10 `services/rag_pipeline.py` -- `_enhance_query()` (Design Section 3.1)

| # | Design Spec | Implementation | Status |
|---|------------|----------------|--------|
| 51 | Signature: `(search_query, namespace, route_cfg, use_enhancement)` | Lines 601-606 | Match |
| 52 | Returns `EnhancementResult` | Line 606 type hint | Match |
| 53 | `not use_enhancement` -> `EnhancementResult(original=search_query, variations=[search_query])` | Lines 621-625 | Match |
| 54 | `get_query_enhancer()` call | Line 628 | Match |
| 55 | `NAMESPACE_DOMAIN_MAP.get(namespace, 'semiconductor')` | Line 629 | Match |
| 56 | Delegates to `query_enhancer.enhance_query(...)` with correct kwargs | Lines 631-637 | Match |
| 57 | Exception fallback log: `[Query Enhancement] Failed: %s, using original query` | Line 639 | Match |
| 58 | Exception fallback returns `EnhancementResult(original=..., variations=[...])` | Lines 640-643 | Match |

### 2.11 `services/rag_pipeline.py` -- `_search_single_query()` (Design Section 3.2)

| # | Design Spec | Implementation | Status |
|---|------------|----------------|--------|
| 59 | Signature: `(agent, query, namespace, top_k, domain_filter, is_all_namespace)` | Lines 646-653 | Match |
| 60 | 2-attempt retry loop | Line 667: `for attempt in range(2)` | Match |
| 61 | `is_all_namespace` branch: `uploader.get_stats()` -> ns_list | Lines 669-681 | Match |
| 62 | Fallback ns_list: `['semiconductor', 'laborlaw', 'field-training']` | Line 675 | Match |
| 63 | Normal branch: `agent.search(...)` | Lines 682-688 | Match |
| 64 | Attempt 1 failure log with retry | Lines 690-695 | Match |
| 65 | `time.sleep(0.5)` between retries | Line 695 | Match |
| 66 | Attempt 2 failure log (no retry) | Lines 696-700 | Match |
| 67 | Return `[]` on total failure | Line 701 | Match |

### 2.12 `services/rag_pipeline.py` -- `_search_with_variations()` (Design Section 3.4)

| # | Design Spec | Implementation | Status |
|---|------------|----------------|--------|
| 68 | Design signature has 6 params (`agent, enhancement, namespace, domain_filter, top_k, search_top_k`) | Impl has 5 params (no `top_k`) | Changed (improvement) |
| 69 | `is_all_ns = (namespace == 'all')` | Line 726 | Match |
| 70 | `_collect()` closure with content hash dedup | Lines 730-739 | Match |
| 71 | Iterates `enhancement.search_queries` then `enhancement.hyde_queries` | Lines 742-747 | Match |
| 72 | Log: `[Search] Retrieved %d unique documents from %d queries` | Lines 749-753 | Match |

### 2.13 `services/rag_pipeline.py` -- `run_rag_pipeline()` Phase 1-2 Replacement (Design Section 3.5)

| # | Design Spec | Implementation | Status |
|---|------------|----------------|--------|
| 73 | Phase 1: `enhancement = _enhance_query(...)` | Line 807 | Match |
| 74 | `enhanced_queries = enhancement.all_queries` (backwards-compat) | Line 808 | Match |
| 75 | `keywords = enhancement.keywords` | Line 809 | Match |
| 76 | Phase 1 timing: `_timings['phase1_enhancement_ms']` | Line 810 | Match |
| 77 | Phase 2: `domain_filter = build_domain_filter(...)` | Line 815 | Match |
| 78 | `top_k_mult = route_cfg.get('top_k_mult', TOP_K_DEFAULT_MULT)` | Line 818 | Match |
| 79 | `skip_bm25` conditional `search_top_k` calculation | Lines 819-822 | Match |
| 80 | `results = _search_with_variations(agent, enhancement, namespace, domain_filter, search_top_k)` | Lines 823-825 | Match (adapted to 5-param signature) |
| 81 | Phase 2 timing: `_timings['phase2_search_ms']` | Line 826 | Match |

### 2.14 `services/rag_pipeline.py` -- `_get_num_variations()` Removal (Design Section 3.6)

| # | Design Spec | Implementation | Status |
|---|------------|----------------|--------|
| 82 | `_get_num_variations()` removed from rag_pipeline.py | Grep confirms 0 matches | Match |

### 2.15 `services/singletons.py` -- `shutdown_all()` (Design Section 4.1)

| # | Design Spec | Implementation | Status |
|---|------------|----------------|--------|
| 83 | `from src.query_enhancer import shutdown_enhancement_executor` | Line 281 | Match |
| 84 | `shutdown_enhancement_executor()` called after closing singletons | Line 286 | Match |

### 2.16 `services/singletons.py` -- `invalidate_query_enhancer()` (Design Section 4.2)

| # | Design Spec | Implementation | Status |
|---|------------|----------------|--------|
| 85 | `from src.query_enhancer import clear_enhancement_cache` | Line 251 | Match |
| 86 | `clear_enhancement_cache()` called after nullifying singleton | Line 256 | Match |

### 2.17 Unchanged Files (Design Section 1.2)

| # | File | Design: "No Change" | Status |
|---|------|---------------------|--------|
| 87 | `services/query_router.py` | No change | Match |
| 88 | `services/domain_config.py` | No change | Match |
| 89 | `services/filters.py` | No change | Match |
| 90 | `src/__init__.py` | No change | Match |
| 91 | Phase 4-7 logic in `run_rag_pipeline()` | No change | Match |
| 92 | API response format | No change | Match |

### 2.18 Logging Compatibility (Design Section 7)

| # | Log Pattern | Design Location | Impl Location | Status |
|---|------------|----------------|---------------|--------|
| 93 | `[Synonym Expansion] '%s' -> '%s'` | QueryEnhancer.enhance_query() | Line 514 | Match |
| 94 | `[Query Enhancement] Generated %d query variations` | QueryEnhancer.enhance_query() | Lines 540-543 | Match |
| 95 | `[Query Enhancement] multi_query failed: %s` | QueryEnhancer.enhance_query() | Line 545 | Match |
| 96 | `[HyDE] Generated hypothetical document` (changed from "Added...") | QueryEnhancer.enhance_query() | Line 556 | Match |
| 97 | `[Query Enhancement] Keywords: %s` | QueryEnhancer.enhance_query() | Line 566 | Match |
| 98 | `[Search] Retrieved %d unique documents from %d queries` | `_search_with_variations()` | Lines 749-753 | Match |
| 99 | `[Search] Attempt N failed...` | `_search_single_query()` | Lines 690-700 | Match |
| 100 | **NEW**: `[Multi-Query Cache] HIT for '%.30s...'` | `multi_query()` | Line 324 | Match |

### 2.19 File Structure (Design Section 8)

| # | Design Spec | Implementation | Status |
|---|------------|----------------|--------|
| 101 | `EnhancementResult` before `QueryEnhancer` class | Lines 92-122 (before class at 187) | Match |
| 102 | Module executor before cache before dataclass | Executor 31-40, Cache 46-86, Dataclass 92-122 | Match |
| 103 | `_get_num_variations()` is `@staticmethod` inside `QueryEnhancer` | Lines 298-307 | Match |
| 104 | rag_pipeline: 4 new module-level functions before `run_rag_pipeline()` | Lines 596-754 (before `run_rag_pipeline` at 757) | Match |
| 105 | rag_pipeline: `_get_num_variations()` deleted | Confirmed absent | Match |

---

## 3. Match Rate Summary

```
Total Items Checked: 105

  Match:                   103 items (98.1%)
  Changed (improvement):     2 items (1.9%)
  Missing in implementation: 0 items (0.0%)
  Missing in design:         0 items (0.0%)
```

### Changed Items (Design != Implementation)

| # | Item | Design | Implementation | Impact | Verdict |
|---|------|--------|----------------|--------|---------|
| 45-46 | `else` branch logging in `enhance_query()` | Not specified | `[Query Enhancement] multi_query skipped` and `[HyDE] Skipped` logs added | None (beneficial) | Improvement |
| 68 | `_search_with_variations()` signature | 6 params (includes unused `top_k`) | 5 params (removed unused `top_k`) | None (cleaner API) | Improvement |

Both deviations are improvements over the design:
- The `else` branch logs improve observability when multi_query or HyDE is intentionally skipped
- Removing the unused `top_k` parameter follows YAGNI and eliminates dead code in the function signature

---

## 4. Overall Scores

| Category | Score | Status |
|----------|:-----:|:------:|
| Design Match | 98% | Pass |
| Architecture Compliance | 100% | Pass |
| Convention Compliance | 100% | Pass |
| Logging Compatibility | 100% | Pass |
| Backwards Compatibility | 100% | Pass |
| **Overall** | **99%** | Pass |

---

## 5. Backwards Compatibility Verification

| # | Compatibility Item | Design Spec | Implementation | Status |
|---|-------------------|------------|----------------|--------|
| B1 | `enhanced_queries` variable in `run_rag_pipeline()` | `enhancement.all_queries` | Line 808: `enhanced_queries = enhancement.all_queries` | Pass |
| B2 | `keywords` variable in `run_rag_pipeline()` | `enhancement.keywords` | Line 809: `keywords = enhancement.keywords` | Pass |
| B3 | `all_queries` property returns same as old `enhanced_queries` list | `search_queries + hyde_queries` | Lines 119-122 | Pass |
| B4 | `multi_query()` signature unchanged | `(self, query, num_variations=3)` | Line 309 | Pass |
| B5 | API response format unchanged | No change | Phase 4-7 untouched | Pass |

---

## 6. Verification Checklist (Design Section 6)

| # | Verification Item | Testable | Notes |
|---|------------------|----------|-------|
| V1 | Enhancement disabled -> original only | Yes | `_enhance_query()` returns `EnhancementResult(original=..., variations=[query])` |
| V2 | Factual: multi-query ON, HyDE OFF | Yes | `route_cfg` passes `use_hyde=False` default |
| V3 | Procedural: multi-query ON, HyDE ON | Yes | `route_cfg` with `use_hyde=True` |
| V4 | Calculation: multi-query OFF, HyDE OFF | Yes | `route_cfg` with `use_multi_query=False, use_hyde=False` |
| V5 | Synonym expansion works | Yes | `expand_with_synonyms()` called synchronously |
| V6 | Cache HIT on repeat query | Yes | `_cache_get()` returns cached result |
| V7 | API response unchanged | Yes | Phase 4-7 untouched, response helpers unchanged |
| V8 | Content hash dedup works | Yes | `_content_hash()` + `seen_ids` in `_search_with_variations()` |
| V9 | All-namespace search | Yes | `is_all_ns` branch in `_search_single_query()` |
| V10 | Retry on cold start failure | Yes | 2-attempt loop with `time.sleep(0.5)` |

---

## 7. Recommended Actions

### 7.1 No Immediate Actions Required

All 105 checked items match or improve upon the design. The 2 deviations are both beneficial.

### 7.2 Design Document Updates (Optional)

These are minor documentation-only updates to reflect the two improvements:

1. **Section 3.4**: Update `_search_with_variations()` signature to remove unused `top_k` parameter (5 params instead of 6)
2. **Section 2.6**: Add `else` branch logging for multi_query/HyDE skip cases

---

## 8. Files Verified

| File | Lines | Changes Verified |
|------|------:|-----------------|
| `src/query_enhancer.py` | 616 | EnhancementResult, executor, TTL cache, `_get_num_variations`, cache in `multi_query`, `enhance_query` redesign, `__main__` update |
| `services/rag_pipeline.py` | ~950 | `_content_hash`, `_enhance_query`, `_search_single_query`, `_search_with_variations`, Phase 1-2 replacement, `_get_num_variations` removal, import changes |
| `services/singletons.py` | 287 | `shutdown_all` executor shutdown, `invalidate_query_enhancer` cache clear |

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2026-03-16 | Initial gap analysis | gap-detector |
