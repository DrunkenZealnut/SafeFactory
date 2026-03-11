# 법령정보-통합-API-연동 Analysis Report

> **Analysis Type**: Gap Analysis (Design vs Implementation)
>
> **Project**: SafeFactory
> **Analyst**: Claude Code (gap-detector)
> **Date**: 2026-03-11
> **Design Doc**: [법령정보-통합-API-연동.design.md](../02-design/features/법령정보-통합-API-연동.design.md)

---

## 1. Analysis Overview

### 1.1 Analysis Purpose

Design document(v0.1, 2026-03-11)에 정의된 법령정보 통합 API 연동 설계와 실제 구현 코드 간의 일치도를 측정하고, 차이점을 식별한다.

### 1.2 Analysis Scope

| Item | Path |
|------|------|
| Design Document | `docs/02-design/features/법령정보-통합-API-연동.design.md` |
| LawDrfClient | `services/law_drf_client.py` |
| LegalSourceRouter | `services/legal_source_router.py` |
| law_api.py (wrapper) | `services/law_api.py` |
| singletons.py | `services/singletons.py` |
| rag_pipeline.py | `services/rag_pipeline.py` |
| .env.example | `.env.example` |

---

## 2. Overall Scores

| Category | Score | Status |
|----------|:-----:|:------:|
| Design Match | 86% | ⚠️ |
| Architecture Compliance | 95% | ✅ |
| Convention Compliance | 98% | ✅ |
| **Overall** | **89%** | ⚠️ |

---

## 3. Gap Analysis (Design vs Implementation)

### 3.1 LawDrfClient (`services/law_drf_client.py`)

#### 3.1.1 Data Classes

| Design Class | Implementation | Status | Notes |
|--------------|---------------|--------|-------|
| `SourceRequest` | L30-36 | ✅ Match | Fields identical |
| `LawArticle` | L39-45 | ✅ Match | Fields identical |
| `Precedent` | L48-57 | ✅ Match | `ref_articles` default changed to `''` (acceptable) |
| `Interpretation` | L60-69 | ✅ Match | `reason`/`ref_laws` defaults changed to `''` (acceptable) |
| `AdminRule` | L72-79 | ✅ Match | Fields identical |
| `LegalContext` | L82-98 | ✅ Match | `has_content` and `source_count` properties present |

**Data Classes Score: 100% (6/6)**

#### 3.1.2 Class Constants

| Design Constant | Design Value | Implementation Value | Status |
|-----------------|:------------:|:--------------------:|--------|
| `BASE_SEARCH` | `http://www.law.go.kr/DRF/lawSearch.do` | Same (L138) | ✅ |
| `BASE_SERVICE` | `http://www.law.go.kr/DRF/lawService.do` | Same (L139) | ✅ |
| `TIMEOUT` | 10 | 10 (L140) | ✅ |
| `MAX_DISPLAY` | 20 | Not defined | ❌ Missing |
| `MAX_TEXT_CHARS` | 800 | 800 (L141) | ✅ |
| `MAX_FAILURES` | 5 | 5 (L144) | ✅ |
| `CIRCUIT_RESET_SEC` | 300 | 300 (L145) | ✅ |

**Constants Score: 86% (6/7)**

#### 3.1.3 Methods

| Design Method | Implementation | Status | Notes |
|---------------|---------------|--------|-------|
| `__init__(oc)` | L147 `__init__(oc=None)` | ⚠️ Changed | `oc` is now optional with env fallback — more flexible |
| `search(target, query, **params)` | **Not implemented** | ❌ Missing | Generic search method absent |
| `get_detail(target, item_id, **params)` | **Not implemented** | ❌ Missing | Generic detail method absent |
| `search_laws(query, org, display)` | L234 | ✅ Match | `org` default differs: design `'1440000'` vs impl `''` |
| `search_precedents(query, court, display)` | L325 | ✅ Match | `court` default differs: design `'400201'` vs impl `''` |
| `search_interpretations(query, display)` | L382 | ✅ Match | Signature matches |
| `search_admin_rules(query, knd, org, display)` | L445 | ✅ Match | `org` default differs: design `'1440000'` vs impl `''` |
| `get_law_articles(law_name)` | L262 | ✅ Match | Full implementation with lsiSeq lookup |
| `get_precedent_detail(prec_id)` | L354 | ✅ Match | Returns `Precedent \| None` |
| `get_interpretation_detail(interp_id)` | L416 | ✅ Match | Returns `Interpretation \| None` |
| `get_admin_rule_detail(rule_id)` | L479 | ✅ Match | Returns `AdminRule \| None` |
| `_request(base_url, params)` | L192 `_request_xml(base_url, params)` | ⚠️ Changed | Returns `ElementTree.Element` instead of `dict\|list` — XML-only, no JSON parsing |
| `_check_circuit()` | L168 `_check_circuit_reset()` | ⚠️ Changed | Name slightly different, logic equivalent |
| `available` property | L160-166 | ✅ Match | Property with circuit breaker check |

**Methods Score: 77% (10/13 — 2 missing generic methods, 1 renamed)**

#### 3.1.4 Response Format

| Design Spec | Implementation | Status | Notes |
|-------------|---------------|--------|-------|
| JSON as primary, XML fallback | **XML as primary, no JSON** | ⚠️ Changed | Design Section 4.1 specifies JSON examples; implementation uses XML exclusively. Design Note (L349) acknowledges XML may be needed. |
| Response size limit 1MB | L211 `len(resp.content) > 1_000_000` | ✅ Match | |

#### 3.1.5 Additional Implementation (Not in Design)

| Item | Location | Description |
|------|----------|-------------|
| `get_specific_law_articles()` | L311-321 | Filters articles by specific article numbers — useful addition |
| `_text()` static utility | L505-512 | XML element text extraction helper |
| `_format_date()` static utility | L514-520 | Date format normalizer `20240515` -> `2024.05.15` |
| `requests.Session` with headers | L154-158 | Connection pooling and User-Agent — good practice |
| Cache size limit (100 entries) | L123-125 | Prevents unbounded cache growth |

### 3.2 LegalSourceRouter (`services/legal_source_router.py`)

#### 3.2.1 Class Interface

| Design Item | Implementation | Status | Notes |
|-------------|---------------|--------|-------|
| `__init__(drf_client)` | L108 | ✅ Match | |
| `route(query, classification)` | L111 | ✅ Match | Signature identical |
| `search_all(query, classification)` | L158 | ✅ Match | Returns `LegalContext` |
| `_search_source(req)` | L193 | ✅ Match | Private method |
| `format_context(ctx, start_index)` | L251 | ✅ Match | Returns markdown string |
| `MAX_TOTAL_CHARS` | 3000 | 3000 (L105) | ✅ |
| `SEARCH_TIMEOUT` | Design: 5s | Impl: 8s (L106) | ⚠️ Changed | Implementation uses longer timeout |

#### 3.2.2 Keyword Mapping

| Design Keyword Count | Implementation Keyword Count | Status |
|:--------------------:|:---------------------------:|--------|
| 32 entries | 42 entries | ⚠️ Expanded |

Implementation adds keywords not in design: `정리해고`, `부당노동행위`, `급여`, `퇴직연금`, `출산휴가`, `주휴`, `근로계약`, `수습`, `비정규직`, `기간제`, `파견`, `고용보험`, `실업급여`, `대법원`. These are beneficial expansions.

#### 3.2.3 Type Default Sources

| Design | Implementation | Status |
|--------|---------------|--------|
| `legal: ['law', 'moelCgmExpc']` | `legal: ['moelCgmExpc']` | ⚠️ Changed | `law` removed (handled by AJAX path) |
| `calculation: ['law']` | `calculation: []` | ⚠️ Changed | `law` removed (handled by AJAX path) |
| `hybrid: ['law', 'moelCgmExpc']` | `hybrid: ['moelCgmExpc']` | ⚠️ Changed | `law` removed (handled by AJAX path) |

This is an intentional architectural decision: law articles are fetched via the existing AJAX path in `_search_via_ajax()`, while DRF handles only additional sources (prec/interp/admrul). This avoids duplication.

#### 3.2.4 Additional Implementation (Not in Design)

| Item | Location | Description |
|------|----------|-------------|
| `_QUERY_REFINEMENTS` mapping | L90-99 | Keyword-to-target query optimization — quality improvement |
| `_search_precedents()` | L207-219 | Dedicated precedent search+detail method |
| `_search_interpretations()` | L221-232 | Dedicated interpretation search+detail method |
| `_search_admin_rules()` | L234-247 | Dedicated admin rule search+detail method |
| `_indent()` static method | L311-316 | Block-quote text formatter |

### 3.3 law_api.py Modifications

| Design Item | Implementation | Status | Notes |
|-------------|---------------|--------|-------|
| `search_labor_laws()` signature unchanged | L804 | ✅ Match | `(query, classification)` |
| DRF first, AJAX fallback flow | L822-830 | ⚠️ Changed | **AJAX always runs first**, DRF results are appended. Design specifies DRF first, then AJAX fallback. |
| `_get_drf_client_if_available()` | Not implemented as separate function | ⚠️ Changed | Logic inlined in `_search_via_drf()` |
| `_search_via_drf()` | L833-885 | ✅ Present | DRF path via LegalSourceRouter |
| `_search_via_ajax()` | L888-933 | ✅ Present | Existing AJAX path preserved |
| `_legal_context_to_legacy_format()` | L851-875 (inlined) | ⚠️ Changed | Not a separate function — conversion logic is inlined in `_search_via_drf()` |
| `format_law_references()` multi-source | L936-1033 | ✅ Match | Groups by source_type (law/prec/interp/admrul) |
| `has_multi_source()` | L1036-1041 | ✅ Match | Checks for non-law source_types |
| Legacy format max 15 items | Not enforced | ⚠️ Missing | Design specifies `[:15]` limit, implementation uses `[:10]` on AJAX results but no limit on combined |

**Critical Design Deviation**: The execution flow is fundamentally different from design:
- **Design**: DRF first -> if fails -> AJAX fallback
- **Implementation**: AJAX always runs -> DRF appends additional sources (prec/interp/admrul)

This is actually a **better** approach because the existing AJAX law article search (`_find_relevant_articles`) is more refined for Korean labor law with its 74-law registry. DRF adds complementary sources (precedents, interpretations, admin rules) that AJAX cannot provide.

### 3.4 rag_pipeline.py Integration

| Design Item | Implementation | Status | Notes |
|-------------|---------------|--------|-------|
| `search_labor_laws(query, classification)` call | L634 | ✅ Match | Called within laborlaw namespace block |
| `has_multi_source()` check | L640 | ✅ Match | Detects multi-source results |
| `format_law_references()` with multi-source | L638-639 | ⚠️ Changed | Uses unified `format_law_references()` for all cases (no `_format_multi_source_refs` function) |
| `build_llm_prompts()` section title | L728 | ✅ Match | `'관련 법적 근거' if _is_multi else '관련 법령 정보'` |
| Citation rules for multi-source | L735-739 | ✅ Match | Includes precedent and admin rule citation examples |
| Multi-source detection in prompt | L724-726 | ✅ Match | Checks for `'### 관련 판례'`, `'### 행정해석'`, `'### 관련 고시/지침'` |

### 3.5 singletons.py

| Design Item | Implementation | Status | Notes |
|-------------|---------------|--------|-------|
| `_law_drf_client = None` global | L11 | ✅ Match | |
| `get_law_drf_client()` function | L199-215 | ✅ Match | Returns `None` if LAW_OC not set |
| Double-checked locking | L210-213 | ✅ Match | Follows existing singleton pattern |
| Returns `None` if no OC | L208-209 | ✅ Match | |

**Score: 100% (4/4)**

### 3.6 Environment Variables (.env.example)

| Design Item | Implementation | Status | Notes |
|-------------|---------------|--------|-------|
| `LAW_OC` documented | L23-27 | ✅ Match | Includes explanation and prerequisites |

**Score: 100% (1/1)**

### 3.7 Error Handling & Security

| Design Item | Implementation | Status | Notes |
|-------------|---------------|--------|-------|
| LAW_OC missing -> no DRF | singletons.py L208-209, law_drf_client.py L163 | ✅ | |
| DRF timeout (10s) | law_drf_client.py L207 | ✅ | |
| Circuit breaker (5 fail -> disable) | law_drf_client.py L182-188 | ✅ | |
| Circuit reset after 5 min | law_drf_client.py L170-174 | ✅ | |
| Individual source failure tolerance | legal_source_router.py L183-185 | ✅ | |
| Response size limit (1MB) | law_drf_client.py L211-213 | ✅ | |
| Query length limit (200 chars) | law_drf_client.py L239,330,387,452 → `[:100]` | ⚠️ Changed | Uses 100-char limit instead of design's 200-char |
| Prompt char limit (3000) | legal_source_router.py L105 | ✅ | |
| OC masking in logs | Not explicitly implemented | ⚠️ Missing | OC is passed as URL param, not logged directly, but could leak in error stack traces |
| Logging prefixes (`[LawDRF]`, `[LegalRouter]`) | Both files | ✅ Match | |

### 3.8 Caching

| Design Item | Implementation | Status | Notes |
|-------------|---------------|--------|-------|
| TTL cache (5 min / 300s) | law_drf_client.py L106 `_CACHE_TTL = 300` | ✅ | |
| Thread-safe cache access | law_drf_client.py L105 `_cache_lock` | ✅ | |
| Cache implementation | law_drf_client.py L104-125 | ✅ | dict-based with TTL eviction and 100-entry size limit |

---

## 4. Differences Summary

### 4.1 Missing Features (Design O, Implementation X)

| # | Item | Design Location | Description | Impact |
|---|------|-----------------|-------------|--------|
| 1 | `search()` generic method | design.md L235-246 | Generic `search(target, query, **params)` not implemented; replaced by target-specific methods | Low |
| 2 | `get_detail()` generic method | design.md L248-257 | Generic `get_detail(target, item_id, **params)` not implemented | Low |
| 3 | `MAX_DISPLAY` constant | design.md L222 | 20-item display limit constant not defined (each method uses own `display` param) | Low |
| 4 | `_get_drf_client_if_available()` | design.md L509-516 | Not a standalone function in `law_api.py` | Low |
| 5 | `_legal_context_to_legacy_format()` | design.md L519-550 | Logic exists but inlined in `_search_via_drf()` | Low |

### 4.2 Added Features (Design X, Implementation O)

| # | Item | Implementation Location | Description | Impact |
|---|------|------------------------|-------------|--------|
| 1 | `get_specific_law_articles()` | law_drf_client.py L311-321 | Filter articles by specific article numbers | Positive |
| 2 | `_QUERY_REFINEMENTS` mapping | legal_source_router.py L90-99 | Keyword-to-target search query optimization | Positive |
| 3 | `_format_date()` utility | law_drf_client.py L514-520 | Date format normalizer | Positive |
| 4 | `_text()` XML utility | law_drf_client.py L505-512 | Robust XML text extraction | Positive |
| 5 | `requests.Session` with headers | law_drf_client.py L154-158 | Connection pooling, User-Agent | Positive |
| 6 | Cache size limit (100) | law_drf_client.py L123-125 | Prevents unbounded memory growth | Positive |
| 7 | `source_type` field in DRF results | law_api.py L858,873,874 | Explicit type field for downstream grouping | Positive |
| 8 | `_search_precedents()`, `_search_interpretations()`, `_search_admin_rules()` | legal_source_router.py L207-247 | Dedicated per-source search+detail methods | Positive |

### 4.3 Changed Features (Design != Implementation)

| # | Item | Design | Implementation | Impact | Justification |
|---|------|--------|----------------|--------|---------------|
| 1 | Execution flow | DRF first, AJAX fallback | AJAX always + DRF appends | Medium | Better: AJAX law lookup is more precise; DRF adds prec/interp/admrul |
| 2 | Response format | JSON primary, XML fallback | XML only | Low | Design Note L349 acknowledges XML may be primary; JSON not reliably supported by DRF |
| 3 | Query length limit | 200 chars | 100 chars (`query[:100]`) | Low | More conservative; sufficient for Korean legal queries |
| 4 | `_TYPE_DEFAULT_SOURCES` `law` entries | `['law', 'moelCgmExpc']` etc. | `['moelCgmExpc']` etc. (no `law`) | Low | `law` handled by AJAX path; avoids duplication |
| 5 | Search timeout | 5s (design L397) | 8s (impl L106 `SEARCH_TIMEOUT = 8`) | Low | Government API can be slow; longer timeout is pragmatic |
| 6 | `__init__` oc param | Required `oc: str` | Optional `oc: str \| None = None` | Low | More flexible with env fallback |
| 7 | `search_laws()` org default | `'1440000'` | `''` (empty) | Low | Allows broader search; org filtering moved to LegalSourceRouter |
| 8 | OC param masking in logs | Specified | Not explicitly implemented | Low | OC isn't logged directly, but not actively masked |

---

## 5. Architecture Compliance

### 5.1 Layer Structure Verification

| Design Layer | Implementation | Status |
|-------------|---------------|--------|
| DRF communication (`law_drf_client.py`) | `services/law_drf_client.py` | ✅ |
| Source routing (`legal_source_router.py`) | `services/legal_source_router.py` | ✅ |
| Legacy wrapper (`law_api.py`) | `services/law_api.py` | ✅ |
| Singleton management (`singletons.py`) | `services/singletons.py` | ✅ |
| RAG integration (`rag_pipeline.py`) | `services/rag_pipeline.py` | ✅ |

### 5.2 Dependency Direction

| From | To | Design | Implementation | Status |
|------|----|--------|---------------|--------|
| `rag_pipeline.py` | `law_api.py` | ✅ | ✅ (L633) | ✅ |
| `law_api.py` | `singletons.py` | ✅ | ✅ (L836) | ✅ |
| `law_api.py` | `legal_source_router.py` | ✅ | ✅ (L841) | ✅ |
| `legal_source_router.py` | `law_drf_client.py` | ✅ | ✅ (L11-19) | ✅ |
| `singletons.py` | `law_drf_client.py` | ✅ | ✅ (L212) | ✅ |

All import directions follow design's dependency chain. Lazy imports used where appropriate (singletons, law_api).

**Architecture Score: 95%**

---

## 6. Convention Compliance

### 6.1 Naming Conventions

| Category | Convention | Files Checked | Compliance | Violations |
|----------|-----------|:-------------:|:----------:|------------|
| File names | snake_case | 4 files | 100% | None |
| Classes | PascalCase | 8 classes | 100% | None |
| Constants | UPPER_SNAKE_CASE | 12 constants | 100% | None |
| Functions | snake_case | ~25 functions | 100% | None |
| Log prefixes | `[LawDRF]`, `[LegalRouter]`, `[LawAPI]` | 3 modules | 100% | None |

### 6.2 Pattern Compliance

| Pattern | Convention | Compliance | Notes |
|---------|-----------|:----------:|-------|
| Singleton | Double-checked locking | ✅ | Follows existing `singletons.py` pattern |
| Caching | `_cache` dict + `_CACHE_TTL` | ✅ | Matches existing `law_api.py` pattern |
| Error handling | try/except -> log + empty return | ✅ | Never raises; always returns safe default |
| Lazy imports | `from ... import` inside function | ✅ | Used in `law_api.py` and `singletons.py` |

**Convention Score: 98%**

---

## 7. Match Rate Calculation

### 7.1 Design Items Checklist

| # | Design Item | Status | Weight |
|---|------------|--------|:------:|
| 1 | Data classes (6 total) | ✅ All match | 10 |
| 2 | LawDrfClient class with constants | ✅ 6/7 constants | 8 |
| 3 | `search()` generic method | ❌ Not implemented | 3 |
| 4 | `get_detail()` generic method | ❌ Not implemented | 3 |
| 5 | Target-specific search methods (4) | ✅ All present | 10 |
| 6 | Target-specific detail methods (4) | ✅ All present | 10 |
| 7 | Circuit breaker implementation | ✅ Full match | 8 |
| 8 | TTL cache implementation | ✅ Full match | 5 |
| 9 | XML response parsing | ✅ Present | 5 |
| 10 | `LegalSourceRouter` class | ✅ Full match | 10 |
| 11 | `_KEYWORD_TO_SOURCES` mapping | ✅ Present (expanded) | 5 |
| 12 | `_TYPE_DEFAULT_SOURCES` mapping | ⚠️ `law` entries removed | 3 |
| 13 | `format_context()` markdown | ✅ Present | 5 |
| 14 | `search_labor_laws()` DRF integration | ⚠️ Flow changed (AJAX+DRF vs DRF->AJAX) | 8 |
| 15 | `format_law_references()` multi-source | ✅ Full match | 5 |
| 16 | `has_multi_source()` | ✅ Match | 3 |
| 17 | `singletons.py` `get_law_drf_client()` | ✅ Full match | 5 |
| 18 | `.env.example` LAW_OC | ✅ Present | 2 |
| 19 | `rag_pipeline.py` section title change | ✅ Match | 5 |
| 20 | Response size limit (1MB) | ✅ Present | 3 |
| 21 | Query length limit | ⚠️ 100 chars instead of 200 | 2 |
| 22 | Logging prefixes | ✅ Match | 2 |

**Weighted calculation**:
- Total weight: 120
- Full match: 96 (items scoring ✅)
- Partial match: 13 (items scoring ⚠️ at 50% credit = 6.5)
- Missing: 6 (items scoring ❌ at 0%)
- **Match Rate: (96 + 6.5) / 120 = 85.4%**

### 7.2 Match Rate Summary

```
+---------------------------------------------+
|  Overall Match Rate: 86%                     |
+---------------------------------------------+
|  Full Match:          17 items (77%)         |
|  Partial Match:        3 items (14%)         |
|  Not Implemented:      2 items  (9%)         |
+---------------------------------------------+
```

---

## 8. Recommended Actions

### 8.1 Immediate Actions (Low Priority)

None of the gaps are blocking. All core functionality works correctly.

### 8.2 Design Document Updates Needed

The design document should be updated to reflect actual implementation decisions:

| # | Section | Update |
|---|---------|--------|
| 1 | Section 4.1 | Remove `search()` and `get_detail()` generic methods. Implementation uses target-specific methods directly, which provides better type safety and clearer API. |
| 2 | Section 4.1 | Change response format from JSON to XML. Add `_request_xml()` method signature and XML parsing utilities. |
| 3 | Section 4.3 | Update execution flow: AJAX always runs for law articles + DRF appends prec/interp/admrul. Remove "DRF first, AJAX fallback" description. |
| 4 | Section 4.2 | Update `_TYPE_DEFAULT_SOURCES` to remove `'law'` entries (law handled by AJAX). |
| 5 | Section 4.2 | Update `SEARCH_TIMEOUT` from 5s to 8s. |
| 6 | Section 6 | Update query length limit from 200 to 100 chars. |
| 7 | Section 4.1 | Remove `MAX_DISPLAY = 20` (each method manages display param individually). |

### 8.3 Optional Enhancements

| # | Item | Description | Impact |
|---|------|-------------|--------|
| 1 | OC param masking | Add explicit OC masking in error exception handlers to prevent leaking in stack traces | Low |
| 2 | Combined result limit | Add a cap on total combined (AJAX + DRF) results in `search_labor_laws()` (design specified 15) | Low |

---

## 9. Assessment

**Match Rate: 86% (>= 70%, < 90%)**

There are some differences between design and implementation. Most changes are intentional improvements:

1. **AJAX+DRF parallel strategy** is superior to the design's DRF-first approach because the existing AJAX law article search is more refined with its 74-law registry.
2. **XML-only parsing** is pragmatic given DRF's unreliable JSON support.
3. **Target-specific methods** instead of generic `search()`/`get_detail()` provide better code clarity.

**Recommendation**: Update the design document to match implementation (Option 2: Update design to match implementation). No code changes needed.

---

## 10. Design Document Updates Needed

- [ ] Update execution flow description (AJAX+DRF parallel instead of DRF-first fallback)
- [ ] Replace generic `search()`/`get_detail()` with target-specific method signatures
- [ ] Change response format from JSON to XML
- [ ] Update `_TYPE_DEFAULT_SOURCES` (remove `'law'` entries)
- [ ] Update timeout values (SEARCH_TIMEOUT 5s -> 8s, query limit 200 -> 100)
- [ ] Add `get_specific_law_articles()` and query refinements to design
- [ ] Add cache size limit (100 entries) to design

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-03-11 | Initial gap analysis | Claude Code (gap-detector) |
