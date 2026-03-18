# gemini-migration Gap Analysis Report

> **Date**: 2026-03-18
> **Feature**: gemini-migration
> **Design**: docs/02-design/features/gemini-migration.design.md

---

## Overall Scores

| Category | Score | Status |
|----------|:-----:|:------:|
| Design Match (Sections 3.1-3.7) | 100% (25/25) | Pass |
| Downstream Compatibility | 37% (4/11 benign) | FAIL |
| **Weighted Overall** | **86%** | Needs Fix |

---

## Design Match: 25/25 (100%)

All 7 design sections implemented exactly as specified. No missing features, no deviations.

| Section | Items | Status |
|---------|:-----:|:------:|
| 3.1 resolve_namespace() | 7/7 | Pass |
| 3.2 get_agent() | 5/5 | Pass |
| 3.3 invalidate_agent() + admin | 5/5 | Pass |
| 3.4 rag_pipeline 3 sites | 4/4 | Pass |
| 3.5 search.py | 2/2 | Pass |
| 3.6 main.py CLI | 1/1 | Pass |
| 3.7 settings.py default | 1/1 | Pass |

---

## CRITICAL: Downstream Bugs (Root Cause: resolve too early)

`resolve_namespace()` applied at line 768 transforms `namespace` from base (`semiconductor-v2`) to resolved (`semiconductor-v2-gemini`). All subsequent dictionary lookups use base namespace keys and fail.

### Bug List

| # | Location | Impact | Severity |
|---|----------|--------|----------|
| B-1 | `rag_pipeline.py:796` NAMESPACE_DOMAIN_MAP lookup | Wrong domain_key | High |
| B-2 | `rag_pipeline.py:624` domain hint for query enhancement | Wrong domain | Medium |
| B-3 | `rag_pipeline.py:835` build_domain_filter | No metadata filter | High |
| B-4 | `rag_pipeline.py:941` Phase 4 RRF config | Wrong config | Medium |
| B-5 | `rag_pipeline.py:963` Phase 5 rerank config | Wrong config | Medium |
| B-6 | `rag_pipeline.py:1150` Safety cross-search trigger | Never triggers | High |
| B-7 | `rag_pipeline.py:1162` MSDS cross-search trigger | Never triggers | High |
| B-8 | `rag_pipeline.py:1223` DOMAIN_PROMPTS lookup | Wrong prompt | High |
| B-9 | `rag_pipeline.py:1224` COT instructions | Wrong instructions | High |
| B-10 | `query_router.py:194` current-page bonus | Never applied | Medium |
| B-11 | `rag_pipeline.py:791` detected vs resolved comparison | Unnecessary override | Low |

### Affected Domains

- **semiconductor**: Unaffected (defaults coincidentally match)
- **field-training, kosha, msds**: Critical breakage

### Fix

Maintain `base_namespace` for dictionary lookups; apply `resolve_namespace()` only at Pinecone search call boundaries.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-03-18 | Initial: 100% design match, 11 downstream bugs found |
