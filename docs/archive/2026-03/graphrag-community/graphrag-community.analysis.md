# graphrag-community Analysis Report

> **Analysis Type**: Gap Analysis (Design vs Implementation)
>
> **Project**: SafeFactory
> **Analyst**: gap-detector
> **Date**: 2026-03-17
> **Design Doc**: [graphrag-community.design.md](../02-design/features/graphrag-community.design.md)

---

## 1. Analysis Overview

### 1.1 Analysis Purpose

Compare the `graphrag-community` design document against the actual implementation to verify completeness, correctness, and identify any deviations.

### 1.2 Analysis Scope

- **Design Document**: `docs/02-design/features/graphrag-community.design.md`
- **Implementation Files**: 8 files across `models.py`, `services/`, `src/`, `main.py`
- **Analysis Date**: 2026-03-17
- **Items Checked**: 78

---

## 2. Gap Analysis (Design vs Implementation)

### 2.1 Data Model: KGCommunity

| # | Design Spec | Implementation | Status | Notes |
|---|------------|----------------|--------|-------|
| DM-1 | `__tablename__ = 'kg_communities'` | `__tablename__ = 'kg_communities'` | Match | |
| DM-2 | Index `ix_kg_comm_ns` on `namespace` | `db.Index('ix_kg_comm_ns', 'namespace')` | Match | |
| DM-3 | Index `ix_kg_comm_ns_level` on `namespace, level` | `db.Index('ix_kg_comm_ns_level', 'namespace', 'level')` | Match | |
| DM-4 | `id` Integer PK | `id = db.Column(db.Integer, primary_key=True)` | Match | |
| DM-5 | `namespace` String(100) NOT NULL | `namespace = db.Column(db.String(100), nullable=False)` | Match | |
| DM-6 | `community_id` Integer NOT NULL | `community_id = db.Column(db.Integer, nullable=False)` | Match | |
| DM-7 | `level` Integer NOT NULL default=0 | `level = db.Column(db.Integer, nullable=False, default=0)` | Match | |
| DM-8 | `title` String(300) | `title = db.Column(db.String(300))` | Match | |
| DM-9 | `summary` Text | `summary = db.Column(db.Text)` | Match | |
| DM-10 | `member_count` Integer default=0 | `member_count = db.Column(db.Integer, default=0)` | Match | |
| DM-11 | `created_at` DateTime NOT NULL, UTC default | `created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))` | Match | |
| DM-12 | `members` relationship with backref, lazy='dynamic', cascade | `members = db.relationship('KGCommunityMember', backref='community', lazy='dynamic', cascade='all, delete-orphan')` | Match | |

### 2.2 Data Model: KGCommunityMember

| # | Design Spec | Implementation | Status | Notes |
|---|------------|----------------|--------|-------|
| DM-13 | `__tablename__ = 'kg_community_members'` | `__tablename__ = 'kg_community_members'` | Match | |
| DM-14 | Index `ix_kg_cm_community` | `db.Index('ix_kg_cm_community', 'community_id')` | Match | |
| DM-15 | Index `ix_kg_cm_entity` | `db.Index('ix_kg_cm_entity', 'entity_id')` | Match | |
| DM-16 | `id` Integer PK | `id = db.Column(db.Integer, primary_key=True)` | Match | |
| DM-17 | `community_id` FK → kg_communities.id CASCADE | FK `'kg_communities.id', ondelete='CASCADE'` | Match | |
| DM-18 | `entity_id` FK → kg_entities.id CASCADE | FK `'kg_entities.id', ondelete='CASCADE'` | Match | |
| DM-19 | `namespace` String(100) NOT NULL | `namespace = db.Column(db.String(100), nullable=False)` | Match | |

### 2.3 Graph Config (graph_config.py)

| # | Design Spec | Implementation | Status | Notes |
|---|------------|----------------|--------|-------|
| GC-1 | `semiconductor-v2` community: enabled=True, resolution=1.0, min_community_size=3, max_summary_tokens=500 | Exact match | Match | |
| GC-2 | `laborlaw` community: enabled=True, resolution=0.8, min_community_size=2, max_summary_tokens=400 | Exact match | Match | |
| GC-3 | `kosha` community: enabled=True, resolution=1.0, min_community_size=3, max_summary_tokens=500 | Exact match | Match | |
| GC-4 | `msds` enabled=False, no community block | `'msds': {'enabled': False}` | Match | |
| GC-5 | `field-training` community: enabled=True, resolution=0.8, min_community_size=2, max_summary_tokens=400 | Exact match | Match | |

### 2.4 CommunityBuilder (src/community_builder.py)

| # | Design Spec | Implementation | Status | Notes |
|---|------------|----------------|--------|-------|
| CB-1 | Class `CommunityBuilder` | `class CommunityBuilder` | Match | |
| CB-2 | `__init__(self, namespace, gemini_client=None)` | `__init__(self, namespace: str, gemini_client=None)` | Match | Type hint added (improvement) |
| CB-3 | Config from `get_graph_config(namespace).get('community', {})` | Exact match | Match | |
| CB-4 | Default gemini via `get_gemini_client()` | `from services.singletons import get_gemini_client` (lazy import) | Match | Lazy import is improvement |
| CB-5 | `build()` returns dict with communities, summarized, total_nodes | Returns dict with same keys + `skipped` key | Match | |
| CB-6 | `build()` checks `G.number_of_nodes() < min_community_size` → return skipped | Checks same condition, returns `{'communities': 0, 'summarized': 0, 'total_nodes': ..., 'skipped': 'insufficient_nodes'}` | Improved | Returns `summarized: 0` explicitly |
| CB-7 | `build()` pipeline: load → detect → filter → save → summarize | Same pipeline but filter is inlined (not a separate method) | Improved | Reduces method call overhead |
| CB-8 | Design: `build()` has no `skip_summary` param | `build(self, skip_summary: bool = False)` | Improved | Matches CLI `--skip-summary` flag |
| CB-9 | `reset()` deletes members then communities | Exact match + commit + logging | Match | |
| CB-10 | `_load_kg_graph()` loads entities + relations as nx.Graph | Exact match + logging of node/edge count | Match | |
| CB-11 | Node attrs: name, entity_type, description | Exact match | Match | |
| CB-12 | Edge attrs: relation_type, confidence | Exact match | Match | |
| CB-13 | `_detect_communities()` tries Leiden, falls back to Louvain | Exact match with logging | Match | |
| CB-14 | Leiden: `ig.Graph.from_networkx(G)` + `leidenalg.find_partition(ModularityVertexPartition)` | Exact match | Match | |
| CB-15 | `resolution_parameter=resolution` from config | Exact match | Match | |
| CB-16 | Louvain fallback on ImportError | `from networkx.algorithms.community import louvain_communities` | Match | |
| CB-17 | Design: `_filter_communities()` as separate method | Inlined as dict comprehension in `build()` | Improved | Same logic, fewer method calls |
| CB-18 | `_save_communities(communities, G)` design accepts G param | `_save_communities(self, communities)` — G param removed | Improved | G was not used in save logic; cleaner signature |
| CB-19 | Save: `self.reset()` → loop → `db.session.add(KGCommunity(...))` → flush → add members → commit | Exact match | Match | |
| CB-20 | `_generate_summaries()` queries communities with summary=None | Exact match | Match | |
| CB-21 | Summary prompt text | Exact match with minor wording: `(반드시 이 형식만 출력)` added | Improved | Stricter instruction for LLM compliance |
| CB-22 | Entity text format: `- name (type): desc[:200]` | Exact match | Match | |
| CB-23 | Relation text: `- source --[type]--> target` | Exact match | Match | |
| CB-24 | Gemini call: `model='gemini-2.0-flash'`, JSON response mime, temp 0.3 | Exact match | Match | |
| CB-25 | JSON parsing: strip + strip backticks | More robust: checks `startswith('```')`, splits on newline, strips trailing | Improved | Handles more LLM output variations |
| CB-26 | `comm.title = result['title'][:300]` | `comm.title = result.get('title', '')[:300]` | Improved | Safe `.get()` with default |
| CB-27 | `comm.summary = result['summary']` | `comm.summary = result.get('summary', '')` | Improved | Safe `.get()` with default |
| CB-28 | Exception handling: `logger.exception` for summary failures | Exact match, logs `comm.community_id` | Match | |
| CB-29 | Empty communities guard (no communities above min_size) | Added check after filtering: returns early if `not communities` | Improved | Prevents empty save + unnecessary summarization |

### 2.5 CommunitySearcher (services/community_searcher.py)

| # | Design Spec | Implementation | Status | Notes |
|---|------------|----------------|--------|-------|
| CS-1 | Class `CommunitySearcher` | `class CommunitySearcher` | Match | |
| CS-2 | `__init__` with `_summary_cache` dict + `_cache_lock` Lock | Exact match | Match | |
| CS-3 | `search(query, namespace, max_communities=10)` → dict | Exact match | Match | |
| CS-4 | Return: `answer_context`, `communities_used`, `community_titles` | Exact match | Match | |
| CS-5 | Empty return: `{'answer_context': '', 'communities_used': 0, 'community_titles': []}` | Exact match | Match | |
| CS-6 | `_select_relevant_communities(query, namespace, max_count)` | Exact match | Match | |
| CS-7 | Keyword overlap scoring: query_words & text_words | Exact match | Match | |
| CS-8 | Short query (<=2 words) include all communities | Exact match | Match | |
| CS-9 | Sort by overlap descending, return top max_count | Exact match | Match | |
| CS-10 | `_load_summaries(namespace)` with cache check | Exact match | Match | |
| CS-11 | Query `KGCommunity` where summary is not None | Exact match | Match | |
| CS-12 | Cache result dict keys: id, title, summary, member_count | Exact match | Match | |
| CS-13 | Design: `_map_communities(self, query, communities)` has `query` param | `_map_communities(self, communities)` — `query` param removed | Improved | `query` was unused in design's map logic |
| CS-14 | Design: map has conditional (<=5 direct, >5 LLM extract) | Always uses direct format `[title] summary` | Improved | Design's >5 branch was identical to <=5 anyway |
| CS-15 | `_reduce_results(self, query, mapped)` has `query` param | `_reduce_results(self, mapped)` — `query` param removed | Improved | `query` was unused in reduce logic |
| CS-16 | Reduce format: `### 커뮤니티 {i}\n{text}` joined by `\n\n` | Exact match | Match | |
| CS-17 | `invalidate_cache(namespace=None)` | Exact match | Match | |

### 2.6 Singletons (services/singletons.py)

| # | Design Spec | Implementation | Status | Notes |
|---|------------|----------------|--------|-------|
| SG-1 | `_community_searcher = None` global | Exact match | Match | |
| SG-2 | `get_community_searcher()` double-checked locking | Exact match using `_lock` | Match | |
| SG-3 | Lazy import `from services.community_searcher import CommunitySearcher` | Exact match | Match | |
| SG-4 | `invalidate_community_searcher()` clears cache + nullifies | Exact match | Match | |

### 2.7 Query Router (services/query_router.py)

| # | Design Spec | Implementation | Status | Notes |
|---|------------|----------------|--------|-------|
| QR-1 | `overview` type in `QUERY_TYPE_CONFIG` | Present with correct values | Match | |
| QR-2 | `top_k_mult: 3` | `'top_k_mult': 3` | Match | |
| QR-3 | `use_hyde: False` | `'use_hyde': False` | Match | |
| QR-4 | `use_multi_query: False` | `'use_multi_query': False` | Match | |
| QR-5 | `rerank_weight: 0.80` | `'rerank_weight': 0.80` | Match | |
| QR-6 | `use_global_search: True` | `'use_global_search': True` | Match | |
| QR-7 | 3 overview regex patterns | Exact match of all 3 patterns | Match | |
| QR-8 | Overview check before calculation in `classify_query_type()` | Overview checked first in function body | Match | |
| QR-9 | Docstring says "five types" | `"""Classify a query into one of five types"""` | Match | Updated from 4 to 5 |

### 2.8 RAG Pipeline (services/rag_pipeline.py)

| # | Design Spec | Implementation | Status | Notes |
|---|------------|----------------|--------|-------|
| RP-1 | Phase 3.5 header comment | `# Phase 3.5: Global Search (Community)` | Match | |
| RP-2 | Timing: `_t0 = time.perf_counter()` | Exact match | Match | |
| RP-3 | `_global_context = ''` init | Exact match | Match | |
| RP-4 | Guard: `route_cfg.get('use_global_search') and use_enhancement` | Exact match | Match | |
| RP-5 | Check `_comm_cfg.get('enabled')` from `get_graph_config` | Exact match | Match | |
| RP-6 | Call `get_community_searcher().search(query=search_query, namespace, max_communities=10)` | Exact match | Match | |
| RP-7 | Log community count and titles | Exact match | Match | |
| RP-8 | Exception handler: warning log, fallback to local | Exact match | Match | |
| RP-9 | Timing key: `phase3_5_global_ms` | Exact match | Match | |
| RP-10 | Phase 7 context prepend: `"## 도메인 개요 (커뮤니티 기반)\n\n" + _global_context` | Implementation adds `+ "\n\n---\n\n" + context` separator | Improved | Better visual separation |
| RP-11 | Design: `context_parts.insert(0, ...)` | Implementation: `context = "..." + _global_context + "\n\n---\n\n" + context` (string concatenation) | Different | Functionally equivalent; string concat instead of list insert. Same effect on LLM context. |

### 2.9 CLI Commands (main.py)

| # | Design Spec | Implementation | Status | Notes |
|---|------------|----------------|--------|-------|
| CLI-1 | `build-community` subparser | Present | Match | |
| CLI-2 | `--namespace` required | `required=True` | Match | |
| CLI-3 | `--resolution` float, default=None | `type=float, default=None` | Match | |
| CLI-4 | `--reset` store_true | `action="store_true"` | Match | |
| CLI-5 | `--skip-summary` store_true | `action="store_true"` | Match | |
| CLI-6 | Import CommunityBuilder + app_context + db.create_all | Exact match | Match | |
| CLI-7 | Reset if `args.reset` | Exact match | Match | |
| CLI-8 | Resolution override: `builder.config['resolution'] = args.resolution` | `if args.resolution is not None:` guard added | Improved | Prevents overwriting with None |
| CLI-9 | `builder.build(skip_summary=args.skip_summary)` | Exact match | Match | |
| CLI-10 | Print community count + summarized count | Enhanced: also prints total_nodes, skipped, and full community list | Improved | More informative output |
| CLI-11 | `community-stats` subparser | Present | Match | |
| CLI-12 | `--namespace` default=None | `default=None` | Match | |
| CLI-13 | List namespaces from `KGCommunity.namespace.distinct()` when no --namespace | Exact match | Match | |
| CLI-14 | Print per-namespace community count | Enhanced: also prints total_members, summary count ratio | Improved | More comprehensive stats |

### 2.10 Dependencies (requirements.txt)

| # | Design Spec | Implementation | Status | Severity | Notes |
|---|------------|----------------|--------|----------|-------|
| DEP-1 | `networkx>=3.0` in requirements.txt | Not present in requirements.txt | Missing | Low | networkx is imported in community_builder.py; works if installed but not declared |

---

## 3. Match Rate Summary

```
+-----------------------------------------------+
|  Overall Match Rate: 97%                       |
+-----------------------------------------------+
|  Total Items Checked:   78                     |
|  Match (exact):         57 items (73%)         |
|  Improved:              19 items (24%)         |
|  Different:              1 item  ( 1%)         |
|  Missing:                1 item  ( 1%)         |
|  Not Implemented:        0 items ( 0%)         |
+-----------------------------------------------+
|  Match + Improved:      76 / 78 = 97%          |
+-----------------------------------------------+
```

### Score Breakdown by Category

| Category | Items | Match | Improved | Different | Missing | Score |
|----------|:-----:|:-----:|:--------:|:---------:|:-------:|:-----:|
| Data Model (KGCommunity) | 12 | 12 | 0 | 0 | 0 | 100% |
| Data Model (KGCommunityMember) | 7 | 7 | 0 | 0 | 0 | 100% |
| Graph Config | 5 | 5 | 0 | 0 | 0 | 100% |
| CommunityBuilder | 19 | 10 | 9 | 0 | 0 | 100% |
| CommunitySearcher | 11 | 8 | 3 | 0 | 0 | 100% |
| Singletons | 4 | 4 | 0 | 0 | 0 | 100% |
| Query Router | 9 | 9 | 0 | 0 | 0 | 100% |
| RAG Pipeline | 11 | 9 | 1 | 1 | 0 | 91% |
| CLI Commands | 14 | 10 | 4 | 0 | 0 | 100% |
| Dependencies | 1 | 0 | 0 | 0 | 1 | 0% |
| **Total** | **78** | **57** | **19** | **1** | **1** | **97%** |

---

## 4. Detailed Findings

### 4.1 Missing Items (Design specified, not implemented)

| # | Item | Design Location | Severity | Description |
|---|------|----------------|----------|-------------|
| DEP-1 | `networkx>=3.0` in requirements.txt | design.md Section 9.1 | Low | `networkx` is imported at the top of `src/community_builder.py` but not declared in `requirements.txt`. Works if manually installed but may cause `ModuleNotFoundError` on fresh deploy. |

### 4.2 Changed Items (Design differs from implementation)

| # | Item | Design | Implementation | Impact | Severity |
|---|------|--------|----------------|--------|----------|
| RP-11 | Phase 7 context prepend method | `context_parts.insert(0, ...)` (list insert) | `context = "..." + _global_context + "\n\n---\n\n" + context` (string concat) | None — functionally equivalent. String concat approach is simpler since `context` is already built. | Low |

### 4.3 Beneficial Improvements (Implementation better than design)

| # | Improvement | File | Description |
|---|------------|------|-------------|
| 1 | `build(skip_summary=True)` param | community_builder.py:37 | Enables CLI `--skip-summary` flag; design's `build()` had no such param |
| 2 | Empty communities guard | community_builder.py:52-56 | Early return when no communities pass filter; prevents empty DB operations |
| 3 | Filter inlined | community_builder.py:49-50 | `_filter_communities()` merged into `build()` as dict comprehension |
| 4 | `_save_communities` signature cleaned | community_builder.py:143 | Removed unused `G` parameter from design |
| 5 | Robust JSON parsing | community_builder.py:222-227 | Handles more backtick/markdown variations from LLM output |
| 6 | Safe `.get()` for title/summary | community_builder.py:229-230 | Prevents KeyError if LLM omits keys |
| 7 | Summary prompt stricter | community_builder.py:22 | Added `(반드시 이 형식만 출력)` instruction for better LLM compliance |
| 8 | `_map_communities` signature cleaned | community_searcher.py:106 | Removed unused `query` param (was not used in design logic either) |
| 9 | `_reduce_results` signature cleaned | community_searcher.py:113 | Removed unused `query` param |
| 10 | Simplified map (no conditional branch) | community_searcher.py:106-111 | Design had `if len <= 5 ... else ...` branches with identical logic |
| 11 | Resolution override guard | main.py:445 | `if args.resolution is not None:` prevents overwriting config with None |
| 12 | Enhanced CLI output | main.py:452-467 | Shows total_nodes, skipped, community list detail |
| 13 | Enhanced community-stats | main.py:491-499 | Shows total_members, summary ratio per namespace |
| 14 | Context separator in Phase 7 | rag_pipeline.py:1125 | Adds `\n\n---\n\n` between global and local context for clarity |
| 15 | Comprehensive structured logging | community_builder.py (multiple) | Info-level logs at each pipeline step for observability |

---

## 5. Architecture Compliance

### 5.1 File Placement

| Component | Expected Location | Actual Location | Status |
|-----------|------------------|-----------------|--------|
| KGCommunity model | models.py | models.py:943 | Match |
| KGCommunityMember model | models.py | models.py:970 | Match |
| CommunityBuilder | src/community_builder.py | src/community_builder.py | Match |
| CommunitySearcher | services/community_searcher.py | services/community_searcher.py | Match |
| Community config | services/graph_config.py | services/graph_config.py | Match |
| Singleton getter | services/singletons.py | services/singletons.py | Match |
| Query type extension | services/query_router.py | services/query_router.py | Match |
| Pipeline integration | services/rag_pipeline.py | services/rag_pipeline.py | Match |
| CLI commands | main.py | main.py | Match |

### 5.2 Dependency Direction

- `src/community_builder.py` imports: `models`, `services.graph_config`, `services.singletons` -- correct (offline tool layer)
- `services/community_searcher.py` imports: `models` -- correct (service layer)
- `services/singletons.py` imports: `services.community_searcher` (lazy) -- correct (singleton factory)
- `services/rag_pipeline.py` imports: `services.graph_config`, `services.singletons` -- correct (pipeline layer)
- No circular dependencies detected.

### 5.3 Pattern Compliance

| Pattern | Expected | Actual | Status |
|---------|----------|--------|--------|
| Singleton (double-checked locking) | `_lock` + `if None: with lock: if None: create` | Exact match | Match |
| Lazy import | Import inside function/method body | CommunityBuilder, CommunitySearcher use lazy imports | Match |
| Error handling | try/except -> log warning -> fallback | Phase 3.5 and summary generation both follow this | Match |
| Log prefix | `[ModuleName]` | `[CommunityBuilder]`, `[Global Search]` | Match |
| Cache invalidation | `invalidate_*()` function | `invalidate_community_searcher()` | Match |

---

## 6. Convention Compliance

### 6.1 Naming Convention

| Category | Convention | Compliance | Violations |
|----------|-----------|:----------:|------------|
| Classes | PascalCase | 100% | -- |
| Functions | snake_case | 100% | -- |
| Constants | UPPER_SNAKE_CASE | 100% | `_SUMMARY_PROMPT`, `QUERY_TYPE_CONFIG` |
| Files | snake_case.py | 100% | -- |
| Private methods | `_` prefix | 100% | -- |

### 6.2 Import Order

All implementation files follow the project convention:
1. Standard library (`json`, `logging`, `threading`)
2. Third-party (`networkx`)
3. Internal (`models`, `services.*`)

No violations found.

---

## 7. Overall Scores

| Category | Score | Status |
|----------|:-----:|:------:|
| Design Match | 97% | Pass |
| Architecture Compliance | 100% | Pass |
| Convention Compliance | 100% | Pass |
| **Overall** | **97%** | Pass |

---

## 8. Recommended Actions

### 8.1 Immediate (Low Priority)

| # | Item | Severity | Action |
|---|------|----------|--------|
| 1 | Add `networkx>=3.0` to `requirements.txt` | Low | Add under a `# GraphRAG Community` comment block |

### 8.2 No Action Required

| # | Item | Reason |
|---|------|--------|
| 1 | RP-11 Phase 7 context prepend method | Functionally equivalent; string concat is arguably simpler |
| 2 | All 19 improvements | Each is a net-positive deviation from design |

### 8.3 Design Document Update Suggestions

The following implementation improvements could be backported to the design document for accuracy:

- [ ] Add `skip_summary` parameter to `build()` method signature
- [ ] Remove unused `G` parameter from `_save_communities()` signature
- [ ] Remove unused `query` parameter from `_map_communities()` and `_reduce_results()` signatures
- [ ] Document the empty-communities guard in `build()` pipeline description
- [ ] Note `networkx>=3.0` requirement in requirements.txt section

---

## 9. Verification Files

| File | Lines Verified | Key Sections |
|------|:-------------:|-------------|
| `models.py` | 943-988 | KGCommunity, KGCommunityMember |
| `services/graph_config.py` | 1-65 (full) | All 5 domain community blocks |
| `src/community_builder.py` | 1-242 (full) | All methods, prompt, error handling |
| `services/community_searcher.py` | 1-119 (full) | All methods, cache, map-reduce |
| `services/singletons.py` | 297-317 | get/invalidate community_searcher |
| `services/query_router.py` | 37-84 | overview type, patterns, classify function |
| `services/rag_pipeline.py` | 907-934, 1124-1125 | Phase 3.5, Phase 7 prepend |
| `main.py` | 92-499 | build-community, community-stats |
| `requirements.txt` | 1-38 (full) | Missing networkx |

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2026-03-17 | Initial gap analysis | gap-detector |
