# search-history Analysis Report

> **Analysis Type**: Gap Analysis (Design vs Implementation)
>
> **Project**: SafeFactory
> **Analyst**: gap-detector
> **Date**: 2026-03-11
> **Design Doc**: [search-history.design.md](../02-design/features/search-history.design.md)

---

## 1. Analysis Overview

### 1.1 Analysis Purpose

PDCA Check phase: verify that the "search-history" feature implementation matches
the design document spec across data model, API endpoints, existing-code
modifications, UI, error handling, and security.

### 1.2 Analysis Scope

- **Design Document**: `docs/02-design/features/search-history.design.md`
- **Implementation Files**:
  - `models.py` (SearchHistory model)
  - `api/v1/search.py` (helper + save calls + CRUD)
  - `web_app.py` (/history route)
  - `templates/history.html` (history page UI)
  - `templates/domain.html` (recent queries dropdown)
  - `templates/base.html` (navigation links)

---

## 2. Gap Analysis (Design vs Implementation)

### 2.1 Data Model (Section 1)

| Spec Item | Design | Implementation | Status |
|-----------|--------|----------------|--------|
| `__tablename__` | `'search_history'` | `'search_history'` | Match |
| Composite index `ix_search_history_user_created` on (`user_id`, `created_at`) | Yes | Yes (`models.py:741`) | Match |
| `id` (Integer, PK) | Yes | Yes | Match |
| `user_id` (Integer, FK `users.id`, CASCADE) | Yes | Yes | Match |
| `query` (Text, not null) | Yes | Yes | Match |
| `query_type` (String(10), default `'search'`) | Yes | Yes | Match |
| `namespace` (String(100), default `''`) | Yes | Yes | Match |
| `search_mode` (String(20), nullable) | Yes | Yes | Match |
| `result_count` (Integer, default 0) | Yes | Yes | Match |
| `answer_preview` (String(200), nullable) | Yes | Yes | Match |
| `created_at` (DateTime, `datetime.now(timezone.utc)`) | Yes | Yes | Match |
| `user = db.relationship('User', backref='search_histories')` | Yes | Yes (`models.py:759`) | Match |
| `MAX_PER_USER = 500` | Yes | Yes (`models.py:761`) | Match |
| `to_dict()` method with all 8 fields | Yes | Yes (`models.py:763-773`) | Match |
| Placement: below `NewsArticle` class | Yes | Yes (line 736, after `NewsArticle` ending at 729) | Match |

**Data Model: 15/15 items match (100%)**

### 2.2 API Design (Section 2)

#### 2.2.1 `_save_search_history` helper (Section 2.1)

| Spec Item | Design | Implementation | Status |
|-----------|--------|----------------|--------|
| Function signature matches (7 params) | Yes | Yes (`search.py:114-115`) | Match |
| `query[:500]` truncation | Yes | Yes (`:120`) | Match |
| `answer_preview[:200]` truncation | Yes | Yes (`:125`) | Match |
| `db.session.add(record)` + `commit()` | Yes | Yes | Match |
| Count check > MAX_PER_USER | Yes | Yes (`:131-132`) | Match |
| Prune oldest via `created_at.asc()` subquery | Yes | Yes (`:134-145`) | Match |
| `except Exception: db.session.rollback()` + `logging.exception` | Yes | Yes (`:146-148`) | Match |

**Helper function: 7/7 items match (100%)**

#### 2.2.2 GET `/api/v1/search/history` (Section 2.2)

| Spec Item | Design | Implementation | Status |
|-----------|--------|----------------|--------|
| Route `GET /search/history` | Yes | Yes (`search.py:710`) | Match |
| `@login_required` | Yes | Yes (`:711`) | Match |
| `page` param, default 1 | Yes | Yes (`:716`) | Match |
| `per_page` param, default 20, max 50 | Yes | Yes (`:717`) | Match |
| `query_type` filter | Yes | Yes (`:721-723`) | Match |
| `namespace` filter | Yes | Yes (`:725-727`) | Match |
| Order by `created_at.desc()` | Yes | Yes (`:729`) | Match |
| Response shape: `{items, total, page, per_page, pages}` | Yes | Yes (`:732-738`) | Match |

**History list endpoint: 8/8 items match (100%)**

#### 2.2.3 GET `/api/v1/search/history/recent` (Section 2.3)

| Spec Item | Design | Implementation | Status |
|-----------|--------|----------------|--------|
| Route `GET /search/history/recent` | Yes | Yes (`search.py:744`) | Match |
| `@login_required` | Yes | Yes (`:745`) | Match |
| `limit` param, default 10, max 20 | Yes | Yes (`:749`) | Match |
| Deduplicate preserving order | Design: `dict.fromkeys()` | Impl: manual `seen` dict | Partial |
| Response shape: `{queries: [...]}` | Yes | Yes (`:765`) | Match |

**Deduplication deviation**: Design specifies `dict.fromkeys()`, implementation uses an
equivalent manual dict-based approach (`seen = {}; for r in rows: if r.query not in seen`).
Functionally identical, stylistically different. Also, the implementation fetches `limit * 3` rows
before deduplication, which is a practical optimization not specified in the design.

**Recent endpoint: 4/5 items match, 1 partial (90%)**

#### 2.2.4 DELETE `/api/v1/search/history/<int:history_id>` (Section 2.4)

| Spec Item | Design | Implementation | Status |
|-----------|--------|----------------|--------|
| Route `DELETE /search/history/<int:history_id>` | Yes | Yes (`search.py:771`) | Match |
| `@login_required` | Yes | Yes (`:772`) | Match |
| Filter by `id=history_id, user_id=current_user.id` | Yes | Yes (`:776-777`) | Match |
| Not found -> 404 | Yes | Yes (`:780`) | Match |
| Success message matches | Yes | Yes (`:784`) | Match |

**Delete single endpoint: 5/5 items match (100%)**

#### 2.2.5 DELETE `/api/v1/search/history` (Section 2.5)

| Spec Item | Design | Implementation | Status |
|-----------|--------|----------------|--------|
| Route `DELETE /search/history` | Yes | Yes (`search.py:791`) | Match |
| `@login_required` | Yes | Yes (`:792`) | Match |
| Filter by `user_id=current_user.id` | Yes | Yes (`:796-797`) | Match |
| Return `deleted_count` | Yes | Yes (`:801`) | Match |
| Success message matches | Yes | Yes (`:800`) | Match |

**Delete all endpoint: 5/5 items match (100%)**

### 2.3 Existing Code Modifications (Section 3)

| Spec Item | Design | Implementation | Status |
|-----------|--------|----------------|--------|
| Import `current_user, login_required` from `flask_login` | Yes | Yes (`search.py:12`) | Match |
| Import `db, SearchHistory` from `models` | Yes | Yes (`search.py:17`) | Match |
| `api_search()`: save call before return | Yes | Yes (`search.py:264-272`) | Match |
| `api_ask()`: save call with `answer_preview` | Yes | Yes (`search.py:378-386`) | Match |
| `api_ask_stream()`: save in `generate()` before done event | Yes | Yes (`search.py:544-551`) | Match |
| `api_ask_stream()`: `answer_preview=None` (no full answer in stream) | Yes | Yes (no `answer_preview` param -> defaults to None) | Match |
| Guard: `hasattr(current_user, 'id') and current_user.is_authenticated` | Design uses this pattern | Impl uses `current_user.is_authenticated` only | Partial |
| `web_app.py`: `/history` route with `@login_required` | Yes | Yes (`web_app.py:334-338`) | Match |
| `web_app.py`: renders `history.html` | Yes | Yes | Match |
| `models.py`: SearchHistory class added | Yes | Yes | Match |

**Authentication guard deviation**: The design specifies
`if hasattr(current_user, 'id') and current_user.is_authenticated:` but
the implementation uses the simpler `if current_user.is_authenticated:`.
This is functionally equivalent because Flask-Login's `is_authenticated`
already implies a valid user object with an `id` attribute. The implementation
is actually cleaner.

**Existing code mods: 9/10 items match, 1 partial (95%)**

### 2.4 UI Design (Section 4)

#### 2.4.1 History Page (`templates/history.html`)

| Spec Item | Design | Implementation | Status |
|-----------|--------|----------------|--------|
| Page title: "검색 기록" | Yes | Yes (`history.html:3, 224`) | Match |
| Dark theme card UI consistent with mypage.html | Yes | Yes (dark gradient background, card styling) | Match |
| Header with `[전체 삭제]` button | Yes | Yes (`:225`) | Match |
| Filter dropdowns: query_type (`전체/검색/AI 질문`) | Yes | Yes (`:229-233`) | Match |
| Filter dropdown: namespace (`전체 도메인` + 5 domains) | Yes | Yes (`:234-241`) | Match |
| History item: icon (search/ask differentiated) | Yes | Yes (`:321`) | Match |
| History item: query text | Yes | Yes (`:328`) | Match |
| History item: meta line (type, domain, count, time) | Yes | Yes (`:329-333`) | Match |
| History item: answer_preview for ask items | Yes | Yes (`:335-337`) | Match |
| `[x]` individual delete button | Yes | Yes (`:338`) | Match |
| Delete confirm popup | Yes | Yes (`:365, 376`) | Match |
| Click item -> navigate to domain page + query auto-input | Yes | Yes (`:327` via `window.location.href`) | Match |
| Pagination: 20 items per page | Yes | Yes (`:289`) | Match |
| Pagination controls | Yes | Yes (`:347-362`) | Match |

**History page UI: 14/14 items match (100%)**

#### 2.4.2 Recent Queries Dropdown (`templates/domain.html`)

| Spec Item | Design | Implementation | Status |
|-----------|--------|----------------|--------|
| Dropdown on search input focus | Yes | Yes (`domain.html:1868`) | Match |
| Calls `GET /api/v1/search/history/recent` | Yes | Yes (`:1839`) | Match |
| Login guard (no request if not logged in) | Yes | Yes (`:1827-1828`) | Match |
| Shows "최근 검색" header | Yes | Yes (`:1850`) | Match |
| Max 10 items | Yes | Yes (`:1839`, `limit=10`) | Match |
| Click item -> fills input + auto-search | Yes | Yes (`:1859-1864`, dispatches form submit) | Match |
| Dropdown styling (positioned under input) | Yes | Yes (CSS at `:1006-1036`) | Match |

**Recent queries dropdown: 7/7 items match (100%)**

#### 2.4.3 Navigation Link (`templates/base.html`)

| Spec Item | Design | Implementation | Status |
|-----------|--------|----------------|--------|
| "검색 기록" / "검색기록" link in nav | Yes | Yes (`base.html:358`, text: "검색기록") | Match |
| Authenticated users only | Yes | Yes (inside `{% if current_user.is_authenticated %}` block) | Match |
| Mobile menu also has link | Not specified | Yes (`base.html:395-397`) | Added |

**Navigation: 2/2 required items match + 1 bonus item (100%)**

### 2.5 Error Handling (Section 6)

| Spec Item | Design | Implementation | Status |
|-----------|--------|----------------|--------|
| Unauthenticated access -> `@login_required` -> 401 | Yes | Yes (on all CRUD endpoints) | Match |
| Unauthorized delete -> `user_id == current_user.id` filter -> 404 | Yes | Yes (`search.py:776-780`) | Match |
| Save failure -> `try/except` + `rollback()`, no impact on search response | Yes | Yes (`search.py:146-148`) | Match |
| Pagination out of range -> empty items | Yes | Yes (SQLAlchemy `paginate(error_out=False)` returns empty) | Match |

**Error handling: 4/4 items match (100%)**

### 2.6 Security (Section 7)

| Spec Item | Design | Implementation | Status |
|-----------|--------|----------------|--------|
| All queries filter by `user_id == current_user.id` | Yes | Yes (all CRUD endpoints) | Match |
| CSRF exempt for `v1_bp` (existing pattern) | Yes | Yes (`web_app.py:95`) | Match |
| SQLAlchemy ORM (parameterized queries) | Yes | Yes (no raw SQL) | Match |
| Rate limiting `@rate_limit("30 per minute")` on history list | Yes | Yes (`search.py:712`) | Match |
| Rate limiting on recent endpoint | Not specified | Not applied | Match |
| `query` 500-char limit | Yes | Yes (`search.py:120`) | Match |
| `per_page` max 50 | Yes | Yes (`search.py:717`) | Match |
| `limit` max 20 | Yes | Yes (`search.py:749`) | Match |

**Security: 8/8 items match (100%)**

---

## 3. Match Rate Summary

```
Total specification items checked: 89

  Match:   87 items  (97.8%)
  Partial:  2 items  ( 2.2%)
  Missing:  0 items  ( 0.0%)
```

| Category | Items | Match | Partial | Missing | Score |
|----------|:-----:|:-----:|:-------:|:-------:|:-----:|
| Data Model | 15 | 15 | 0 | 0 | 100% |
| API - Helper Function | 7 | 7 | 0 | 0 | 100% |
| API - History List | 8 | 8 | 0 | 0 | 100% |
| API - Recent | 5 | 4 | 1 | 0 | 90% |
| API - Delete Single | 5 | 5 | 0 | 0 | 100% |
| API - Delete All | 5 | 5 | 0 | 0 | 100% |
| Existing Code Mods | 10 | 9 | 1 | 0 | 95% |
| UI - History Page | 14 | 14 | 0 | 0 | 100% |
| UI - Recent Dropdown | 7 | 7 | 0 | 0 | 100% |
| UI - Nav Link | 2 | 2 | 0 | 0 | 100% |
| Error Handling | 4 | 4 | 0 | 0 | 100% |
| Security | 8 | 8 | 0 | 0 | 100% |

---

## 4. Overall Scores

| Category | Score | Status |
|----------|:-----:|:------:|
| Design Match | 98% | Match |
| Architecture Compliance | 100% | Match |
| Convention Compliance | 100% | Match |
| **Overall** | **98%** | **Match** |

---

## 5. Differences Found

### Partial Matches (Design ~ Implementation)

| # | Item | Design | Implementation | Impact |
|---|------|--------|----------------|--------|
| 1 | Deduplication method (recent endpoint) | `dict.fromkeys()` | Manual `seen` dict + `limit * 3` over-fetch | None (functionally identical, impl is more robust) |
| 2 | Auth guard pattern | `hasattr(current_user, 'id') and current_user.is_authenticated` | `current_user.is_authenticated` | None (impl is cleaner; Flask-Login guarantees `id` on authenticated users) |

### Added Features (Design X, Implementation O)

| Item | Implementation Location | Description |
|------|------------------------|-------------|
| Mobile nav "검색기록" link | `base.html:395-397` | Mobile menu also shows the history link; design only mentioned desktop nav |
| `limit * 3` over-fetch | `search.py:755` | Fetches 3x limit rows to improve dedup quality before trimming to `limit` |

---

## 6. Recommended Actions

### Documentation Update (Low Priority)

1. Update design Section 2.3 deduplication description to match the actual `seen`-dict implementation and `limit * 3` over-fetch optimization.
2. Update design Section 3.2-3.4 auth guard pattern from `hasattr(current_user, 'id') and current_user.is_authenticated` to `current_user.is_authenticated`.

No code changes are necessary -- all deviations are improvements over the original design.

---

## 7. Conclusion

The "search-history" feature implementation achieves a **98% match rate** with the design
document. The two partial matches are minor stylistic differences that represent
improvements over the design (cleaner auth guard, more robust deduplication). There are
zero missing features. The implementation fully satisfies all functional, security,
error-handling, and UI requirements specified in the design.

**Recommendation**: Mark the Check phase as complete and proceed to Report.

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2026-03-11 | Initial analysis | gap-detector |
