# Code Review Reference: Language-Specific Patterns

This reference covers common anti-patterns and review focus areas by language/framework.
Read this when reviewing code in a specific language to catch language-specific issues.

## JavaScript / TypeScript

### Common Issues
- `==` instead of `===` (type coercion bugs)
- Missing `await` on async functions (silent promise drops)
- Mutating state directly in React (use setState/spread)
- Memory leaks from uncleared intervals, event listeners, or subscriptions
- `any` type overuse defeating TypeScript's purpose
- Prototype pollution via unchecked object merging
- `eval()` or `new Function()` usage (security risk)

### React Specific
- Missing dependency arrays in `useEffect` (infinite re-renders or stale closures)
- Creating objects/functions inside render without `useMemo`/`useCallback`
- Direct DOM manipulation instead of refs
- Key props using array index (unstable keys cause reconciliation bugs)
- Not cleaning up effects (return cleanup function)

### Node.js Specific
- Unhandled promise rejections (crash in Node 15+)
- Synchronous file I/O in request handlers
- Missing rate limiting on public endpoints
- SQL injection via string concatenation
- Path traversal in file operations

## Python

### Common Issues
- Mutable default arguments (`def foo(items=[])` — shared across calls)
- Bare `except:` catching everything including `SystemExit`
- Not using context managers for resources (`with open(...)`)
- Late binding closures in loops
- f-strings with user input in logging (format string attacks)
- `pickle` with untrusted data (arbitrary code execution)

### Django / FastAPI Specific
- Missing CSRF protection
- `raw()` SQL queries without parameterization
- N+1 queries (use `select_related` / `prefetch_related`)
- Missing authentication decorators on views
- Exposing internal IDs without authorization checks

## Go

### Common Issues
- Ignored errors (`val, _ := someFunc()`)
- Goroutine leaks (no way to signal goroutine to stop)
- Race conditions on shared state (missing mutex/channel)
- `defer` in loops (resource leak until function returns)
- Nil pointer dereference on interface types
- String concatenation in loops (use `strings.Builder`)

## Rust

### Common Issues
- `unwrap()` in production code (use `?` or proper error handling)
- Unnecessary `clone()` to satisfy borrow checker
- Blocking in async context
- Unsafe blocks without justification
- Missing error context (use `anyhow` or `thiserror`)

## Java / Kotlin

### Common Issues
- Unclosed resources (use try-with-resources / `use {}`)
- `equals()` vs `==` for object comparison
- Mutable collections exposed from getters
- Missing null checks (or missing `@Nullable` annotations)
- Thread safety issues with shared mutable state
- Spring: missing `@Transactional` on service methods that modify data

## SQL

### Common Issues
- SQL injection via string concatenation
- Missing indexes on frequently queried columns
- `SELECT *` in production queries
- Missing `WHERE` clause on `UPDATE`/`DELETE`
- N+1 query patterns (should be JOINs or batch queries)
- Missing transactions for multi-statement operations
