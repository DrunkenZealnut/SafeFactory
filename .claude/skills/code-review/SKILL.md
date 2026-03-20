---
name: code-review
description: >
  AI-powered code review skill inspired by CodeRabbit. Analyzes git diffs, pull requests,
  staged changes, or specific files to provide structured, actionable code review feedback.
  Use this skill whenever the user asks for: code review, PR review, diff review, reviewing
  changes, checking code quality, finding bugs in changes, security review of code, or
  anything related to reviewing code modifications. Also trigger when users say things like
  "review my PR", "check my changes", "look at this diff", "review before I merge",
  "what do you think of these changes", or "/review". This skill should be your go-to
  whenever code quality assessment is needed, even if the user doesn't explicitly say "review".
---

# Code Review

An AI-powered code reviewer for Claude Code, inspired by CodeRabbit. It analyzes your git
diffs, PRs, staged changes, or individual files and delivers structured, actionable feedback
covering bugs, security, performance, style, and maintainability.

## Quick Start

When this skill triggers, follow these steps:

1. **Determine the review scope** — what exactly should be reviewed
2. **Gather the diff / code** — using git commands or file reads
3. **Analyze the changes** — applying the review checklist
4. **Output the review** — in the structured format described below

## Step 1: Determine Review Scope

Figure out what the user wants reviewed. Common scenarios:

| User says | Scope | How to gather |
|-----------|-------|---------------|
| "review my PR" / "review PR #123" | Pull request diff | `gh pr diff <number>` or `git diff main...HEAD` |
| "review my changes" / "review staged" | Staged changes | `git diff --cached` |
| "review" (no qualifier) | All uncommitted changes | `git diff HEAD` (staged + unstaged) |
| "review this file" / "review src/foo.ts" | Specific file(s) | Read the file(s) directly |
| "review last commit" | Most recent commit | `git diff HEAD~1..HEAD` |
| "review branch X" | Branch diff against base | `git diff main...X` |
| "review commits A..B" | Commit range | `git diff A..B` |

If the scope is ambiguous, check `git status` first and ask the user to clarify only if
there are multiple plausible interpretations. When in doubt, default to reviewing all
uncommitted changes (`git diff HEAD`).

### Detecting the base branch

Not every project uses `main` as the default branch. Run this to detect it:

```bash
git symbolic-ref refs/remotes/origin/HEAD 2>/dev/null | sed 's@^refs/remotes/origin/@@'
```

If that fails, fall back to checking if `main` or `master` exists.

## Step 2: Gather Context

Good reviews need context beyond just the diff. Gather:

1. **The diff itself** — using the appropriate git command from Step 1
2. **Full file context for changed files** — read the complete file when a diff is complex,
   because understanding surrounding code is essential for a quality review
3. **Related files** — if the diff modifies an interface, check its implementations;
   if it modifies a function, check its callers. Use `grep` / `glob` as needed.
4. **Project context** — check for:
   - `package.json`, `Cargo.toml`, `pyproject.toml`, etc. to understand the stack
   - Linter configs (`.eslintrc`, `ruff.toml`, `.prettierrc`, etc.) to align with project style rules
   - Test files related to changed code
   - CI/CD config (`.github/workflows/`, etc.)

Don't spend more than a few tool calls on context gathering. The goal is enough context to
give informed feedback, not a complete codebase audit.

## Step 3: Analyze Changes

For each changed file, evaluate against the review checklist below. Not every item applies
to every change — use judgment about which categories are relevant.

### Review Checklist

#### 1. Bugs & Correctness
- Logic errors, off-by-one mistakes, wrong conditions
- Null/undefined/None handling — can anything blow up unexpectedly?
- Race conditions in concurrent code
- Incorrect type usage or implicit coercions
- Edge cases: empty inputs, boundary values, Unicode, large data
- State management issues (stale closures, mutation of shared state)

#### 2. Security
- Injection vulnerabilities (SQL, XSS, command injection, path traversal)
- Authentication & authorization gaps
- Secrets or credentials in code (API keys, passwords, tokens)
- Insufficient input validation or output encoding
- Insecure cryptographic practices
- Sensitive data in logs or error messages
- Dependency vulnerabilities (known CVEs when identifiable)

#### 3. Performance
- Unnecessary computation inside loops or hot paths
- N+1 query patterns
- Memory leaks or unbounded growth
- Missing pagination for large datasets
- Blocking I/O on the main thread
- Redundant API calls or database queries
- Missing caching where it would help

#### 4. Error Handling
- Uncaught exceptions or promise rejections
- Silent failures (empty catch blocks, swallowed errors)
- Missing validation of external inputs (API responses, user input, file data)
- Unhelpful error messages (missing context, wrong error type)
- Missing cleanup in failure paths (open handles, temporary files)

#### 5. Code Quality
- Naming clarity — do names convey intent?
- Code duplication — DRY violations
- Complexity — functions doing too many things
- Dead code, commented-out code, debug leftovers
- Magic numbers or strings without explanation
- Consistent patterns with the rest of the codebase

#### 6. Maintainability
- Readability — would a new teammate understand this in 6 months?
- Missing or misleading comments/docs
- Test coverage — are new code paths tested?
- Coupling — does this change create unwanted dependencies?
- Breaking changes to public APIs

#### 7. Best Practices (language/framework specific)
- Idiomatic usage for the language and framework
- Deprecated APIs or patterns
- Type safety (if applicable)
- Proper resource cleanup (connections, file handles, subscriptions)

### Severity Levels

Each finding gets a severity:

- **critical** — Must fix. Bugs, security vulnerabilities, data loss risks.
- **warning** — Should fix. Performance issues, poor error handling, logic concerns.
- **suggestion** — Nice to have. Style improvements, refactoring opportunities.
- **nitpick** — Optional. Minor style/naming preferences, trivial improvements.

Aim to be helpful, not pedantic. A good review has a mix of severities, but if the code
is solid, it's perfectly fine to say so with just a few suggestions or nitpicks. Don't
manufacture problems where there aren't any.

## Step 4: Output the Review

Structure the review output as follows. Adapt the depth to the size of the diff — a
one-line change doesn't need 50 lines of review.

### Review Output Format

```markdown
## Code Review Summary

**Scope**: [what was reviewed — e.g., "PR #42: Add user authentication"]
**Files changed**: [count]
**Overall assessment**: [one-line verdict — e.g., ✅ Looks good / ⚠️ Needs changes / 🚨 Major concerns]

---

### Walkthrough

[Brief narrative (2-5 sentences) explaining what the changes do at a high level.
 Think of this as explaining the PR to a teammate who hasn't seen it yet.]

---

### Findings

#### `filename` (lines X-Y) — 🚨 critical: Title

[Explanation of the issue. Include the relevant code snippet if it helps clarity.]

**Suggested fix:**
```language
// concrete code suggestion
```

---

#### `filename` (line X) — ⚠️ warning: Title

[Explanation]

---

[Repeat for each finding, grouped by file]

---

### Praise

[Optionally note 1-2 things done well — good patterns, clean abstractions, etc.]

### Summary

| Severity | Count |
|----------|-------|
| 🚨 Critical | N |
| ⚠️ Warning | N |
| 💡 Suggestion | N |
| 🔍 Nitpick | N |

[One-line conclusion: safe to merge? needs work? specific follow-up needed?]
```

### Output Guidelines

- **Lead with the walkthrough** — help the reader understand what the changes do before
  diving into problems.
- **Be specific** — reference exact file names, line numbers, variable names. Vague
  feedback like "this could be better" is useless.
- **Show, don't just tell** — when suggesting a fix, include a code snippet.
- **Praise good patterns** — if you see something well-done, say so. Reviews
  shouldn't be purely negative.
- **Group by file** — makes it easy for the developer to address findings file by file.
- **Calibrate depth to size** — a 5-line diff gets a short review. A 500-line diff gets
  a thorough one.
- **Match the user's language** — if they write in Korean, respond in Korean. If English,
  respond in English.
- **Emoji severity markers** — use 🚨 critical, ⚠️ warning, 💡 suggestion, 🔍 nitpick
  for visual scanning.

## Special Modes

### Security-Focused Review (`/review --security`)

If the user asks for a security review specifically, prioritize security checklist items
and also check:
- Authentication & authorization logic flow
- Input sanitization and output encoding at boundaries
- Cryptographic practices (hardcoded keys, weak algorithms, insecure random)
- Dependency vulnerabilities (check `package-lock.json`, `Cargo.lock`, etc.)
- Sensitive data exposure in logs, errors, or API responses
- CORS, CSP, and other security headers (for web apps)

### Quick Review (`/review --quick`)

If the user asks for a "quick review" or the diff is small (< 50 lines), keep the output
concise: walkthrough + only critical/warning findings + one-line summary. Skip nitpicks.

### Pre-merge Checklist (`/review --merge`)

If the user says "is this ready to merge?" or similar, add a pre-merge checklist at the end:

```markdown
### Pre-merge Checklist
- [ ] All critical findings addressed
- [ ] Tests exist and pass for new code paths
- [ ] No TODOs, FIXMEs, or debug code left in
- [ ] Breaking changes documented (if any)
- [ ] Types / interfaces consistent across the change
- [ ] Error handling covers failure modes
- [ ] No secrets or credentials in the diff
```

## Working with GitHub PRs

When reviewing a GitHub PR, gather extra context:

```bash
# Get PR metadata
gh pr view <number> --json title,body,labels,baseRefName,headRefName,reviews,comments

# Get the diff
gh pr diff <number>

# Check CI status
gh pr checks <number>
```

Include the PR description in your analysis — it tells you the *intent* behind the changes,
which helps assess whether the implementation matches the goal.

If CI checks are failing, mention that prominently at the top of the review.

If there are existing review comments, read them to avoid duplicating feedback that's
already been given.

## Working with Large Diffs

For PRs with > 500 lines of diff:

1. Start with `git diff --stat` to see which files changed and by how much
2. Prioritize reviewing files that are:
   - Core logic (not generated code, not test fixtures)
   - Security-sensitive (auth, payments, permissions)
   - Public API surfaces
3. Group your review by logical area rather than strictly by file
4. Note any files you skipped and why (e.g., "Skipped auto-generated migration files")

## Configuration (Optional)

If users want to customize the review behavior, they can create a `.codereview.yml`
or `.codereview.json` in their project root. When this file exists, read it and adapt
the review accordingly.

```yaml
# .codereview.yml example
language: ko           # review output language (default: match user)
severity_threshold: warning  # only show findings at this level or above
focus:
  - security
  - performance
ignore:
  - "**/*.test.ts"     # skip test files
  - "migrations/"      # skip migration files
custom_rules:
  - "All API endpoints must have rate limiting"
  - "Database queries must use parameterized statements"
  - "React components must have PropTypes or TypeScript types"
```

When this config exists, respect its settings during the review.

## Tips for Best Results

- For large PRs (> 500 lines of diff), the file-by-file approach keeps things manageable.
- When the diff modifies tests, review the tests too — are they testing the right things?
  Are edge cases covered? Are the assertions meaningful or just snapshot tests?
- If you see a pattern repeated across files (e.g., the same bug in multiple places),
  call it out once clearly with a note about all affected locations rather than repeating
  the same finding N times.
- For refactoring PRs, focus more on maintainability and less on functionality (since
  behavior should be unchanged). Verify that the refactoring doesn't subtly change behavior.
- For dependency updates, check the changelog for breaking changes and security advisories.
