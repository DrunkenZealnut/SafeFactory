#!/usr/bin/env python3
"""
diff_stats.py — Quick stats about a git diff.

Usage:
    python3 scripts/diff_stats.py [<git-diff-args>...]

Examples:
    python3 scripts/diff_stats.py                    # uncommitted changes
    python3 scripts/diff_stats.py --cached           # staged changes
    python3 scripts/diff_stats.py HEAD~1..HEAD        # last commit
    python3 scripts/diff_stats.py main...feature      # branch diff

Outputs a JSON summary with file counts, line changes, and file types.
"""

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


_EMPTY_STATS = {
    "files_changed": 0,
    "total_lines_added": 0,
    "total_lines_removed": 0,
    "net_change": 0,
    "file_types": {},
    "files": [],
    "size_category": "small",
}


def _error_result(msg: str) -> dict:
    return {**_EMPTY_STATS, "error": msg}


def get_diff_stats(diff_args: list[str]) -> dict:
    """Parse git diff --numstat output into structured stats."""
    cmd = ["git", "diff", "--numstat"] + diff_args
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return _error_result("git diff command timed out")
    except FileNotFoundError:
        return _error_result("git is not installed or not in PATH")

    if result.returncode != 0:
        return _error_result(result.stderr.strip())

    files = []
    total_added = 0
    total_removed = 0
    ext_counter = Counter()

    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            continue

        added, removed, filepath = parts

        # Binary files show "-" for added/removed
        added = int(added) if added != "-" else 0
        removed = int(removed) if removed != "-" else 0

        ext = Path(filepath).suffix or "(no extension)"
        ext_counter[ext] += 1

        files.append({
            "path": filepath,
            "added": added,
            "removed": removed,
            "extension": ext,
        })
        total_added += added
        total_removed += removed

    return {
        "files_changed": len(files),
        "total_lines_added": total_added,
        "total_lines_removed": total_removed,
        "net_change": total_added - total_removed,
        "file_types": dict(ext_counter.most_common()),
        "files": sorted(files, key=lambda f: f["added"] + f["removed"], reverse=True),
        "size_category": (
            "small" if total_added + total_removed < 50
            else "medium" if total_added + total_removed < 300
            else "large" if total_added + total_removed < 1000
            else "very_large"
        ),
    }


def main():
    """Run diff stats from command line."""
    diff_args = sys.argv[1:]
    stats = get_diff_stats(diff_args)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
