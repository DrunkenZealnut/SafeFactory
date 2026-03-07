"""Compare two RAG eval results JSON files and report improvements.

Usage:
    python -m scripts.eval.compare_results results_before.json results_after.json
"""

import json
import sys


def compare(before_path: str, after_path: str):
    with open(before_path, 'r', encoding='utf-8') as f:
        before = json.load(f)
    with open(after_path, 'r', encoding='utf-8') as f:
        after = json.load(f)

    b = before.get('overall', {})
    a = after.get('overall', {})

    print(f"\n{'='*65}")
    print("RAG Pipeline A/B Comparison Report")
    print(f"{'='*65}")
    print(f"Before: top_k={before.get('top_k')}  ({before.get('timestamp', 'N/A')})")
    print(f"After:  top_k={after.get('top_k')}  ({after.get('timestamp', 'N/A')})")

    metrics = [
        ('avg_keyword_hit', 'Keyword Hit Rate'),
        ('avg_recall_at_k', 'Recall@K'),
        ('avg_mrr', 'MRR'),
        ('avg_ndcg_at_k', 'NDCG@K'),
        ('avg_wall_ms', 'Avg Wall Time (ms)'),
    ]

    print(f"\n{'Metric':<25} {'Before':>10} {'After':>10} {'Change':>12}")
    print("-" * 60)
    for key, label in metrics:
        bv = b.get(key, 0)
        av = a.get(key, 0)
        change = av - bv
        if key == 'avg_wall_ms':
            print(f"{label:<25} {bv:>10} {av:>10} {change:>+10}ms")
        else:
            pct = (change / bv * 100) if bv > 0 else 0
            print(f"{label:<25} {bv:>10.4f} {av:>10.4f} {change:>+10.4f} ({pct:>+.1f}%)")

    # Failure rate comparison (Anthropic primary metric)
    b_fail = b.get('retrieval_failure_rate', 1.0 - b.get('avg_recall_at_k', 0))
    a_fail = a.get('retrieval_failure_rate', 1.0 - a.get('avg_recall_at_k', 0))
    reduction = ((b_fail - a_fail) / b_fail * 100) if b_fail > 0 else 0

    print(f"\n{'='*65}")
    print(f"Retrieval Failure Rate: {b_fail:.2%} -> {a_fail:.2%} ({reduction:+.1f}% reduction)")
    print(f"Anthropic target: -67% reduction (5.7% -> 1.9%)")
    print(f"{'='*65}")

    # Per-domain comparison
    b_domains = before.get('domain_metrics', {})
    a_domains = after.get('domain_metrics', {})
    all_domains = sorted(set(list(b_domains.keys()) + list(a_domains.keys())))

    if all_domains:
        print(f"\nPer-Domain Recall@K:")
        print(f"{'Domain':<20} {'Before':>10} {'After':>10} {'Change':>12}")
        print("-" * 55)
        for domain in all_domains:
            bv = b_domains.get(domain, {}).get('avg_recall_at_k', 0)
            av = a_domains.get(domain, {}).get('avg_recall_at_k', 0)
            change = av - bv
            pct = (change / bv * 100) if bv > 0 else 0
            print(f"{domain:<20} {bv:>10.4f} {av:>10.4f} {change:>+10.4f} ({pct:>+.1f}%)")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python -m scripts.eval.compare_results <before.json> <after.json>")
        sys.exit(1)
    compare(sys.argv[1], sys.argv[2])
