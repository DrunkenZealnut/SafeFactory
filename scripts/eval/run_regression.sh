#!/bin/bash
# RAG Pipeline Regression Detection Script
# Compares current eval results against the most recent baseline.
#
# Usage:
#   ./scripts/eval/run_regression.sh                    # Run eval + compare
#   ./scripts/eval/run_regression.sh --baseline-only     # Just save new baseline
#
# Requires: venv activated, OPENAI_API_KEY + PINECONE_API_KEY set

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
EVAL_DIR="$SCRIPT_DIR"
DATE=$(date +%Y-%m-%d)
NEW_BASELINE="$EVAL_DIR/baseline_${DATE}.json"

cd "$PROJECT_DIR"

echo "========================================"
echo "  RAG Pipeline Regression Check"
echo "  $(date)"
echo "========================================"

# Find existing baseline
PREV_BASELINE=$(ls -t "$EVAL_DIR"/baseline_*.json 2>/dev/null | head -1)

if [ -z "$PREV_BASELINE" ]; then
    echo "No previous baseline found. Running first baseline..."
fi

# Run eval
echo ""
echo "Running evaluation..."
python -m scripts.eval.eval_pipeline \
    --top-k 20 \
    --embedding text-embedding-3-small \
    --output "$NEW_BASELINE"

echo ""

# Compare if previous baseline exists
if [ -n "$PREV_BASELINE" ] && [ "$PREV_BASELINE" != "$NEW_BASELINE" ]; then
    echo "========================================"
    echo "  Regression Comparison"
    echo "  Previous: $(basename "$PREV_BASELINE")"
    echo "  Current:  $(basename "$NEW_BASELINE")"
    echo "========================================"

    python -c "
import json, sys

with open('$PREV_BASELINE') as f:
    prev = json.load(f)
with open('$NEW_BASELINE') as f:
    curr = json.load(f)

po = prev.get('overall', {})
co = curr.get('overall', {})

metrics = ['avg_keyword_hit', 'avg_recall_at_k', 'avg_mrr', 'avg_ndcg_at_k', 'retrieval_failure_rate']
labels = ['Keyword Hit', 'Recall@K', 'MRR', 'NDCG@K', 'Failure Rate']

print()
print(f'{\"Metric\":20s}  {\"Previous\":>10s}  {\"Current\":>10s}  {\"Delta\":>10s}  Status')
print('-' * 70)

regression = False
for metric, label in zip(metrics, labels):
    p = po.get(metric, 0)
    c = co.get(metric, 0)
    delta = c - p
    # For failure rate, lower is better
    is_worse = delta < -0.02 if metric != 'retrieval_failure_rate' else delta > 0.02

    status = '🔴 REGRESS' if is_worse else ('🟢 IMPROVED' if abs(delta) > 0.01 else '⚪ STABLE')
    if is_worse:
        regression = True
    print(f'{label:20s}  {p:10.2%}  {c:10.2%}  {delta:+10.2%}  {status}')

print()

# Domain comparison
pd = prev.get('domain_metrics', {})
cd = curr.get('domain_metrics', {})
all_domains = sorted(set(pd.keys()) | set(cd.keys()))
if all_domains:
    print('Per-Domain Recall@K:')
    for d in all_domains:
        pr = pd.get(d, {}).get('avg_recall_at_k', 0)
        cr = cd.get(d, {}).get('avg_recall_at_k', 0)
        delta = cr - pr
        status = '🔴' if delta < -0.05 else ('🟢' if delta > 0.05 else '⚪')
        print(f'  {d:25s}  {pr:.2%} → {cr:.2%}  ({delta:+.2%}) {status}')

print()
if regression:
    print('⚠️  REGRESSION DETECTED — review changes before deploying')
    sys.exit(1)
else:
    print('✅  No regression detected')
"
else
    echo "First baseline saved. Run again after changes to compare."
fi

echo ""
echo "Baseline saved: $NEW_BASELINE"
