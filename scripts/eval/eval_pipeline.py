"""RAG Pipeline Evaluation Script.

Runs the golden dataset through the RAG pipeline and computes
retrieval quality metrics: Recall@K, MRR, NDCG@K, keyword hit rate,
and per-phase latency statistics.

Usage:
    python -m scripts.eval.eval_pipeline [--top-k 5] [--output results.json]
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_golden_dataset(path: str = None) -> Dict[str, List[Dict]]:
    """Load golden dataset from JSON file."""
    if path is None:
        path = Path(__file__).parent / 'golden_dataset.json'
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['domains']


def keyword_hit_rate(keywords: List[str], content: str) -> float:
    """Fraction of expected keywords found in retrieved content."""
    if not keywords:
        return 0.0
    content_lower = content.lower()
    hits = sum(1 for kw in keywords if kw.lower() in content_lower)
    return hits / len(keywords)


def answer_contains_rate(expected: List[str], answer: str) -> float:
    """Fraction of expected key facts found in the generated answer.

    Unlike keyword_hit_rate (which checks retrieved context), this checks
    the final LLM-generated answer for expected content.
    Returns -1.0 if expected list is empty (no ground truth).
    """
    if not expected:
        return -1.0
    if not answer:
        return 0.0
    answer_lower = answer.lower()
    hits = sum(1 for fact in expected if fact.lower() in answer_lower)
    return hits / len(expected)


def recall_at_k(expected_sources: List[str], retrieved_sources: List[str], k: int) -> float:
    """Recall@K: fraction of expected sources found in top-K results.

    If expected_sources is empty, returns keyword_hit_rate-based proxy instead.
    """
    if not expected_sources:
        return -1.0  # Sentinel: caller should use keyword proxy
    retrieved_set = set(s.lower() for s in retrieved_sources[:k])
    expected_set = set(s.lower() for s in expected_sources)
    if not expected_set:
        return 0.0
    return len(expected_set & retrieved_set) / len(expected_set)


def mrr(expected_sources: List[str], retrieved_sources: List[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant result."""
    if not expected_sources:
        return -1.0
    expected_set = set(s.lower() for s in expected_sources)
    for i, src in enumerate(retrieved_sources):
        if src.lower() in expected_set:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(expected_sources: List[str], retrieved_sources: List[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at K."""
    if not expected_sources:
        return -1.0
    expected_set = set(s.lower() for s in expected_sources)

    # Binary relevance: 1 if in expected set, 0 otherwise
    dcg = 0.0
    for i, src in enumerate(retrieved_sources[:k]):
        rel = 1.0 if src.lower() in expected_set else 0.0
        dcg += rel / math.log2(i + 2)  # log2(rank+1), rank is 1-indexed

    # Ideal DCG: all relevant docs at the top
    ideal_count = min(len(expected_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))

    return dcg / idcg if idcg > 0 else 0.0


def _generate_answer(pipeline_result: Dict) -> str:
    """Generate LLM answer from pipeline context for answer quality evaluation.

    Uses the same LLM call path as the production /ask endpoint.
    """
    from services.rag_pipeline import build_llm_messages

    messages = pipeline_result.get('messages')
    if not messages:
        messages = build_llm_messages(
            query=pipeline_result.get('query', ''),
            sources=pipeline_result.get('sources', []),
            context=pipeline_result.get('context', ''),
            namespace=pipeline_result.get('namespace', ''),
            safety_references=pipeline_result.get('safety_references'),
            msds_references=pipeline_result.get('msds_references'),
        )

    # Use OpenAI for eval answer generation (Gemini key may be unavailable)
    from services.singletons import get_openai_client
    client = get_openai_client()
    resp = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=messages,
        temperature=0.3,
        max_tokens=1500,
    )
    return resp.choices[0].message.content or ''


def run_evaluation(top_k: int = 5, dataset_path: str = None,
                   eval_answer: bool = False) -> Dict[str, Any]:
    """Run full evaluation across all domains.

    Args:
        top_k: Number of results to retrieve.
        dataset_path: Path to golden dataset JSON.
        eval_answer: If True, also generate LLM answers and check answer_contains.

    Returns a dict with per-query results and aggregate metrics.
    """
    from services.rag_pipeline import run_rag_pipeline

    domains = load_golden_dataset(dataset_path)
    all_results = []
    domain_metrics = {}
    latency_totals = {}

    for domain_name, queries in domains.items():
        domain_results = []
        logger.info("=== Evaluating domain: %s (%d queries) ===", domain_name, len(queries))

        for q in queries:
            query_id = q['id']
            query_text = q['query']
            namespace = q['namespace']
            expected_kw = q.get('expected_keywords', [])
            expected_src = q.get('expected_sources', [])

            logger.info("  [%s] %s", query_id, query_text[:60])

            t0 = time.perf_counter()
            try:
                pipeline = run_rag_pipeline({
                    'query': query_text,
                    'namespace': namespace,
                    'top_k': top_k,
                    'use_enhancement': True,
                    'debug': True,
                })
            except Exception as e:
                logger.error("  [%s] Pipeline failed: %s", query_id, e)
                domain_results.append({
                    'id': query_id, 'error': str(e),
                    'keyword_hit': 0.0, 'recall': 0.0, 'mrr': 0.0, 'ndcg': 0.0,
                })
                continue
            wall_ms = round((time.perf_counter() - t0) * 1000)

            if pipeline.get('early_response'):
                logger.warning("  [%s] Early response (no results)", query_id)
                domain_results.append({
                    'id': query_id, 'early_response': True,
                    'keyword_hit': 0.0, 'recall': 0.0, 'mrr': 0.0, 'ndcg': 0.0,
                    'wall_ms': wall_ms,
                })
                continue

            sources = pipeline.get('sources', [])
            retrieved_files = [s.get('source_file', '') for s in sources]
            all_content = ' '.join(s.get('content_text', '') for s in sources)

            # Compute retrieval metrics
            kw_hit = keyword_hit_rate(expected_kw, all_content)
            r_at_k = recall_at_k(expected_src, retrieved_files, top_k)
            m = mrr(expected_src, retrieved_files)
            n_at_k = ndcg_at_k(expected_src, retrieved_files, top_k)

            # If no expected_sources, use keyword hit as proxy for recall
            if r_at_k < 0:
                r_at_k = kw_hit
            if m < 0:
                m = kw_hit
            if n_at_k < 0:
                n_at_k = kw_hit

            # Answer quality metric (v2): check if answer contains expected facts
            expected_answer = q.get('expected_answer_contains', [])
            answer_hit = -1.0
            if expected_answer and eval_answer:
                try:
                    answer_text = _generate_answer(pipeline)
                    answer_hit = answer_contains_rate(expected_answer, answer_text)
                    logger.info("    answer_hit=%.2f (%d/%d facts)",
                                answer_hit, int(answer_hit * len(expected_answer)), len(expected_answer))
                except Exception as e:
                    logger.warning("    answer generation failed: %s", e)

            latencies = pipeline.get('latencies', {})
            query_type = pipeline.get('query_type', 'unknown')

            result = {
                'id': query_id,
                'query': query_text,
                'namespace': namespace,
                'query_type': query_type,
                'source_count': len(sources),
                'keyword_hit': round(kw_hit, 4),
                'recall_at_k': round(r_at_k, 4),
                'mrr': round(m, 4),
                'ndcg_at_k': round(n_at_k, 4),
                'answer_hit': round(answer_hit, 4) if answer_hit >= 0 else None,
                'wall_ms': wall_ms,
                'latencies': latencies,
                'retrieved_files': retrieved_files[:5],
            }
            domain_results.append(result)

            # Accumulate latencies
            for phase, ms in latencies.items():
                if phase not in latency_totals:
                    latency_totals[phase] = []
                latency_totals[phase].append(ms)

            logger.info("    kw_hit=%.2f recall@%d=%.2f mrr=%.2f ndcg@%d=%.2f wall=%dms",
                        kw_hit, top_k, r_at_k, m, top_k, n_at_k, wall_ms)

        # Domain aggregate
        if domain_results:
            valid = [r for r in domain_results if 'error' not in r and not r.get('early_response')]
            if valid:
                dm = {
                    'count': len(valid),
                    'avg_keyword_hit': round(sum(r['keyword_hit'] for r in valid) / len(valid), 4),
                    'avg_recall_at_k': round(sum(r['recall_at_k'] for r in valid) / len(valid), 4),
                    'avg_mrr': round(sum(r['mrr'] for r in valid) / len(valid), 4),
                    'avg_ndcg_at_k': round(sum(r['ndcg_at_k'] for r in valid) / len(valid), 4),
                    'avg_wall_ms': round(sum(r['wall_ms'] for r in valid) / len(valid)),
                }
                # Answer quality (only if eval_answer was enabled)
                with_answer = [r for r in valid if r.get('answer_hit') is not None]
                if with_answer:
                    dm['avg_answer_hit'] = round(sum(r['answer_hit'] for r in with_answer) / len(with_answer), 4)
                domain_metrics[domain_name] = dm

        all_results.extend(domain_results)

    # Overall aggregate
    valid_all = [r for r in all_results if 'error' not in r and not r.get('early_response')]
    overall = {}
    if valid_all:
        overall = {
            'total_queries': len(valid_all),
            'avg_keyword_hit': round(sum(r['keyword_hit'] for r in valid_all) / len(valid_all), 4),
            'avg_recall_at_k': round(sum(r['recall_at_k'] for r in valid_all) / len(valid_all), 4),
            'avg_mrr': round(sum(r['mrr'] for r in valid_all) / len(valid_all), 4),
            'avg_ndcg_at_k': round(sum(r['ndcg_at_k'] for r in valid_all) / len(valid_all), 4),
            'avg_wall_ms': round(sum(r['wall_ms'] for r in valid_all) / len(valid_all)),
            'retrieval_failure_rate': round(1.0 - sum(r['recall_at_k'] for r in valid_all) / len(valid_all), 4),
        }
        with_answer_all = [r for r in valid_all if r.get('answer_hit') is not None]
        if with_answer_all:
            overall['avg_answer_hit'] = round(
                sum(r['answer_hit'] for r in with_answer_all) / len(with_answer_all), 4)

    # Latency breakdown
    latency_stats = {}
    for phase, values in latency_totals.items():
        latency_stats[phase] = {
            'avg_ms': round(sum(values) / len(values)),
            'max_ms': round(max(values)),
            'min_ms': round(min(values)),
            'p50_ms': round(sorted(values)[len(values) // 2]),
        }

    return {
        'top_k': top_k,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'overall': overall,
        'domain_metrics': domain_metrics,
        'latency_stats': latency_stats,
        'results': all_results,
    }


def print_report(report: Dict[str, Any]):
    """Print a human-readable evaluation report."""
    print("\n" + "=" * 70)
    print("RAG Pipeline Evaluation Report")
    print("=" * 70)
    print(f"Date: {report['timestamp']}  |  top_k: {report['top_k']}")

    overall = report.get('overall', {})
    if overall:
        print(f"\nOverall ({overall['total_queries']} queries):")
        print(f"  Keyword Hit Rate : {overall['avg_keyword_hit']:.2%}")
        print(f"  Recall@K         : {overall['avg_recall_at_k']:.2%}")
        print(f"  MRR              : {overall['avg_mrr']:.4f}")
        print(f"  NDCG@K           : {overall['avg_ndcg_at_k']:.4f}")
        print(f"  Avg Wall Time    : {overall['avg_wall_ms']}ms")
        if 'retrieval_failure_rate' in overall:
            print(f"  Failure Rate     : {overall['retrieval_failure_rate']:.2%}  (Anthropic baseline: 5.7%)")
        if 'avg_answer_hit' in overall:
            print(f"  Answer Hit Rate  : {overall['avg_answer_hit']:.2%}  (expected facts found in answer)")

    print("\nPer-Domain:")
    for domain, metrics in report.get('domain_metrics', {}).items():
        print(f"  {domain} ({metrics['count']} queries):")
        line = (f"    KW Hit={metrics['avg_keyword_hit']:.2%}  "
                f"Recall={metrics['avg_recall_at_k']:.2%}  "
                f"MRR={metrics['avg_mrr']:.4f}  "
                f"NDCG={metrics['avg_ndcg_at_k']:.4f}  "
                f"Avg={metrics['avg_wall_ms']}ms")
        if 'avg_answer_hit' in metrics:
            line += f"  AnswerHit={metrics['avg_answer_hit']:.2%}"
        print(line)

    latency = report.get('latency_stats', {})
    if latency:
        print("\nLatency Breakdown (ms):")
        for phase in sorted(latency.keys()):
            s = latency[phase]
            print(f"  {phase:30s}  avg={s['avg_ms']:5d}  p50={s['p50_ms']:5d}  max={s['max_ms']:5d}")

    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RAG Pipeline Evaluation')
    parser.add_argument('--top-k', type=int, default=20, help='Top K for retrieval (Anthropic recommends 20)')
    parser.add_argument('--dataset', type=str, default=None, help='Path to golden dataset JSON')
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')
    parser.add_argument('--eval-answer', action='store_true',
                        help='Also generate LLM answers and check expected_answer_contains (slower, costs API tokens)')
    parser.add_argument('--embedding', type=str, default=None,
                        help='Override embedding model (e.g. text-embedding-3-small). '
                             'Useful when default Gemini key is unavailable.')
    args = parser.parse_args()

    # Override embedding model if specified (patches settings cache)
    if args.embedding:
        from services.settings import _cache, _cache_lock
        with _cache_lock:
            _cache['embedding_model'] = args.embedding
        logger.info("Embedding model override: %s", args.embedding)

    report = run_evaluation(top_k=args.top_k, dataset_path=args.dataset, eval_answer=args.eval_answer)
    print_report(report)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {args.output}")
