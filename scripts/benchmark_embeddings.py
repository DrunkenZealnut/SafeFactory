"""
A/B 비교 벤치마크: OpenAI text-embedding-3-small vs Gemini Embedding 2

동일 쿼리셋으로 두 임베딩 모델의 검색 품질을 정량 비교합니다.

전제:
  - semiconductor-v2: OpenAI 임베딩으로 인제스트 완료
  - semiconductor-v2-gemini: Gemini 임베딩으로 인제스트 완료

Usage:
    python scripts/benchmark_embeddings.py [--top-k 5]
"""

import json
import os
import sys
import time
import argparse
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from pinecone import Pinecone
from src.embedding_generator import EmbeddingGenerator

DOMAIN_NAMESPACE_MAP = {
    "semiconductor-v2": ("semiconductor-v2", "semiconductor-v2-gemini"),
    "laborlaw-v2": ("laborlaw-v2", "laborlaw-v2-gemini"),
    "counsel": ("counsel", "counsel-gemini"),
    "precedent": ("precedent", "precedent-gemini"),
}


def search_with_model(index, gen: EmbeddingGenerator, query: str,
                      namespace: str, top_k: int, task_type: str = None) -> Dict:
    """Run a single search and return results with timing."""
    t0 = time.time()
    emb = gen.generate(query, task_type=task_type)
    result = index.query(
        vector=emb.embedding,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
    )
    latency = (time.time() - t0) * 1000

    matches = []
    for m in result.matches:
        meta = m.metadata or {}
        preview = (meta.get("content_preview") or meta.get("chunk_text")
                   or meta.get("text") or meta.get("content") or "")[:80]
        matches.append({
            "id": m.id,
            "score": round(m.score, 4),
            "preview": preview,
        })

    return {
        "top_ids": [m["id"] for m in matches],
        "top_scores": [m["score"] for m in matches],
        "matches": matches,
        "latency_ms": round(latency, 1),
    }


def compute_keyword_hit(matches: List[Dict], keywords: List[str]) -> float:
    """Top-5 결과에서 expected_keywords가 얼마나 등장하는지 비율."""
    if not keywords:
        return 1.0
    all_text = " ".join(m.get("preview", "") for m in matches).lower()
    hits = sum(1 for kw in keywords if kw.lower() in all_text)
    return hits / len(keywords)


def print_summary(results: List[Dict]):
    """Print aggregated comparison metrics."""
    oai_scores, gem_scores = [], []
    oai_lats, gem_lats = [], []
    overlaps = []
    oai_kw_hits, gem_kw_hits = [], []

    for r in results:
        oai = r["openai"]
        gem = r["gemini"]

        oai_scores.append(oai["top_scores"][0] if oai["top_scores"] else 0)
        gem_scores.append(gem["top_scores"][0] if gem["top_scores"] else 0)
        oai_lats.append(oai["latency_ms"])
        gem_lats.append(gem["latency_ms"])

        oai_set = set(oai["top_ids"])
        gem_set = set(gem["top_ids"])
        overlaps.append(len(oai_set & gem_set) / max(len(oai_set | gem_set), 1))

        oai_kw_hits.append(r.get("openai_keyword_hit", 0))
        gem_kw_hits.append(r.get("gemini_keyword_hit", 0))

    n = len(results)
    print("\n" + "=" * 60)
    print("  A/B 비교 결과 요약")
    print("=" * 60)
    print(f"  쿼리 수: {n}")
    print()
    print(f"  {'메트릭':<24} {'OpenAI':>12} {'Gemini':>12} {'차이':>10}")
    print(f"  {'-'*24} {'-'*12} {'-'*12} {'-'*10}")

    avg_oai_score = sum(oai_scores) / n
    avg_gem_score = sum(gem_scores) / n
    print(f"  {'Top-1 평균 유사도':<24} {avg_oai_score:>12.4f} {avg_gem_score:>12.4f} {avg_gem_score - avg_oai_score:>+10.4f}")

    avg_oai_lat = sum(oai_lats) / n
    avg_gem_lat = sum(gem_lats) / n
    print(f"  {'평균 Latency (ms)':<24} {avg_oai_lat:>12.1f} {avg_gem_lat:>12.1f} {avg_gem_lat - avg_oai_lat:>+10.1f}")

    avg_overlap = sum(overlaps) / n * 100
    print(f"  {'Top-5 Jaccard Overlap':<24} {'-':>12} {'-':>12} {avg_overlap:>9.1f}%")

    avg_oai_kw = sum(oai_kw_hits) / n * 100
    avg_gem_kw = sum(gem_kw_hits) / n * 100
    print(f"  {'키워드 적중률 (%)':<24} {avg_oai_kw:>11.1f}% {avg_gem_kw:>11.1f}% {avg_gem_kw - avg_oai_kw:>+9.1f}%")

    print()
    print(f"  비용 (1K 쿼리 기준):")
    print(f"    OpenAI: ~$0.0006  ($0.02/1M tokens × ~30 tokens/query)")
    print(f"    Gemini: ~$0.006   ($0.20/1M tokens × ~30 tokens/query)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="OpenAI vs Gemini 임베딩 A/B 벤치마크")
    parser.add_argument("--top-k", type=int, default=5, help="검색 결과 수")
    parser.add_argument("--queries", type=str, default="scripts/benchmark_queries.json")
    parser.add_argument("--output", type=str, default="scripts/benchmark_results.json")
    parser.add_argument("--domain", type=str, default="semiconductor-v2",
                        help="벤치마크할 도메인 (default: semiconductor-v2)")
    args = parser.parse_args()

    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "document-index")

    if not all([openai_key, gemini_key, pinecone_key]):
        print("ERROR: OPENAI_API_KEY, GEMINI_API_KEY, PINECONE_API_KEY가 모두 필요합니다.")
        sys.exit(1)

    # Initialize
    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index(index_name)

    openai_gen = EmbeddingGenerator(api_key=openai_key, model="text-embedding-3-small")
    gemini_gen = EmbeddingGenerator(api_key=gemini_key, model="gemini-embedding-2-preview", dimensions=1536)

    print(f"OpenAI: {openai_gen.get_model_info()}")
    print(f"Gemini: {gemini_gen.get_model_info()}")
    print()

    # Load queries
    with open(args.queries, encoding="utf-8") as f:
        all_queries = json.load(f)["queries"]

    # Resolve namespaces from domain
    if args.domain not in DOMAIN_NAMESPACE_MAP:
        print(f"ERROR: 지원하지 않는 도메인 '{args.domain}'")
        print(f"지원 도메인: {', '.join(DOMAIN_NAMESPACE_MAP.keys())}")
        sys.exit(1)

    openai_ns, gemini_ns = DOMAIN_NAMESPACE_MAP[args.domain]

    # Filter to target domain
    queries = [q for q in all_queries if q["domain"] == args.domain]
    if not queries:
        print(f"도메인 '{args.domain}'에 해당하는 쿼리가 없습니다. 전체 쿼리로 실행합니다.")
        queries = all_queries

    print(f"벤치마크 쿼리: {len(queries)}개\n")

    # Check target namespace exists
    stats = index.describe_index_stats()
    if gemini_ns not in stats.namespaces:
        print(f"ERROR: '{gemini_ns}' 네임스페이스가 없습니다.")
        print(f"먼저 인제스트를 실행하세요: python scripts/ingest_gemini_test.py")
        sys.exit(1)

    gem_count = stats.namespaces[gemini_ns].vector_count
    oai_count = stats.namespaces.get(openai_ns, type("", (), {"vector_count": 0})).vector_count
    print(f"벡터 수 — OpenAI ({openai_ns}): {oai_count} | Gemini ({gemini_ns}): {gem_count}\n")

    # Run benchmark
    results = []
    for i, q in enumerate(queries):
        print(f"[{i+1}/{len(queries)}] {q['query']}")

        oai_result = search_with_model(
            index, openai_gen, q["query"], openai_ns, args.top_k
        )
        gem_result = search_with_model(
            index, gemini_gen, q["query"], gemini_ns, args.top_k,
            task_type="RETRIEVAL_QUERY"
        )

        keywords = q.get("expected_keywords", [])
        oai_kw = compute_keyword_hit(oai_result["matches"], keywords)
        gem_kw = compute_keyword_hit(gem_result["matches"], keywords)

        entry = {
            "query_id": q["id"],
            "query": q["query"],
            "domain": q["domain"],
            "openai": oai_result,
            "gemini": gem_result,
            "openai_keyword_hit": oai_kw,
            "gemini_keyword_hit": gem_kw,
        }
        results.append(entry)

        # Per-query summary
        oai_top = oai_result["top_scores"][0] if oai_result["top_scores"] else 0
        gem_top = gem_result["top_scores"][0] if gem_result["top_scores"] else 0
        winner = "Gemini" if gem_top > oai_top else "OpenAI" if oai_top > gem_top else "Tie"
        print(f"  OpenAI: {oai_top:.4f} ({oai_result['latency_ms']:.0f}ms) | "
              f"Gemini: {gem_top:.4f} ({gem_result['latency_ms']:.0f}ms) | "
              f"Winner: {winner}")

    # Summary
    print_summary(results)

    # Save results
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n상세 결과 저장: {args.output}")


if __name__ == "__main__":
    main()
