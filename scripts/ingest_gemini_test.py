"""
Gemini Embedding 2 테스트 인제스트 스크립트

기존 semiconductor-v2 네임스페이스의 벡터 메타데이터(content_preview)를
Gemini Embedding 2 (1536D MRL)로 재임베딩하여
semiconductor-v2-gemini 네임스페이스에 업로드합니다.

Usage:
    python scripts/ingest_gemini_test.py [--dry-run] [--limit 100] [--batch-size 50]
"""

import os
import sys
import time
import argparse
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from pinecone import Pinecone
from src.embedding_generator import EmbeddingGenerator

SOURCE_NAMESPACE = "semiconductor-v2"
TARGET_NAMESPACE = "semiconductor-v2-gemini"
EMBEDDING_MODEL = "gemini-embedding-2-preview"
DIMENSIONS = 1536


def fetch_existing_vectors(index, namespace: str, limit: int = 0):
    """Fetch all vector IDs and metadata from a namespace."""
    all_ids = []
    for ids_batch in index.list(namespace=namespace):
        all_ids.extend(ids_batch)
        if limit and len(all_ids) >= limit:
            all_ids = all_ids[:limit]
            break

    print(f"  총 {len(all_ids)}개 벡터 ID 수집 완료")
    return all_ids


def fetch_metadata_batch(index, ids: list, namespace: str):
    """Fetch metadata for a batch of vector IDs."""
    result = index.fetch(ids=ids, namespace=namespace)
    vectors = []
    for vid, data in result.vectors.items():
        metadata = data.metadata or {}
        text = metadata.get("content_preview", "")
        if text:
            vectors.append((vid, text, metadata))
    return vectors


def main():
    parser = argparse.ArgumentParser(description="Gemini Embedding 2 테스트 인제스트")
    parser.add_argument("--dry-run", action="store_true", help="실제 업로드 없이 시뮬레이션")
    parser.add_argument("--limit", type=int, default=0, help="처리할 최대 벡터 수 (0=전체)")
    parser.add_argument("--batch-size", type=int, default=50, help="배치 크기")
    args = parser.parse_args()

    gemini_key = os.getenv("GEMINI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "document-index")

    if not gemini_key:
        print("ERROR: GEMINI_API_KEY가 설정되지 않았습니다.")
        sys.exit(1)
    if not pinecone_key:
        print("ERROR: PINECONE_API_KEY가 설정되지 않았습니다.")
        sys.exit(1)

    # Initialize
    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index(index_name)
    gemini_gen = EmbeddingGenerator(
        api_key=gemini_key,
        model=EMBEDDING_MODEL,
        dimensions=DIMENSIONS,
    )

    print(f"모델: {gemini_gen.get_model_info()}")
    print(f"소스: {SOURCE_NAMESPACE} → 타겟: {TARGET_NAMESPACE}")
    print(f"Dry run: {args.dry_run}")
    print()

    # 1. Fetch all vector IDs from source namespace
    print("1단계: 소스 네임스페이스에서 벡터 ID 수집...")
    all_ids = fetch_existing_vectors(index, SOURCE_NAMESPACE, limit=args.limit)

    if not all_ids:
        print("소스 네임스페이스에 벡터가 없습니다.")
        return

    # 2. Process in batches
    print(f"\n2단계: Gemini 임베딩 생성 및 업로드 (배치 크기: {args.batch_size})...")
    total_uploaded = 0
    total_skipped = 0
    total_errors = 0
    start_time = time.time()

    for batch_start in range(0, len(all_ids), args.batch_size):
        batch_ids = all_ids[batch_start:batch_start + args.batch_size]

        # Fetch metadata from source
        vectors = fetch_metadata_batch(index, batch_ids, SOURCE_NAMESPACE)

        if not vectors:
            total_skipped += len(batch_ids)
            continue

        # Generate Gemini embeddings
        upsert_batch = []
        for vid, text, metadata in vectors:
            try:
                result = gemini_gen.generate(text, task_type="RETRIEVAL_DOCUMENT")
                upsert_batch.append({
                    "id": vid,
                    "values": result.embedding,
                    "metadata": metadata,
                })
            except Exception as e:
                print(f"  ✗ 임베딩 실패 [{vid[:20]}...]: {str(e)[:80]}")
                total_errors += 1

        # Upsert to target namespace
        if upsert_batch and not args.dry_run:
            try:
                index.upsert(vectors=upsert_batch, namespace=TARGET_NAMESPACE)
                total_uploaded += len(upsert_batch)
            except Exception as e:
                print(f"  ✗ Upsert 실패: {str(e)[:100]}")
                total_errors += len(upsert_batch)
        elif upsert_batch:
            total_uploaded += len(upsert_batch)

        # Progress
        processed = batch_start + len(batch_ids)
        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        print(f"  진행: {processed}/{len(all_ids)} ({processed/len(all_ids)*100:.1f}%) "
              f"| 업로드: {total_uploaded} | 에러: {total_errors} "
              f"| {rate:.1f} vectors/s")

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'=' * 50}")
    print(f"완료!")
    print(f"  총 벡터: {len(all_ids)}")
    print(f"  업로드: {total_uploaded}")
    print(f"  건너뜀 (텍스트 없음): {total_skipped}")
    print(f"  에러: {total_errors}")
    print(f"  소요 시간: {elapsed:.1f}초")
    if args.dry_run:
        print(f"  ⚠️ DRY RUN 모드 — 실제 업로드 없음")


if __name__ == "__main__":
    main()
