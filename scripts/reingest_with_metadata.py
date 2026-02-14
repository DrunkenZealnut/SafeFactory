#!/usr/bin/env python3
"""Re-ingest laborlaw and field-training documents with new domain-specific metadata.

Deletes existing vectors in target namespaces and re-processes documents
through the updated pipeline that includes domain metadata extraction.

Usage:
    python scripts/reingest_with_metadata.py                  # Both namespaces
    python scripts/reingest_with_metadata.py --namespace laborlaw
    python scripts/reingest_with_metadata.py --namespace field-training
    python scripts/reingest_with_metadata.py --dry-run        # Preview only
"""
import sys
import os
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

from src.agent import PineconeAgent


NAMESPACE_CONFIG = {
    'laborlaw': {
        'folder': 'documents/laborlaw',
        'description': '노동법 문서',
    },
    'field-training': {
        'folder': 'documents/현장실습',
        'description': '현장실습 안전교육 문서',
    },
}


def reingest_namespace(agent, namespace: str, dry_run: bool = False):
    """Re-ingest a single namespace with domain metadata."""
    config = NAMESPACE_CONFIG[namespace]
    folder = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        config['folder']
    )

    print(f"\n{'='*60}")
    print(f"[{namespace}] {config['description']}")
    print(f"  폴더: {folder}")
    print(f"{'='*60}")

    if not os.path.exists(folder):
        print(f"  ❌ 폴더가 존재하지 않습니다: {folder}")
        return

    # Count files
    file_count = sum(len(files) for _, _, files in os.walk(folder))
    print(f"  📁 파일 수: {file_count}")

    if dry_run:
        print(f"  🔍 [DRY-RUN] 실제 삭제/업로드를 수행하지 않습니다.")
        return

    # Step 1: Delete existing vectors
    print(f"\n  1️⃣  기존 벡터 삭제 중...")
    try:
        agent.pinecone_uploader.index.delete(delete_all=True, namespace=namespace)
        print(f"     ✅ namespace '{namespace}' 벡터 삭제 완료")
        time.sleep(2)  # Wait for deletion to propagate
    except Exception as e:
        print(f"     ⚠️  삭제 실패 (빈 네임스페이스일 수 있음): {e}")

    # Step 2: Re-process documents
    print(f"\n  2️⃣  문서 재처리 중...")
    start_time = time.time()
    try:
        result = agent.process_folder(
            folder_path=folder,
            namespace=namespace,
            recursive=True,
            batch_size=50,
            verbose=True
        )
        elapsed = time.time() - start_time

        print(f"\n  📊 결과:")
        print(f"     처리 파일: {result.total_files}")
        print(f"     업로드 벡터: {result.uploaded_vectors}")
        print(f"     실패: {result.failed_uploads}")
        print(f"     소요 시간: {elapsed:.1f}초")
    except Exception as e:
        print(f"  ❌ 처리 실패: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Re-ingest documents with domain metadata')
    parser.add_argument('--namespace', choices=['laborlaw', 'field-training'],
                        help='Re-ingest specific namespace only')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview without actual processing')
    args = parser.parse_args()

    print("🚀 도메인 메타데이터 재수집 스크립트")
    print("=" * 60)

    # Initialize agent
    print("\n  에이전트 초기화 중...")
    agent = PineconeAgent(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "semiconductor-lithography"),
        create_index_if_not_exists=False,
        track_metadata=True
    )
    print("  ✅ 에이전트 준비 완료")

    # Determine which namespaces to process
    if args.namespace:
        namespaces = [args.namespace]
    else:
        namespaces = list(NAMESPACE_CONFIG.keys())

    # Process each namespace
    for ns in namespaces:
        reingest_namespace(agent, ns, dry_run=args.dry_run)

    # Show final stats
    if not args.dry_run:
        print(f"\n{'='*60}")
        print("📊 최종 인덱스 현황:")
        try:
            stats = agent.pinecone_uploader.get_stats()
            for ns, info in sorted(stats.get('namespaces', {}).items()):
                print(f"  {ns}: {info['vector_count']} vectors")
            print(f"  총 벡터: {stats.get('total_vector_count', 'N/A')}")
        except Exception as e:
            print(f"  통계 조회 실패: {e}")

    print(f"\n✅ 완료!")


if __name__ == "__main__":
    main()
