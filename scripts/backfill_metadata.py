#!/usr/bin/env python3
"""
Backfill Metadata Script
Fetches existing vectors from Pinecone and populates the metadata database.
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pinecone_uploader import PineconeUploader
from src.metadata_manager import MetadataManager
from src.file_loader import FileType


def get_file_type_from_extension(filename: str) -> str:
    """Determine file type from extension."""
    ext = Path(filename).suffix.lower()

    # Image extensions
    if ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']:
        return 'image'
    # Markdown extensions
    elif ext in ['.md', '.markdown']:
        return 'markdown'
    # JSON extension
    elif ext == '.json':
        return 'json'
    else:
        return 'markdown'  # Default


def backfill_namespace(
    uploader: PineconeUploader,
    metadata_manager: MetadataManager,
    namespace: str,
    verbose: bool = True
):
    """
    Backfill metadata for a specific namespace.

    Args:
        uploader: PineconeUploader instance
        metadata_manager: MetadataManager instance
        namespace: Namespace to backfill
        verbose: Whether to print progress
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing namespace: {namespace or '(default)'}")
        print(f"{'='*60}")

    # Get namespace stats
    stats = uploader.get_stats()
    namespace_info = stats.get('namespaces', {}).get(namespace)

    if not namespace_info:
        if verbose:
            print(f"⚠️ No vectors found in namespace: {namespace}")
        return 0

    vector_count = namespace_info.vector_count
    if verbose:
        print(f"📊 Total vectors in namespace: {vector_count}")

    # Fetch all vector IDs with metadata using list() and fetch()
    if verbose:
        print(f"🔍 Fetching vector metadata...")

    # Group vectors by source_file
    file_vectors: Dict[str, List[str]] = defaultdict(list)
    file_metadata: Dict[str, dict] = {}

    try:
        # List all vector IDs in the namespace (paginated)
        vector_ids = []
        pagination_token = None

        while True:
            if pagination_token:
                list_result = uploader.index.list(
                    namespace=namespace,
                    pagination_token=pagination_token
                )
            else:
                list_result = uploader.index.list(namespace=namespace)

            # Get vector IDs from this page
            if hasattr(list_result, 'vectors'):
                page_ids = [v.id for v in list_result.vectors]
            elif isinstance(list_result, dict) and 'vectors' in list_result:
                page_ids = [v['id'] if isinstance(v, dict) else v.id for v in list_result['vectors']]
            else:
                # Try to get ids directly
                page_ids = list_result if isinstance(list_result, list) else []

            vector_ids.extend(page_ids)

            if verbose:
                print(f"  Fetched {len(page_ids)} vector IDs (total: {len(vector_ids)})...")

            # Check if there are more pages
            pagination_token = getattr(list_result, 'pagination', None)
            if hasattr(pagination_token, 'next'):
                pagination_token = pagination_token.next
            elif isinstance(list_result, dict) and 'pagination' in list_result:
                pagination_token = list_result['pagination'].get('next')
            else:
                pagination_token = None

            if not pagination_token:
                break

        if verbose:
            print(f"✓ Total vector IDs: {len(vector_ids)}")

        # Fetch metadata in batches
        batch_size = 100
        total_fetched = 0

        for i in range(0, len(vector_ids), batch_size):
            batch_ids = vector_ids[i:i + batch_size]

            try:
                # Fetch vectors with metadata
                fetch_result = uploader.index.fetch(ids=batch_ids, namespace=namespace)

                # Process fetched vectors
                vectors = fetch_result.get('vectors', {})
                for vector_id, vector_data in vectors.items():
                    metadata = vector_data.get('metadata', {})
                    source_file = metadata.get('source_file')

                    if source_file and vector_id:
                        file_vectors[source_file].append(vector_id)
                        if source_file not in file_metadata:
                            file_metadata[source_file] = metadata
                        total_fetched += 1

                if verbose and (i + batch_size) % 500 == 0:
                    print(f"  Processed {min(i + batch_size, len(vector_ids))}/{len(vector_ids)} vectors...")

            except Exception as e:
                print(f"⚠️ Error fetching batch {i//batch_size}: {e}")
                continue

        if verbose:
            print(f"✓ Fetched metadata for {total_fetched} vectors")
            print(f"📁 Found {len(file_vectors)} unique files")

    except Exception as e:
        print(f"❌ Error fetching vectors: {e}")
        import traceback
        traceback.print_exc()
        return 0

    # Save metadata for each file
    saved_count = 0
    error_count = 0

    for source_file, vector_ids in file_vectors.items():
        try:
            metadata = file_metadata[source_file]
            file_type = metadata.get('file_type', 'markdown')

            # Try to find the file locally to get hash and size
            file_path = None
            file_hash = ""
            file_size = 0

            # Try common paths
            possible_paths = [
                source_file,
                os.path.join('documents', source_file),
                os.path.join('documents/semiconductor', source_file),
                os.path.join('documents/laborlaw', source_file),
                os.path.join('documents/현장실습', source_file),
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    file_path = os.path.abspath(path)
                    file_hash = MetadataManager.calculate_file_hash(file_path)
                    file_size = os.path.getsize(file_path)
                    break

            # If file not found locally, use source_file as path
            if not file_path:
                file_path = source_file
                if verbose:
                    print(f"⚠️ File not found locally: {source_file}")

            # Determine file type from extension if not in metadata
            if not file_type or file_type == 'unknown':
                file_type = get_file_type_from_extension(source_file)

            # Estimate chunk count (vectors for same file)
            chunk_count = len(vector_ids)
            vector_count = len(vector_ids)

            # Save to database
            success = metadata_manager.insert_metadata(
                namespace=namespace,
                source_file=source_file,
                file_type=file_type,
                file_path=file_path,
                chunk_count=chunk_count,
                vector_count=vector_count,
                vector_ids=vector_ids,
                status='completed',
                file_hash=file_hash,
                file_size=file_size
            )

            if success:
                saved_count += 1
                if verbose:
                    print(f"✓ {source_file}: {vector_count} vectors")
            else:
                error_count += 1
                if verbose:
                    print(f"✗ Failed to save: {source_file}")

        except Exception as e:
            error_count += 1
            if verbose:
                print(f"✗ Error processing {source_file}: {e}")

    if verbose:
        print(f"\n{'='*60}")
        print(f"Results for namespace: {namespace or '(default)'}")
        print(f"{'='*60}")
        print(f"✓ Saved: {saved_count} files")
        if error_count > 0:
            print(f"✗ Errors: {error_count} files")

    return saved_count


def main():
    """Main function to backfill metadata from Pinecone."""
    # Load environment variables
    load_dotenv()

    pinecone_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "document-index")

    if not pinecone_key:
        print("❌ Error: PINECONE_API_KEY not found in .env file")
        sys.exit(1)

    print("="*60)
    print("Pinecone Metadata Backfill Script")
    print("="*60)

    # Initialize components
    print("\n🔧 Initializing components...")

    try:
        uploader = PineconeUploader(
            api_key=pinecone_key,
            index_name=index_name,
            create_if_not_exists=False
        )
        print(f"✓ Connected to Pinecone index: {index_name}")
    except Exception as e:
        print(f"❌ Failed to connect to Pinecone: {e}")
        sys.exit(1)

    try:
        metadata_manager = MetadataManager()
        print(f"✓ Connected to metadata database")
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        sys.exit(1)

    # Get all namespaces
    stats = uploader.get_stats()
    namespaces = list(stats.get('namespaces', {}).keys())

    if not namespaces:
        print("\n⚠️ No namespaces found in Pinecone index")
        metadata_manager.close()
        return

    print(f"\n📊 Found {len(namespaces)} namespace(s):")
    for ns in namespaces:
        ns_info = stats['namespaces'][ns]
        ns_name = ns if ns else "(default)"
        print(f"  - {ns_name}: {ns_info['vector_count']} vectors")

    # Process each namespace
    total_saved = 0
    for namespace in namespaces:
        total_saved += backfill_namespace(uploader, metadata_manager, namespace)

    # Show final statistics
    print(f"\n{'='*60}")
    print("Final Statistics")
    print(f"{'='*60}")

    for namespace in namespaces:
        stats = metadata_manager.get_stats(namespace)
        ns_name = namespace if namespace else "(default)"
        print(f"\nNamespace: {ns_name}")
        print(f"  Total files: {stats.get('total_files', 0)}")
        print(f"  Total chunks: {stats.get('total_chunks', 0)}")
        print(f"  Total vectors: {stats.get('total_vectors', 0)}")
        print(f"  Completed: {stats.get('completed', 0)}")
        print(f"  Failed: {stats.get('failed', 0)}")

    # Show overall stats
    overall_stats = metadata_manager.get_stats()
    print(f"\n{'='*60}")
    print("Overall Database Statistics")
    print(f"{'='*60}")
    print(f"Total files: {overall_stats.get('total_files', 0)}")
    print(f"Total chunks: {overall_stats.get('total_chunks', 0)}")
    print(f"Total vectors: {overall_stats.get('total_vectors', 0)}")
    print(f"Total size: {overall_stats.get('total_size', 0)} bytes")

    # Close connections
    metadata_manager.close()
    print(f"\n✓ Backfill complete! Saved {total_saved:,} new records")


if __name__ == "__main__":
    main()
