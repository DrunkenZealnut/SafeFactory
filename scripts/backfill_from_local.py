#!/usr/bin/env python3
"""
Backfill Metadata from Local Files
Scans local document folders and populates metadata database
by checking which files exist in Pinecone.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.file_loader import FileLoader
from src.metadata_manager import MetadataManager
from src.pinecone_uploader import PineconeUploader


def backfill_folder(
    folder_path: str,
    namespace: str,
    metadata_manager: MetadataManager,
    uploader: PineconeUploader,
    verbose: bool = True
):
    """
    Backfill metadata for files in a folder.

    Args:
        folder_path: Path to folder to scan
        namespace: Pinecone namespace
        metadata_manager: MetadataManager instance
        uploader: PineconeUploader instance
        verbose: Print progress
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing folder: {folder_path}")
        print(f"Namespace: {namespace}")
        print(f"{'='*60}")

    # Load all files
    loader = FileLoader(folder_path, recursive=True)
    summary = loader.get_file_summary()

    if verbose:
        print(f"\n📁 Files found:")
        print(f"   - Images: {summary['images']}")
        print(f"   - Markdown: {summary['markdown']}")
        print(f"   - JSON: {summary['json']}")
        print(f"   - Total: {summary['total']}")

    saved_count = 0
    skipped_count = 0
    error_count = 0

    for loaded_file in loader.load_all():
        try:
            source_file = loaded_file.path
            file_path = str(Path(loaded_file.path).resolve())

            # Check if already in database
            existing = metadata_manager.get_file_metadata(namespace, source_file)
            if existing and existing['status'] == 'completed':
                skipped_count += 1
                if verbose and skipped_count % 100 == 0:
                    print(f"  Skipped {skipped_count} existing records...")
                continue

            # Calculate file metadata
            file_hash = MetadataManager.calculate_file_hash(file_path)
            file_size = os.path.getsize(file_path)
            file_type = loaded_file.file_type.value

            # Try to find vectors for this file in Pinecone
            # Use zero vector with source_file filter (filter does all matching)
            try:
                zero_vector = [0.0] * 1536
                results = uploader.query(
                    vector=zero_vector,
                    top_k=100,
                    namespace=namespace,
                    filter={"source_file": source_file},
                    include_metadata=True
                )

                vector_ids = [r['id'] for r in results]
                vector_count = len(vector_ids)
                chunk_count = vector_count

                if vector_count == 0:
                    # File not in Pinecone, skip
                    if verbose:
                        print(f"⏭️  {source_file}: Not in Pinecone (skipped)")
                    continue

            except Exception as e:
                # If query fails, assume file is not in Pinecone
                if verbose:
                    print(f"⚠️  {source_file}: Query failed ({e})")
                continue

            # Save metadata
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
                print(f"✗ Error processing {loaded_file.filename}: {e}")

    if verbose:
        print(f"\n{'='*60}")
        print(f"Results:")
        print(f"{'='*60}")
        print(f"✓ Saved: {saved_count} files")
        print(f"⏭️  Skipped (existing): {skipped_count} files")
        if error_count > 0:
            print(f"✗ Errors: {error_count} files")


def main():
    """Main function."""
    load_dotenv()

    pinecone_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "document-index")

    if not pinecone_key:
        print("❌ Error: PINECONE_API_KEY not found in .env file")
        sys.exit(1)

    print("="*60)
    print("Backfill Metadata from Local Files")
    print("="*60)

    # Initialize components
    print("\n🔧 Initializing components...")

    try:
        metadata_manager = MetadataManager()
        print("✓ Connected to metadata database")
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        sys.exit(1)

    try:
        uploader = PineconeUploader(
            api_key=pinecone_key,
            index_name=index_name,
            create_if_not_exists=False
        )
        print(f"✓ Connected to Pinecone: {index_name}")
    except Exception as e:
        print(f"❌ Failed to connect to Pinecone: {e}")
        metadata_manager.close()
        sys.exit(1)

    # Define folder-namespace mappings
    folders = [
        ("documents/laborlaw", "laborlaw"),
        ("documents/현장실습", "field-training"),
    ]

    # Check for semiconductor folder
    if os.path.exists("documents/semiconductor"):
        folders.append(("documents/semiconductor", "semiconductor"))
    elif os.path.exists("documents/ncs"):
        folders.append(("documents/ncs", "semiconductor"))

    # Process each folder
    for folder_path, namespace in folders:
        if not os.path.exists(folder_path):
            print(f"\n⚠️ Folder not found: {folder_path} (skipped)")
            continue

        backfill_folder(
            folder_path=folder_path,
            namespace=namespace,
            metadata_manager=metadata_manager,
            uploader=uploader,
            verbose=True
        )

    # Show final statistics
    print(f"\n{'='*60}")
    print("Final Database Statistics")
    print(f"{'='*60}")

    overall_stats = metadata_manager.get_stats()
    print(f"\nOverall:")
    print(f"  Total files: {overall_stats.get('total_files', 0)}")
    print(f"  Total chunks: {overall_stats.get('total_chunks', 0)}")
    print(f"  Total vectors: {overall_stats.get('total_vectors', 0)}")
    print(f"  Completed: {overall_stats.get('completed', 0)}")
    print(f"  Failed: {overall_stats.get('failed', 0)}")

    print(f"\nBy namespace:")
    for folder_path, namespace in folders:
        if os.path.exists(folder_path):
            stats = metadata_manager.get_stats(namespace)
            print(f"  {namespace}: {stats.get('total_files', 0)} files, {stats.get('total_vectors', 0)} vectors")

    metadata_manager.close()
    print("\n✓ Backfill complete!")


if __name__ == "__main__":
    main()
