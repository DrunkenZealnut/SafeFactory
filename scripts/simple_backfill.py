#!/usr/bin/env python3
"""
Simple Metadata Backfill
Manually creates metadata records for existing Pinecone vectors without re-processing files.
This is useful when files are already in Pinecone but metadata database needs to be populated.
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metadata_manager import MetadataManager
from src.file_loader import FileLoader


def estimate_metadata(namespace: str, folder_path: str, metadata_manager: MetadataManager):
    """
    Create estimated metadata records for files in a folder.

    This doesn't query Pinecone but creates database records based on local files,
    assuming they are already uploaded to Pinecone.
    """

    if not os.path.exists(folder_path):
        print(f"⚠️ Folder not found: {folder_path}")
        return 0

    print(f"\n{'='*60}")
    print(f"Creating metadata for: {folder_path}")
    print(f"Namespace: {namespace}")
    print(f"{'='*60}")

    loader = FileLoader(folder_path, recursive=True)
    summary = loader.get_file_summary()

    print(f"\n📁 Files found:")
    print(f"   Images: {summary['images']}")
    print(f"   Markdown: {summary['markdown']}")
    print(f"   JSON: {summary['json']}")
    print(f"   Total: {summary['total']}")

    saved_count = 0
    skipped_count = 0

    for loaded_file in loader.load_all():
        try:
            source_file = loaded_file.path
            file_path = str(Path(loaded_file.path).resolve())

            # Check if already exists
            existing = metadata_manager.file_exists(namespace, source_file)
            if existing and existing['status'] == 'completed':
                skipped_count += 1
                if skipped_count % 100 == 0:
                    print(f"  Skipped {skipped_count} existing records...")
                continue

            # Calculate file metadata
            file_hash = MetadataManager.calculate_file_hash(file_path)
            file_size = os.path.getsize(file_path)
            file_type = loaded_file.file_type.value

            # Estimate chunks (rough estimate)
            # Images: 1 chunk, Markdown/JSON: based on size
            if file_type == 'image':
                chunk_count = 1
                vector_count = 1
            else:
                # Rough estimate: 1 chunk per 2KB of text
                chunk_count = max(1, file_size // 2048)
                vector_count = chunk_count

            # Save metadata without vector IDs
            # We'll mark as completed since data is already in Pinecone
            success = metadata_manager.insert_metadata(
                namespace=namespace,
                source_file=source_file,
                file_type=file_type,
                file_path=file_path,
                chunk_count=chunk_count,
                vector_count=vector_count,
                vector_ids=[],  # Empty for now
                status='completed'
            )

            if success:
                saved_count += 1
                if saved_count % 100 == 0:
                    print(f"  Saved {saved_count} records...")

        except Exception as e:
            print(f"✗ Error: {loaded_file.filename}: {e}")

    print(f"\n{'='*60}")
    print(f"✓ Saved: {saved_count} records")
    print(f"⏭️  Skipped: {skipped_count} existing records")
    print(f"{'='*60}")

    return saved_count


def main():
    """Main function."""
    load_dotenv()

    print("="*60)
    print("Simple Metadata Backfill")
    print("Creates metadata records for existing Pinecone data")
    print("="*60)

    # Initialize metadata manager
    try:
        metadata_manager = MetadataManager()
        print("\n✓ Connected to metadata database")
    except Exception as e:
        print(f"\n❌ Failed to connect to database: {e}")
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
    total_saved = 0
    for folder_path, namespace in folders:
        saved = estimate_metadata(namespace, folder_path, metadata_manager)
        total_saved += saved

    # Show final statistics
    print(f"\n{'='*60}")
    print("Final Database Statistics")
    print(f"{'='*60}")

    overall_stats = metadata_manager.get_stats()
    print(f"\nOverall:")
    print(f"  Total files: {overall_stats.get('total_files', 0):,}")
    print(f"  Total vectors (estimated): {overall_stats.get('total_vectors', 0) or 0:,}")
    print(f"  Completed: {overall_stats.get('completed', 0):,}")

    print(f"\nBy namespace:")
    for folder_path, namespace in folders:
        if os.path.exists(folder_path):
            stats = metadata_manager.get_stats(namespace)
            files = stats.get('total_files', 0)
            vectors = stats.get('total_vectors', 0) or 0
            print(f"  {namespace:20s}: {files:,} files, ~{vectors:,} vectors")

    metadata_manager.close()
    print(f"\n✓ Backfill complete! Saved {total_saved:,} new records")


if __name__ == "__main__":
    main()
