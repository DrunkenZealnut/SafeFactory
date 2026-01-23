#!/usr/bin/env python3
"""
Prepare data for Pinecone upload (offline mode).
This script processes files and saves them as JSON for later upload.
"""

import os
import json
import base64
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, asdict
from enum import Enum


class FileType(str, Enum):
    IMAGE = "image"
    MARKDOWN = "markdown"
    JSON = "json"


@dataclass
class ProcessedFile:
    path: str
    filename: str
    file_type: str
    content: str
    metadata: Dict


def scan_folder(folder_path: str) -> Dict[str, int]:
    """Scan folder and return file counts."""
    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
    MARKDOWN_EXTENSIONS = {'.md', '.markdown'}
    JSON_EXTENSIONS = {'.json'}

    counts = {'images': 0, 'markdown': 0, 'json': 0}

    folder = Path(folder_path)
    for file_path in folder.rglob('*'):
        if not file_path.is_file():
            continue
        ext = file_path.suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            counts['images'] += 1
        elif ext in MARKDOWN_EXTENSIONS:
            counts['markdown'] += 1
        elif ext in JSON_EXTENSIONS:
            counts['json'] += 1

    return counts


def load_files(folder_path: str) -> List[ProcessedFile]:
    """Load all supported files from folder."""
    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
    MARKDOWN_EXTENSIONS = {'.md', '.markdown'}
    JSON_EXTENSIONS = {'.json'}

    files = []
    folder = Path(folder_path)

    for file_path in folder.rglob('*'):
        if not file_path.is_file():
            continue

        # Skip pinecone_agent folder
        if 'pinecone_agent' in str(file_path):
            continue

        ext = file_path.suffix.lower()

        try:
            if ext in IMAGE_EXTENSIONS:
                with open(file_path, 'rb') as f:
                    content = base64.b64encode(f.read()).decode('utf-8')
                files.append(ProcessedFile(
                    path=str(file_path),
                    filename=file_path.name,
                    file_type=FileType.IMAGE.value,
                    content=content,
                    metadata={
                        'extension': ext,
                        'size_bytes': file_path.stat().st_size,
                        'relative_path': str(file_path.relative_to(folder))
                    }
                ))
            elif ext in MARKDOWN_EXTENSIONS:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                files.append(ProcessedFile(
                    path=str(file_path),
                    filename=file_path.name,
                    file_type=FileType.MARKDOWN.value,
                    content=content,
                    metadata={
                        'extension': ext,
                        'size_bytes': len(content.encode('utf-8')),
                        'relative_path': str(file_path.relative_to(folder)),
                        'line_count': content.count('\n') + 1
                    }
                ))
            elif ext in JSON_EXTENSIONS:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                files.append(ProcessedFile(
                    path=str(file_path),
                    filename=file_path.name,
                    file_type=FileType.JSON.value,
                    content=content,
                    metadata={
                        'extension': ext,
                        'size_bytes': len(content.encode('utf-8')),
                        'relative_path': str(file_path.relative_to(folder))
                    }
                ))
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return files


def chunk_text(text: str, max_chars: int = 1500) -> List[str]:
    """Simple text chunking by paragraphs."""
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) + 2 <= max_chars:
            current_chunk = f"{current_chunk}\n\n{para}" if current_chunk else para
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = para

    if current_chunk:
        chunks.append(current_chunk)

    return chunks if chunks else [text[:max_chars]]


def prepare_chunks(files: List[ProcessedFile]) -> List[Dict]:
    """Prepare chunks from files for Pinecone upload."""
    all_chunks = []

    for file in files:
        if file.file_type == FileType.IMAGE.value:
            # For images, we need image description (will be added later with API)
            all_chunks.append({
                'id': f"{file.filename}_0",
                'content': f"[이미지 파일: {file.filename}]",
                'source_file': file.path,
                'file_type': file.file_type,
                'chunk_index': 0,
                'needs_description': True,
                'base64_content': file.content,
                'metadata': file.metadata
            })
        elif file.file_type == FileType.MARKDOWN.value:
            chunks = chunk_text(file.content)
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    'id': f"{file.filename}_{i}",
                    'content': chunk,
                    'source_file': file.path,
                    'file_type': file.file_type,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'needs_description': False,
                    'metadata': file.metadata
                })
        elif file.file_type == FileType.JSON.value:
            # Keep JSON as single chunk if small enough
            if len(file.content) <= 2000:
                all_chunks.append({
                    'id': f"{file.filename}_0",
                    'content': file.content,
                    'source_file': file.path,
                    'file_type': file.file_type,
                    'chunk_index': 0,
                    'total_chunks': 1,
                    'needs_description': False,
                    'metadata': file.metadata
                })
            else:
                chunks = chunk_text(file.content)
                for i, chunk in enumerate(chunks):
                    all_chunks.append({
                        'id': f"{file.filename}_{i}",
                        'content': chunk,
                        'source_file': file.path,
                        'file_type': file.file_type,
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'needs_description': False,
                        'metadata': file.metadata
                    })

    return all_chunks


def main():
    import sys

    if len(sys.argv) < 2:
        folder_path = "/sessions/hopeful-brave-pasteur/mnt/반도체_재료_02._반도체용_리소그래피_재료_제조_LM1903060402_14v1_"
    else:
        folder_path = sys.argv[1]

    print(f"📁 폴더 스캔 중: {folder_path}")
    counts = scan_folder(folder_path)
    print(f"   - 이미지: {counts['images']}개")
    print(f"   - 마크다운: {counts['markdown']}개")
    print(f"   - JSON: {counts['json']}개")

    print("\n📥 파일 로딩 중...")
    files = load_files(folder_path)
    print(f"   로드된 파일: {len(files)}개")

    print("\n✂️ 청킹 처리 중...")
    chunks = prepare_chunks(files)
    print(f"   생성된 청크: {len(chunks)}개")

    # Save to JSON
    output_file = os.path.join(os.path.dirname(__file__), "prepared_data.json")

    # Remove base64 content for smaller file size (keep only metadata)
    chunks_for_save = []
    for chunk in chunks:
        save_chunk = chunk.copy()
        if 'base64_content' in save_chunk:
            # Keep reference but truncate actual content
            save_chunk['base64_preview'] = save_chunk['base64_content'][:100] + "..."
            del save_chunk['base64_content']
        chunks_for_save.append(save_chunk)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'folder_path': folder_path,
            'file_counts': counts,
            'total_files': len(files),
            'total_chunks': len(chunks),
            'chunks': chunks_for_save
        }, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 데이터 준비 완료!")
    print(f"   출력 파일: {output_file}")

    # Also save full chunks with base64 for actual upload
    full_output = os.path.join(os.path.dirname(__file__), "chunks_full.json")
    with open(full_output, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False)
    print(f"   전체 데이터: {full_output}")

    return chunks


if __name__ == "__main__":
    main()
