"""
File Loader Module
Handles loading of image, markdown, and JSON files from a specified folder.
"""

import os
import json
import base64
from pathlib import Path
from typing import Dict, List, Optional, Generator
from dataclasses import dataclass
from enum import Enum


class FileType(Enum):
    IMAGE = "image"
    MARKDOWN = "markdown"
    JSON = "json"


@dataclass
class LoadedFile:
    """Represents a loaded file with its content and metadata."""
    path: str
    filename: str
    file_type: FileType
    content: str  # For text files or base64 for images
    raw_content: Optional[bytes] = None  # Original bytes for images
    metadata: Optional[Dict] = None


class FileLoader:
    """Loads and processes files from a specified folder."""

    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
    MARKDOWN_EXTENSIONS = {'.md', '.markdown'}
    JSON_EXTENSIONS = {'.json'}

    def __init__(self, folder_path: str, recursive: bool = True):
        """
        Initialize the FileLoader.

        Args:
            folder_path: Path to the folder to scan
            recursive: Whether to scan subdirectories
        """
        self.folder_path = Path(folder_path)
        self.recursive = recursive

        if not self.folder_path.exists():
            raise ValueError(f"Folder does not exist: {folder_path}")
        if not self.folder_path.is_dir():
            raise ValueError(f"Path is not a directory: {folder_path}")

    def _get_file_type(self, file_path: Path) -> Optional[FileType]:
        """Determine the file type based on extension."""
        ext = file_path.suffix.lower()

        if ext in self.IMAGE_EXTENSIONS:
            return FileType.IMAGE
        elif ext in self.MARKDOWN_EXTENSIONS:
            return FileType.MARKDOWN
        elif ext in self.JSON_EXTENSIONS:
            return FileType.JSON
        return None

    def _load_image(self, file_path: Path) -> LoadedFile:
        """Load an image file and encode it as base64."""
        with open(file_path, 'rb') as f:
            raw_content = f.read()

        base64_content = base64.b64encode(raw_content).decode('utf-8')

        return LoadedFile(
            path=str(file_path),
            filename=file_path.name,
            file_type=FileType.IMAGE,
            content=base64_content,
            raw_content=raw_content,
            metadata={
                'extension': file_path.suffix.lower(),
                'size_bytes': len(raw_content),
                'relative_path': str(file_path.relative_to(self.folder_path))
            }
        )

    def _load_markdown(self, file_path: Path) -> LoadedFile:
        """Load a markdown file. Auto-detects marker _meta.json in the same directory."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Auto-detect marker _meta.json in the same directory
        marker_meta = None
        meta_candidates = list(file_path.parent.glob('*_meta.json'))
        if meta_candidates:
            try:
                with open(meta_candidates[0], 'r', encoding='utf-8') as mf:
                    marker_meta = json.load(mf)
            except (json.JSONDecodeError, OSError):
                pass

        return LoadedFile(
            path=str(file_path),
            filename=file_path.name,
            file_type=FileType.MARKDOWN,
            content=content,
            metadata={
                'extension': file_path.suffix.lower(),
                'size_bytes': len(content.encode('utf-8')),
                'relative_path': str(file_path.relative_to(self.folder_path)),
                'line_count': content.count('\n') + 1,
                'marker_meta': marker_meta
            }
        )

    def _load_json(self, file_path: Path) -> LoadedFile:
        """Load a JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Validate JSON
        try:
            parsed = json.loads(content)
            is_valid = True
        except json.JSONDecodeError:
            parsed = None
            is_valid = False

        return LoadedFile(
            path=str(file_path),
            filename=file_path.name,
            file_type=FileType.JSON,
            content=content,
            metadata={
                'extension': file_path.suffix.lower(),
                'size_bytes': len(content.encode('utf-8')),
                'relative_path': str(file_path.relative_to(self.folder_path)),
                'is_valid_json': is_valid,
                'json_type': type(parsed).__name__ if parsed is not None else None
            }
        )

    def scan_files(self) -> Generator[Path, None, None]:
        """Scan the folder for supported files."""
        pattern = '**/*' if self.recursive else '*'

        for file_path in self.folder_path.glob(pattern):
            if file_path.is_file() and self._get_file_type(file_path):
                # Skip _meta.json files (PDF structure metadata, not user content)
                if file_path.name.endswith('_meta.json'):
                    continue
                yield file_path

    def load_file(self, file_path: Path) -> Optional[LoadedFile]:
        """Load a single file based on its type."""
        file_type = self._get_file_type(file_path)

        if file_type == FileType.IMAGE:
            return self._load_image(file_path)
        elif file_type == FileType.MARKDOWN:
            return self._load_markdown(file_path)
        elif file_type == FileType.JSON:
            return self._load_json(file_path)

        return None

    def load_all(self) -> Generator[LoadedFile, None, None]:
        """Load all supported files from the folder."""
        for file_path in self.scan_files():
            try:
                loaded = self.load_file(file_path)
                if loaded:
                    yield loaded
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    def get_file_summary(self) -> Dict[str, int]:
        """Get a summary of files in the folder."""
        summary = {
            'images': 0,
            'markdown': 0,
            'json': 0,
            'total': 0
        }

        for file_path in self.scan_files():
            file_type = self._get_file_type(file_path)
            if file_type == FileType.IMAGE:
                summary['images'] += 1
            elif file_type == FileType.MARKDOWN:
                summary['markdown'] += 1
            elif file_type == FileType.JSON:
                summary['json'] += 1
            summary['total'] += 1

        return summary


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        folder = "."

    loader = FileLoader(folder)
    summary = loader.get_file_summary()
    print(f"File Summary: {summary}")

    for loaded_file in loader.load_all():
        print(f"Loaded: {loaded_file.filename} ({loaded_file.file_type.value})")
