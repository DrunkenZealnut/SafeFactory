"""Filesystem tree scanner with TTL caching for admin document management."""

import logging
import os
import time
import unicodedata
from pathlib import Path
from threading import Lock

from services.domain_config import DOCUMENTS_PATH

logger = logging.getLogger(__name__)

_cache = {}
_cache_lock = Lock()
_CACHE_TTL = 60  # seconds

HIDDEN_PREFIXES = ('.', '~$')


def _normalize(text):
    """Normalize unicode to NFC for cross-platform compatibility."""
    return unicodedata.normalize('NFC', text)


def _is_hidden(name):
    """Check if a file/folder should be hidden from the tree."""
    return any(name.startswith(p) for p in HIDDEN_PREFIXES)


def _validate_path(relative_path):
    """Validate and resolve a relative path safely.

    Returns the resolved absolute Path.
    Raises ValueError on path traversal or missing path.
    """
    relative_path = _normalize(relative_path)
    doc_root = DOCUMENTS_PATH.resolve()
    target = (DOCUMENTS_PATH / relative_path).resolve()

    if not target.is_relative_to(doc_root):
        raise ValueError('잘못된 경로입니다.')

    if target.exists():
        return target

    # Try NFD variant for macOS filesystem
    nfd_path = unicodedata.normalize('NFD', relative_path)
    target_nfd = (DOCUMENTS_PATH / nfd_path).resolve()
    if target_nfd.exists() and target_nfd.is_relative_to(doc_root):
        return target_nfd

    raise ValueError('경로를 찾을 수 없습니다.')


def _count_children(dir_path, file_types=None):
    """Count immediate child directories and filtered files."""
    dir_count = 0
    file_count = 0
    try:
        with os.scandir(dir_path) as entries:
            for entry in entries:
                if _is_hidden(entry.name):
                    continue
                if entry.is_dir(follow_symlinks=False):
                    dir_count += 1
                elif entry.is_file(follow_symlinks=False):
                    if file_types:
                        ext = entry.name.rsplit('.', 1)[-1].lower() if '.' in entry.name else ''
                        if ext not in file_types:
                            continue
                    file_count += 1
    except PermissionError:
        logger.warning("Permission denied: %s", dir_path)
    return dir_count, file_count


def scan_directory(relative_path='', file_types=None):
    """Scan a directory and return its immediate children.

    Args:
        relative_path: Path relative to DOCUMENTS_PATH.
        file_types: Optional set of extensions to include (e.g. {'md', 'pdf'}).

    Returns:
        dict with 'directories' and 'files' lists.
    """
    cache_key = f"{relative_path}|{','.join(sorted(file_types or []))}"

    with _cache_lock:
        cached = _cache.get(cache_key)
        if cached:
            entry_time, entry_data = cached
            if time.time() - entry_time < _CACHE_TTL:
                return entry_data

    abs_path = _validate_path(relative_path)

    directories = []
    files = []

    try:
        with os.scandir(abs_path) as entries:
            for entry in entries:
                name = _normalize(entry.name)
                if _is_hidden(name):
                    continue

                entry_rel = f"{relative_path}/{name}".lstrip('/')

                if entry.is_dir(follow_symlinks=False):
                    dir_count, file_count = _count_children(entry.path, file_types)
                    directories.append({
                        'path': entry_rel,
                        'name': name,
                        'type': 'directory',
                        'file_count': file_count,
                        'dir_count': dir_count,
                        'has_children': (dir_count + file_count) > 0,
                    })
                elif entry.is_file(follow_symlinks=False):
                    ext = name.rsplit('.', 1)[-1].lower() if '.' in name else ''
                    if file_types and ext not in file_types:
                        continue
                    try:
                        stat = entry.stat()
                        size = stat.st_size
                        modified = stat.st_mtime
                    except OSError:
                        size = 0
                        modified = 0
                    files.append({
                        'path': entry_rel,
                        'name': name,
                        'type': 'file',
                        'extension': ext,
                        'size': size,
                        'modified': modified,
                    })
    except PermissionError:
        logger.warning("Permission denied: %s", abs_path)

    directories.sort(key=lambda d: d['name'])
    files.sort(key=lambda f: f['name'])

    result = {'directories': directories, 'files': files}

    with _cache_lock:
        # Double-check: another thread may have populated while we scanned
        cached = _cache.get(cache_key)
        if not cached or time.time() - cached[0] >= _CACHE_TTL:
            _cache[cache_key] = (time.time(), result)
        else:
            result = cached[1]

    return result


def invalidate_cache():
    """Clear the entire filetree cache."""
    with _cache_lock:
        _cache.clear()


# Lowercase only – callers must compare via ext.lower()
_IMAGE_EXTENSIONS = {'.jpeg', '.jpg', '.png', '.gif', '.webp', '.bmp'}


def scan_document_folders():
    """Find all learning-material folders (those containing a *_meta.json).

    Yields dicts with folder metadata, one per learning material:
        path        – relative to DOCUMENTS_PATH (e.g. 'laborlaw/laws/근로기준법_...')
        name        – folder name
        has_md      – True if a .md file exists
        has_pdf     – True if a .pdf file exists
        image_count – number of image files
        total_size  – sum of all file sizes in the folder (bytes)
    """
    doc_root = DOCUMENTS_PATH.resolve()

    for dirpath, dirnames, filenames in os.walk(doc_root):
        # Skip hidden directories
        dirnames[:] = [d for d in dirnames if not _is_hidden(d)]

        # Check for _meta.json marker
        has_meta = any(
            _normalize(f).endswith('_meta.json') for f in filenames
        )
        if not has_meta:
            continue

        rel_path = _normalize(os.path.relpath(dirpath, doc_root))
        folder_name = _normalize(os.path.basename(dirpath))

        has_md = False
        has_pdf = False
        image_count = 0
        total_size = 0

        for fname in filenames:
            if _is_hidden(fname):
                continue
            norm_name = _normalize(fname)
            ext = os.path.splitext(norm_name)[1].lower()

            try:
                total_size += os.path.getsize(os.path.join(dirpath, fname))
            except OSError:
                pass

            if ext in ('.md', '.markdown'):
                has_md = True
            elif ext == '.pdf':
                has_pdf = True
            elif ext in _IMAGE_EXTENSIONS:
                image_count += 1

        yield {
            'path': rel_path,
            'name': folder_name,
            'has_md': has_md,
            'has_pdf': has_pdf,
            'image_count': image_count,
            'total_size': total_size,
        }
