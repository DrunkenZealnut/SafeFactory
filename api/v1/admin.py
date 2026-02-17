"""Admin dashboard API endpoints."""

import logging
import os
import re
from datetime import datetime, timedelta, timezone
from functools import wraps

from flask import request
from flask_login import current_user, login_required

from api.response import error_response, success_response
from api.v1 import v1_bp
from models import (
    AdminLog, Category, Comment, Document, Post, PostAttachment, PostLike,
    SocialAccount, User, db,
)

# ---------------------------------------------------------------------------
# Admin decorator
# ---------------------------------------------------------------------------


def admin_required(f):
    """Require authenticated user with admin role."""
    @wraps(f)
    @login_required
    def decorated(*args, **kwargs):
        if current_user.role != 'admin':
            return error_response('관리자 권한이 필요합니다.', 403)
        return f(*args, **kwargs)
    return decorated


def _log_action(action, target_type=None, target_id=None, details=None):
    """Record an admin action to the audit log."""
    log = AdminLog(
        admin_id=current_user.id,
        action=action,
        target_type=target_type,
        target_id=target_id,
        details=details,
    )
    db.session.add(log)


# ---------------------------------------------------------------------------
# Stats – overview / vectors / community / users
# ---------------------------------------------------------------------------

@v1_bp.route('/admin/stats/overview', methods=['GET'])
@admin_required
def admin_stats_overview():
    """Aggregate statistics for the admin dashboard."""
    total_users = User.query.count()
    total_posts = Post.query.filter_by(is_deleted=False).count()
    total_comments = Comment.query.filter_by(is_deleted=False).count()
    # Learning materials = folder-level Documents
    total_documents = Document.query.filter_by(
        status='indexed', file_type='folder',
    ).count()
    # Vector count from file-level only (folder-level is aggregated duplicate)
    total_vectors = db.session.query(
        db.func.coalesce(db.func.sum(Document.vector_count), 0)
    ).filter(
        Document.file_type != 'folder',
    ).scalar()
    namespace_count = db.session.query(
        db.func.count(db.distinct(Document.namespace))
    ).filter(
        Document.file_type != 'folder',
    ).scalar()

    return success_response(data={
        'total_users': total_users,
        'total_posts': total_posts,
        'total_comments': total_comments,
        'total_documents': total_documents,
        'total_vectors': total_vectors,
        'namespaces': namespace_count,
    })


@v1_bp.route('/admin/stats/vectors', methods=['GET'])
@admin_required
def admin_stats_vectors():
    """Vector distribution by namespace.

    Excludes folder-level Documents to avoid double-counting vectors
    (folder entries contain aggregated totals of their child files).
    """
    rows = (
        db.session.query(
            Document.namespace,
            db.func.count(Document.id).label('doc_count'),
            db.func.coalesce(db.func.sum(Document.vector_count), 0).label('vec_count'),
        )
        .filter(Document.status == 'indexed')
        .filter(Document.file_type != 'folder')
        .group_by(Document.namespace)
        .all()
    )
    return success_response(data=[
        {'namespace': r.namespace or '(기본)', 'documents': r.doc_count, 'vectors': r.vec_count}
        for r in rows
    ])


@v1_bp.route('/admin/stats/community', methods=['GET'])
@admin_required
def admin_stats_community():
    """Community activity statistics (last 30 days)."""
    since = datetime.now(timezone.utc) - timedelta(days=30)

    # Daily posts
    post_rows = (
        db.session.query(
            db.func.date(Post.created_at).label('day'),
            db.func.count(Post.id).label('cnt'),
        )
        .filter(Post.created_at >= since, Post.is_deleted == False)  # noqa: E712
        .group_by(db.func.date(Post.created_at))
        .all()
    )

    # Daily comments
    comment_rows = (
        db.session.query(
            db.func.date(Comment.created_at).label('day'),
            db.func.count(Comment.id).label('cnt'),
        )
        .filter(Comment.created_at >= since, Comment.is_deleted == False)  # noqa: E712
        .group_by(db.func.date(Comment.created_at))
        .all()
    )

    # Category distribution
    cat_rows = (
        db.session.query(
            Category.name,
            db.func.count(Post.id).label('cnt'),
        )
        .join(Post, Post.category_id == Category.id)
        .filter(Post.is_deleted == False)  # noqa: E712
        .group_by(Category.name)
        .all()
    )

    return success_response(data={
        'posts_trend': [{'date': str(r.day), 'posts': r.cnt} for r in post_rows],
        'comments_trend': [{'date': str(r.day), 'comments': r.cnt} for r in comment_rows],
        'category_distribution': [{'category': r.name, 'count': r.cnt} for r in cat_rows],
    })


@v1_bp.route('/admin/stats/users', methods=['GET'])
@admin_required
def admin_stats_users():
    """User registration and provider statistics."""
    since = datetime.now(timezone.utc) - timedelta(days=30)

    # Daily new users
    user_rows = (
        db.session.query(
            db.func.date(User.created_at).label('day'),
            db.func.count(User.id).label('cnt'),
        )
        .filter(User.created_at >= since)
        .group_by(db.func.date(User.created_at))
        .all()
    )

    # Provider breakdown
    provider_rows = (
        db.session.query(
            SocialAccount.provider,
            db.func.count(SocialAccount.id).label('cnt'),
        )
        .group_by(SocialAccount.provider)
        .all()
    )

    active_users = User.query.filter(
        User.last_login >= since, User.is_active == True  # noqa: E712
    ).count()

    return success_response(data={
        'users_trend': [{'date': str(r.day), 'new_users': r.cnt} for r in user_rows],
        'providers': [{'provider': r.provider, 'count': r.cnt} for r in provider_rows],
        'active_users_30d': active_users,
    })


# ---------------------------------------------------------------------------
# Document management (catalog + Pinecone sync)
# ---------------------------------------------------------------------------

@v1_bp.route('/admin/documents', methods=['GET'])
@admin_required
def admin_documents_list():
    """List documents with filters and pagination."""
    page = max(1, request.args.get('page', 1, type=int))
    per_page = min(100, max(1, request.args.get('per_page', 20, type=int)))
    namespace = request.args.get('namespace', '').strip()
    status = request.args.get('status', '').strip()
    search = request.args.get('search', '').strip()
    sort_by = request.args.get('sort_by', 'created_at').strip()
    sort_order = request.args.get('sort_order', 'desc').strip()

    query = Document.query

    if namespace:
        query = query.filter_by(namespace=namespace)
    if status:
        query = query.filter_by(status=status)
    if search:
        like = f'%{search}%'
        query = query.filter(
            db.or_(Document.filename.ilike(like), Document.source_file.ilike(like))
        )

    # Sortable columns whitelist
    sortable = {
        'filename': Document.filename,
        'namespace': Document.namespace,
        'file_type': Document.file_type,
        'vector_count': Document.vector_count,
        'status': Document.status,
        'created_at': Document.created_at,
    }
    col = sortable.get(sort_by, Document.created_at)
    query = query.order_by(col.asc() if sort_order == 'asc' else col.desc())
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)

    return success_response(data={
        'documents': [d.to_dict() for d in pagination.items],
        'total': pagination.total,
        'pages': pagination.pages,
        'current_page': page,
    })


@v1_bp.route('/admin/documents/<int:doc_id>', methods=['GET'])
@admin_required
def admin_document_detail(doc_id):
    """Get document details."""
    doc = db.session.get(Document, doc_id)
    if not doc:
        return error_response('문서를 찾을 수 없습니다.', 404)
    return success_response(data=doc.to_dict())


@v1_bp.route('/admin/documents/sync-filesystem', methods=['POST'])
@admin_required
def admin_documents_sync_filesystem():
    """Sync physical filesystem into the Document catalog table.

    Scans the documents/ directory for folders containing *_meta.json
    (the marker for a fully-processed learning material) and creates
    or updates Document rows accordingly.
    """
    import unicodedata

    from services.domain_config import get_namespace_for_path
    from services.filetree import scan_document_folders

    try:
        created = 0
        updated = 0
        orphaned = 0
        skipped = 0

        # Phase 1: Scan filesystem and upsert Document rows
        disk_keys = set()
        for folder in scan_document_folders():
            source_file = 'documents/' + folder['path']
            source_file = unicodedata.normalize('NFC', source_file)
            namespace = get_namespace_for_path(folder['path'])
            disk_keys.add((namespace, source_file))

            existing = Document.query.filter_by(
                namespace=namespace, source_file=source_file,
            ).first()

            if existing:
                changed = False
                if existing.file_size != folder['total_size']:
                    existing.file_size = folder['total_size']
                    changed = True
                if existing.status == 'orphaned':
                    existing.status = (
                        'indexed' if existing.vector_count else 'unindexed'
                    )
                    changed = True
                if changed:
                    updated += 1
                else:
                    skipped += 1
            else:
                doc = Document(
                    namespace=namespace,
                    source_file=source_file,
                    filename=folder['name'],
                    file_type='folder',
                    file_size=folder['total_size'],
                    vector_count=0,
                    status='unindexed',
                    uploaded_by=current_user.id,
                )
                db.session.add(doc)
                created += 1

        # Phase 2: Mark orphaned folder-level documents only.
        # Only check documents originally created by filesystem sync
        # (file_type='folder'). Leave Pinecone-synced file-level entries
        # untouched since they use a different source_file format.
        for doc in Document.query.filter(
            Document.file_type == 'folder',
            Document.status != 'orphaned',
        ).all():
            if (doc.namespace, doc.source_file) not in disk_keys:
                doc.status = 'orphaned'
                orphaned += 1

        _log_action('sync_filesystem', details={
            'created': created, 'updated': updated,
            'skipped': skipped, 'orphaned': orphaned,
        })
        db.session.commit()

        return success_response(
            data={
                'created': created, 'updated': updated,
                'skipped': skipped, 'orphaned': orphaned,
            },
            message=(
                f'{created}개 학습자료 등록, {updated}개 갱신'
                + (f', {orphaned}개 고아 처리' if orphaned else '')
            ),
        )
    except Exception as e:
        db.session.rollback()
        logging.exception('Filesystem sync failed')
        return error_response('파일시스템 동기화 중 오류가 발생했습니다.', 500)


@v1_bp.route('/admin/documents/sync', methods=['POST'])
@admin_required
def admin_documents_sync():
    """Sync Pinecone vector stats and update folder-level Documents.

    Lightweight version: uses describe_index_stats() (~1 second) to get
    namespace-level totals, then aggregates existing DB file-level Documents
    into folder-level entries.  No slow list()+fetch() enumeration.

    The heavy per-vector enumeration is only needed for initial DB population
    and is available via the ?full=true query parameter.
    """
    import unicodedata

    full_mode = request.args.get('full', '').lower() == 'true'

    try:
        from services.singletons import get_uploader

        uploader = get_uploader()
        stats = uploader.get_stats()
        namespaces = stats.get('namespaces', {})

        # ── Quick stats sync (default) ──────────────────────────
        # Update namespace-level totals from Pinecone stats
        ns_info = {}
        for ns_name, info in namespaces.items():
            vec_count = info.get('vector_count', 0)
            ns_info[ns_name or '(기본)'] = vec_count

        # If full mode requested AND no file-level data exists, do heavy sync
        file_level_count = Document.query.filter(
            Document.file_type != 'folder',
        ).count()

        if full_mode and file_level_count == 0:
            return _full_pinecone_sync(uploader, namespaces)

        # ── Auto-populate new & updated namespaces ─────────────
        # Detect namespaces in Pinecone that have no file-level Documents
        # OR have more vectors than currently tracked in the DB.
        db_ns_vectors = {}
        for row in (
            db.session.query(
                Document.namespace,
                db.func.coalesce(db.func.sum(Document.vector_count), 0),
            )
            .filter(Document.file_type != 'folder')
            .group_by(Document.namespace)
            .all()
        ):
            db_ns_vectors[row[0]] = row[1]

        new_ns_created = 0
        updated_ns_created = 0
        for ns_name, info in namespaces.items():
            pinecone_vec_count = info.get('vector_count', 0)
            if pinecone_vec_count == 0:
                continue

            db_vec_count = db_ns_vectors.get(ns_name, 0)
            # Skip if DB already tracks all vectors (within 5% tolerance)
            if db_vec_count > 0 and db_vec_count >= pinecone_vec_count * 0.95:
                continue

            is_new_ns = ns_name not in db_ns_vectors

            source_file_vectors = _enumerate_namespace_sources(
                uploader, ns_name,
            )
            for sf, sf_vec_count in source_file_vectors.items():
                sf = unicodedata.normalize('NFC', sf)

                # Skip if this source_file already exists with correct count
                existing = Document.query.filter_by(
                    namespace=ns_name, source_file=sf,
                ).filter(Document.file_type != 'folder').first()
                if existing:
                    if existing.vector_count != sf_vec_count:
                        existing.vector_count = sf_vec_count
                        existing.status = 'indexed'
                        updated_ns_created += 1
                    continue

                filename = sf.rsplit('/', 1)[-1] if '/' in sf else sf
                ext = (filename.rsplit('.', 1)[-1].lower()
                       if '.' in filename else '')
                doc = Document(
                    namespace=ns_name,
                    source_file=sf,
                    filename=filename,
                    file_type=ext or None,
                    vector_count=sf_vec_count,
                    status='indexed',
                    uploaded_by=current_user.id,
                )
                db.session.add(doc)
                if is_new_ns:
                    new_ns_created += 1
                else:
                    updated_ns_created += 1

        # ── DB-only folder aggregation ──────────────────────────
        # Aggregate file-level Documents into folder-level ones
        folder_docs = Document.query.filter_by(file_type='folder').all()
        folder_by_sf = {d.source_file: d for d in folder_docs}
        folder_sfs = set(folder_by_sf.keys())

        file_docs = (
            Document.query
            .filter(Document.file_type != 'folder')
            .filter(Document.vector_count > 0)
            .all()
        )

        folder_vectors = {}
        matched_files = 0
        for fd in file_docs:
            sf = unicodedata.normalize('NFC', fd.source_file)
            folder_sf = _resolve_folder_source_file_fuzzy(sf, folder_sfs)
            if folder_sf:
                folder_vectors[folder_sf] = (
                    folder_vectors.get(folder_sf, 0) + fd.vector_count
                )
                matched_files += 1

        # ── Filename-based fallback for unmatched folders ────────
        # When Pinecone paths use translated folder names (e.g.
        # English) but the disk uses Korean names, path matching
        # fails.  Fall back to checking if a file-level doc's
        # filename (sans extension) matches a folder's leaf name.
        unmatched_folders = {
            sf: doc for sf, doc in folder_by_sf.items()
            if folder_vectors.get(sf, 0) == 0
        }
        if unmatched_folders:
            # Build leaf-name → folder_sf lookup
            leaf_to_folder = {}
            for fsf, doc in unmatched_folders.items():
                leaf = unicodedata.normalize('NFC', fsf.rsplit('/', 1)[-1])
                leaf_to_folder[(doc.namespace, leaf)] = fsf

            for fd in file_docs:
                fname = unicodedata.normalize('NFC', fd.filename)
                stem = fname.rsplit('.', 1)[0] if '.' in fname else fname
                key = (fd.namespace, stem)
                if key in leaf_to_folder:
                    fsf = leaf_to_folder[key]
                    folder_vectors[fsf] = (
                        folder_vectors.get(fsf, 0) + fd.vector_count
                    )
                    matched_files += 1

        folder_updated = 0
        for folder_sf, doc in folder_by_sf.items():
            total_vec = folder_vectors.get(folder_sf, 0)
            if total_vec > 0:
                changed = False
                if doc.vector_count != total_vec:
                    doc.vector_count = total_vec
                    changed = True
                if doc.status != 'indexed':
                    doc.status = 'indexed'
                    changed = True
                if changed:
                    folder_updated += 1

        _log_action('sync_documents', details={
            'mode': 'quick',
            'namespaces': ns_info,
            'file_docs': len(file_docs),
            'matched_files': matched_files,
            'folder_updated': folder_updated,
            'new_ns_created': new_ns_created,
            'updated_ns_created': updated_ns_created,
        })
        db.session.commit()

        total_vectors = sum(ns_info.values())
        msg_parts = [f'Pinecone 통계 동기화 완료: 총 {total_vectors:,}개 벡터']
        if new_ns_created:
            msg_parts.append(f'새 네임스페이스에서 {new_ns_created}개 문서 등록')
        if updated_ns_created:
            msg_parts.append(f'기존 네임스페이스에서 {updated_ns_created}개 문서 갱신')
        if folder_updated:
            msg_parts.append(f'{folder_updated}개 학습자료 상태 갱신')
        return success_response(
            data={
                'namespaces': ns_info,
                'total_vectors': total_vectors,
                'folder_updated': folder_updated,
                'matched_files': matched_files,
                'new_ns_created': new_ns_created,
                'updated_ns_created': updated_ns_created,
            },
            message=', '.join(msg_parts),
        )
    except Exception as e:
        db.session.rollback()
        logging.exception('Document sync failed')
        return error_response('문서 동기화 중 오류가 발생했습니다.', 500)


def _full_pinecone_sync(uploader, namespaces):
    """Heavy sync: enumerate ALL vectors via list()+fetch().

    Only used when file-level Documents don't exist yet (initial population).
    Takes 15-30+ minutes for large indexes.
    """
    import unicodedata
    from concurrent.futures import ThreadPoolExecutor

    created = 0
    skipped = 0

    active_ns = {
        ns: info for ns, info in namespaces.items()
        if info.get('vector_count', 0) > 0
    }

    ns_results = {}
    with ThreadPoolExecutor(max_workers=min(len(active_ns), 10) or 1) as ns_exec:
        ns_futures = {
            ns_exec.submit(_enumerate_namespace_sources, uploader, ns): ns
            for ns in active_ns
        }
        for future in ns_futures:
            ns_name = ns_futures[future]
            try:
                ns_results[ns_name] = future.result()
            except Exception as exc:
                logging.warning('Namespace %s sync failed: %s', ns_name, exc)
                ns_results[ns_name] = {}

    for ns_name, source_file_vectors in ns_results.items():
        for sf, sf_vec_count in source_file_vectors.items():
            sf = unicodedata.normalize('NFC', sf)

            existing = Document.query.filter_by(
                namespace=ns_name, source_file=sf,
            ).first()
            if existing:
                if existing.vector_count != sf_vec_count:
                    existing.vector_count = sf_vec_count
                skipped += 1
                continue

            filename = sf.rsplit('/', 1)[-1] if '/' in sf else sf
            ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
            doc = Document(
                namespace=ns_name,
                source_file=sf,
                filename=filename,
                file_type=ext or None,
                vector_count=sf_vec_count,
                status='indexed',
                uploaded_by=current_user.id,
            )
            db.session.add(doc)
            created += 1

    _log_action('sync_documents', details={
        'mode': 'full', 'created': created, 'skipped': skipped,
    })
    db.session.commit()

    return success_response(
        data={'created': created, 'skipped': skipped},
        message=f'전체 동기화 완료: {created}개 문서 등록 ({skipped}개 이미 존재)',
    )


def _resolve_folder_source_file(source_file):
    """Convert a Pinecone source_file path to a folder-level relative path.

    Handles both absolute and relative paths.  Returns the folder path in
    'documents/...' format (matching filesystem sync), or None.
    """
    import unicodedata
    from pathlib import PurePosixPath

    sf = source_file

    # Extract the part after '/documents/' (absolute) or 'documents/' (relative)
    marker = '/documents/'
    idx = sf.find(marker)
    if idx >= 0:
        remainder = sf[idx + len(marker):]
    elif sf.startswith('documents/'):
        remainder = sf[len('documents/'):]
    else:
        return None

    # remainder: e.g. "laborlaw/laws/FOLDER/FILE.ext"
    rp = PurePosixPath(remainder)

    # If it's a file (has extension), take the parent folder
    if rp.suffix:
        folder_rel = str(rp.parent)
    else:
        folder_rel = remainder

    if not folder_rel or folder_rel == '.':
        return None

    folder_sf = 'documents/' + folder_rel
    return unicodedata.normalize('NFC', folder_sf)


# Regex for timestamp prefix: YYYYMMDD_HHMMSS_
_TIMESTAMP_PREFIX_RE = re.compile(r'^\d{8}_\d{6}_')


def _resolve_folder_source_file_fuzzy(source_file, folder_sfs):
    """Try exact match first, then fuzzy match by stripping prefixes.

    Matching strategies (in order):
    1. Exact path match
    2. Strip timestamp prefix (YYYYMMDD_HHMMSS_) from folder name
    3. Suffix matching — one folder name is a suffix of the other
    4. Cross-parent leaf name match — same folder name under different parent
       (handles directory moves like ncs/data → ncs/old_markdown)

    Args:
        source_file: raw Pinecone source_file path
        folder_sfs: set of known folder-level source_file strings

    Returns:
        matched folder source_file string, or None
    """
    import unicodedata

    exact = _resolve_folder_source_file(source_file)
    if exact and exact in folder_sfs:
        return exact

    if not exact:
        return None

    # Split into parent directory + folder name
    parts = exact.rsplit('/', 1)
    if len(parts) != 2:
        return None

    parent, folder_name = parts

    # Strategy 2: Strip timestamp prefix (YYYYMMDD_HHMMSS_)
    stripped = _TIMESTAMP_PREFIX_RE.sub('', folder_name).rstrip('_')
    if stripped != folder_name:
        candidate = unicodedata.normalize('NFC', f'{parent}/{stripped}')
        if candidate in folder_sfs:
            return candidate

        # Strategy 3: Suffix matching within the same parent directory
        # After timestamp stripping, check if stripped name is a suffix of
        # any known folder name (or vice versa) under the same parent.
        for known in folder_sfs:
            if not known.startswith(parent + '/'):
                continue
            known_name = known.rsplit('/', 1)[-1]
            # One must contain the other, with minimum 10 chars overlap
            if len(stripped) >= 10 and len(known_name) >= 10:
                if known_name.endswith(stripped) or stripped.endswith(known_name):
                    return known

    # Strategy 4: Cross-parent leaf name match
    # Handles directory reorganization (e.g. ncs/data/X → ncs/old_markdown/X,
    # or laborlaw/cases/korean/X → laborlaw/cases/X).
    # Match by exact leaf folder name across all known folders that share
    # the same top-level domain (first path segment after 'documents/').
    leaf = _TIMESTAMP_PREFIX_RE.sub('', folder_name).rstrip('_') or folder_name
    if len(leaf) < 8:
        return None  # Too short, risk of false matches

    # Determine top-level domain: documents/DOMAIN/...
    exact_parts = exact.split('/')
    domain = exact_parts[1] if len(exact_parts) > 2 else None
    if not domain:
        return None

    for known in folder_sfs:
        known_parts = known.split('/')
        known_domain = known_parts[1] if len(known_parts) > 2 else None
        if known_domain != domain:
            continue
        known_leaf = known.rsplit('/', 1)[-1]
        if known_leaf == leaf:
            return known

    return None


def _enumerate_namespace_sources(uploader, ns_name, max_workers=10):
    """Enumerate all source files in a namespace via list() + parallel fetch().

    Falls back to search-based discovery if the list API is unavailable.

    Returns:
        dict mapping source_file -> vector_count
    """
    import logging
    from concurrent.futures import ThreadPoolExecutor

    source_file_vectors = {}

    def _fetch_batch(batch_ids):
        """Fetch a batch of vector IDs and extract source_file metadata."""
        local = {}
        try:
            fetch_res = uploader.index.fetch(ids=batch_ids, namespace=ns_name)
            for _vid, vec_data in fetch_res.vectors.items():
                meta = vec_data.metadata or {}
                sf = meta.get('source_file', '')
                if sf:
                    local[sf] = local.get(sf, 0) + 1
        except Exception as exc:
            logging.warning('Pinecone fetch error during sync: %s', exc)
        return local

    try:
        # Step 1: collect all vector IDs (list is fast, returns only IDs)
        all_ids = []
        for id_list in uploader.index.list(namespace=ns_name):
            if not id_list:
                continue
            all_ids.extend(
                v.id if hasattr(v, 'id') else str(v) for v in id_list
            )

        if not all_ids:
            return source_file_vectors

        # Step 2: parallel fetch in batches of 100
        batches = [all_ids[i:i + 100] for i in range(0, len(all_ids), 100)]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_fetch_batch, b) for b in batches]
            for future in futures:
                for sf, cnt in future.result().items():
                    source_file_vectors[sf] = (
                        source_file_vectors.get(sf, 0) + cnt
                    )

    except Exception as exc:
        logging.warning('Pinecone list API failed, using search fallback: %s', exc)
        source_file_vectors = _discover_sources_via_search(uploader, ns_name)

    return source_file_vectors


def _discover_sources_via_search(uploader, ns_name):
    """Fallback: discover source files via broad semantic searches.

    Used when the Pinecone list/fetch API is unavailable.
    """
    from services.singletons import get_agent

    agent = get_agent()
    source_counts = {}
    queries = [
        '문서', '정보', 'data', 'document', '공정', '기술',
        '안전', '보건', '법률', '규정', '화학', '물질',
        '반도체', 'semiconductor', '제조', '설비',
    ]
    for q in queries:
        try:
            results = agent.search(query=q, top_k=100, namespace=ns_name)
        except Exception:
            continue
        for r in results:
            meta = r.get('metadata', {})
            sf = meta.get('source_file', '')
            if sf:
                source_counts[sf] = source_counts.get(sf, 0) + 1
    return source_counts


@v1_bp.route('/admin/documents/update-folder-status', methods=['POST'])
@admin_required
def admin_documents_update_folder_status():
    """Update folder-level Document status using existing DB data.

    Aggregates vector counts from file-level Document entries (created by
    Pinecone sync) into folder-level entries (created by filesystem sync)
    using fuzzy path matching.  No Pinecone API calls — purely DB-based,
    completes in under a second.
    """
    import unicodedata

    try:
        # 1. Get all folder-level Documents (targets)
        folder_docs = Document.query.filter_by(file_type='folder').all()
        if not folder_docs:
            return success_response(
                data={'updated': 0, 'unindexed': 0},
                message='폴더형 학습자료가 없습니다. 먼저 파일 동기화를 실행하세요.',
            )

        folder_by_sf = {d.source_file: d for d in folder_docs}
        folder_sfs = set(folder_by_sf.keys())

        # 2. Get all file-level Documents with vectors (sources)
        file_docs = (
            Document.query
            .filter(Document.file_type != 'folder')
            .filter(Document.vector_count > 0)
            .all()
        )

        # 3. Map file-level → folder-level and aggregate vector counts
        folder_vectors = {}  # folder_sf → total_vector_count
        matched_files = 0

        for fd in file_docs:
            sf = unicodedata.normalize('NFC', fd.source_file)
            folder_sf = _resolve_folder_source_file_fuzzy(sf, folder_sfs)
            if folder_sf:
                folder_vectors[folder_sf] = (
                    folder_vectors.get(folder_sf, 0) + fd.vector_count
                )
                matched_files += 1

        # ── Filename-based fallback for unmatched folders ────────
        # When Pinecone paths use translated folder names (e.g.
        # English) but the disk uses Korean names, path matching
        # fails.  Fall back to checking if a file-level doc's
        # filename (sans extension) matches a folder's leaf name.
        unmatched_folders = {
            sf: doc for sf, doc in folder_by_sf.items()
            if folder_vectors.get(sf, 0) == 0
        }
        if unmatched_folders:
            leaf_to_folder = {}
            for fsf, doc in unmatched_folders.items():
                leaf = unicodedata.normalize('NFC', fsf.rsplit('/', 1)[-1])
                leaf_to_folder[(doc.namespace, leaf)] = fsf

            for fd in file_docs:
                fname = unicodedata.normalize('NFC', fd.filename)
                stem = fname.rsplit('.', 1)[0] if '.' in fname else fname
                key = (fd.namespace, stem)
                if key in leaf_to_folder:
                    fsf = leaf_to_folder[key]
                    folder_vectors[fsf] = (
                        folder_vectors.get(fsf, 0) + fd.vector_count
                    )
                    matched_files += 1

        # 4. Update folder Documents with aggregated counts
        updated = 0
        for folder_sf, doc in folder_by_sf.items():
            total_vec = folder_vectors.get(folder_sf, 0)
            if total_vec > 0:
                changed = False
                if doc.vector_count != total_vec:
                    doc.vector_count = total_vec
                    changed = True
                if doc.status != 'indexed':
                    doc.status = 'indexed'
                    changed = True
                if changed:
                    updated += 1

        # 5. Duplicate folder detection: if an unindexed folder has the same
        # leaf name as an indexed folder in the same namespace, inherit its
        # status.  Handles directory reorganizations (e.g. ncs/data → ncs/old_markdown).
        indexed_by_leaf = {}
        for folder_sf, doc in folder_by_sf.items():
            if doc.status == 'indexed' and doc.vector_count > 0:
                leaf = folder_sf.rsplit('/', 1)[-1]
                key = (doc.namespace, leaf)
                indexed_by_leaf[key] = doc

        duplicates = 0
        for folder_sf, doc in folder_by_sf.items():
            if doc.status != 'unindexed':
                continue
            leaf = folder_sf.rsplit('/', 1)[-1]
            key = (doc.namespace, leaf)
            source = indexed_by_leaf.get(key)
            if source and len(leaf) >= 8:
                doc.vector_count = source.vector_count
                doc.status = 'indexed'
                duplicates += 1

        still_unindexed = sum(
            1 for doc in folder_by_sf.values()
            if doc.status == 'unindexed'
        )

        _log_action('update_folder_status', details={
            'folders': len(folder_docs),
            'file_docs_with_vectors': len(file_docs),
            'matched_files': matched_files,
            'updated': updated,
            'duplicates': duplicates,
            'still_unindexed': still_unindexed,
        })
        db.session.commit()

        return success_response(
            data={
                'folders': len(folder_docs),
                'updated': updated,
                'duplicates': duplicates,
                'still_unindexed': still_unindexed,
                'matched_files': matched_files,
            },
            message=(
                f'{updated + duplicates}개 학습자료 색인 확인'
                + (f' (중복 {duplicates}개 포함)' if duplicates else '')
                + (f', {still_unindexed}개 미색인' if still_unindexed else '')
            ),
        )
    except Exception as e:
        db.session.rollback()
        logging.exception('Folder status update failed')
        return error_response('폴더 상태 업데이트 중 오류가 발생했습니다.', 500)


@v1_bp.route('/admin/documents/<int:doc_id>', methods=['DELETE'])
@admin_required
def admin_document_delete(doc_id):
    """Delete a document from both DB and Pinecone."""
    doc = db.session.get(Document, doc_id)
    if not doc:
        return error_response('문서를 찾을 수 없습니다.', 404)

    try:
        from services.singletons import get_uploader
        uploader = get_uploader()
        uploader.delete_by_filter(
            filter={'source_file': doc.source_file},
            namespace=doc.namespace,
        )
    except Exception as e:
        logging.warning('Pinecone document delete failed for %s (ns=%s): %s — orphan vectors may remain',
                        doc.source_file, doc.namespace, e)

    _log_action('delete_document', 'document', doc_id, {
        'filename': doc.filename, 'namespace': doc.namespace,
    })
    db.session.delete(doc)
    db.session.commit()

    return success_response(message='문서가 삭제되었습니다.')


@v1_bp.route('/admin/documents/namespace/<path:ns>', methods=['DELETE'])
@admin_required
def admin_namespace_delete(ns):
    """Delete an entire namespace from Pinecone and remove catalog entries."""
    try:
        from services.singletons import get_uploader
        uploader = get_uploader()
        uploader.index.delete(delete_all=True, namespace=ns)
    except Exception as e:
        logging.warning('Pinecone namespace delete failed for %s: %s — orphan vectors may remain', ns, e)

    count = Document.query.filter_by(namespace=ns).delete()
    _log_action('delete_namespace', 'namespace', details={'namespace': ns, 'documents': count})
    db.session.commit()

    return success_response(message=f"네임스페이스 '{ns}'가 삭제되었습니다. ({count}개 문서)")


# ---------------------------------------------------------------------------
# Post management
# ---------------------------------------------------------------------------

@v1_bp.route('/admin/posts', methods=['GET'])
@admin_required
def admin_posts_list():
    """List all posts including soft-deleted ones."""
    page = max(1, request.args.get('page', 1, type=int))
    per_page = min(100, max(1, request.args.get('per_page', 20, type=int)))
    search = request.args.get('search', '').strip()
    show_deleted = request.args.get('deleted', '').lower() == 'true'
    category_slug = request.args.get('category', '').strip()

    query = Post.query
    if not show_deleted:
        query = query.filter_by(is_deleted=False)
    if category_slug:
        cat = Category.query.filter_by(slug=category_slug).first()
        if cat:
            query = query.filter_by(category_id=cat.id)
    if search:
        like = f'%{search}%'
        query = query.filter(
            db.or_(Post.title.ilike(like), Post.content.ilike(like))
        )

    query = query.order_by(Post.created_at.desc())
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)

    posts = []
    for p in pagination.items:
        d = p.to_dict(include_content=False)
        d['is_deleted'] = p.is_deleted
        d['user_id'] = p.user_id
        posts.append(d)

    return success_response(data={
        'posts': posts,
        'total': pagination.total,
        'pages': pagination.pages,
        'current_page': page,
    })


@v1_bp.route('/admin/posts/<int:post_id>/pin', methods=['PUT'])
@admin_required
def admin_post_pin(post_id):
    """Toggle pin status on a post."""
    post = db.session.get(Post, post_id)
    if not post:
        return error_response('게시글을 찾을 수 없습니다.', 404)

    post.is_pinned = not post.is_pinned
    _log_action('pin_post' if post.is_pinned else 'unpin_post', 'post', post_id)
    db.session.commit()

    action = '고정' if post.is_pinned else '고정 해제'
    return success_response(
        data={'is_pinned': post.is_pinned},
        message=f'게시글이 {action}되었습니다.',
    )


@v1_bp.route('/admin/posts/<int:post_id>/restore', methods=['PUT'])
@admin_required
def admin_post_restore(post_id):
    """Restore a soft-deleted post."""
    post = db.session.get(Post, post_id)
    if not post:
        return error_response('게시글을 찾을 수 없습니다.', 404)
    if not post.is_deleted:
        return error_response('삭제된 게시글이 아닙니다.', 400)

    post.is_deleted = False
    _log_action('restore_post', 'post', post_id)
    db.session.commit()

    return success_response(message='게시글이 복원되었습니다.')


@v1_bp.route('/admin/posts/<int:post_id>/hard', methods=['DELETE'])
@admin_required
def admin_post_hard_delete(post_id):
    """Permanently delete a post and its attachments/comments/likes."""
    post = db.session.get(Post, post_id)
    if not post:
        return error_response('게시글을 찾을 수 없습니다.', 404)

    # Delete attachment files from disk
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    upload_dir = os.path.join(base_dir, 'static', 'uploads', 'community')
    for att in post.attachments:
        try:
            # Use basename to prevent path traversal
            safe_filename = os.path.basename(att.filename)
            filepath = os.path.join(upload_dir, safe_filename)
            # Verify path is within upload_dir
            real_upload_dir = os.path.realpath(upload_dir)
            real_filepath = os.path.realpath(filepath)
            if os.path.commonpath([real_upload_dir, real_filepath]) == real_upload_dir and os.path.exists(filepath):
                os.remove(filepath)
        except OSError:
            pass

    _log_action('hard_delete_post', 'post', post_id, {'title': post.title})
    db.session.delete(post)  # cascade deletes comments, likes, attachments
    db.session.commit()

    return success_response(message='게시글이 완전히 삭제되었습니다.')


# ---------------------------------------------------------------------------
# Comment management
# ---------------------------------------------------------------------------

@v1_bp.route('/admin/comments', methods=['GET'])
@admin_required
def admin_comments_list():
    """List all comments with pagination."""
    page = max(1, request.args.get('page', 1, type=int))
    per_page = min(100, max(1, request.args.get('per_page', 20, type=int)))

    query = Comment.query.order_by(Comment.created_at.desc())
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)

    comments = []
    for c in pagination.items:
        d = c.to_dict()
        d['is_deleted'] = c.is_deleted
        d['user_id'] = c.user_id
        comments.append(d)

    return success_response(data={
        'comments': comments,
        'total': pagination.total,
        'pages': pagination.pages,
        'current_page': page,
    })


@v1_bp.route('/admin/comments/<int:comment_id>/hard', methods=['DELETE'])
@admin_required
def admin_comment_hard_delete(comment_id):
    """Permanently delete a comment."""
    comment = db.session.get(Comment, comment_id)
    if not comment:
        return error_response('댓글을 찾을 수 없습니다.', 404)

    _log_action('hard_delete_comment', 'comment', comment_id)
    db.session.delete(comment)
    db.session.commit()

    return success_response(message='댓글이 완전히 삭제되었습니다.')


# ---------------------------------------------------------------------------
# Category management
# ---------------------------------------------------------------------------

@v1_bp.route('/admin/categories', methods=['GET'])
@admin_required
def admin_categories_list():
    """List all categories (including inactive)."""
    post_counts = (
        db.session.query(
            Post.category_id,
            db.func.count(Post.id).label('post_count'),
        )
        .filter(Post.is_deleted == False)  # noqa: E712
        .group_by(Post.category_id)
        .subquery()
    )
    cats = (
        db.session.query(Category, db.func.coalesce(post_counts.c.post_count, 0))
        .outerjoin(post_counts, Category.id == post_counts.c.category_id)
        .order_by(Category.sort_order)
        .all()
    )
    result = []
    for c, count in cats:
        d = c.to_dict()
        d['is_active'] = c.is_active
        d['post_count'] = count
        result.append(d)
    return success_response(data=result)


@v1_bp.route('/admin/categories', methods=['POST'])
@admin_required
def admin_category_create():
    """Create a new category."""
    data = request.get_json(silent=True)
    if not data:
        return error_response('요청 데이터가 없습니다.', 400)

    name = (data.get('name') or '').strip()
    slug = (data.get('slug') or '').strip()
    if not name or not slug:
        return error_response('이름과 슬러그는 필수입니다.', 400)

    if Category.query.filter_by(slug=slug).first():
        return error_response('이미 존재하는 슬러그입니다.', 400)

    cat = Category(
        name=name,
        slug=slug,
        description=(data.get('description') or '').strip() or None,
        color=data.get('color', '#00d4ff'),
        icon=data.get('icon'),
        sort_order=data.get('sort_order', 0),
        is_active=data.get('is_active', True),
    )
    db.session.add(cat)
    _log_action('create_category', 'category', details={'name': name, 'slug': slug})
    db.session.commit()

    return success_response(data=cat.to_dict(), message='카테고리가 생성되었습니다.')


@v1_bp.route('/admin/categories/<int:cat_id>', methods=['PUT'])
@admin_required
def admin_category_update(cat_id):
    """Update a category."""
    cat = db.session.get(Category, cat_id)
    if not cat:
        return error_response('카테고리를 찾을 수 없습니다.', 404)

    data = request.get_json(silent=True)
    if not data:
        return error_response('요청 데이터가 없습니다.', 400)

    if 'name' in data:
        cat.name = (data['name'] or '').strip() or cat.name
    if 'description' in data:
        cat.description = (data['description'] or '').strip() or None
    if 'color' in data:
        cat.color = data['color']
    if 'icon' in data:
        cat.icon = data['icon']
    if 'sort_order' in data:
        cat.sort_order = data['sort_order']
    if 'is_active' in data:
        cat.is_active = bool(data['is_active'])

    _log_action('update_category', 'category', cat_id)
    db.session.commit()

    return success_response(data=cat.to_dict(), message='카테고리가 수정되었습니다.')


@v1_bp.route('/admin/categories/<int:cat_id>', methods=['DELETE'])
@admin_required
def admin_category_deactivate(cat_id):
    """Deactivate (soft-delete) a category."""
    cat = db.session.get(Category, cat_id)
    if not cat:
        return error_response('카테고리를 찾을 수 없습니다.', 404)

    cat.is_active = False
    _log_action('deactivate_category', 'category', cat_id, {'name': cat.name})
    db.session.commit()

    return success_response(message='카테고리가 비활성화되었습니다.')


# ---------------------------------------------------------------------------
# User management
# ---------------------------------------------------------------------------

@v1_bp.route('/admin/users', methods=['GET'])
@admin_required
def admin_users_list():
    """List users with search and pagination."""
    page = max(1, request.args.get('page', 1, type=int))
    per_page = min(100, max(1, request.args.get('per_page', 20, type=int)))
    search = request.args.get('search', '').strip()
    sort = request.args.get('sort', 'latest')

    # Subqueries to avoid 2N+1 per-user count queries
    post_counts = (
        db.session.query(
            Post.user_id,
            db.func.count(Post.id).label('cnt'),
        )
        .filter(Post.is_deleted == False)  # noqa: E712
        .group_by(Post.user_id)
        .subquery()
    )
    comment_counts = (
        db.session.query(
            Comment.user_id,
            db.func.count(Comment.id).label('cnt'),
        )
        .filter(Comment.is_deleted == False)  # noqa: E712
        .group_by(Comment.user_id)
        .subquery()
    )

    query = (
        db.session.query(
            User,
            db.func.coalesce(post_counts.c.cnt, 0).label('post_count'),
            db.func.coalesce(comment_counts.c.cnt, 0).label('comment_count'),
        )
        .outerjoin(post_counts, User.id == post_counts.c.user_id)
        .outerjoin(comment_counts, User.id == comment_counts.c.user_id)
    )

    if search:
        like = f'%{search}%'
        query = query.filter(
            db.or_(User.name.ilike(like), User.email.ilike(like))
        )

    if sort == 'name':
        query = query.order_by(User.name.asc())
    elif sort == 'login':
        query = query.order_by(User.last_login.desc().nullslast())
    else:
        query = query.order_by(User.created_at.desc())

    pagination = query.paginate(page=page, per_page=per_page, error_out=False)

    users = []
    for u, post_count, comment_count in pagination.items:
        d = u.to_dict()
        d['post_count'] = post_count
        d['comment_count'] = comment_count
        users.append(d)

    return success_response(data={
        'users': users,
        'total': pagination.total,
        'pages': pagination.pages,
        'current_page': page,
    })


@v1_bp.route('/admin/users/<int:user_id>', methods=['GET'])
@admin_required
def admin_user_detail(user_id):
    """Get user detail with social accounts and activity."""
    user = db.session.get(User, user_id)
    if not user:
        return error_response('사용자를 찾을 수 없습니다.', 404)

    data = user.to_dict()
    data['social_accounts'] = [
        {'provider': sa.provider, 'created_at': sa.created_at.isoformat() if sa.created_at else None}
        for sa in user.social_accounts
    ]
    data['post_count'] = Post.query.filter_by(user_id=user.id, is_deleted=False).count()
    data['comment_count'] = Comment.query.filter_by(user_id=user.id, is_deleted=False).count()

    return success_response(data=data)


@v1_bp.route('/admin/users/<int:user_id>/role', methods=['PUT'])
@admin_required
def admin_user_role(user_id):
    """Change user role (user ↔ admin)."""
    if user_id == current_user.id:
        return error_response('자신의 역할은 변경할 수 없습니다.', 400)

    user = db.session.get(User, user_id)
    if not user:
        return error_response('사용자를 찾을 수 없습니다.', 404)

    data = request.get_json(silent=True)
    new_role = (data.get('role') or '').strip() if data else ''
    if new_role not in ('user', 'admin'):
        return error_response("역할은 'user' 또는 'admin'이어야 합니다.", 400)

    old_role = user.role
    user.role = new_role
    _log_action('change_role', 'user', user_id, {
        'old_role': old_role, 'new_role': new_role, 'name': user.name,
    })
    db.session.commit()

    return success_response(
        data={'role': user.role},
        message=f'{user.name}의 역할이 {new_role}로 변경되었습니다.',
    )


@v1_bp.route('/admin/users/<int:user_id>/active', methods=['PUT'])
@admin_required
def admin_user_active(user_id):
    """Toggle user active status."""
    if user_id == current_user.id:
        return error_response('자신의 계정은 비활성화할 수 없습니다.', 400)

    user = db.session.get(User, user_id)
    if not user:
        return error_response('사용자를 찾을 수 없습니다.', 404)

    user.is_active = not user.is_active
    action = 'activate_user' if user.is_active else 'deactivate_user'
    _log_action(action, 'user', user_id, {'name': user.name})
    db.session.commit()

    status = '활성화' if user.is_active else '비활성화'
    return success_response(
        data={'is_active': user.is_active},
        message=f'{user.name} 계정이 {status}되었습니다.',
    )


# ---------------------------------------------------------------------------
# Activity logs
# ---------------------------------------------------------------------------

@v1_bp.route('/admin/logs', methods=['GET'])
@admin_required
def admin_logs_list():
    """List admin activity logs."""
    page = max(1, request.args.get('page', 1, type=int))
    per_page = min(100, max(1, request.args.get('per_page', 30, type=int)))
    action_filter = request.args.get('action', '').strip()
    target_filter = request.args.get('target_type', '').strip()

    query = AdminLog.query

    if action_filter:
        query = query.filter_by(action=action_filter)
    if target_filter:
        query = query.filter_by(target_type=target_filter)

    query = query.order_by(AdminLog.created_at.desc())
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)

    return success_response(data={
        'logs': [l.to_dict() for l in pagination.items],
        'total': pagination.total,
        'pages': pagination.pages,
        'current_page': page,
    })


# ---------------------------------------------------------------------------
# File tree (filesystem-based document browser)
# ---------------------------------------------------------------------------

@v1_bp.route('/admin/filetree', methods=['GET'])
@admin_required
def admin_filetree():
    """Return immediate children of a directory in the documents folder."""
    from services.filetree import scan_directory

    path = request.args.get('path', '').strip()
    file_types_str = request.args.get('file_types', '').strip()
    file_types = set(file_types_str.split(',')) if file_types_str else None

    try:
        data = scan_directory(path, file_types)
        return success_response(data={
            'path': path,
            'name': path.rsplit('/', 1)[-1] if path else 'documents',
            **data,
        })
    except ValueError as e:
        return error_response(str(e), 400)


@v1_bp.route('/admin/filetree/refresh', methods=['POST'])
@admin_required
def admin_filetree_refresh():
    """Invalidate the filetree cache and force a re-scan on next request."""
    from services.filetree import invalidate_cache

    invalidate_cache()
    _log_action('refresh_filetree')
    db.session.commit()
    return success_response(message='파일 트리 캐시가 갱신되었습니다.')
