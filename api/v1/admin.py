"""Admin dashboard API endpoints."""

import logging
import os
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from functools import wraps

from flask import request
from flask_login import current_user, login_required

from api.response import error_response, escape_like as _escape_like, success_response
from api.v1 import v1_bp
from models import (
    AdminLog, Category, Comment, Document, NewsArticle, Post, PostAttachment,
    PostLike, SocialAccount, SystemSetting, User, db,
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


def _safe_commit(error_msg='작업 처리 중 오류가 발생했습니다.'):
    """Commit session with rollback on failure. Returns error response or None."""
    try:
        db.session.commit()
        return None
    except Exception:
        db.session.rollback()
        logging.exception('DB commit failed')
        return error_response(error_msg, 500)


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
    total_news = NewsArticle.query.count()

    return success_response(data={
        'total_users': total_users,
        'total_posts': total_posts,
        'total_comments': total_comments,
        'total_documents': total_documents,
        'total_vectors': total_vectors,
        'namespaces': namespace_count,
        'total_news': total_news,
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
        .filter(Post.created_at >= since, Post.is_deleted.is_(False))
        .group_by(db.func.date(Post.created_at))
        .all()
    )

    # Daily comments
    comment_rows = (
        db.session.query(
            db.func.date(Comment.created_at).label('day'),
            db.func.count(Comment.id).label('cnt'),
        )
        .filter(Comment.created_at >= since, Comment.is_deleted.is_(False))
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
        .filter(Post.is_deleted.is_(False))
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
    page = min(10000, max(1, request.args.get('page', 1, type=int)))
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
        like = f'%{_escape_like(search)}%'
        query = query.filter(
            db.or_(
                Document.filename.ilike(like, escape='\\'),
                Document.source_file.ilike(like, escape='\\'),
            )
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

    from concurrent.futures import ThreadPoolExecutor, as_completed

    created = 0
    updated = 0
    skipped = 0

    active_ns = {
        ns: info for ns, info in namespaces.items()
        if info.get('vector_count', 0) > 0
    }

    # Process each namespace and commit immediately to limit memory usage
    with ThreadPoolExecutor(max_workers=min(len(active_ns), 10) or 1) as ns_exec:
        ns_futures = {
            ns_exec.submit(_enumerate_namespace_sources, uploader, ns): ns
            for ns in active_ns
        }
        for future in as_completed(ns_futures):
            ns_name = ns_futures[future]
            try:
                source_file_vectors = future.result()
            except Exception as exc:
                logging.warning('Namespace %s sync failed: %s', ns_name, exc)
                continue

            for sf, sf_vec_count in source_file_vectors.items():
                sf = unicodedata.normalize('NFC', sf)

                existing = Document.query.filter_by(
                    namespace=ns_name, source_file=sf,
                ).first()
                if existing:
                    if existing.vector_count != sf_vec_count:
                        existing.vector_count = sf_vec_count
                        updated += 1
                    else:
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

            # Commit per namespace to free memory and avoid giant transactions
            try:
                db.session.commit()
            except Exception:
                db.session.rollback()
                logging.exception('Sync commit failed for namespace %s', ns_name)
            finally:
                del source_file_vectors

    _log_action('sync_documents', details={
        'mode': 'full', 'created': created, 'updated': updated, 'skipped': skipped,
    })
    try:
        db.session.commit()
    except Exception:
        db.session.rollback()
        logging.exception('Full Pinecone sync final commit failed')
        return error_response('전체 동기화 중 오류가 발생했습니다.', 500)

    return success_response(
        data={'created': created, 'updated': updated, 'skipped': skipped},
        message=f'전체 동기화 완료: {created}개 등록, {updated}개 갱신, {skipped}개 변경없음',
    )


def _resolve_folder_source_file(source_file):
    """Convert a Pinecone source_file path to a folder-level relative path.

    Handles both absolute and relative paths.  Returns the folder path in
    'documents/...' format (matching filesystem sync), or None.
    """

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
    err = _safe_commit('문서 삭제 중 오류가 발생했습니다.')
    if err:
        return err

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
    err = _safe_commit('네임스페이스 삭제 중 오류가 발생했습니다.')
    if err:
        return err

    return success_response(message=f"네임스페이스 '{ns}'가 삭제되었습니다. ({count}개 문서)")


# ---------------------------------------------------------------------------
# News management (admin-specific list; CUD endpoints in api/v1/news.py)
# ---------------------------------------------------------------------------

@v1_bp.route('/admin/news', methods=['GET'])
@admin_required
def admin_news_list():
    """List all news articles including unpublished ones."""
    page = min(10000, max(1, request.args.get('page', 1, type=int)))
    per_page = min(100, max(1, request.args.get('per_page', 20, type=int)))
    search = request.args.get('search', '').strip()
    category = request.args.get('category', '').strip()

    query = NewsArticle.query

    if category and category in NewsArticle.CATEGORIES:
        query = query.filter_by(category=category)
    if search:
        like = f'%{_escape_like(search)}%'
        query = query.filter(
            db.or_(
                NewsArticle.title.ilike(like, escape='\\'),
                NewsArticle.summary.ilike(like, escape='\\'),
            )
        )

    query = query.order_by(NewsArticle.created_at.desc())
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)

    return success_response(data={
        'articles': [
            {**a.to_dict(include_content=False), 'is_published': a.is_published}
            for a in pagination.items
        ],
        'total': pagination.total,
        'pages': pagination.pages,
        'current_page': page,
    })


# ---------------------------------------------------------------------------
# Post management
# ---------------------------------------------------------------------------

@v1_bp.route('/admin/posts', methods=['GET'])
@admin_required
def admin_posts_list():
    """List all posts including soft-deleted ones."""
    page = min(10000, max(1, request.args.get('page', 1, type=int)))
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
        like = f'%{_escape_like(search)}%'
        query = query.filter(
            db.or_(
                Post.title.ilike(like, escape='\\'),
                Post.content.ilike(like, escape='\\'),
            )
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
    err = _safe_commit('게시글 고정 처리 중 오류가 발생했습니다.')
    if err:
        return err

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
    err = _safe_commit('게시글 복원 중 오류가 발생했습니다.')
    if err:
        return err

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
    err = _safe_commit('게시글 완전 삭제 중 오류가 발생했습니다.')
    if err:
        return err

    return success_response(message='게시글이 완전히 삭제되었습니다.')


# ---------------------------------------------------------------------------
# Comment management
# ---------------------------------------------------------------------------

@v1_bp.route('/admin/comments', methods=['GET'])
@admin_required
def admin_comments_list():
    """List all comments with pagination."""
    page = min(10000, max(1, request.args.get('page', 1, type=int)))
    per_page = min(100, max(1, request.args.get('per_page', 20, type=int)))

    query = Comment.query.order_by(Comment.created_at.desc())
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)

    comments = []
    for c in pagination.items:
        d = c.to_dict(include_replies=False)
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
    err = _safe_commit('댓글 삭제 중 오류가 발생했습니다.')
    if err:
        return err

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
        .filter(Post.is_deleted.is_(False))
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
    err = _safe_commit('카테고리 생성 중 오류가 발생했습니다.')
    if err:
        return err

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
    err = _safe_commit('카테고리 수정 중 오류가 발생했습니다.')
    if err:
        return err

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
    err = _safe_commit('카테고리 비활성화 중 오류가 발생했습니다.')
    if err:
        return err

    return success_response(message='카테고리가 비활성화되었습니다.')


# ---------------------------------------------------------------------------
# User management
# ---------------------------------------------------------------------------

@v1_bp.route('/admin/users', methods=['GET'])
@admin_required
def admin_users_list():
    """List users with search and pagination."""
    page = min(10000, max(1, request.args.get('page', 1, type=int)))
    per_page = min(100, max(1, request.args.get('per_page', 20, type=int)))
    search = request.args.get('search', '').strip()
    sort = request.args.get('sort', 'latest')

    # Subqueries to avoid 2N+1 per-user count queries
    post_counts = (
        db.session.query(
            Post.user_id,
            db.func.count(Post.id).label('cnt'),
        )
        .filter(Post.is_deleted.is_(False))
        .group_by(Post.user_id)
        .subquery()
    )
    comment_counts = (
        db.session.query(
            Comment.user_id,
            db.func.count(Comment.id).label('cnt'),
        )
        .filter(Comment.is_deleted.is_(False))
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
        like = f'%{_escape_like(search)}%'
        query = query.filter(
            db.or_(
                User.name.ilike(like, escape='\\'),
                User.email.ilike(like, escape='\\'),
            )
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
    err = _safe_commit('역할 변경 중 오류가 발생했습니다.')
    if err:
        return err

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
    err = _safe_commit('계정 상태 변경 중 오류가 발생했습니다.')
    if err:
        return err

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
    page = min(10000, max(1, request.args.get('page', 1, type=int)))
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
    err = _safe_commit('캐시 갱신 중 오류가 발생했습니다.')
    if err:
        return err
    return success_response(message='파일 트리 캐시가 갱신되었습니다.')


# ---------------------------------------------------------------------------
# System settings
# ---------------------------------------------------------------------------

# Provider → Model mapping (only shown if API key is configured in .env)
_PROVIDER_MODELS = {
    'openai': {
        'label': 'OpenAI',
        'env_key': 'OPENAI_API_KEY',
        'llm_models': [
            {'value': 'gpt-4o-mini', 'label': 'GPT-4o Mini (빠름, 균형)'},
            {'value': 'gpt-4o', 'label': 'GPT-4o (고품질)'},
            {'value': 'gpt-4.1', 'label': 'GPT-4.1 (최고품질)'},
        ],
    },
    'gemini': {
        'label': 'Google Gemini',
        'env_key': 'GEMINI_API_KEY',
        'llm_models': [
            {'value': 'gemini-2.5-pro', 'label': 'Gemini 2.5 Pro (최고품질)'},
            {'value': 'gemini-2.5-flash', 'label': 'Gemini 2.5 Flash (빠름, 추론)'},
            {'value': 'gemini-2.5-flash-lite', 'label': 'Gemini 2.5 Flash Lite (경량)'},
            {'value': 'gemini-2.0-flash', 'label': 'Gemini 2.0 Flash (안정)'},
            {'value': 'gemini-1.5-flash', 'label': 'Gemini 1.5 Flash (균형)'},
            {'value': 'gemini-1.5-pro', 'label': 'Gemini 1.5 Pro (고품질)'},
        ],
    },
    'anthropic': {
        'label': 'Anthropic Claude',
        'env_key': 'ANTHROPIC_API_KEY',
        'llm_models': [
            {'value': 'claude-sonnet-4-20250514', 'label': 'Claude Sonnet 4 (균형)'},
            {'value': 'claude-haiku-4-5-20251001', 'label': 'Claude Haiku 4.5 (빠름, 저렴)'},
            {'value': 'claude-3-5-sonnet-20241022', 'label': 'Claude 3.5 Sonnet (안정)'},
            {'value': 'claude-3-5-haiku-20241022', 'label': 'Claude 3.5 Haiku (경량)'},
        ],
    },
}

# Build combined model list from all providers
_ALL_LLM_MODELS = []
for _pinfo in _PROVIDER_MODELS.values():
    _ALL_LLM_MODELS.extend([m['value'] for m in _pinfo['llm_models']])

# Calculator setting validation rules
_CALC_RATE_KEYS = {
    'calc.np_rate', 'calc.hi_rate', 'calc.ltc_rate',
    'calc.ei_employee', 'calc.ei_employer_base',
    'calc.ei_under_150', 'calc.ei_priority', 'calc.ei_150_to_999', 'calc.ei_over_1000',
    'calc.ia_commute', 'calc.ia_wage_claim', 'calc.ia_asbestos',
}
_CALC_AMOUNT_KEYS = {
    'calc.np_max_income', 'calc.np_min_income',
    'calc.hi_max_income', 'calc.hi_min_income',
    'calc.hi_max_premium', 'calc.hi_min_premium',
    'calc.min_wage_2026', 'calc.min_wage_2025',
}
_CALC_YEAR_KEYS = {'calc.min_wage_year', 'calc.rates_year'}


def _validate_calc_setting(key: str, value: str) -> str | None:
    """Validate a calculator setting value. Returns error message or None."""
    if key in _CALC_RATE_KEYS:
        try:
            v = float(value)
            if not (0.0 <= v <= 1.0):
                return f"'{key}' 요율은 0~1 범위여야 합니다. (입력: {value})"
        except ValueError:
            return f"'{key}' 요율은 숫자여야 합니다. (입력: {value})"
    elif key in _CALC_AMOUNT_KEYS:
        try:
            v = int(value)
            if v < 0:
                return f"'{key}' 금액은 0 이상이어야 합니다. (입력: {value})"
        except ValueError:
            return f"'{key}' 금액은 정수여야 합니다. (입력: {value})"
    elif key in _CALC_YEAR_KEYS:
        try:
            v = int(value)
            if not (2020 <= v <= 2099):
                return f"'{key}' 연도는 2020~2099 범위여야 합니다. (입력: {value})"
        except ValueError:
            return f"'{key}' 연도는 정수여야 합니다. (입력: {value})"
    return None


# Whitelist of allowed setting keys
_ALLOWED_SETTING_KEYS = {
    'llm_answer_model', 'llm_answer_temperature', 'llm_answer_provider',
    'llm_query_model', 'llm_query_provider',
    'llm_context_model', 'llm_context_provider',
    'embedding_model', 'reranker_type',
    # Calculator rates
    'calc.np_rate', 'calc.np_max_income', 'calc.np_min_income',
    'calc.hi_rate', 'calc.hi_max_income', 'calc.hi_min_income',
    'calc.hi_max_premium', 'calc.hi_min_premium',
    'calc.ltc_rate',
    'calc.ei_employee', 'calc.ei_employer_base',
    'calc.ei_under_150', 'calc.ei_priority', 'calc.ei_150_to_999', 'calc.ei_over_1000',
    'calc.ia_commute', 'calc.ia_wage_claim', 'calc.ia_asbestos',
    'calc.min_wage_year', 'calc.min_wage_2026', 'calc.min_wage_2025',
    'calc.rates_updated_at', 'calc.rates_year',
}

# Valid values per key (empty means any value accepted)
_VALID_SETTING_VALUES = {
    'llm_answer_model': _ALL_LLM_MODELS,
    'llm_answer_provider': list(_PROVIDER_MODELS.keys()),
    'llm_query_model': _ALL_LLM_MODELS,
    'llm_query_provider': list(_PROVIDER_MODELS.keys()),
    'llm_context_model': _ALL_LLM_MODELS,
    'llm_context_provider': list(_PROVIDER_MODELS.keys()),
    'embedding_model': [
        'text-embedding-3-small', 'text-embedding-3-large',
        'text-embedding-ada-002',
    ],
    'reranker_type': [
        'pinecone', 'local', 'lightweight',
    ],
}


@v1_bp.route('/admin/settings', methods=['GET'])
@admin_required
def admin_settings_list():
    """List all system settings grouped by category."""
    settings = SystemSetting.query.order_by(
        SystemSetting.category, SystemSetting.key,
    ).all()
    grouped: dict[str, list] = {}
    for s in settings:
        grouped.setdefault(s.category, []).append(s.to_dict())
    return success_response(data=grouped)


@v1_bp.route('/admin/settings', methods=['PUT'])
@admin_required
def admin_settings_update():
    """Bulk-update system settings with validation."""
    from services.settings import invalidate_cache as invalidate_settings_cache
    from services.singletons import (
        invalidate_context_optimizer,
        invalidate_query_enhancer,
        invalidate_reranker,
    )

    data = request.get_json(silent=True)
    if not data or 'settings' not in data:
        return error_response('설정 데이터가 없습니다.', 400)

    updated = []
    for item in data['settings']:
        key = str(item.get('key', '')).strip()
        value = str(item.get('value', '')).strip()

        if key not in _ALLOWED_SETTING_KEYS:
            continue
        if not value:
            return error_response(f"'{key}' 값이 비어 있습니다.", 400)
        if key in _VALID_SETTING_VALUES and value not in _VALID_SETTING_VALUES[key]:
            return error_response(
                f"'{key}'의 값 '{value}'은(는) 허용되지 않습니다. "
                f"허용 값: {_VALID_SETTING_VALUES[key]}", 400,
            )
        # Temperature range check
        if key == 'llm_answer_temperature':
            try:
                temp = float(value)
                if not (0.0 <= temp <= 1.0):
                    return error_response('Temperature는 0.0~1.0 범위여야 합니다.', 400)
            except ValueError:
                return error_response('Temperature는 숫자여야 합니다.', 400)

        # Calculator rate validation
        if key.startswith('calc.'):
            err = _validate_calc_setting(key, value)
            if err:
                return error_response(err, 400)

        setting = SystemSetting.query.filter_by(key=key).first()
        if setting:
            if setting.value != value:
                old_value = setting.value
                setting.value = value
                setting.updated_by = current_user.id
                setting.updated_at = datetime.now(timezone.utc)
                updated.append({'key': key, 'old': old_value, 'new': value})
        else:
            # Upsert: create new setting row for newly introduced keys
            new_setting = SystemSetting(
                key=key,
                value=value,
                description=key,
                category='llm' if 'llm' in key else 'general',
                updated_by=current_user.id,
                updated_at=datetime.now(timezone.utc),
            )
            db.session.add(new_setting)
            updated.append({'key': key, 'old': '', 'new': value})

    if updated:
        _log_action('update_settings', 'system_setting', details={
            'changes': updated,
        })
        db.session.commit()

        # Invalidate caches so new values take effect immediately
        invalidate_settings_cache()
        changed_keys = {u['key'] for u in updated}
        if changed_keys & {'llm_query_model', 'llm_query_provider'}:
            invalidate_query_enhancer()
        if changed_keys & {'llm_context_model', 'llm_context_provider'}:
            invalidate_context_optimizer()
        if 'reranker_type' in changed_keys:
            invalidate_reranker()

    return success_response(
        data={'updated': len(updated)},
        message=f'{len(updated)}개 설정이 변경되었습니다.',
    )


@v1_bp.route('/admin/calculator-rates', methods=['GET'])
@admin_required
def admin_calculator_rates():
    """Return current calculator rates with freshness info."""
    from calculator.rates import get_all_rates, get_rates_freshness

    rates = get_all_rates()
    freshness = get_rates_freshness()

    return success_response(data={
        'rates': rates,
        'freshness': freshness,
    })


@v1_bp.route('/admin/settings/models', methods=['GET'])
@admin_required
def admin_settings_available_models():
    """Return available model options grouped by provider (filtered by .env API keys)."""
    available_providers = []
    for provider_key, provider_info in _PROVIDER_MODELS.items():
        if os.getenv(provider_info['env_key']):
            available_providers.append({
                'key': provider_key,
                'label': provider_info['label'],
                'llm_models': provider_info['llm_models'],
            })

    return success_response(data={
        'providers': available_providers,
        'embedding_models': [
            {'value': 'text-embedding-3-small', 'label': 'Small (1536D, 저렴)'},
            {'value': 'text-embedding-3-large', 'label': 'Large (3072D, 고품질)'},
            {'value': 'text-embedding-ada-002', 'label': 'Ada-002 (1536D, 레거시)'},
        ],
        'reranker_types': [
            {'value': 'pinecone', 'label': 'Pinecone API (bge-reranker-v2-m3)'},
            {'value': 'local', 'label': '로컬 Cross-Encoder (multilingual)'},
            {'value': 'lightweight', 'label': '키워드 기반 (경량)'},
        ],
    })


@v1_bp.route('/admin/settings/test-connection', methods=['POST'])
@admin_required
def admin_test_connection():
    """Test API key connectivity for a given provider."""
    data = request.get_json(silent=True)
    if not data:
        return error_response('요청 데이터가 없습니다.', 400)

    provider = (data.get('provider') or '').strip()

    if provider == 'openai':
        try:
            from services.singletons import get_openai_client
            client = get_openai_client()
            models = client.models.list()
            model_count = sum(1 for _ in models)
            return success_response(
                data={'provider': 'openai', 'status': 'connected', 'models': model_count},
                message=f'OpenAI 연결 성공 (모델 {model_count}개 확인)',
            )
        except Exception:
            logging.exception('OpenAI connection test failed')
            return error_response('OpenAI 연결 실패', 502, details={'provider': 'openai', 'status': 'failed'})

    elif provider == 'gemini':
        try:
            from services.singletons import get_gemini_client
            client = get_gemini_client()
            models = list(client.models.list())
            model_count = len(models)
            return success_response(
                data={'provider': 'gemini', 'status': 'connected', 'models': model_count},
                message=f'Gemini 연결 성공 (모델 {model_count}개 확인)',
            )
        except Exception:
            logging.exception('Gemini connection test failed')
            return error_response('Gemini 연결 실패', 502, details={'provider': 'gemini', 'status': 'failed'})

    elif provider == 'anthropic':
        try:
            from services.singletons import get_anthropic_client
            client = get_anthropic_client()
            # Simple connectivity check: list models
            models = client.models.list()
            model_count = len(models.data)
            return success_response(
                data={'provider': 'anthropic', 'status': 'connected', 'models': model_count},
                message=f'Anthropic 연결 성공 (모델 {model_count}개 확인)',
            )
        except Exception:
            logging.exception('Anthropic connection test failed')
            return error_response('Anthropic 연결 실패', 502, details={'provider': 'anthropic', 'status': 'failed'})

    elif provider == 'pinecone':
        try:
            from services.singletons import get_uploader
            uploader = get_uploader()
            stats = uploader.get_stats()
            vec_count = stats.get('total_vector_count', 0)
            return success_response(
                data={'provider': 'pinecone', 'status': 'connected', 'vectors': vec_count},
                message=f'Pinecone 연결 성공 (벡터: {vec_count:,}개)',
            )
        except Exception:
            logging.exception('Pinecone connection test failed')
            return error_response('Pinecone 연결 실패', 502, details={'provider': 'pinecone', 'status': 'failed'})

    else:
        return error_response(f"알 수 없는 제공자: {provider}", 400)
