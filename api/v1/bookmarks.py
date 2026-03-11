"""User bookmark CRUD endpoints."""

import logging

from flask import request
from flask_login import current_user, login_required

from api.v1 import v1_bp
from api.response import success_response, error_response
from api import rate_limit
from models import db, UserBookmark


@v1_bp.route('/bookmarks', methods=['POST'])
@login_required
@rate_limit("30 per minute")
def api_bookmark_create():
    """Add a document to user's bookmarks."""
    try:
        data = request.get_json(silent=True)
        if not data:
            return error_response('요청 데이터가 없습니다.', 400)

        source_file = (data.get('source_file') or '').strip()
        if not source_file:
            return error_response('source_file은 필수입니다.', 400)

        namespace = (data.get('namespace') or '').strip()
        title = (data.get('title') or '').strip()
        if not title:
            title = source_file.split('/')[-1]  # fallback to filename
        file_type = (data.get('file_type') or '').strip() or None

        # Check per-user limit
        count = UserBookmark.query.filter_by(user_id=current_user.id).count()
        if count >= UserBookmark.MAX_PER_USER:
            return error_response(
                f'북마크는 최대 {UserBookmark.MAX_PER_USER}개까지 저장할 수 있습니다.', 400
            )

        # Check duplicate
        existing = UserBookmark.query.filter_by(
            user_id=current_user.id, source_file=source_file
        ).first()
        if existing:
            return success_response(data=existing.to_dict(), message='이미 저장된 자료입니다.')

        bookmark = UserBookmark(
            user_id=current_user.id,
            source_file=source_file,
            namespace=namespace,
            title=title[:300],
            file_type=file_type,
        )
        db.session.add(bookmark)
        db.session.commit()
        return success_response(data=bookmark.to_dict(), message='자료가 저장되었습니다.')

    except Exception:
        db.session.rollback()
        logging.exception('[Bookmark] Create failed')
        return error_response('자료 저장 중 오류가 발생했습니다.', 500)


@v1_bp.route('/bookmarks', methods=['GET'])
@login_required
@rate_limit("30 per minute")
def api_bookmark_list():
    """Return paginated bookmarks for the current user."""
    try:
        page = max(1, request.args.get('page', 1, type=int))
        per_page = min(max(1, request.args.get('per_page', 20, type=int)), 50)

        q = UserBookmark.query.filter_by(user_id=current_user.id)

        namespace = request.args.get('namespace', '').strip()
        if namespace:
            q = q.filter_by(namespace=namespace)

        sort = request.args.get('sort', 'newest').strip()
        if sort == 'title':
            q = q.order_by(UserBookmark.title.asc())
        else:
            q = q.order_by(UserBookmark.created_at.desc())

        pagination = q.paginate(page=page, per_page=per_page, error_out=False)

        return success_response(data={
            'items': [b.to_dict() for b in pagination.items],
            'total': pagination.total,
            'page': pagination.page,
            'per_page': pagination.per_page,
            'pages': pagination.pages,
        })
    except Exception:
        logging.exception('[Bookmark] List failed')
        return error_response('자료 목록 조회 중 오류가 발생했습니다.', 500)


@v1_bp.route('/bookmarks/<int:bookmark_id>', methods=['DELETE'])
@login_required
def api_bookmark_delete(bookmark_id):
    """Delete a single bookmark owned by the current user."""
    try:
        record = UserBookmark.query.filter_by(
            id=bookmark_id, user_id=current_user.id
        ).first()
        if not record:
            return error_response('자료를 찾을 수 없습니다.', 404)

        db.session.delete(record)
        db.session.commit()
        return success_response(message='자료가 삭제되었습니다.')
    except Exception:
        db.session.rollback()
        logging.exception('[Bookmark] Delete failed')
        return error_response('자료 삭제 중 오류가 발생했습니다.', 500)


@v1_bp.route('/bookmarks', methods=['DELETE'])
@login_required
def api_bookmark_delete_all():
    """Delete all bookmarks for the current user."""
    try:
        deleted = UserBookmark.query.filter_by(
            user_id=current_user.id
        ).delete()
        db.session.commit()
        return success_response(
            message='전체 자료가 삭제되었습니다.',
            data={'deleted_count': deleted},
        )
    except Exception:
        db.session.rollback()
        logging.exception('[Bookmark] Delete all failed')
        return error_response('자료 삭제 중 오류가 발생했습니다.', 500)


@v1_bp.route('/bookmarks/check-batch', methods=['POST'])
@login_required
def api_bookmark_check_batch():
    """Check bookmark status for multiple source_files at once."""
    try:
        data = request.get_json(silent=True)
        if not data:
            return error_response('요청 데이터가 없습니다.', 400)

        source_files = data.get('source_files', [])
        if not source_files or not isinstance(source_files, list):
            return error_response('source_files 배열이 필요합니다.', 400)

        # Limit batch size
        source_files = source_files[:100]

        bookmarked = UserBookmark.query.filter(
            UserBookmark.user_id == current_user.id,
            UserBookmark.source_file.in_(source_files)
        ).all()

        bookmarked_map = {b.source_file: b.id for b in bookmarked}

        return success_response(data={'bookmarked': bookmarked_map})
    except Exception:
        logging.exception('[Bookmark] Check batch failed')
        return error_response('북마크 확인 중 오류가 발생했습니다.', 500)
