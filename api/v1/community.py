"""Community board API endpoints."""

import logging
import os
import uuid
from datetime import datetime, timedelta, timezone

from flask import request
from flask_login import current_user, login_required
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import joinedload

from api.response import error_response, escape_like, success_response
from api.v1 import v1_bp
from models import Category, Comment, Post, PostAttachment, PostLike, _safe_url, db

# ---------------------------------------------------------------------------
# File upload configuration
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPLOAD_DIR = os.path.join(_BASE_DIR, 'static', 'uploads', 'community')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'pdf', 'docx', 'xlsx', 'hwp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


def _allowed_file(filename):
    if '.' not in filename:
        return False
    # Reject double extensions (e.g. malicious.php.jpg)
    parts = filename.rsplit('/', 1)[-1].split('.')
    extensions = [p.lower() for p in parts[1:]]
    return len(extensions) == 1 and extensions[0] in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------

@v1_bp.route('/community/categories', methods=['GET'])
def api_community_categories():
    """List active categories."""
    categories = Category.query.filter_by(is_active=True).order_by(Category.sort_order).all()
    return success_response(data=[c.to_dict() for c in categories])


# ---------------------------------------------------------------------------
# Posts – list / detail / create / update / delete
# ---------------------------------------------------------------------------

@v1_bp.route('/community/posts', methods=['GET'])
def api_community_posts():
    """List posts with pagination, search, category filter, and sort."""
    try:
        page = min(10000, max(1, request.args.get('page', 1, type=int)))
        per_page = min(50, max(1, request.args.get('per_page', 20, type=int)))
        category_slug = request.args.get('category', '').strip()
        search_query = request.args.get('search', '').strip()
        sort = request.args.get('sort', 'latest')

        # Subqueries to avoid N+1 per-post count queries
        like_counts = (
            db.session.query(
                PostLike.post_id,
                db.func.count(PostLike.id).label('cnt'),
            )
            .group_by(PostLike.post_id)
            .subquery()
        )
        comment_counts = (
            db.session.query(
                Comment.post_id,
                db.func.count(Comment.id).label('cnt'),
            )
            .filter(Comment.is_deleted.is_(False))
            .group_by(Comment.post_id)
            .subquery()
        )

        query = (
            db.session.query(
                Post,
                db.func.coalesce(like_counts.c.cnt, 0).label('like_count'),
                db.func.coalesce(comment_counts.c.cnt, 0).label('comment_count'),
            )
            .outerjoin(like_counts, Post.id == like_counts.c.post_id)
            .outerjoin(comment_counts, Post.id == comment_counts.c.post_id)
            .filter(Post.is_deleted.is_(False))  # noqa: E712
        )

        if category_slug:
            cat = Category.query.filter_by(slug=category_slug, is_active=True).first()
            if cat:
                query = query.filter(Post.category_id == cat.id)

        if search_query:
            like_pattern = f'%{escape_like(search_query)}%'
            query = query.filter(
                db.or_(
                    Post.title.ilike(like_pattern, escape='\\'),
                    Post.content.ilike(like_pattern, escape='\\'),
                )
            )

        # Pinned first, then sort
        if sort == 'popular':
            query = query.order_by(
                Post.is_pinned.desc(),
                like_counts.c.cnt.desc().nullslast(),
                Post.created_at.desc(),
            )
        elif sort == 'views':
            query = query.order_by(
                Post.is_pinned.desc(), Post.view_count.desc(), Post.created_at.desc(),
            )
        else:
            query = query.order_by(Post.is_pinned.desc(), Post.created_at.desc())

        pagination = query.paginate(page=page, per_page=per_page, error_out=False)

        return success_response(data={
            'posts': [
                p.to_dict(_like_count=lc, _comment_count=cc)
                for p, lc, cc in pagination.items
            ],
            'total': pagination.total,
            'pages': pagination.pages,
            'current_page': page,
        })
    except Exception:
        logging.exception('Community list failed')
        return error_response('게시글 목록 조회 중 오류가 발생했습니다.', 500)


@v1_bp.route('/community/posts/popular', methods=['GET'])
def api_community_popular_posts():
    """Top 10 popular posts by like count in the last 7 days."""
    since = datetime.now(timezone.utc) - timedelta(days=7)

    like_sub = (
        db.session.query(
            PostLike.post_id,
            db.func.count(PostLike.id).label('cnt'),
        )
        .filter(PostLike.created_at >= since)
        .group_by(PostLike.post_id)
        .subquery()
    )

    # Total like counts (all-time) for to_dict display
    all_likes = (
        db.session.query(
            PostLike.post_id,
            db.func.count(PostLike.id).label('cnt'),
        )
        .group_by(PostLike.post_id)
        .subquery()
    )

    comment_counts = (
        db.session.query(
            Comment.post_id,
            db.func.count(Comment.id).label('cnt'),
        )
        .filter(Comment.is_deleted.is_(False))
        .group_by(Comment.post_id)
        .subquery()
    )

    rows = (
        db.session.query(
            Post,
            db.func.coalesce(all_likes.c.cnt, 0).label('like_count'),
            db.func.coalesce(comment_counts.c.cnt, 0).label('comment_count'),
        )
        .join(like_sub, Post.id == like_sub.c.post_id)
        .outerjoin(all_likes, Post.id == all_likes.c.post_id)
        .outerjoin(comment_counts, Post.id == comment_counts.c.post_id)
        .filter(Post.is_deleted.is_(False))
        .order_by(like_sub.c.cnt.desc())
        .limit(10)
        .all()
    )

    return success_response(data=[
        p.to_dict(_like_count=lc, _comment_count=cc) for p, lc, cc in rows
    ])


@v1_bp.route('/community/posts/<int:post_id>', methods=['GET'])
def api_community_post_detail(post_id):
    """Get post detail with comments. Increments view count."""
    post = Post.query.filter_by(id=post_id, is_deleted=False).first()
    if not post:
        return error_response('게시글을 찾을 수 없습니다.', 404)

    # Atomic view-count increment (non-critical, suppress errors)
    try:
        Post.query.filter_by(id=post_id).update({Post.view_count: Post.view_count + 1})
        db.session.commit()
        db.session.refresh(post)
    except Exception:
        db.session.rollback()

    data = post.to_dict(include_content=True)

    # Load all comments in a single query, build tree in memory
    all_comments = (
        Comment.query
        .filter_by(post_id=post_id, is_deleted=False)
        .options(joinedload(Comment.author))
        .order_by(Comment.created_at.asc())
        .all()
    )
    comment_map = {}
    for c in all_comments:
        cd = {
            'id': c.id,
            'post_id': c.post_id,
            'author': {
                'id': c.author.id,
                'name': c.author.name,
                'profile_image': _safe_url(c.author.profile_image),
            } if c.author else None,
            'parent_id': c.parent_id,
            'content': c.content,
            'is_deleted': c.is_deleted,
            'created_at': c.created_at.isoformat() if c.created_at else None,
            'updated_at': c.updated_at.isoformat() if c.updated_at else None,
            'replies': [],
        }
        comment_map[c.id] = cd

    top_comments = []
    for c in all_comments:
        cd = comment_map[c.id]
        if c.parent_id and c.parent_id in comment_map:
            comment_map[c.parent_id]['replies'].append(cd)
        else:
            top_comments.append(cd)

    data['comments'] = top_comments

    data['liked_by_me'] = False
    if current_user.is_authenticated:
        data['liked_by_me'] = PostLike.query.filter_by(
            post_id=post_id, user_id=current_user.id,
        ).first() is not None

    return success_response(data=data)


@v1_bp.route('/community/posts', methods=['POST'])
@login_required
def api_community_create_post():
    """Create a new post."""
    try:
        data = request.get_json(silent=True)
        if not data:
            return error_response('요청 데이터가 없습니다.', 400)

        title = (data.get('title') or '').strip()
        content = (data.get('content') or '').strip()
        category_id = data.get('category_id')

        if not title or len(title) < 2:
            return error_response('제목은 2자 이상이어야 합니다.', 400)
        if len(title) > 200:
            return error_response('제목은 200자 이하여야 합니다.', 400)
        if not content:
            return error_response('내용을 입력해주세요.', 400)
        if len(content) > 50000:
            return error_response('내용은 50,000자 이하여야 합니다.', 400)

        category = db.session.get(Category, category_id)
        if not category or not category.is_active:
            return error_response('유효하지 않은 카테고리입니다.', 400)
        if category.slug == 'notice' and current_user.role != 'admin':
            return error_response('공지사항은 관리자만 작성할 수 있습니다.', 403)

        post = Post(
            category_id=category_id,
            user_id=current_user.id,
            title=title,
            content=content,
        )
        db.session.add(post)
        db.session.commit()

        return success_response(
            data=post.to_dict(include_content=True),
            message='게시글이 작성되었습니다.',
        )
    except Exception:
        db.session.rollback()
        logging.exception('Post creation failed')
        return error_response('게시글 작성 중 오류가 발생했습니다.', 500)


@v1_bp.route('/community/posts/<int:post_id>', methods=['PUT'])
@login_required
def api_community_update_post(post_id):
    """Update an existing post (owner only)."""
    post = Post.query.filter_by(id=post_id, is_deleted=False).first()
    if not post:
        return error_response('게시글을 찾을 수 없습니다.', 404)
    if post.user_id != current_user.id:
        return error_response('수정 권한이 없습니다.', 403)

    data = request.get_json(silent=True)
    if not data:
        return error_response('요청 데이터가 없습니다.', 400)

    title = (data.get('title') or '').strip()
    content = (data.get('content') or '').strip()

    if title:
        if len(title) < 2 or len(title) > 200:
            return error_response('제목은 2~200자여야 합니다.', 400)
        post.title = title
    if content:
        if len(content) > 50000:
            return error_response('내용은 50,000자 이하여야 합니다.', 400)
        post.content = content
    if 'category_id' in data:
        cat = db.session.get(Category, data['category_id'])
        if not cat or not cat.is_active:
            return error_response('유효하지 않은 카테고리입니다.', 400)
        if cat.slug == 'notice' and current_user.role != 'admin':
            return error_response('공지사항은 관리자만 작성할 수 있습니다.', 403)
        post.category_id = data['category_id']

    db.session.commit()
    return success_response(
        data=post.to_dict(include_content=True),
        message='게시글이 수정되었습니다.',
    )


@v1_bp.route('/community/posts/<int:post_id>', methods=['DELETE'])
@login_required
def api_community_delete_post(post_id):
    """Soft-delete a post (owner or admin)."""
    post = Post.query.filter_by(id=post_id, is_deleted=False).first()
    if not post:
        return error_response('게시글을 찾을 수 없습니다.', 404)
    if post.user_id != current_user.id and current_user.role != 'admin':
        return error_response('삭제 권한이 없습니다.', 403)

    post.is_deleted = True
    db.session.commit()
    return success_response(message='게시글이 삭제되었습니다.')


# ---------------------------------------------------------------------------
# Comments
# ---------------------------------------------------------------------------

@v1_bp.route('/community/posts/<int:post_id>/comments', methods=['POST'])
@login_required
def api_community_create_comment(post_id):
    """Add a comment to a post."""
    post = Post.query.filter_by(id=post_id, is_deleted=False).first()
    if not post:
        return error_response('게시글을 찾을 수 없습니다.', 404)

    data = request.get_json(silent=True)
    content = (data.get('content') or '').strip() if data else ''
    parent_id = data.get('parent_id') if data else None

    if not content:
        return error_response('댓글 내용을 입력해주세요.', 400)
    if len(content) > 2000:
        return error_response('댓글은 2000자 이하여야 합니다.', 400)

    if parent_id:
        parent = Comment.query.filter_by(
            id=parent_id, post_id=post_id, is_deleted=False,
        ).first()
        if not parent:
            return error_response('상위 댓글을 찾을 수 없습니다.', 404)

    comment = Comment(
        post_id=post_id,
        user_id=current_user.id,
        parent_id=parent_id,
        content=content,
    )
    db.session.add(comment)
    db.session.commit()
    return success_response(data=comment.to_dict(include_replies=False), message='댓글이 작성되었습니다.')


@v1_bp.route('/community/comments/<int:comment_id>', methods=['DELETE'])
@login_required
def api_community_delete_comment(comment_id):
    """Soft-delete a comment (owner or admin)."""
    comment = Comment.query.filter_by(id=comment_id, is_deleted=False).first()
    if not comment:
        return error_response('댓글을 찾을 수 없습니다.', 404)
    if comment.user_id != current_user.id and current_user.role != 'admin':
        return error_response('삭제 권한이 없습니다.', 403)

    comment.is_deleted = True
    db.session.commit()
    return success_response(message='댓글이 삭제되었습니다.')


# ---------------------------------------------------------------------------
# Likes
# ---------------------------------------------------------------------------

@v1_bp.route('/community/posts/<int:post_id>/like', methods=['POST'])
@login_required
def api_community_toggle_like(post_id):
    """Toggle like on a post."""
    post = Post.query.filter_by(id=post_id, is_deleted=False).first()
    if not post:
        return error_response('게시글을 찾을 수 없습니다.', 404)

    # Use INSERT + IntegrityError to avoid race condition between
    # filter_by().first() and add()/delete() on concurrent requests.
    try:
        like = PostLike(post_id=post_id, user_id=current_user.id)
        db.session.add(like)
        db.session.commit()
        db.session.refresh(post)
        return success_response(data={'liked': True, 'like_count': post.like_count})
    except IntegrityError:
        db.session.rollback()
        PostLike.query.filter_by(post_id=post_id, user_id=current_user.id).delete()
        db.session.commit()
        db.session.refresh(post)
        return success_response(data={'liked': False, 'like_count': post.like_count})


# ---------------------------------------------------------------------------
# File attachments
# ---------------------------------------------------------------------------

@v1_bp.route('/community/posts/<int:post_id>/attachments', methods=['POST'])
@login_required
def api_community_upload_attachment(post_id):
    """Upload a file attachment to a post."""
    post = Post.query.filter_by(id=post_id, is_deleted=False).first()
    if not post:
        return error_response('게시글을 찾을 수 없습니다.', 404)
    if post.user_id != current_user.id:
        return error_response('권한이 없습니다.', 403)

    if 'file' not in request.files:
        return error_response('파일이 없습니다.', 400)

    file = request.files['file']
    if not file.filename:
        return error_response('파일명이 없습니다.', 400)
    if not _allowed_file(file.filename):
        return error_response(
            f'허용되지 않는 파일 형식입니다. ({", ".join(sorted(ALLOWED_EXTENSIONS))})', 400,
        )

    # Check file size
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    if size > MAX_FILE_SIZE:
        return error_response('파일 크기는 10MB 이하여야 합니다.', 400)

    ext = file.filename.rsplit('.', 1)[1].lower()
    safe_filename = f'{uuid.uuid4().hex}.{ext}'

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    file.save(file_path)

    attachment = PostAttachment(
        post_id=post_id,
        filename=safe_filename,
        original_filename=file.filename,
        file_size=size,
        mime_type=file.content_type or 'application/octet-stream',
    )
    try:
        db.session.add(attachment)
        db.session.commit()
    except Exception:
        os.remove(file_path)
        db.session.rollback()
        logging.exception('Attachment DB commit failed')
        return error_response('첨부파일 저장 중 오류가 발생했습니다.', 500)

    return success_response(data=attachment.to_dict(), message='파일이 업로드되었습니다.')
