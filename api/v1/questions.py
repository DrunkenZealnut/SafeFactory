"""Shared questions & likes API endpoints."""

import hashlib
import logging
from datetime import datetime, timedelta, timezone

from flask import request
from flask_login import current_user, login_required
from sqlalchemy.exc import IntegrityError

from api.response import error_response, success_response
from api.v1 import v1_bp
from models import db, SharedQuestion, QuestionLike
from api import rate_limit


@v1_bp.route('/questions/share', methods=['POST'])
@login_required
@rate_limit("30 per minute")
def api_question_share():
    """Share a question publicly after receiving AI answer."""
    try:
        data = request.get_json(silent=True)
        if not data:
            return error_response('요청 데이터가 없습니다.', 400)

        query = (data.get('query') or '').strip()
        if not query:
            return error_response('질문 내용은 필수입니다.', 400)
        if len(query) > 500:
            return error_response('질문은 500자 이하로 입력해주세요.', 400)

        namespace = (data.get('namespace') or '').strip()
        answer_preview = (data.get('answer_preview') or '').strip()
        if answer_preview:
            answer_preview = answer_preview[:300]
        answer_full = (data.get('answer_full') or '').strip() or None
        if answer_full:
            answer_full = answer_full[:10000]

        # Daily share limit
        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0,
        )
        today_count = db.session.query(SharedQuestion).filter(
            SharedQuestion.user_id == current_user.id,
            SharedQuestion.created_at >= today_start,
        ).count()
        if today_count >= SharedQuestion.DAILY_SHARE_LIMIT:
            return error_response(
                f'하루 최대 {SharedQuestion.DAILY_SHARE_LIMIT}개까지 공유할 수 있습니다.', 400
            )

        # Duplicate check via query_hash
        query_hash = hashlib.md5(query.lower().encode('utf-8')).hexdigest()
        existing = db.session.query(SharedQuestion).filter_by(
            user_id=current_user.id, query_hash=query_hash,
        ).first()
        if existing:
            return success_response(
                data=existing.to_dict(liked_by_me=False),
                message='이미 공유한 질문입니다.',
            )

        sq = SharedQuestion(
            user_id=current_user.id,
            query=query[:500],
            query_hash=query_hash,
            namespace=namespace,
            answer_preview=answer_preview,
            answer_full=answer_full,
        )
        db.session.add(sq)
        db.session.commit()
        return success_response(
            data=sq.to_dict(liked_by_me=False),
            message='질문이 공유되었습니다.',
        )

    except Exception:
        db.session.rollback()
        logging.exception('[Question] Share failed')
        return error_response('질문 공유 중 오류가 발생했습니다.', 500)


@v1_bp.route('/questions/wordcloud', methods=['GET'])
@rate_limit("30 per minute")
def api_question_wordcloud():
    """Get keyword cloud data extracted from shared questions."""
    try:
        from services.keyword_extractor import extract_keywords

        namespace = request.args.get('namespace', '').strip()
        period = request.args.get('period', 'all').strip()
        limit = min(max(1, request.args.get('limit', 80, type=int)), 100)

        q = db.session.query(
            SharedQuestion.query, SharedQuestion.like_count,
        ).filter_by(is_hidden=False)

        if namespace:
            q = q.filter(SharedQuestion.namespace == namespace)

        if period == '7d':
            since = datetime.now(timezone.utc) - timedelta(days=7)
            q = q.filter(SharedQuestion.created_at >= since)
        elif period == '30d':
            since = datetime.now(timezone.utc) - timedelta(days=30)
            q = q.filter(SharedQuestion.created_at >= since)

        rows = q.all()
        keywords = extract_keywords(
            [(row.query, row.like_count) for row in rows],
            limit=limit,
        )

        return success_response(data={
            'keywords': keywords,
            'total_questions': len(rows),
        })
    except Exception:
        logging.exception('[Question] Wordcloud failed')
        return error_response('워드 클라우드 데이터 조회 중 오류가 발생했습니다.', 500)


@v1_bp.route('/questions/popular', methods=['GET'])
@rate_limit("60 per minute")
def api_question_popular():
    """Get popular shared questions, with optional pagination and sorting."""
    try:
        namespace = request.args.get('namespace', '').strip()
        sort = request.args.get('sort', 'likes').strip()
        include_answer = request.args.get('include_answer', '0') == '1'
        page = request.args.get('page', None, type=int)

        q = db.session.query(SharedQuestion).filter_by(is_hidden=False)
        if namespace:
            q = q.filter_by(namespace=namespace)

        if sort == 'recent':
            q = q.order_by(SharedQuestion.created_at.desc())
        else:
            q = q.order_by(
                SharedQuestion.like_count.desc(),
                SharedQuestion.created_at.desc(),
            )

        # Paginated mode (when page param is provided)
        if page is not None:
            page = max(1, page)
            per_page = min(max(1, request.args.get('per_page', 20, type=int)), 50)
            pagination = q.paginate(page=page, per_page=per_page, error_out=False)
            questions = pagination.items
        else:
            # Legacy mode (limit only)
            limit = min(max(1, request.args.get('limit', 10, type=int)), 20)
            questions = q.limit(limit).all()

        # Check which ones the current user has liked
        liked_ids = set()
        if hasattr(current_user, 'id') and current_user.is_authenticated:
            qids = [sq.id for sq in questions]
            if qids:
                liked = QuestionLike.query.filter(
                    QuestionLike.user_id == current_user.id,
                    QuestionLike.question_id.in_(qids),
                ).all()
                liked_ids = {lk.question_id for lk in liked}

        data = {
            'questions': [
                sq.to_dict(
                    liked_by_me=(sq.id in liked_ids),
                    include_answer=include_answer,
                )
                for sq in questions
            ],
        }
        if page is not None:
            data['total'] = pagination.total
            data['page'] = pagination.page
            data['per_page'] = pagination.per_page
            data['pages'] = pagination.pages

        return success_response(data=data)
    except Exception:
        logging.exception('[Question] Popular list failed')
        return error_response('인기 질문 조회 중 오류가 발생했습니다.', 500)


@v1_bp.route('/questions/<int:question_id>/like', methods=['POST'])
@login_required
@rate_limit("60 per minute")
def api_question_toggle_like(question_id):
    """Toggle like on a shared question (PostLike pattern)."""
    sq = db.session.query(SharedQuestion).filter_by(id=question_id, is_hidden=False).first()
    if not sq:
        return error_response('질문을 찾을 수 없습니다.', 404)

    try:
        like = QuestionLike(question_id=question_id, user_id=current_user.id)
        db.session.add(like)
        db.session.flush()
        sq.like_count = QuestionLike.query.filter_by(question_id=question_id).count()
        db.session.commit()
        return success_response(data={'liked': True, 'like_count': sq.like_count})
    except IntegrityError:
        db.session.rollback()
        QuestionLike.query.filter_by(
            question_id=question_id, user_id=current_user.id,
        ).delete()
        sq = db.session.get(SharedQuestion, question_id)
        sq.like_count = QuestionLike.query.filter_by(question_id=question_id).count()
        db.session.commit()
        return success_response(data={'liked': False, 'like_count': sq.like_count})


@v1_bp.route('/questions/<int:question_id>', methods=['DELETE'])
@login_required
def api_question_delete(question_id):
    """Delete own shared question."""
    try:
        sq = db.session.query(SharedQuestion).filter_by(
            id=question_id, user_id=current_user.id,
        ).first()
        if not sq:
            return error_response('질문을 찾을 수 없습니다.', 404)

        db.session.delete(sq)
        db.session.commit()
        return success_response(message='질문이 삭제되었습니다.')
    except Exception:
        db.session.rollback()
        logging.exception('[Question] Delete failed')
        return error_response('질문 삭제 중 오류가 발생했습니다.', 500)


@v1_bp.route('/questions/my', methods=['GET'])
@login_required
@rate_limit("30 per minute")
def api_question_my():
    """List questions shared by the current user."""
    try:
        page = max(1, request.args.get('page', 1, type=int))
        per_page = min(max(1, request.args.get('per_page', 20, type=int)), 50)

        q = db.session.query(SharedQuestion).filter_by(user_id=current_user.id)
        q = q.order_by(SharedQuestion.created_at.desc())

        pagination = q.paginate(page=page, per_page=per_page, error_out=False)

        include_answer = request.args.get('include_answer', '0') == '1'
        return success_response(data={
            'items': [sq.to_dict(liked_by_me=False, include_answer=include_answer) for sq in pagination.items],
            'total': pagination.total,
            'page': pagination.page,
            'per_page': pagination.per_page,
            'pages': pagination.pages,
        })
    except Exception:
        logging.exception('[Question] My list failed')
        return error_response('내 질문 목록 조회 중 오류가 발생했습니다.', 500)
