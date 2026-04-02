"""Answer feedback API endpoints."""

import hashlib
import logging

from flask import request
from flask_login import current_user, login_required

from api.response import error_response, success_response
from api.v1 import v1_bp
from models import AnswerFeedback, db


@v1_bp.route('/feedback', methods=['POST'])
@login_required
def submit_feedback():
    """Submit feedback on an AI answer."""
    data = request.get_json(silent=True)
    if not data:
        return error_response('요청 데이터가 없습니다.', 400)

    query = (data.get('query') or '').strip()
    answer = (data.get('answer') or '').strip()
    feedback_type = (data.get('feedback_type') or '').strip()

    if not query or not answer:
        return error_response('질문과 답변은 필수입니다.', 400)
    if feedback_type not in AnswerFeedback.FEEDBACK_TYPES:
        return error_response(f'유효하지 않은 피드백 타입: {feedback_type}', 400)

    namespace = data.get('namespace', '')
    query_hash = hashlib.md5(f"{query}{namespace}".encode()).hexdigest()

    existing = AnswerFeedback.query.filter_by(
        user_id=current_user.id, query_hash=query_hash,
    ).first()
    if existing:
        return error_response('이미 이 답변에 피드백을 제출하셨습니다.', 409)

    fb = AnswerFeedback(
        user_id=current_user.id,
        query=query,
        query_hash=query_hash,
        answer=answer,
        namespace=namespace,
        source_count=data.get('source_count', 0),
        confidence_score=data.get('confidence_score'),
        feedback_type=feedback_type,
        comment=(data.get('comment') or '').strip() or None,
    )
    db.session.add(fb)
    db.session.commit()
    logging.info("[Feedback] User %d submitted %s for ns=%s", current_user.id, feedback_type, namespace)

    return success_response(data={'id': fb.id}, message='피드백이 제출되었습니다.')
