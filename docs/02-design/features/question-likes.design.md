# Design: 질문 좋아요 기능 (Question Likes)

> **Feature**: question-likes
> **Date**: 2026-03-11
> **Project**: SafeFactory
> **Version**: 1.0
> **Plan Reference**: `docs/01-plan/features/question-likes.plan.md`

---

## 1. Data Model

### 1.1 SharedQuestion Model (`models.py`)

SearchHistory 모델 뒤, UserBookmark 모델 앞에 추가.

```python
class SharedQuestion(db.Model):
    """Publicly shared AI questions that other users can like."""

    __tablename__ = 'shared_questions'
    __table_args__ = (
        db.UniqueConstraint('user_id', 'query_hash', name='uq_user_query_hash'),
        db.Index('ix_shared_questions_namespace_likes', 'namespace', 'like_count'),
        db.Index('ix_shared_questions_user', 'user_id'),
    )

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False,
    )
    query = db.Column(db.Text, nullable=False)
    query_hash = db.Column(db.String(32), nullable=False)  # MD5 of normalized query
    namespace = db.Column(db.String(100), nullable=False, default='')
    answer_preview = db.Column(db.String(300), nullable=True)
    like_count = db.Column(db.Integer, nullable=False, default=0)
    is_hidden = db.Column(db.Boolean, nullable=False, default=False)
    created_at = db.Column(
        db.DateTime, nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    user = db.relationship('User', backref='shared_questions')
    likes = db.relationship('QuestionLike', backref='question', cascade='all, delete-orphan')

    DAILY_SHARE_LIMIT = 10

    def to_dict(self, liked_by_me=False):
        return {
            'id': self.id,
            'query': self.query,
            'namespace': self.namespace,
            'answer_preview': self.answer_preview,
            'like_count': self.like_count,
            'author': {
                'id': self.user.id,
                'name': self.user.name,
            } if self.user else None,
            'liked_by_me': liked_by_me,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }
```

**설계 결정**:
- `query_hash`: 동일 사용자가 같은 질문을 중복 공유 방지 (MD5 of `query.strip().lower()`)
- `like_count` 캐시 컬럼: 인기 질문 정렬 시 매번 COUNT 쿼리 대신 인덱스 활용
- `is_hidden`: 관리자 숨김 처리 (Phase 1 간단 관리)
- `answer_preview`: 300자까지 저장 (검색 결과에서 미리보기용)
- `to_dict(liked_by_me)`: 현재 사용자의 좋아요 여부를 파라미터로 전달

### 1.2 QuestionLike Model (`models.py`)

SharedQuestion 바로 뒤에 추가.

```python
class QuestionLike(db.Model):
    """Like on a shared question (unique per user+question)."""

    __tablename__ = 'question_likes'
    __table_args__ = (
        db.UniqueConstraint('question_id', 'user_id', name='uq_question_user_like'),
    )

    id = db.Column(db.Integer, primary_key=True)
    question_id = db.Column(
        db.Integer, db.ForeignKey('shared_questions.id', ondelete='CASCADE'), nullable=False,
    )
    user_id = db.Column(
        db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False,
    )
    created_at = db.Column(
        db.DateTime, nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    user = db.relationship('User')
```

**설계 결정**: `PostLike` 패턴과 동일 — UniqueConstraint로 중복 방지, INSERT + IntegrityError로 toggle 구현.

---

## 2. API Endpoints (`api/v1/questions.py`)

### 2.1 신규 파일 생성

`api/v1/questions.py` — 모든 엔드포인트는 `@login_required` 필수 (popular 제외).

```python
import hashlib
import logging
from datetime import datetime, timezone, timedelta

from flask import request
from flask_login import current_user, login_required
from sqlalchemy.exc import IntegrityError

from api.response import error_response, success_response
from api.v1 import v1_bp
from models import db, SharedQuestion, QuestionLike
from services.rate_limiter import rate_limit
```

### 2.2 POST `/api/v1/questions/share` — 질문 공유

```python
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

        # Daily share limit
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        today_count = SharedQuestion.query.filter(
            SharedQuestion.user_id == current_user.id,
            SharedQuestion.created_at >= today_start,
        ).count()
        if today_count >= SharedQuestion.DAILY_SHARE_LIMIT:
            return error_response(
                f'하루 최대 {SharedQuestion.DAILY_SHARE_LIMIT}개까지 공유할 수 있습니다.', 400
            )

        # Duplicate check via query_hash
        query_hash = hashlib.md5(query.lower().encode('utf-8')).hexdigest()
        existing = SharedQuestion.query.filter_by(
            user_id=current_user.id, query_hash=query_hash,
        ).first()
        if existing:
            return success_response(data=existing.to_dict(liked_by_me=False), message='이미 공유한 질문입니다.')

        sq = SharedQuestion(
            user_id=current_user.id,
            query=query[:500],
            query_hash=query_hash,
            namespace=namespace,
            answer_preview=answer_preview,
        )
        db.session.add(sq)
        db.session.commit()
        return success_response(data=sq.to_dict(liked_by_me=False), message='질문이 공유되었습니다.')

    except Exception:
        db.session.rollback()
        logging.exception('[Question] Share failed')
        return error_response('질문 공유 중 오류가 발생했습니다.', 500)
```

### 2.3 GET `/api/v1/questions/popular` — 인기 질문 목록

```python
@v1_bp.route('/questions/popular', methods=['GET'])
@rate_limit("60 per minute")
def api_question_popular():
    """Get popular shared questions for a namespace, sorted by like_count."""
    try:
        namespace = request.args.get('namespace', '').strip()
        limit = min(max(1, request.args.get('limit', 10, type=int)), 20)

        q = SharedQuestion.query.filter_by(is_hidden=False)
        if namespace:
            q = q.filter_by(namespace=namespace)

        questions = q.order_by(
            SharedQuestion.like_count.desc(),
            SharedQuestion.created_at.desc(),
        ).limit(limit).all()

        # Check which ones the current user has liked
        liked_ids = set()
        if hasattr(current_user, 'id') and current_user.is_authenticated:
            qids = [sq.id for sq in questions]
            if qids:
                liked = QuestionLike.query.filter(
                    QuestionLike.user_id == current_user.id,
                    QuestionLike.question_id.in_(qids),
                ).all()
                liked_ids = {l.question_id for l in liked}

        return success_response(data={
            'questions': [
                sq.to_dict(liked_by_me=(sq.id in liked_ids))
                for sq in questions
            ],
        })
    except Exception:
        logging.exception('[Question] Popular list failed')
        return error_response('인기 질문 조회 중 오류가 발생했습니다.', 500)
```

**Query Parameters**:
- `namespace` (string, optional) — 도메인 필터
- `limit` (int, default=10, max=20) — 최대 개수

### 2.4 POST `/api/v1/questions/<id>/like` — 좋아요 토글

```python
@v1_bp.route('/questions/<int:question_id>/like', methods=['POST'])
@login_required
@rate_limit("60 per minute")
def api_question_toggle_like(question_id):
    """Toggle like on a shared question (PostLike pattern)."""
    sq = SharedQuestion.query.filter_by(id=question_id, is_hidden=False).first()
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
        sq = SharedQuestion.query.get(question_id)
        sq.like_count = QuestionLike.query.filter_by(question_id=question_id).count()
        db.session.commit()
        return success_response(data={'liked': False, 'like_count': sq.like_count})
```

**설계 결정**: `PostLike` 패턴(`community.py:417-438`)과 동일. INSERT 시도 → IntegrityError면 이미 좋아요 상태이므로 삭제(toggle). `like_count` 캐시를 실제 COUNT로 갱신.

### 2.5 DELETE `/api/v1/questions/<id>` — 본인 질문 삭제

```python
@v1_bp.route('/questions/<int:question_id>', methods=['DELETE'])
@login_required
def api_question_delete(question_id):
    """Delete own shared question."""
    try:
        sq = SharedQuestion.query.filter_by(
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
```

### 2.6 GET `/api/v1/questions/my` — 내가 공유한 질문

```python
@v1_bp.route('/questions/my', methods=['GET'])
@login_required
@rate_limit("30 per minute")
def api_question_my():
    """List questions shared by the current user."""
    try:
        page = max(1, request.args.get('page', 1, type=int))
        per_page = min(max(1, request.args.get('per_page', 20, type=int)), 50)

        q = SharedQuestion.query.filter_by(user_id=current_user.id)
        q = q.order_by(SharedQuestion.created_at.desc())

        pagination = q.paginate(page=page, per_page=per_page, error_out=False)

        return success_response(data={
            'items': [sq.to_dict(liked_by_me=False) for sq in pagination.items],
            'total': pagination.total,
            'page': pagination.page,
            'per_page': pagination.per_page,
            'pages': pagination.pages,
        })
    except Exception:
        logging.exception('[Question] My list failed')
        return error_response('내 질문 목록 조회 중 오류가 발생했습니다.', 500)
```

---

## 3. Blueprint Registration

### 3.1 `api/v1/__init__.py` 수정

```python
from api.v1 import questions  # noqa: E402, F401
```

기존 import 목록 끝 (`bookmarks` 뒤)에 추가.

---

## 4. UI: domain.html 수정

### 4.1 CSS 추가

기존 `.btn-bookmark.active` 스타일 뒤에 추가:

```css
/* Popular questions section */
.popular-questions {
    margin-top: 12px;
    padding: 14px;
    background: rgba(var(--primary-color-rgb), 0.04);
    border: 1px solid rgba(var(--primary-color-rgb), 0.1);
    border-radius: 12px;
}
.popular-questions-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--sf-text-2);
    margin-bottom: 10px;
}
.pq-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 10px;
    border-radius: 8px;
    cursor: pointer;
    transition: background 0.2s;
    font-size: 0.88rem;
    color: var(--sf-text-2);
}
.pq-item:hover {
    background: rgba(var(--primary-color-rgb), 0.08);
    color: var(--sf-text-1);
}
.pq-item-query {
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.pq-like-btn {
    display: flex;
    align-items: center;
    gap: 3px;
    padding: 3px 8px;
    border-radius: 6px;
    border: 1px solid var(--sf-border);
    background: transparent;
    color: var(--sf-text-3);
    font-size: 0.78rem;
    cursor: pointer;
    transition: all 0.2s;
    flex-shrink: 0;
}
.pq-like-btn:hover {
    border-color: #ef4444;
    color: #ef4444;
}
.pq-like-btn.liked {
    border-color: #ef4444;
    color: #ef4444;
    background: rgba(239, 68, 68, 0.08);
}
.pq-author {
    font-size: 0.72rem;
    color: var(--sf-text-4);
    flex-shrink: 0;
}

/* Share question button (after AI answer) */
.btn-share-question {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 6px 14px;
    border-radius: 8px;
    border: 1px solid rgba(var(--primary-color-rgb), 0.3);
    background: rgba(var(--primary-color-rgb), 0.06);
    color: var(--primary-color);
    font-size: 0.82rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
    font: inherit;
    margin-left: 8px;
}
.btn-share-question:hover {
    background: rgba(var(--primary-color-rgb), 0.12);
}
.btn-share-question.shared {
    border-color: rgba(76, 175, 80, 0.4);
    color: #4caf50;
    background: rgba(76, 175, 80, 0.08);
    cursor: default;
}
```

### 4.2 인기 질문 영역 (HTML)

기존 샘플 질문 `.hint` 영역 바로 뒤에 추가:

```html
<!-- Popular Questions -->
<div class="popular-questions" id="popularQuestions" style="display:none;">
    <div class="popular-questions-title">🔥 인기 질문</div>
    <div id="popularQuestionsList"></div>
</div>
```

### 4.3 JavaScript — 인기 질문 로드 + 좋아요 토글

`renderAskResults` 함수 앞에 추가:

```javascript
// ========================================
// Popular Questions (Question Likes)
// ========================================
async function loadPopularQuestions() {
    try {
        const res = await fetch(`/api/v1/questions/popular?namespace=${encodeURIComponent(DEFAULT_NAMESPACE)}&limit=10`);
        const json = await res.json();
        if (!json.success || !json.data.questions.length) return;

        const container = document.getElementById('popularQuestionsList');
        const section = document.getElementById('popularQuestions');
        section.style.display = '';

        container.innerHTML = json.data.questions.map(q => `
            <div class="pq-item">
                <span class="pq-item-query" onclick="askPopularQuestion('${escapeAttr(q.query)}')">${escapeHtml(q.query)}</span>
                <span class="pq-author">${escapeHtml(q.author ? q.author.name : '')}</span>
                ${isLoggedIn ? `
                    <button class="pq-like-btn ${q.liked_by_me ? 'liked' : ''}"
                            onclick="event.stopPropagation();toggleQuestionLike(this,${q.id})">
                        ${q.liked_by_me ? '♥' : '♡'} <span>${q.like_count}</span>
                    </button>
                ` : `
                    <span class="pq-like-btn" style="cursor:default;">♡ <span>${q.like_count}</span></span>
                `}
            </div>
        `).join('');
    } catch (e) {
        console.warn('Popular questions load failed:', e);
    }
}

function askPopularQuestion(query) {
    const textarea = document.getElementById('askQuery');
    textarea.value = query;
    textarea.focus();
    // Auto-submit
    document.getElementById('askForm').dispatchEvent(new Event('submit'));
}

async function toggleQuestionLike(btn, questionId) {
    if (!isLoggedIn) return;
    try {
        const res = await fetch(`/api/v1/questions/${questionId}/like`, {method: 'POST'});
        const json = await res.json();
        if (json.success) {
            const countEl = btn.querySelector('span');
            if (json.data.liked) {
                btn.classList.add('liked');
                btn.innerHTML = '♥ <span>' + json.data.like_count + '</span>';
            } else {
                btn.classList.remove('liked');
                btn.innerHTML = '♡ <span>' + json.data.like_count + '</span>';
            }
        }
    } catch (e) {
        console.warn('Question like failed:', e);
    }
}
```

### 4.4 JavaScript — 질문 공유 버튼

`renderAskResults` 함수 내부에서, AI 답변 헤더에 "공유하기" 버튼 추가.

기존 `renderAskResults`의 AI 답변 컨테이너 헤더:
```javascript
<div class="ai-answer-header">
    <span>🤖</span> AI 답변
    <button class="copy-md-btn" ...>Markdown 복사</button>
</div>
```

이 헤더에 공유 버튼 추가:
```javascript
const shareBtn = isLoggedIn
    ? `<button class="btn-share-question" id="btnShareQuestion"
             onclick="shareQuestion()">📤 질문 공유</button>`
    : '';
```

수정된 헤더:
```javascript
<div class="ai-answer-header">
    <span>🤖</span> AI 답변
    ${shareBtn}
    <button class="copy-md-btn" onclick="copyMarkdown(this, _lastAnswerMd)">...</button>
</div>
```

공유 함수:
```javascript
// Store the last asked query and namespace for sharing
let _lastAskedQuery = '';
let _lastAskedNamespace = '';
let _lastAnswerPreview = '';

async function shareQuestion() {
    if (!isLoggedIn || !_lastAskedQuery) return;
    const btn = document.getElementById('btnShareQuestion');
    if (btn.classList.contains('shared')) return;

    try {
        const res = await fetch('/api/v1/questions/share', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                query: _lastAskedQuery,
                namespace: _lastAskedNamespace,
                answer_preview: _lastAnswerPreview,
            })
        });
        const json = await res.json();
        if (json.success) {
            btn.classList.add('shared');
            btn.innerHTML = '✅ 공유됨';
            showToast(json.message || '질문이 공유되었습니다.');
            // Refresh popular questions
            loadPopularQuestions();
        }
    } catch (e) {
        showToast('질문 공유에 실패했습니다.', 'error');
    }
}
```

### 4.5 JavaScript — 질문/답변 정보 저장 (공유에 사용)

AI Ask 폼 submit 핸들러에서 질문 정보를 저장하는 코드 추가.

기존 `askForm` submit 핸들러의 query 추출 부분 뒤:
```javascript
// 기존 코드: const askQuery = document.getElementById('askQuery').value;
// 추가:
_lastAskedQuery = askQuery;
_lastAskedNamespace = document.getElementById('askNamespace').value || DEFAULT_NAMESPACE;
```

`renderAskResults` 함수 호출 직전 (done 이벤트 처리 시):
```javascript
_lastAnswerPreview = (answerText || '').substring(0, 300);
```

### 4.6 JavaScript — 페이지 로드 시 인기 질문 로드

기존 `loadStats();` 호출 뒤에 추가:
```javascript
loadPopularQuestions();
```

---

## 5. 파일 변경 목록

| 파일 | 작업 | 설명 |
|------|------|------|
| `models.py` | **수정** | SharedQuestion, QuestionLike 모델 추가 (SearchHistory 뒤) |
| `api/v1/questions.py` | **신규** | 질문 공유/인기/좋아요/삭제/내 질문 API (5개 엔드포인트) |
| `api/v1/__init__.py` | **수정** | `from api.v1 import questions` 추가 |
| `templates/domain.html` | **수정** | CSS + 인기 질문 HTML + 공유 버튼 + JS 로직 |

---

## 6. 구현 순서

1. `models.py` — SharedQuestion, QuestionLike 모델 추가
2. `api/v1/questions.py` — 5개 API 엔드포인트 구현
3. `api/v1/__init__.py` — import 추가
4. `templates/domain.html` — CSS + 인기 질문 영역 + 공유 버튼 + JS 함수

---

## 7. 주의사항

- **CSRF**: `v1_bp`는 이미 CSRF exempt (`web_app.py:95`)이므로 API 호출 시 CSRF 토큰 불필요
- **인증**: share/like/delete/my는 `@login_required`, popular은 공개 (비로그인도 조회 가능)
- **보안**: `user_id`는 항상 `current_user.id`에서 가져옴. `query_hash`는 중복 방지용 (보안 목적 아님)
- **성능**: `like_count` 캐시 컬럼으로 인기 질문 정렬 시 JOIN 없이 인덱스 활용
- **일관성**: namespace 값은 Pinecone의 실제 namespace (`semiconductor-v2`, `laborlaw`, `kosha` 등) 사용
- **SQLite 호환**: `db.create_all()` 자동 테이블 생성으로 별도 마이그레이션 불필요
- **PostLike 패턴**: 좋아요 토글은 INSERT + IntegrityError 패턴으로 race condition 방지
- **PR 생성 필요**: 구현 완료 후 PR을 생성하여 코드 리뷰 진행
