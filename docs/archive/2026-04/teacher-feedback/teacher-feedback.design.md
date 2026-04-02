# Design: 교사 답변 피드백 시스템

> **Feature**: teacher-feedback
> **Date**: 2026-04-02
> **Project**: SafeFactory
> **Version**: 1.0
> **Plan Reference**: `docs/01-plan/features/teacher-feedback.plan.md`

---

## 1. Data Model

### 1.1 AnswerFeedback Model (`models.py`)

`SearchHistory` 패턴을 따르며, `user_id` FK + `created_at` 인덱스 + `to_dict()` 직렬화 패턴 재사용.

```python
class AnswerFeedback(db.Model):
    """Teacher feedback on AI answer quality."""

    __tablename__ = 'answer_feedbacks'
    __table_args__ = (
        db.Index('ix_feedback_created', 'created_at'),
        db.Index('ix_feedback_ns_status', 'namespace', 'status'),
        db.UniqueConstraint('user_id', 'query_hash', name='uq_feedback_user_query'),
    )

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False,
    )
    query = db.Column(db.Text, nullable=False)
    query_hash = db.Column(db.String(32), nullable=False)  # MD5 of query+namespace for dedup
    answer = db.Column(db.Text, nullable=False)
    namespace = db.Column(db.String(100), nullable=False, default='')
    source_count = db.Column(db.Integer, nullable=False, default=0)
    confidence_score = db.Column(db.Float, nullable=True)

    feedback_type = db.Column(db.String(20), nullable=False)
    comment = db.Column(db.Text, nullable=True)

    status = db.Column(db.String(20), nullable=False, default='pending')
    admin_note = db.Column(db.Text, nullable=True)

    created_at = db.Column(
        db.DateTime, nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    resolved_at = db.Column(db.DateTime, nullable=True)

    user = db.relationship('User', backref='answer_feedbacks')

    FEEDBACK_TYPES = ('inaccurate', 'incomplete', 'irrelevant', 'unclear')
    STATUSES = ('pending', 'resolved', 'dismissed')

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'user_name': self.user.name if self.user else None,
            'query': self.query,
            'answer': self.answer[:300],  # 미리보기만 (목록용)
            'namespace': self.namespace,
            'source_count': self.source_count,
            'confidence_score': self.confidence_score,
            'feedback_type': self.feedback_type,
            'comment': self.comment,
            'status': self.status,
            'admin_note': self.admin_note,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
        }

    def to_dict_full(self):
        """Full dict including complete answer text (상세 조회용)."""
        d = self.to_dict()
        d['answer'] = self.answer
        return d
```

**인덱스 설계:**
- `ix_feedback_created`: 최신순 정렬
- `ix_feedback_ns_status`: namespace + status 복합 필터 (관리자 조회)
- `uq_feedback_user_query`: 동일 사용자의 동일 질문 중복 신고 방지 (MD5 해시)

**`query_hash` 생성:**
```python
import hashlib
query_hash = hashlib.md5(f"{query.strip()}{namespace}".encode()).hexdigest()
```

### 1.2 DB 테이블 생성

기존 `web_app.py`의 `db.create_all()` 호출 시 자동 생성. 별도 마이그레이션 불필요 (SQLite + create_all 패턴).

---

## 2. API Design

모든 엔드포인트는 `v1_bp` 블루프린트에 등록. `api/response.py`의 `success_response`/`error_response` 사용.

### 2.1 피드백 제출 (`api/v1/feedback.py` — 신규 파일)

```python
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

    # 중복 체크
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

    return success_response(data={'id': fb.id}, message='피드백이 제출되었습니다.')
```

### 2.2 관리자 피드백 목록 (`api/v1/admin.py`에 추가)

```python
@v1_bp.route('/admin/feedback', methods=['GET'])
@admin_required
def admin_feedback_list():
    """List answer feedbacks with pagination and filters."""
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 20, type=int), 100)
    namespace = request.args.get('namespace', '')
    status = request.args.get('status', '')
    feedback_type = request.args.get('feedback_type', '')

    q = AnswerFeedback.query
    if namespace:
        q = q.filter_by(namespace=namespace)
    if status:
        q = q.filter_by(status=status)
    if feedback_type:
        q = q.filter_by(feedback_type=feedback_type)

    q = q.order_by(AnswerFeedback.created_at.desc())
    pagination = q.paginate(page=page, per_page=per_page, error_out=False)

    return success_response(data={
        'items': [fb.to_dict() for fb in pagination.items],
        'total': pagination.total,
        'page': page,
        'pages': pagination.pages,
    })
```

### 2.3 관리자 피드백 상태 업데이트

```python
@v1_bp.route('/admin/feedback/<int:fb_id>', methods=['PUT'])
@admin_required
def admin_feedback_update(fb_id):
    """Update feedback status (resolve/dismiss)."""
    fb = db.session.get(AnswerFeedback, fb_id)
    if not fb:
        return error_response('피드백을 찾을 수 없습니다.', 404)

    data = request.get_json(silent=True) or {}
    new_status = data.get('status', '')
    if new_status not in AnswerFeedback.STATUSES:
        return error_response(f'유효하지 않은 상태: {new_status}', 400)

    fb.status = new_status
    fb.admin_note = data.get('admin_note') or fb.admin_note
    if new_status in ('resolved', 'dismissed'):
        fb.resolved_at = datetime.now(timezone.utc)

    db.session.commit()
    return success_response(data=fb.to_dict())
```

### 2.4 Golden Dataset 내보내기

```python
@v1_bp.route('/admin/feedback/export', methods=['GET'])
@admin_required
def admin_feedback_export():
    """Export feedbacks as Golden Dataset compatible JSON."""
    status_filter = request.args.get('status', '')
    q = AnswerFeedback.query
    if status_filter:
        q = q.filter_by(status=status_filter)
    q = q.order_by(AnswerFeedback.created_at.desc())
    feedbacks = q.all()

    # Group by namespace → golden dataset format
    domains = {}
    for fb in feedbacks:
        ns = fb.namespace or 'unknown'
        if ns not in domains:
            domains[ns] = []
        domains[ns].append({
            'id': f'fb-{fb.id:04d}',
            'query': fb.query,
            'namespace': fb.namespace,
            'feedback_type': fb.feedback_type,
            'teacher_comment': fb.comment,
            'original_answer_excerpt': fb.answer[:500],
            'confidence_score': fb.confidence_score,
            'source_count': fb.source_count,
            'created_at': fb.created_at.isoformat() if fb.created_at else None,
        })

    result = {
        'version': '2.0',
        'source': 'teacher-feedback',
        'exported_at': datetime.now(timezone.utc).isoformat(),
        'total': len(feedbacks),
        'domains': domains,
    }

    return jsonify(result)
```

---

## 3. Frontend Design

### 3.1 피드백 버튼 위치 (`templates/domain.html`)

`renderAskResults()` 함수에서 `shareBtn` 옆에 피드백 버튼 추가:

```javascript
// 기존 shareBtn 행 (line ~1779)
const shareBtn = isLoggedIn
    ? `<button class="btn-share-question" id="btnShareQuestion" onclick="shareQuestion()">📤 질문 공유</button>`
    : '';

// 추가: 피드백 버튼
const feedbackBtn = isLoggedIn
    ? `<button class="btn-feedback" id="btnFeedback" onclick="openFeedbackModal()">👎 부정확 신고</button>`
    : '';
```

`ai-answer-header`에 `feedbackBtn` 삽입 (shareBtn 뒤):
```javascript
let html = calcHtml + `
    <div class="ai-answer-container">
        <div class="ai-answer-header">
            <span>🤖</span> AI 답변
            ${shareBtn}
            ${feedbackBtn}
            <button class="copy-md-btn" ...>...</button>
        </div>
        ...
    </div>
`;
```

### 3.2 피드백 모달 HTML

PDF 모달(`#pdfModal`) 패턴을 따라 `{% endblock %}` 직전에 추가:

```html
<div class="pdf-modal-overlay" id="feedbackModal">
    <div class="pdf-modal" style="max-width: 480px;">
        <div class="pdf-modal-header">
            <span class="pdf-modal-title">답변 피드백</span>
            <button class="pdf-modal-close" onclick="closeFeedbackModal()">&times;</button>
        </div>
        <div class="pdf-modal-body" style="padding: 20px;">
            <p style="margin-bottom: 16px; color: var(--sf-text-2);">어떤 문제가 있나요?</p>

            <div id="feedbackTypeGroup" style="display: flex; flex-direction: column; gap: 8px; margin-bottom: 16px;">
                <label class="feedback-radio">
                    <input type="radio" name="feedbackType" value="inaccurate">
                    <span>❌ 사실과 다른 내용 (부정확)</span>
                </label>
                <label class="feedback-radio">
                    <input type="radio" name="feedbackType" value="incomplete">
                    <span>📝 중요한 정보 누락 (불완전)</span>
                </label>
                <label class="feedback-radio">
                    <input type="radio" name="feedbackType" value="irrelevant">
                    <span>🔀 질문과 무관한 답변 (관련없음)</span>
                </label>
                <label class="feedback-radio">
                    <input type="radio" name="feedbackType" value="unclear">
                    <span>❓ 이해하기 어려움 (이해불가)</span>
                </label>
            </div>

            <textarea id="feedbackComment" rows="3" placeholder="추가 의견 (선택): 올바른 답변이나 빠진 내용을 적어주세요"
                style="width: 100%; padding: 10px; border: 1px solid var(--sf-border); border-radius: 8px;
                       background: var(--sf-bg-2); color: var(--sf-text-1); resize: vertical;"></textarea>

            <div style="display: flex; justify-content: flex-end; gap: 8px; margin-top: 16px;">
                <button onclick="closeFeedbackModal()"
                    style="padding: 8px 16px; border-radius: 8px; border: 1px solid var(--sf-border);
                           background: transparent; color: var(--sf-text-2); cursor: pointer;">취소</button>
                <button id="btnSubmitFeedback" onclick="submitFeedback()"
                    style="padding: 8px 16px; border-radius: 8px; border: none;
                           background: #ef4444; color: white; cursor: pointer;">제출</button>
            </div>
        </div>
    </div>
</div>
```

### 3.3 피드백 CSS

기존 `.btn-share-question` 패턴 복제 + 색상 변경:

```css
.btn-feedback {
    font-size: 0.8rem;
    padding: 4px 10px;
    border-radius: 6px;
    border: 1px solid rgba(239, 68, 68, 0.3);
    background: rgba(239, 68, 68, 0.1);
    color: #ef4444;
    cursor: pointer;
    transition: all 0.2s;
}
.btn-feedback:hover {
    background: rgba(239, 68, 68, 0.2);
}
.btn-feedback.submitted {
    opacity: 0.5;
    pointer-events: none;
}
.feedback-radio {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    border-radius: 8px;
    cursor: pointer;
    transition: background 0.15s;
}
.feedback-radio:hover {
    background: var(--sf-bg-2);
}
```

### 3.4 피드백 JavaScript

```javascript
// 마지막 답변 메타데이터 저장 (renderAskResults에서 갱신)
let _lastAskMetadata = null;

function openFeedbackModal() {
    document.getElementById('feedbackModal').classList.add('active');
    // Reset form
    document.querySelectorAll('input[name="feedbackType"]').forEach(r => r.checked = false);
    document.getElementById('feedbackComment').value = '';
}

function closeFeedbackModal() {
    document.getElementById('feedbackModal').classList.remove('active');
}

async function submitFeedback() {
    const typeEl = document.querySelector('input[name="feedbackType"]:checked');
    if (!typeEl) {
        alert('피드백 유형을 선택해주세요.');
        return;
    }

    const btn = document.getElementById('btnSubmitFeedback');
    btn.disabled = true;
    btn.textContent = '제출 중...';

    try {
        const resp = await fetch('/api/v1/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: _lastAskedQuery,
                answer: window._lastAnswerMd || '',
                namespace: _lastAskedNamespace || DEFAULT_NAMESPACE,
                source_count: (_lastAskMetadata && _lastAskMetadata.source_count) || 0,
                confidence_score: (_lastAskMetadata && _lastAskMetadata.confidence) ?
                    _lastAskMetadata.confidence.score : null,
                feedback_type: typeEl.value,
                comment: document.getElementById('feedbackComment').value.trim() || null,
            }),
        });

        const result = await resp.json();
        if (result.success) {
            closeFeedbackModal();
            const fbBtn = document.getElementById('btnFeedback');
            if (fbBtn) {
                fbBtn.textContent = '✅ 신고 완료';
                fbBtn.classList.add('submitted');
            }
        } else {
            alert(result.error || '피드백 제출에 실패했습니다.');
        }
    } catch (e) {
        alert('네트워크 오류가 발생했습니다.');
    } finally {
        btn.disabled = false;
        btn.textContent = '제출';
    }
}
```

**`_lastAskMetadata` 갱신 위치:** `renderAskResults()` 함수 시작 부분에 추가:
```javascript
function renderAskResults(resultsContainer, metadata, answerText, calculations) {
    _lastAskMetadata = metadata;  // ← 추가
    // ... 기존 코드
}
```

---

## 4. Admin UI Design

### 4.1 관리자 패널 탭 추가 (`templates/admin.html`)

기존 탭 네비게이션에 "답변 피드백" 탭 추가. `admin-` CSS 클래스 접두사 규칙 준수.

**탭 버튼:**
```html
<button class="admin-tab-btn" data-tab="feedback" onclick="switchTab('feedback')">
    답변 피드백 <span class="admin-badge" id="feedbackBadge">0</span>
</button>
```

**탭 콘텐츠:**
```html
<div class="admin-tab-content" id="tab-feedback" style="display:none;">
    <div class="admin-toolbar">
        <select id="feedbackNsFilter" onchange="loadFeedbacks()">
            <option value="">전체 도메인</option>
            <option value="semiconductor-v2">반도체</option>
            <option value="field-training">현장실습</option>
            <option value="kosha">안전보건</option>
        </select>
        <select id="feedbackStatusFilter" onchange="loadFeedbacks()">
            <option value="">전체 상태</option>
            <option value="pending" selected>대기중</option>
            <option value="resolved">해결됨</option>
            <option value="dismissed">무시됨</option>
        </select>
        <button onclick="exportFeedbacks('json')">📥 JSON 내보내기</button>
    </div>
    <div id="feedbackList" class="admin-list"></div>
    <div id="feedbackPagination" class="admin-pagination"></div>
</div>
```

**피드백 항목 렌더링:**
```javascript
function renderFeedbackItem(fb) {
    const typeLabels = {
        inaccurate: '❌ 부정확', incomplete: '📝 불완전',
        irrelevant: '🔀 관련없음', unclear: '❓ 이해불가'
    };
    const statusColors = {
        pending: '#f59e0b', resolved: '#10b981', dismissed: '#6b7280'
    };
    return `
        <div class="admin-list-item" style="border-left: 3px solid ${statusColors[fb.status] || '#6b7280'};">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span><strong>${typeLabels[fb.feedback_type] || fb.feedback_type}</strong>
                    · ${fb.namespace || '전체'} · ${fb.user_name || '익명'}</span>
                <span style="color: var(--sf-text-3); font-size: 0.85rem;">${formatDate(fb.created_at)}</span>
            </div>
            <div style="margin-bottom: 6px;"><strong>Q:</strong> ${escapeHtml(fb.query)}</div>
            <div style="margin-bottom: 6px; color: var(--sf-text-2); font-size: 0.9rem;">
                A: ${escapeHtml(fb.answer)}...
            </div>
            ${fb.comment ? `<div style="margin-bottom: 8px; padding: 8px; background: var(--sf-bg-2); border-radius: 6px;">
                💬 교사: ${escapeHtml(fb.comment)}</div>` : ''}
            ${fb.status === 'pending' ? `
                <div style="display: flex; gap: 8px;">
                    <button onclick="updateFeedbackStatus(${fb.id}, 'resolved')"
                        style="padding: 4px 12px; border-radius: 6px; border: 1px solid #10b981; background: rgba(16,185,129,0.1); color: #10b981; cursor: pointer;">해결</button>
                    <button onclick="updateFeedbackStatus(${fb.id}, 'dismissed')"
                        style="padding: 4px 12px; border-radius: 6px; border: 1px solid #6b7280; background: transparent; color: #6b7280; cursor: pointer;">무시</button>
                </div>
            ` : `<span style="color: ${statusColors[fb.status]};">● ${fb.status}</span>`}
        </div>
    `;
}
```

---

## 5. Implementation Order

| Step | File | Change | Dependencies |
|------|------|--------|-------------|
| 1 | `models.py` | `AnswerFeedback` 모델 추가 | None |
| 2 | `api/v1/feedback.py` | 피드백 제출 API (신규 파일) | Step 1 |
| 3 | `api/v1/__init__.py` | `from api.v1 import feedback` 추가 | Step 2 |
| 4 | `api/v1/admin.py` | 관리자 피드백 목록/상태변경/내보내기 API | Step 1 |
| 5 | `templates/domain.html` | 피드백 버튼 + 모달 + CSS + JS | Step 2 |
| 6 | `templates/admin.html` | 피드백 탭 + JS | Step 4 |

---

## 6. Error Handling

| Error Case | Response | HTTP |
|------------|----------|------|
| 비로그인 사용자 피드백 시도 | `login_required` → 401 redirect | 401 |
| 필수 필드 누락 (query, answer, feedback_type) | `error_response('필수입니다', 400)` | 400 |
| 잘못된 feedback_type | `error_response('유효하지 않은', 400)` | 400 |
| 동일 query 중복 신고 | `error_response('이미 제출', 409)` | 409 |
| 관리자가 아닌 사용자의 관리 API 접근 | `admin_required` → 403 | 403 |
| 존재하지 않는 피드백 ID | `error_response('찾을 수 없습니다', 404)` | 404 |

---

## 7. Test Verification

구현 완료 후 수동 검증 체크리스트:

- [ ] 로그인 사용자가 AI 답변 후 "부정확 신고" 버튼이 표시되는지 확인
- [ ] 버튼 클릭 → 모달 열림 → 유형 선택 → 제출 → 성공 메시지
- [ ] 동일 질문에 대한 중복 제출 시 409 에러 표시
- [ ] 비로그인 시 버튼이 표시되지 않는지 확인
- [ ] 응급 답변(emergency)에는 버튼이 표시되지 않는지 확인
- [ ] 관리자 페이지에서 피드백 목록이 표시되는지 확인
- [ ] namespace/status 필터 동작 확인
- [ ] 해결/무시 버튼 동작 확인
- [ ] JSON 내보내기 → Golden Dataset 호환 형식 확인
- [ ] SQLite DB에 `answer_feedbacks` 테이블 자동 생성 확인
