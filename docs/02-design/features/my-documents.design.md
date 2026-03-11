# Design: 나의 자료 등록 (My Documents Bookmark)

> **Feature**: my-documents
> **Date**: 2026-03-11
> **Project**: SafeFactory
> **Version**: 1.0
> **Plan Reference**: `docs/01-plan/features/my-documents.plan.md`

---

## 1. Data Model

### 1.1 UserBookmark Model (`models.py`)

```python
class UserBookmark(db.Model):
    """User's bookmarked documents from search results."""

    __tablename__ = 'user_bookmarks'
    __table_args__ = (
        db.UniqueConstraint('user_id', 'source_file', name='uq_user_bookmark'),
        db.Index('ix_user_bookmarks_user_created', 'user_id', 'created_at'),
        db.Index('ix_user_bookmarks_namespace', 'user_id', 'namespace'),
    )

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False,
    )
    source_file = db.Column(db.String(500), nullable=False)
    namespace = db.Column(db.String(100), nullable=False, default='')
    title = db.Column(db.String(300), nullable=False)
    file_type = db.Column(db.String(20), nullable=True)
    created_at = db.Column(
        db.DateTime, nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    user = db.relationship('User', backref='bookmarks')

    MAX_PER_USER = 200

    def to_dict(self):
        return {
            'id': self.id,
            'source_file': self.source_file,
            'namespace': self.namespace,
            'title': self.title,
            'file_type': self.file_type,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }
```

**설계 결정**:
- `source_file`이 Pinecone metadata에서 항상 제공되므로 문서 고유 식별자로 사용
- `title`은 검색 결과의 `filename` 또는 `ncs_document_title`에서 추출
- `memo` 컬럼은 Phase 1에서 제외 (YAGNI 원칙)
- `file_type`은 아이콘 표시용 (image, markdown, json 등)

---

## 2. API Endpoints (`api/v1/bookmarks.py`)

### 2.1 신규 파일 생성

`api/v1/bookmarks.py` — 모든 엔드포인트는 `@login_required` 필수.

### 2.2 POST `/api/v1/bookmarks` — 북마크 추가

```python
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
```

**Request Body**:
```json
{
  "source_file": "ncs/data/반도체제조/LM1234/chunk_01.md",
  "namespace": "semiconductor-v2",
  "title": "CVD 공정 개요",
  "file_type": "markdown"
}
```

### 2.3 GET `/api/v1/bookmarks` — 목록 조회

```python
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
```

**Query Parameters**:
- `page` (int, default=1)
- `per_page` (int, default=20, max=50)
- `namespace` (string, optional) — 도메인 필터
- `sort` (string, default='newest') — `newest` | `title`

### 2.4 DELETE `/api/v1/bookmarks/<id>` — 개별 삭제

```python
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
```

### 2.5 DELETE `/api/v1/bookmarks` — 전체 삭제

```python
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
```

### 2.6 POST `/api/v1/bookmarks/check-batch` — 일괄 상태 확인

```python
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
```

**Request Body**:
```json
{
  "source_files": [
    "ncs/data/반도체제조/LM1234/chunk_01.md",
    "laborlaw/laws/근로기준법.md"
  ]
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "bookmarked": {
      "ncs/data/반도체제조/LM1234/chunk_01.md": 42
    }
  }
}
```

`bookmarked` 맵의 value는 bookmark ID (삭제 시 사용). 키가 없으면 해당 문서는 북마크되지 않은 상태.

---

## 3. Blueprint Registration

### 3.1 `api/v1/__init__.py` 수정

```python
from api.v1 import bookmarks  # noqa: E402, F401
```

기존 import 목록 끝에 추가.

---

## 4. Web Route

### 4.1 `web_app.py` 수정

```python
@app.route('/my-documents')
@login_required
def my_documents():
    """User's bookmarked documents page."""
    return render_template('my_documents.html')
```

`/history` 라우트 바로 아래에 추가.

---

## 5. Navigation 수정 (`templates/base.html`)

### 5.1 Desktop Nav (로그인 상태)

`검색기록` 링크 바로 뒤에 추가:

```html
<a href="/my-documents" class="sf-nav-link">나의 자료</a>
```

위치: 기존 `<a href="/history" class="sf-nav-link">검색기록</a>` 바로 뒤.

### 5.2 Mobile Menu (로그인 상태)

```html
<a href="/my-documents" class="sf-mobile-auth-link">나의 자료</a>
```

위치: 기존 `검색기록` 링크 바로 뒤.

---

## 6. 검색 결과 카드 북마크 버튼 (`templates/domain.html`)

### 6.1 CSS 추가

```css
/* Bookmark button on result cards */
.btn-bookmark {
    position: absolute;
    top: 16px;
    right: 16px;
    width: 36px;
    height: 36px;
    border-radius: 10px;
    border: 1px solid var(--sf-border);
    background: var(--sf-card-bg);
    color: var(--sf-text-3);
    font-size: 1.1rem;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s;
    z-index: 2;
}
.btn-bookmark:hover {
    border-color: var(--primary-color);
    color: var(--primary-color);
}
.btn-bookmark.active {
    background: rgba(var(--primary-rgb), 0.1);
    border-color: var(--primary-color);
    color: var(--primary-color);
}
```

### 6.2 Result Card 수정

기존 `.result-card` 클래스에 `position: relative` 추가 (이미 적용되어 있으면 불필요).

카드 HTML 템플릿에 북마크 버튼 추가 (로그인 시만):

```javascript
// card.innerHTML 구성 시, 기존 내용 앞에 추가
const bookmarkBtn = isLoggedIn
    ? `<button class="btn-bookmark ${isBookmarked(result.source_file) ? 'active' : ''}"
         onclick="event.stopPropagation();toggleBookmark(this,'${escapeAttr(result.source_file)}','${escapeAttr(safeFilename)}','${escapeAttr(namespace)}','${escapeAttr(safeFileType)}')"
         title="나의 자료에 저장">
         ${isBookmarked(result.source_file) ? '★' : '☆'}
       </button>`
    : '';
```

### 6.3 JavaScript 로직

```javascript
// Bookmark state cache (populated by check-batch API)
let _bookmarkMap = {};  // source_file -> bookmark_id
const isLoggedIn = {{ 'true' if current_user.is_authenticated else 'false' }};

function isBookmarked(sourceFile) {
    return sourceFile in _bookmarkMap;
}

// After search results are rendered, check bookmark status
async function checkBookmarkStatus(results) {
    if (!isLoggedIn || !results.length) return;
    const sourceFiles = results.map(r => r.source_file).filter(Boolean);
    if (!sourceFiles.length) return;
    try {
        const res = await fetch('/api/v1/bookmarks/check-batch', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({source_files: sourceFiles})
        });
        const json = await res.json();
        if (json.success && json.data.bookmarked) {
            _bookmarkMap = {..._bookmarkMap, ...json.data.bookmarked};
            // Update all bookmark buttons
            document.querySelectorAll('.btn-bookmark').forEach(btn => {
                const sf = btn.dataset.sourceFile;
                if (sf && sf in _bookmarkMap) {
                    btn.classList.add('active');
                    btn.textContent = '★';
                }
            });
        }
    } catch (e) {
        console.warn('Bookmark check failed:', e);
    }
}

async function toggleBookmark(btn, sourceFile, title, namespace, fileType) {
    if (!isLoggedIn) return;

    const wasActive = btn.classList.contains('active');

    if (wasActive) {
        // Remove bookmark
        const bookmarkId = _bookmarkMap[sourceFile];
        if (!bookmarkId) return;
        try {
            const res = await fetch(`/api/v1/bookmarks/${bookmarkId}`, {method: 'DELETE'});
            const json = await res.json();
            if (json.success) {
                delete _bookmarkMap[sourceFile];
                btn.classList.remove('active');
                btn.textContent = '☆';
            }
        } catch (e) {
            console.warn('Bookmark delete failed:', e);
        }
    } else {
        // Add bookmark
        try {
            const res = await fetch('/api/v1/bookmarks', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({source_file: sourceFile, title, namespace, file_type: fileType})
            });
            const json = await res.json();
            if (json.success && json.data) {
                _bookmarkMap[sourceFile] = json.data.id;
                btn.classList.add('active');
                btn.textContent = '★';
            }
        } catch (e) {
            console.warn('Bookmark create failed:', e);
        }
    }
}
```

**호출 시점**: 검색 결과 렌더 후 `checkBookmarkStatus(data.data.results)` 호출.

### 6.4 `escapeAttr` 헬퍼

```javascript
function escapeAttr(str) {
    if (!str) return '';
    return str.replace(/'/g, "\\'").replace(/"/g, '&quot;');
}
```

---

## 7. 나의 자료 페이지 (`templates/my_documents.html`)

### 7.1 페이지 구조

- `{% extends "base.html" %}`
- 다크 테마 (history.html과 동일한 `background: linear-gradient(135deg, #1a1a2e, #16213e)`)
- `max-width: 800px` 컨테이너

### 7.2 UI 구성

```
┌─────────────────────────────────────────┐
│ 나의 자료                    [전체 삭제] │
├─────────────────────────────────────────┤
│ [전체 ▼] [최신순 ▼]                      │
├─────────────────────────────────────────┤
│ ★ CVD 공정 개요                    [✕]  │
│   📁 반도체 · markdown · 2분 전         │
├─────────────────────────────────────────┤
│ ★ 근로기준법 제56조                [✕]  │
│   📁 노동법 · markdown · 1시간 전       │
├─────────────────────────────────────────┤
│         < 1  2  3 >                     │
└─────────────────────────────────────────┘
```

### 7.3 핵심 HTML/CSS/JS 패턴

history.html의 구조를 그대로 참조:
- `.container`, `.page-header`, `.filters`, `.bookmark-list`, `.pagination` 레이아웃
- `.bookmark-item` 카드 (`.history-item`과 동일 스타일)
- 필터: namespace 드롭다운 + sort 드롭다운
- 개별 삭제 버튼 (`.btn-delete-item`)
- 전체 삭제 버튼 (`.btn-delete-all`)
- 페이지네이션 (동일 패턴)
- 빈 상태: "저장한 자료가 없습니다."

### 7.4 JavaScript 동작

```javascript
const NAMESPACE_LABELS = {
    '': '전체',
    'semiconductor-v2': '반도체',
    'laborlaw': '노동법',
    'field-training': '현장훈련',
    'kosha': '안전보건',
    'msds': '화학물질'
};
const NAMESPACE_PATHS = {
    'semiconductor-v2': '/semiconductor',
    'laborlaw': '/laborlaw',
    'field-training': '/field-training',
    'kosha': '/safeguide',
    'msds': '/msds'
};

let currentPage = 1;

async function loadBookmarks(page) {
    if (page) currentPage = page;
    else currentPage = 1;

    const ns = document.getElementById('filterNamespace').value;
    const sort = document.getElementById('filterSort').value;
    let url = `/api/v1/bookmarks?page=${currentPage}&per_page=20`;
    if (ns) url += `&namespace=${encodeURIComponent(ns)}`;
    if (sort) url += `&sort=${sort}`;

    // ... fetch, renderBookmarks, renderPagination (history.html 패턴 동일)
}

function renderBookmarks(data) {
    // 각 항목: title, namespace label, file_type, timeAgo(created_at)
    // 클릭 시: NAMESPACE_PATHS[namespace] + '?q=' + encodeURIComponent(title)
    // 삭제 버튼: deleteOne(id)
}

async function deleteOne(id) {
    if (!confirm('이 자료를 삭제하시겠습니까?')) return;
    await fetch(`/api/v1/bookmarks/${id}`, {method: 'DELETE'});
    loadBookmarks(currentPage);
}

async function deleteAll() {
    if (!confirm('전체 자료를 삭제하시겠습니까?\n이 작업은 되돌릴 수 없습니다.')) return;
    await fetch('/api/v1/bookmarks', {method: 'DELETE'});
    loadBookmarks(1);
}

document.addEventListener('DOMContentLoaded', () => loadBookmarks(1));
```

---

## 8. 파일 변경 목록

| 파일 | 작업 | 설명 |
|------|------|------|
| `models.py` | **수정** | UserBookmark 모델 추가 (SearchHistory 아래) |
| `api/v1/bookmarks.py` | **신규** | 북마크 CRUD + check-batch API (6개 엔드포인트) |
| `api/v1/__init__.py` | **수정** | `from api.v1 import bookmarks` 추가 |
| `web_app.py` | **수정** | `/my-documents` 라우트 추가 |
| `templates/my_documents.html` | **신규** | 나의 자료 페이지 (history.html 패턴 참조) |
| `templates/domain.html` | **수정** | 검색 결과 카드에 북마크 버튼 + JS 로직 추가 |
| `templates/base.html` | **수정** | 네비게이션에 "나의 자료" 링크 추가 (데스크톱+모바일) |

---

## 9. 구현 순서

1. `models.py` — UserBookmark 모델 추가
2. `api/v1/bookmarks.py` — CRUD API 구현
3. `api/v1/__init__.py` — import 추가
4. `web_app.py` — `/my-documents` 라우트 추가
5. `templates/my_documents.html` — 나의 자료 페이지 생성
6. `templates/base.html` — 네비게이션 링크 추가
7. `templates/domain.html` — 검색 결과 카드 북마크 버튼 추가

---

## 10. 주의사항

- **CSRF**: `v1_bp`는 이미 CSRF exempt (`web_app.py:95`)이므로 API 호출 시 CSRF 토큰 불필요
- **인증**: 모든 API는 `@login_required` 데코레이터 사용, 미인증 시 401 응답
- **보안**: `user_id`는 항상 `current_user.id`에서 가져옴 (클라이언트 제공 불가)
- **성능**: `check-batch` API로 검색 결과 렌더 후 1회 호출로 여러 문서 상태 확인 (N+1 방지)
- **일관성**: namespace 값은 Pinecone의 실제 namespace (`semiconductor-v2`, `laborlaw`, `kosha` 등) 사용
- **SQLite 호환**: `db.create_all()` 자동 테이블 생성으로 별도 마이그레이션 불필요
