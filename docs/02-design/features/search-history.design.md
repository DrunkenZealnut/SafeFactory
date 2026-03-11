# Design: 사용자별 검색기록 저장 및 불러오기

> **Feature**: search-history
> **Date**: 2026-03-11
> **Project**: SafeFactory
> **Version**: 1.0
> **Plan Reference**: `docs/01-plan/features/search-history.plan.md`

---

## 1. Data Model

### 1.1 SearchHistory Model (`models.py`)

```python
class SearchHistory(db.Model):
    """Per-user search and ask history."""

    __tablename__ = 'search_history'
    __table_args__ = (
        db.Index('ix_search_history_user_created', 'user_id', 'created_at'),
    )

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False,
    )
    query = db.Column(db.Text, nullable=False)
    query_type = db.Column(db.String(10), nullable=False, default='search')  # 'search' | 'ask'
    namespace = db.Column(db.String(100), nullable=False, default='')
    search_mode = db.Column(db.String(20), nullable=True)  # 'vector' | 'hybrid' | 'keyword'
    result_count = db.Column(db.Integer, nullable=False, default=0)
    answer_preview = db.Column(db.String(200), nullable=True)  # AI 답변 첫 200자 (ask만)
    created_at = db.Column(
        db.DateTime, nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    user = db.relationship('User', backref='search_histories')

    MAX_PER_USER = 500

    def to_dict(self):
        return {
            'id': self.id,
            'query': self.query,
            'query_type': self.query_type,
            'namespace': self.namespace,
            'search_mode': self.search_mode,
            'result_count': self.result_count,
            'answer_preview': self.answer_preview,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }
```

**인덱스 설계:**
- `ix_search_history_user_created`: (`user_id`, `created_at`) — 사용자별 최신순 조회 + 오래된 기록 삭제용
- `user_id` FK에 `ondelete='CASCADE'` — 사용자 삭제 시 기록도 삭제

### 1.2 DB 테이블 생성

`web_app.py`의 `db.create_all()` 호출 시 자동 생성 (기존 패턴 유지, 별도 마이그레이션 불필요).

---

## 2. API Design

모든 엔드포인트는 기존 `v1_bp` 블루프린트(`/api/v1`)에 추가합니다.
기존 `api/response.py`의 `success_response`, `error_response` 패턴을 따릅니다.

### 2.1 기록 저장 (내부 함수)

검색/질문 API 내에서 호출되는 헬퍼 함수. 별도 엔드포인트 아님.

```python
def _save_search_history(user_id, query, query_type, namespace='', search_mode=None, result_count=0, answer_preview=None):
    """Save search history for logged-in user. Enforces MAX_PER_USER limit."""
    try:
        record = SearchHistory(
            user_id=user_id,
            query=query[:500],  # 쿼리 길이 제한
            query_type=query_type,
            namespace=namespace,
            search_mode=search_mode,
            result_count=result_count,
            answer_preview=answer_preview[:200] if answer_preview else None,
        )
        db.session.add(record)
        db.session.commit()

        # 사용자당 MAX_PER_USER 초과 시 오래된 기록 삭제
        count = SearchHistory.query.filter_by(user_id=user_id).count()
        if count > SearchHistory.MAX_PER_USER:
            excess = count - SearchHistory.MAX_PER_USER
            old_ids = (
                db.session.query(SearchHistory.id)
                .filter_by(user_id=user_id)
                .order_by(SearchHistory.created_at.asc())
                .limit(excess)
                .all()
            )
            if old_ids:
                SearchHistory.query.filter(
                    SearchHistory.id.in_([r.id for r in old_ids])
                ).delete(synchronize_session=False)
                db.session.commit()
    except Exception:
        db.session.rollback()
        logging.exception('[SearchHistory] Failed to save')
```

**호출 위치:**
- `api_search()` — 성공 응답 반환 직전, `result_count = len(formatted_results)`
- `api_ask()` — 성공 응답 반환 직전, `answer_preview = answer[:200]`
- `api_ask_stream()` — 스트리밍 완료 후 `generate()` 내부에서 done 이벤트 직전

### 2.2 기록 조회

```
GET /api/v1/search/history?page=1&per_page=20&query_type=ask&namespace=laborlaw
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page` | int | 1 | 페이지 번호 |
| `per_page` | int | 20 | 페이지당 항목 수 (최대 50) |
| `query_type` | string | — | 필터: `search` 또는 `ask` |
| `namespace` | string | — | 필터: 도메인 네임스페이스 |

**Response:**
```json
{
  "success": true,
  "data": {
    "items": [
      {
        "id": 123,
        "query": "반도체 공정 안전",
        "query_type": "search",
        "namespace": "",
        "search_mode": "vector",
        "result_count": 5,
        "answer_preview": null,
        "created_at": "2026-03-11T07:30:00+00:00"
      }
    ],
    "total": 45,
    "page": 1,
    "per_page": 20,
    "pages": 3
  }
}
```

**인증:** `@login_required` — 비로그인 시 401
**권한:** `current_user.id`만 조회 (다른 사용자 기록 접근 불가)

### 2.3 최근 검색어

```
GET /api/v1/search/history/recent?limit=10
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | int | 10 | 반환 개수 (최대 20) |

**Response:**
```json
{
  "success": true,
  "data": {
    "queries": ["반도체 공정 안전", "유해화학물질", "최저임금 계산"]
  }
}
```

**로직:**
- `current_user`의 최근 기록에서 `query` 컬럼을 `created_at DESC`로 조회
- Python `dict.fromkeys()`로 중복 제거 (순서 유지)
- 최대 `limit`개 반환

### 2.4 개별 기록 삭제

```
DELETE /api/v1/search/history/<int:history_id>
```

**Response (성공):**
```json
{
  "success": true,
  "message": "검색 기록이 삭제되었습니다."
}
```

**검증:**
- `history_id`에 해당하는 기록이 `current_user.id` 소유인지 확인
- 타인의 기록 삭제 시도 → 404

### 2.5 전체 기록 삭제

```
DELETE /api/v1/search/history
```

**Response:**
```json
{
  "success": true,
  "message": "전체 검색 기록이 삭제되었습니다.",
  "data": { "deleted_count": 45 }
}
```

---

## 3. 기존 코드 수정 사항

### 3.1 `api/v1/search.py` 수정

| 위치 | 변경 내용 |
|------|----------|
| 파일 상단 import | `from flask_login import current_user, login_required` + `from models import db, SearchHistory` 추가 |
| `api_search()` 함수 | `return success_response(...)` 직전에 `_save_search_history()` 호출 |
| `api_ask()` 함수 | `return success_response(...)` 직전에 `_save_search_history()` 호출 (answer_preview 포함) |
| `api_ask_stream()` `generate()` 내부 | done 이벤트 yield 직전에 `_save_search_history()` 호출 |
| 파일 하단 | 기록 CRUD 엔드포인트 4개 추가 |

### 3.2 `api_search()` 기록 저장 삽입 위치

```python
# 기존 코드: return success_response(...) 앞에 삽입
# --- 검색기록 저장 ---
if hasattr(current_user, 'id') and current_user.is_authenticated:
    _save_search_history(
        user_id=current_user.id,
        query=query,
        query_type='search',
        namespace=namespace,
        search_mode=search_mode,
        result_count=len(formatted_results),
    )

return success_response(data={...})
```

### 3.3 `api_ask()` 기록 저장 삽입 위치

```python
# 기존 코드: return success_response(data=resp_data) 앞에 삽입
if hasattr(current_user, 'id') and current_user.is_authenticated:
    _save_search_history(
        user_id=current_user.id,
        query=query,
        query_type='ask',
        namespace=namespace,
        result_count=len(sources),
        answer_preview=answer[:200] if answer else None,
    )

return success_response(data=resp_data)
```

### 3.4 `api_ask_stream()` 기록 저장 삽입 위치

```python
# generate() 내부, done 이벤트 yield 직전
from flask_login import current_user
if hasattr(current_user, 'id') and current_user.is_authenticated:
    _save_search_history(
        user_id=current_user.id,
        query=query,
        query_type='ask',
        namespace=namespace,
        result_count=len(sources),
    )

# Send done event
done_event = json.dumps({...})
yield f"data: {done_event}\n\n"
```

> 참고: 스트리밍 응답에서는 전체 answer 문자열이 없으므로 `answer_preview`는 `None`으로 저장.

### 3.5 `models.py` 수정

파일 하단(`NewsArticle` 클래스 아래)에 `SearchHistory` 클래스 추가.

### 3.6 `web_app.py` 수정

| 위치 | 변경 내용 |
|------|----------|
| 라우트 추가 | `/history` — 검색기록 페이지 |
| import 추가 | 없음 (기존 `render_template` 사용) |

```python
@app.route('/history')
@login_required
def history():
    return render_template('history.html')
```

---

## 4. UI Design

### 4.1 검색기록 페이지 (`templates/history.html`)

**레이아웃:** 기존 `mypage.html` 스타일과 일관된 다크 테마 카드 UI

**구성 요소:**
```
┌──────────────────────────────────────────┐
│  검색 기록                    [전체 삭제] │
├──────────────────────────────────────────┤
│  필터: [전체 ▼] [전체 도메인 ▼]          │
├──────────────────────────────────────────┤
│  🔍 반도체 공정 안전                  [×] │
│  검색 · 반도체 · 5건 · 3분 전            │
├──────────────────────────────────────────┤
│  💬 유해화학물질 취급 시 주의사항     [×] │
│  AI 질문 · 노동법 · 3건 · 1시간 전      │
│  "유해화학물질을 취급할 때는 다음..."     │
├──────────────────────────────────────────┤
│         [1] [2] [3] ... [다음 >]         │
└──────────────────────────────────────────┘
```

**기능:**
- 각 기록 클릭 → 해당 도메인 페이지로 이동 + 쿼리 자동 입력 후 재검색
- `[×]` 버튼 → 개별 삭제 (confirm 팝업)
- `[전체 삭제]` → 전체 삭제 (confirm 팝업)
- 필터 드롭다운: query_type(`전체`/`검색`/`AI 질문`), namespace(`전체 도메인`/도메인별)
- 페이지네이션: 20건씩

### 4.2 검색창 최근 검색어 (`templates/domain.html` 수정)

**동작:**
- 검색 입력창 focus 시 → `GET /api/v1/search/history/recent` 호출
- 로그인 상태에서만 활성화 (비로그인 시 요청 안 함)
- 최근 검색어 드롭다운 표시 (최대 10개)
- 항목 클릭 → 입력창에 쿼리 입력 + 자동 검색 실행

```
┌─────────────────────────────────┐
│  🔍 검색어를 입력하세요...      │
├─────────────────────────────────┤
│  최근 검색                       │
│  ├ 반도체 공정 안전              │
│  ├ 유해화학물질 취급             │
│  └ 최저임금 계산법               │
└─────────────────────────────────┘
```

### 4.3 네비게이션 링크

`templates/base.html`에 마이페이지 드롭다운 또는 네비게이션에 "검색 기록" 링크 추가.
로그인한 사용자에게만 표시.

---

## 5. Implementation Order

| Step | 파일 | 작업 내용 | 의존성 |
|------|------|----------|--------|
| 1 | `models.py` | `SearchHistory` 모델 클래스 추가 | 없음 |
| 2 | `api/v1/search.py` | `_save_search_history()` 헬퍼 함수 추가 | Step 1 |
| 3 | `api/v1/search.py` | `api_search()`, `api_ask()`, `api_ask_stream()`에 저장 호출 삽입 | Step 2 |
| 4 | `api/v1/search.py` | 기록 조회/삭제/최근 검색어 API 엔드포인트 추가 | Step 1 |
| 5 | `web_app.py` | `/history` 라우트 추가 | 없음 |
| 6 | `templates/history.html` | 검색기록 페이지 UI 구현 | Step 4, 5 |
| 7 | `templates/domain.html` | 검색창 최근 검색어 드롭다운 추가 | Step 4 |

---

## 6. Error Handling

| 시나리오 | 처리 |
|----------|------|
| 비로그인 사용자가 기록 API 접근 | `@login_required` → 401 응답 |
| 타인의 기록 삭제 시도 | 기록 조회 시 `user_id == current_user.id` 필터 → 404 |
| 기록 저장 실패 (DB 오류) | `try/except` + `db.session.rollback()` — 검색 응답에 영향 없음 |
| 페이지네이션 범위 초과 | 빈 `items` 리스트 반환 |

---

## 7. Security Considerations

| 항목 | 대응 |
|------|------|
| 권한 분리 | 모든 기록 쿼리에 `user_id == current_user.id` 필터 적용 |
| CSRF | API 블루프린트(`v1_bp`)는 기존 CSRF exempt 설정 유지 |
| SQL Injection | SQLAlchemy ORM 사용으로 파라미터 바인딩 자동 적용 |
| Rate Limiting | 기록 조회 API에 `@rate_limit("30 per minute")` 적용 |
| 입력 검증 | `query` 500자 제한, `per_page` 최대 50, `limit` 최대 20 |

---

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-03-11 | — | Initial design |
