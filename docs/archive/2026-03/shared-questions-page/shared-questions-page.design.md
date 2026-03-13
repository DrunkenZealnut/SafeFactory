# Design: 공유 질문 전용 페이지 (Shared Questions Page)

> **Feature**: shared-questions-page
> **Date**: 2026-03-13
> **Plan**: [shared-questions-page.plan.md](../../01-plan/features/shared-questions-page.plan.md)

---

## 1. 설계 변경사항 (Plan 대비 추가)

사용자 요구: **"공유한 질문에 대한 답변도 저장하고, 질문을 클릭하면 답변을 보여줘야 합니다."**

| 항목 | Plan 기존 | Design 변경 |
|------|----------|------------|
| 답변 저장 | `answer_preview` (300자) 만 저장 | `answer_full` (Text, 전체 마크다운) 컬럼 추가 |
| 질문 클릭 동작 | 도메인 페이지로 이동 → AI 재질의 | **질문 페이지 내에서 답변 아코디언 펼침** |
| DB 변경 | 없음 | `SharedQuestion` 테이블에 `answer_full` 컬럼 추가 |

---

## 2. 데이터 모델 변경

### 2.1 SharedQuestion 테이블 수정

```python
# models.py — SharedQuestion 클래스에 컬럼 추가
answer_full = db.Column(db.Text, nullable=True)  # 전체 AI 답변 (마크다운)
```

**마이그레이션**: SQLite `ALTER TABLE` — 컬럼 추가만이므로 무중단 적용 가능

```sql
ALTER TABLE shared_questions ADD COLUMN answer_full TEXT;
```

### 2.2 to_dict 수정

```python
def to_dict(self, liked_by_me=False, include_answer=False):
    d = {
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
    if include_answer:
        d['answer_full'] = self.answer_full
    return d
```

`include_answer=False`가 기본값이므로 기존 API(popular, wordcloud 등) 응답 크기에 영향 없음.

---

## 3. API 설계

### 3.1 수정: POST /api/v1/questions/share

**변경**: `answer_full` 파라미터 추가 수신

```python
answer_full = (data.get('answer_full') or '').strip()
if answer_full:
    answer_full = answer_full[:10000]  # 최대 10,000자 제한
```

**기존 호환**: `answer_full` 미전송 시 None 저장 (기존 클라이언트 영향 없음)

### 3.2 수정: GET /api/v1/questions/popular

**변경**: `page`/`per_page`/`sort` 파라미터 추가, `include_answer` 파라미터 추가

```
GET /api/v1/questions/popular?page=1&per_page=20&sort=likes&namespace=&include_answer=1
```

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `page` | int | 1 | 페이지 번호 (없으면 기존 limit 방식) |
| `per_page` | int | 20 | 페이지당 개수 (max 50) |
| `sort` | string | `likes` | `likes` (인기순) / `recent` (최신순) |
| `namespace` | string | `''` | 도메인 필터 (빈값=전체) |
| `include_answer` | int | 0 | 1이면 `answer_full` 포함 |
| `limit` | int | 10 | (레거시) page 미사용 시 기존 방식 |

**응답 (page 사용 시)**:
```json
{
  "success": true,
  "data": {
    "questions": [...],
    "total": 150,
    "page": 1,
    "per_page": 20,
    "pages": 8
  }
}
```

**하위 호환**: `page` 미전달 시 기존 `limit` 방식 동작 (도메인 인기질문·워드클라우드 영향 없음)

### 3.3 기존 API (변경 없음)

| API | 변경 | 이유 |
|-----|------|------|
| `POST /questions/<id>/like` | 없음 | 그대로 사용 |
| `DELETE /questions/<id>` | 없음 | 그대로 사용 |
| `GET /questions/my` | `include_answer` 파라미터 추가 | 내 질문에서도 답변 확인 |
| `GET /questions/wordcloud` | 없음 | 키워드 추출에 answer 불필요 |

---

## 4. 프론트엔드 설계

### 4.1 신규 파일: templates/questions.html

**구조**:
```
questions.html
├── 헤더 (타이틀 + 워드클라우드 크로스 링크)
├── 탭: [전체 질문] [내 질문]
├── 필터 바
│   ├── 도메인 필터 (전체/반도체/현장실습/안전보건/MSDS)
│   └── 정렬 (인기순/최신순)
├── 질문 목록
│   ├── 질문 카드 (클릭 → 아코디언 답변 펼침)
│   │   ├── 질문 텍스트
│   │   ├── 도메인 뱃지 + 작성자 + 시간
│   │   ├── 좋아요 버튼 (♡/♥ + count)
│   │   └── 답변 영역 (아코디언, 마크다운 렌더링)
│   └── ...
├── 페이지네이션
└── 스크립트
```

### 4.2 질문 카드 상세 설계

```
┌─────────────────────────────────────────────────────┐
│ ▶ CVD 공정과 PVD 공정의 차이점은 무엇인가요?            │
│   🔬 반도체  ·  홍길동  ·  3일 전              ♥ 18  │
├─────────────────────────────────────────────────────┤ ← 클릭 시 펼침
│ 🤖 AI 답변                                          │
│ CVD(Chemical Vapor Deposition)와                     │
│ PVD(Physical Vapor Deposition)의 주요 차이점:         │
│                                                     │
│ 1. **CVD**: 화학 반응을 이용하여 기판 위에...            │
│ 2. **PVD**: 물리적 방법(스퍼터링 등)으로...              │
│ ...                                                 │
│                                    [🔍 도메인에서 재검색] │
└─────────────────────────────────────────────────────┘
```

**인터랙션**:
1. 질문 카드 클릭 → 아코디언 펼침/접힘 (토글)
2. 답변 영역은 `marked.js`로 마크다운 렌더링 + `DOMPurify` XSS 방지
3. 답변이 없는 경우(`answer_full` = null) → "답변이 저장되지 않았습니다. 도메인에서 직접 질문해보세요." + 도메인 링크 버튼
4. "도메인에서 재검색" 버튼 → `/{domain}?q={query}` 이동

### 4.3 "내 질문" 탭

```
┌─────────────────────────────────────────────────────┐
│ ▶ 반도체 수율 향상 방법에 대해 알려주세요                │
│   🔬 반도체  ·  3일 전  ·  ♥ 20           [🗑️ 삭제]  │
├─────────────────────────────────────────────────────┤
│ 🤖 AI 답변 ...                                      │
└─────────────────────────────────────────────────────┘
```

- 로그인 필수 (미로그인 시 로그인 유도 메시지)
- 삭제 버튼 (확인 다이얼로그 → DELETE API)
- 좋아요 수 표시 (토글은 불필요 — 내 질문이므로)

### 4.4 좋아요 인터랙션

```
비로그인: ♡ 18 → 클릭 → "로그인 후 이용 가능합니다" 토스트
로그인:   ♡ 18 → 클릭 → ♥ 19 (POST /api/v1/questions/{id}/like)
         ♥ 19 → 클릭 → ♡ 18 (토글)
```

### 4.5 페이지네이션

`history.html` 패턴 재사용: `<< 1 2 3 4 5 >>` 숫자 페이지네이션.

---

## 5. 수정 파일 상세

### 5.1 models.py

| 변경 | 내용 |
|------|------|
| `SharedQuestion` | `answer_full = db.Column(db.Text, nullable=True)` 추가 |
| `to_dict()` | `include_answer` 파라미터 추가, `answer_full` 조건부 포함 |

### 5.2 api/v1/questions.py

| 함수 | 변경 |
|------|------|
| `api_question_share()` | `answer_full` 수신 및 저장 (max 10,000자) |
| `api_question_popular()` | `page`/`per_page`/`sort`/`include_answer` 파라미터 추가. `page` 미전달 시 기존 `limit` 방식 유지 |
| `api_question_my()` | `include_answer` 파라미터 추가 |

### 5.3 web_app.py

```python
@app.route('/questions')
def questions():
    """Shared questions page."""
    return render_template('questions.html')
```

### 5.4 templates/base.html

네비게이션 탭 추가 (클라우드 앞 또는 뒤):
```html
<a href="/questions" class="sf-nav-tab {% block nav_questions %}{% endblock %}">❓ 질문</a>
```

모바일 메뉴에도 동일 추가.

### 5.5 templates/wordcloud.html

헤더에 크로스 링크 추가:
```html
<a href="/questions" class="...">📋 질문 목록 보기</a>
```

### 5.6 templates/domain.html

`shareQuestion()` 함수에서 `answer_full` 전송 추가:
```javascript
body: JSON.stringify({
    query: _lastAskedQuery,
    namespace: _lastAskedNamespace,
    answer_preview: _lastAnswerPreview,
    answer_full: window._lastAnswerMd || '',  // ← 추가
})
```

---

## 6. 구현 순서

```
1. models.py          — answer_full 컬럼 추가 + to_dict 수정
2. DB 마이그레이션     — ALTER TABLE shared_questions ADD COLUMN answer_full TEXT
3. api/v1/questions.py — share API에 answer_full 저장, popular API에 페이지네이션·정렬·include_answer 추가
4. domain.html         — shareQuestion()에서 answer_full 전송
5. web_app.py          — /questions 라우트 추가
6. questions.html      — 전용 페이지 템플릿 (필터·질문카드·아코디언답변·좋아요·내질문탭·페이지네이션)
7. base.html           — 네비게이션 "❓ 질문" 탭 추가
8. wordcloud.html      — 크로스 링크 추가
```

---

## 7. 외부 의존성

| 라이브러리 | 용도 | 출처 |
|-----------|------|------|
| `marked.js` | 마크다운 → HTML 렌더링 | CDN (이미 domain.html에서 사용 중) |
| `DOMPurify` | XSS 방지 | CDN (이미 domain.html에서 사용 중) |

**추가 의존성 없음** — 기존 CDN 라이브러리 재사용.

---

## 8. 위험 요소 및 완화

| 위험 | 완화 |
|------|------|
| `answer_full` 저장으로 DB 용량 증가 | 10,000자 제한 + SQLite Text 타입은 효율적 |
| `include_answer=1` 응답 크기 증가 | 목록에서는 `include_answer=0`(기본), 아코디언 펼침 시 개별 로드 또는 첫 로드 시 포함 |
| `popular` API 변경 시 기존 사용처 | `page` 미전달 → 기존 `limit` 방식 유지 (하위 호환 보장) |
| 기존 공유 질문의 `answer_full` = null | "답변이 저장되지 않았습니다" 폴백 UI 표시 + 도메인 재검색 링크 |
| 마크다운 렌더링 XSS | DOMPurify 적용 (기존 domain.html 패턴 동일) |

---

## 9. 데이터 로딩 전략

질문 목록에서 답변 전체를 한꺼번에 로드하면 응답이 커지므로:

**방안: 첫 로드 시 답변 포함**
- `include_answer=1`로 20개씩 페이지네이션 → 한 번 로드로 아코디언 즉시 펼침
- 평균 답변 1,000자 가정 시 20개 = ~20KB (허용 범위)
- 장점: 추가 API 호출 없이 즉시 답변 표시, 구현 단순

---

## 10. 영향도 요약

| 항목 | 수량 |
|------|------|
| 신규 파일 | 1개 (`templates/questions.html`) |
| 수정 파일 | 6개 (`models.py`, `api/v1/questions.py`, `web_app.py`, `base.html`, `wordcloud.html`, `domain.html`) |
| DB 변경 | 1건 (ALTER TABLE — 컬럼 추가) |
| 예상 코드량 | ~400줄 (템플릿 ~300줄, API ~60줄, 모델 ~10줄, 라우트+기타 ~30줄) |
