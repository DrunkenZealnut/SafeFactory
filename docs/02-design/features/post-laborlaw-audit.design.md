# 주석 처리 이후 전체 검색 모듈 점검 Design

> Plan 참조: `docs/01-plan/features/post-laborlaw-audit.plan.md`

## 1. 변경 범위

| 파일 | 작업 | 변경 유형 |
|------|------|-----------|
| `templates/admin.html` | FOLDER_ICONS laborlaw 키 주석 처리 | 1줄 수정 |
| `src/semantic_chunker.py` | 3개 스텁 함수 + 1개 속성 + 1개 변수 정리 | dead code 제거 |
| `services/rag_pipeline.py` | laborlaw 메타데이터 키 순회 주석 처리 | dead code 제거 |
| `services/domain_config.py` | stale 주석 정리 | 주석 수정 |

---

## 2. 상세 변경 사항

### 2.1 `templates/admin.html` (L1046)

```javascript
// 변경 전
laborlaw:'⚖️', ncs:'🔬', ...

// 변경 후
// [LABORLAW_DISABLED] laborlaw:'⚖️',
ncs:'🔬', ...
```

### 2.2 `src/semantic_chunker.py`

| 대상 | 라인 | 작업 |
|------|------|------|
| `LABORLAW_SECTION_PATTERNS` | ~L439 | 주석 처리된 빈 배열 → `# [LABORLAW_DISABLED]` 마커와 함께 전체 제거 또는 빈 리스트 유지 |
| `_split_by_laborlaw_structure()` | ~L446 | 스텁 함수 본문을 `pass` + `return []`로 단순화 (이미 되어 있으면 유지) |
| `_extract_laborlaw_metadata()` | ~L662 | 스텁 확인 (이미 `return {}`) |
| `_classify_laborlaw_category()` | ~L702 | 스텁 확인 (이미 `return 'general'`) |
| `laborlaw_metadata = {}` | ~L919 | 미사용 할당 제거 |

### 2.3 `services/rag_pipeline.py` (L1103-1107)

```python
# 변경 전 — laborlaw 메타데이터 키 순회
for key in ('content_type', 'law_name', 'law_number', 'law_date',
            'law_category', 'article_number', 'case_collection'):
    if metadata.get(key):
        source_entry[key] = metadata[key]

# 변경 후 — 주석 처리
# [LABORLAW_DISABLED] Laborlaw metadata keys (re-enable when laborlaw is active)
# for key in ('content_type', 'law_name', 'law_number', 'law_date',
#             'law_category', 'article_number', 'case_collection'):
#     if metadata.get(key):
#         source_entry[key] = metadata[key]
```

### 2.4 `services/domain_config.py` (L240 부근)

stale 주석에서 laborlaw 예시 제거.

---

## 3. 구현 체크리스트

1. [ ] `templates/admin.html` — FOLDER_ICONS laborlaw 키 주석 처리
2. [ ] `src/semantic_chunker.py` — `laborlaw_metadata = {}` 미사용 할당 제거
3. [ ] `services/rag_pipeline.py` — laborlaw 메타데이터 키 순회 주석 처리
4. [ ] `services/domain_config.py` — stale 주석 정리
5. [ ] 앱 기동 테스트 (`python -c "from web_app import app"`)

---

## 4. 검증 기준

- `python -c "from web_app import app"` 성공
- `grep -c "LABORLAW_DISABLED" *.py` 로 모든 마커 일관성 확인
- 기존 13개 파일의 `[LABORLAW_DISABLED]` 마커 유지
