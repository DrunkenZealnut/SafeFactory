# Plan: 나의 자료목록에서 AI 채팅하기

## Executive Summary

| 항목 | 내용 |
|------|------|
| Feature | my-docs-ai-chat |
| 작성일 | 2026-03-21 |
| 예상 기간 | 1일 |
| 난이도 | Medium |

### Value Delivered

| 관점 | 내용 |
|------|------|
| **Problem** | 북마크한 자료에 대해 질문하려면 학습 페이지로 이동해야 하고, 전체 DB 검색이라 관계없는 결과가 섞임 |
| **Solution** | 나의 자료 페이지에서 직접 AI 채팅 — 북마크된 자료(source_file)만 대상으로 RAG 검색 |
| **Function UX Effect** | "내가 모아둔 자료"에 집중하여 질문 → 정확한 맥락의 답변 |
| **Core Value** | 개인화된 학습 경험 — 큐레이션한 자료로 맞춤형 AI 튜터 |

---

## 1. 현재 구조 (AS-IS)

```text
나의 자료 페이지
├── 검색 + NCS 브라우저 (자료 추가용)
├── 북마크 목록
└── PDF 뷰어

AI 질문은 /learn 페이지에서만 가능
→ 전체 namespace 대상 검색 (내 자료 필터 없음)
```

## 2. 목표 구조 (TO-BE)

```text
나의 자료 페이지
├── 검색 + NCS 브라우저 (자료 추가용)
├── 💬 AI 채팅 영역 ← NEW
│   ├── 질문 입력
│   ├── "내 자료 N개를 기반으로 답변합니다" 표시
│   ├── SSE 스트리밍 답변
│   └── 출처 표시 (북마크된 자료 중 매칭된 것)
├── 북마크 목록
└── PDF 뷰어
```

---

## 3. 구현 항목

### 3.1 [P1] 백엔드 — source_file 필터 지원

기존 `/api/v1/ask/stream`에 `source_files` 파라미터 추가:
- `source_files: ["path/to/file1", "path/to/file2", ...]`
- Pinecone 쿼리 시 `filter: {"source_file": {"$in": source_files}}`
- 빈 배열이면 기존 동작 (전체 검색)

수정 파일: `services/rag_pipeline.py` (Pinecone 쿼리에 filter 추가)

### 3.2 [P1] 프론트엔드 — 채팅 UI

`my_documents.html`에 추가:
- 질문 입력 textarea + 질문하기 버튼
- "내 자료 N개를 기반으로 답변합니다" 안내
- SSE 스트리밍 답변 표시 (marked.js + DOMPurify)
- 출처 표시

### 3.3 [P1] 채팅 JS — 북마크 source_files를 API에 전달

1. 현재 사용자의 북마크 목록에서 `source_file` 배열 추출
2. `/api/v1/ask/stream` POST 시 `source_files` 포함
3. SSE 응답 파싱 + 실시간 렌더링

---

## 4. 영향 범위

| 파일 | 변경 내용 |
|------|-----------|
| `services/rag_pipeline.py` | Pinecone 검색에 source_file 필터 추가 |
| `api/v1/search.py` | source_files 파라미터 전달 |
| `templates/my_documents.html` | AI 채팅 UI + JS |

---

## 5. 제외 사항 (YAGNI)
- 대화 히스토리 저장 — 첫 버전은 단발 질문
- 자료 선택 후 질문 (특정 자료만) — 전체 북마크 대상
- 채팅 내보내기 — 불필요
