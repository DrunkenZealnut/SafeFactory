# Plan: Reranker 토큰 제한 이슈 수정

> Feature: reranker-token-limit-fix
> Created: 2026-03-08
> Status: Plan

---

## Executive Summary

| 관점 | 설명 |
|------|------|
| **Problem** | Pinecone reranker(bge-reranker-v2-m3)의 query+document 쌍이 1024 토큰 제한을 초과하여 reranking 실패 및 검색 결과 누락 발생 |
| **Solution** | document content truncation 로직을 query 길이를 고려하여 동적으로 조정, 1024 토큰 제한 내로 보장 |
| **Function UX Effect** | Contextual Retrieval 적용 후에도 모든 쿼리에서 reranking이 정상 동작하여 검색 품질 유지 |
| **Core Value** | reranker 실패로 인한 early response(빈 결과) 제거, 검색 안정성 100% 달성 |

---

## 1. Root Cause

### 에러 메시지
```
INVALID_ARGUMENT: Request contains a query+document pair with 1053 tokens,
which exceeds the maximum token limit of 1024 for each query+document pair.
```

### 코드 위치
`src/reranker.py:293` — `PineconeReranker.rerank()`
```python
# 현재: 고정 3000 chars truncation
documents.append(content[:3000] if content else "")
```

### 원인
- `bge-reranker-v2-m3` 제한: query + document ≤ 1024 tokens
- 한국어: 3000 chars ÷ 3 = ~1000 tokens (document만)
- query ~50 tokens 추가 → 총 ~1050 tokens → **1024 초과**
- Contextual Retrieval prefix(50-100 tokens)가 추가되면서 document가 더 길어짐

---

## 2. Fix Strategy

### 변경 파일
- `src/reranker.py` — `PineconeReranker.rerank()` 메서드

### 변경 내용
1. query 토큰 수 추정 (`len(query) // 3`)
2. document 허용 토큰 = 1024 - query 토큰 - safety margin(24)
3. document chars = 허용 토큰 × 3
4. 기존 `content[:3000]` → `content[:max_doc_chars]`

### 예상 코드
```python
MAX_PAIR_TOKENS = 1024
SAFETY_MARGIN = 24  # 토크나이저 오차 여유

query_tokens_est = len(query) // 3 + 1
max_doc_tokens = MAX_PAIR_TOKENS - query_tokens_est - SAFETY_MARGIN
max_doc_chars = max(300, max_doc_tokens * 3)  # 최소 300자 보장

documents.append(content[:max_doc_chars] if content else "")
```

---

## 3. Success Criteria

| Metric | Target |
|--------|--------|
| `semi-001` (CVD) 쿼리 reranking 성공 | Early response 제거 |
| semiconductor-v2 전체 4 쿼리 eval 통과 | Failure Rate 0% |
| 기존 다른 도메인 eval 영향 없음 | 동일 또는 개선 |
