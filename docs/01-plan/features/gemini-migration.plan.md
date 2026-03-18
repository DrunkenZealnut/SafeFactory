# Gemini Embedding Migration Planning Document

> **Summary**: OpenAI text-embedding-3-small에서 Gemini Embedding 2로 전체 검색 파이프라인 전환
>
> **Project**: SafeFactory
> **Author**: zealnutkim
> **Date**: 2026-03-18
> **Status**: Draft

---

## Executive Summary

| Perspective | Content |
|-------------|---------|
| **Problem** | OpenAI 임베딩 대비 검색 유사도가 0.49~0.54로 낮아 RAG 응답 품질이 제한됨 |
| **Solution** | Gemini Embedding 2 (MRL 1536D)로 전환하여 검색 품질 대폭 개선 |
| **Function/UX Effect** | 검색 유사도 +0.25 향상 (0.53→0.78), 더 정확한 문서 검색 및 AI 답변 품질 개선 |
| **Core Value** | 코드 변경 최소화로 전체 검색 품질을 39% 향상, 비용 증가 무시 가능 수준 |

---

## 1. Overview

### 1.1 Purpose

전체 RAG 파이프라인의 임베딩 모델을 OpenAI text-embedding-3-small에서 Gemini Embedding 2로 전환하여 검색 품질을 개선한다.

### 1.2 Background

A/B 벤치마크 결과 Gemini Embedding 2가 모든 도메인에서 압도적 우위를 보임:

| 도메인 | OpenAI 유사도 | Gemini 유사도 | 차이 | 승률 |
|--------|:---:|:---:|:---:|:---:|
| semiconductor-v2 | 0.5273 | 0.7449 | +0.2176 | 5/5 |
| laborlaw-v2 | 0.5340 | 0.7886 | +0.2546 | 5/5 |
| counsel | 0.5036 | 0.7403 | +0.2367 | 5/5 |
| precedent | 0.5374 | 0.8123 | +0.2749 | 5/5 |

- **20전 20승**, 평균 유사도 +0.245 향상
- Jaccard Overlap 2~14%: 두 모델이 완전히 다른(더 관련성 높은) 문서를 검색
- Gemini 네임스페이스 이미 인제스트 완료 (7개 네임스페이스)

### 1.3 Related Documents

- 벤치마크 결과: `scripts/benchmark_results.json`
- 인제스트 스크립트: `scripts/ingest_gemini_test.py`
- 이전 설계: `docs/archive/2026-03/google-embedding-model-test/`

---

## 2. Scope

### 2.1 In Scope

- [x] Gemini 네임스페이스 인제스트 (이미 완료)
- [ ] `singletons.py`의 `get_agent()`가 admin 설정의 `embedding_model`을 참조하도록 수정
- [ ] 네임스페이스 라우팅: 모델에 따라 `-gemini` 접미사 자동 매핑
- [ ] Admin 설정 기본값을 `gemini-embedding-2-preview`로 변경
- [ ] 기존 OpenAI 네임스페이스 보존 (롤백 가능)

### 2.2 Out of Scope

- OpenAI 네임스페이스 삭제 (당분간 보존)
- Pinecone 인덱스 차원 변경 (MRL 1536D로 동일 차원 유지)
- SemanticChunker 임베딩 모델 변경 (청킹 단계는 OpenAI 유지 가능)
- Gemini Embedding 001 지원 (추후 검토)

---

## 3. Requirements

### 3.1 Functional Requirements

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| FR-01 | `get_agent()` 싱글톤이 `embedding_model` 설정을 반영 | High | Pending |
| FR-02 | Gemini 모델 선택 시 자동으로 `-gemini` 접미사 네임스페이스 검색 | High | Pending |
| FR-03 | Admin 패널에서 embedding model 변경 시 싱글톤 캐시 무효화 | High | Pending |
| FR-04 | `RETRIEVAL_QUERY` / `RETRIEVAL_DOCUMENT` task_type 자동 적용 | Medium | Already Done |
| FR-05 | CLI (`main.py`)에서 `--embedding-model gemini-embedding-2-preview` 지원 | Medium | Pending |
| FR-06 | 기존 OpenAI 네임스페이스로 즉시 롤백 가능 | High | Pending |

### 3.2 Non-Functional Requirements

| Category | Criteria | Measurement Method |
|----------|----------|-------------------|
| Performance | 검색 레이턴시 < 1000ms | 벤치마크 스크립트 |
| Compatibility | 기존 1536D 인덱스 호환 | MRL dimension=1536 |
| Rollback | Admin에서 OpenAI로 즉시 전환 가능 | 설정 변경 테스트 |

---

## 4. Success Criteria

### 4.1 Definition of Done

- [ ] Admin에서 `gemini-embedding-2-preview` 선택 시 Gemini 네임스페이스로 검색됨
- [ ] 웹 앱 검색/Q&A가 Gemini 임베딩으로 정상 동작
- [ ] 롤백: `text-embedding-3-small`로 변경 시 OpenAI 네임스페이스로 즉시 복귀
- [ ] CLI에서 Gemini 모델로 인제스트 가능

### 4.2 Quality Criteria

- [ ] 기존 모든 검색 기능 정상 동작 (cross-search 포함)
- [ ] 싱글톤 캐시 무효화 정상 작동
- [ ] 에러 없이 프로덕션 배포 완료

---

## 5. Risks and Mitigation

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Gemini API 장애 | High | Low | Admin에서 OpenAI로 즉시 롤백 |
| 네임스페이스 매핑 오류 | High | Medium | 매핑 테이블 + 존재 여부 검증 로직 |
| 싱글톤 캐시 무효화 누락 | Medium | Medium | `embedding_model` 변경 시 `_agent = None` 처리 |
| Gemini API 비용 증가 | Low | Low | 절대 금액 미미 ($0.006/1K 쿼리) |
| HybridSearch BM25 영향 | Low | Low | BM25는 텍스트 기반이라 임베딩 무관 |

---

## 6. Architecture Considerations

### 6.1 핵심 변경 지점

```
[현재]
singletons.py: get_agent() → PineconeAgent(embedding_model="text-embedding-3-small")  # 하드코딩
domain_config.py: namespace = "semiconductor-v2"                                        # 고정

[변경 후]
singletons.py: get_agent() → PineconeAgent(embedding_model=get_setting('embedding_model'))
domain_config.py: namespace = resolve_namespace("semiconductor-v2", embedding_model)
                                                  → "semiconductor-v2-gemini" (Gemini일 때)
```

### 6.2 네임스페이스 매핑 전략

**Option A**: `-gemini` 접미사 자동 추가 (선택)
```python
def resolve_namespace(base_ns: str, model: str) -> str:
    if model.startswith("gemini-embedding"):
        return f"{base_ns}-gemini"
    return base_ns
```

**Option B**: 네임스페이스 리네임 (비권장 — Pinecone은 rename 미지원, 재인제스트 필요)

### 6.3 영향 범위

| 파일 | 변경 내용 |
|------|-----------|
| `services/singletons.py` | `get_agent()`에 `embedding_model` 설정 반영 + 캐시 무효화 |
| `services/domain_config.py` | `resolve_namespace()` 함수 추가 |
| `services/rag_pipeline.py` | 네임스페이스 해석 시 `resolve_namespace()` 적용 |
| `services/settings.py` | 기본값 변경 + `embedding_model` 변경 시 agent 무효화 |
| `api/v1/search.py` | 네임스페이스 해석에 `resolve_namespace()` 적용 |
| `main.py` | CLI `--embedding-model` 선택지에 Gemini 추가 |

### 6.4 현재 Gemini 네임스페이스 현황

| OpenAI 네임스페이스 | Gemini 네임스페이스 | 벡터 수 | 상태 |
|---------------------|---------------------|---------|------|
| semiconductor-v2 (10,034) | semiconductor-v2-gemini | 10,022 | Ready |
| laborlaw-v2 (9,088) | laborlaw-v2-gemini | 9,079 | Ready |
| counsel (1,244) | counsel-gemini | 1,244 | Ready |
| precedent (6,540) | precedent-gemini | 6,534 | Ready |
| field-training (92) | field-training-gemini | 92 | Ready |
| kosha (205) | kosha-gemini | 205 | Ready |
| safeguide (308) | safeguide-gemini | 308 | Ready |

**모든 7개 네임스페이스 인제스트 완료.**

---

## 7. Convention Prerequisites

### 7.1 Existing Project Conventions

- [x] `CLAUDE.md` has coding conventions section
- [x] Singleton pattern with double-checked locking (`services/singletons.py`)
- [x] Admin settings via `get_setting()` / `set_setting()`
- [x] Cache invalidation by nullifying singletons

### 7.2 Environment Variables

| Variable | Purpose | Status |
|----------|---------|--------|
| `GEMINI_API_KEY` | Gemini Embedding API 호출 | Already exists |
| `OPENAI_API_KEY` | OpenAI Embedding (롤백용) | Already exists |

---

## 8. Implementation Order

1. `services/domain_config.py` — `resolve_namespace()` 추가
2. `services/singletons.py` — `get_agent()`가 `embedding_model` 설정 참조
3. `services/settings.py` — `embedding_model` 변경 시 `_agent` 캐시 무효화
4. `services/rag_pipeline.py` — 네임스페이스 resolve 적용
5. `api/v1/search.py` — 네임스페이스 resolve 적용
6. `main.py` — CLI Gemini 모델 선택지 추가
7. Admin 기본값 변경 → 프로덕션 배포

---

## 9. Next Steps

1. [ ] Write design document (`gemini-migration.design.md`)
2. [ ] Implementation
3. [ ] 프로덕션 배포 및 모니터링

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-03-18 | Initial draft | zealnutkim |
