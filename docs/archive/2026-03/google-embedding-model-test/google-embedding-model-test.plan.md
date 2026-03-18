# Plan: Gemini Embedding 2 테스트

> **Feature**: google-embedding-model-test
> **Created**: 2026-03-18
> **Updated**: 2026-03-18
> **Status**: Draft
> **Level**: Dynamic

---

## Executive Summary

| 항목 | 내용 |
|------|------|
| Feature | Gemini Embedding 2 A/B 비교 테스트 |
| 시작일 | 2026-03-18 |
| 예상 기간 | 3-5일 |
| 영향 범위 | 임베딩 생성, 벡터 검색, RAG 파이프라인 |

### Value Delivered (4-Perspective)

| 관점 | 설명 |
|------|------|
| **Problem** | OpenAI text-embedding-3-small(1536D) 단일 모델 종속. 한국어 도메인 임베딩 품질 비교 데이터 부재 |
| **Solution** | Google Gemini Embedding 2 Preview(3072D, MRL로 1536D 호환) 통합 후 동일 인덱스에서 A/B 비교 테스트 |
| **Function UX Effect** | 5개 도메인 25개 쿼리셋 정량 비교로 검색 품질 향상 가능성 검증 |
| **Core Value** | 멀티모달 임베딩 기반 마련 + 벤더 종속 탈피 + 한국어 검색 품질 최적화 |

---

## 1. 배경 및 목적

### 1.1 현재 상태

- **임베딩 모델**: OpenAI `text-embedding-3-small` (1536D)
- **사용처**:
  - 문서 처리 파이프라인 (`src/embedding_generator.py`) — 문서 청크 → 벡터 변환
  - 검색 쿼리 임베딩 (`src/agent.py:453`) — 사용자 쿼리 → 벡터 변환
  - 하이브리드 검색의 벡터 검색 부분
- **벡터 DB**: Pinecone (AWS us-east-1, Serverless) — 1536D 인덱스
- **admin 설정**: `embedding_model` 키 존재, OpenAI 모델만 지원 (`services/settings.py:22`)
- **지원 모델 목록** (admin UI): `text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002`
- **Gemini SDK**: `google-genai` 이미 설치, `GEMINI_API_KEY` 보유, `singletons.py`에 `get_gemini_client()` 존재

### 1.2 Gemini Embedding 2 모델 사양

| 항목 | 내용 |
|------|------|
| **모델 ID** | `gemini-embedding-2-preview` |
| **출시일** | 2026-03-10 (Public Preview) |
| **타입** | **멀티모달** (텍스트, 이미지, 비디오, 오디오, PDF) |
| **기본 차원** | 3,072D |
| **MRL 조절** | 128 ~ 3,072 (권장: 768, 1536, 3072) |
| **입력 토큰** | **8,192** (기존 text-embedding-004의 4배) |
| **언어** | 100+ 언어 지원 |
| **가격 (텍스트)** | $0.20/1M tokens (무료 티어 존재) |
| **가격 (이미지)** | $0.45/1M tokens (~$0.00012/image) |
| **Task Types** | RETRIEVAL_QUERY, RETRIEVAL_DOCUMENT, SEMANTIC_SIMILARITY, CLASSIFICATION 등 |

**비교 모델 참고**:

| 모델 | 차원 | 입력 토큰 | 가격/1M tokens | 상태 |
|------|------|----------|---------------|------|
| OpenAI `text-embedding-3-small` | 1536 | 8,191 | $0.02 | Stable |
| Gemini `gemini-embedding-001` | 3072 (MRL) | 2,048 | $0.15 | Stable (GA) |
| Gemini `gemini-embedding-2-preview` | 3072 (MRL) | **8,192** | $0.20 | **Preview** |
| ~~`text-embedding-004`~~ | 768 | 2,048 | - | **Deprecated** (2026-01-14) |

### 1.3 Gemini Embedding 2를 선택하는 이유

1. **1536D 호환**: `output_dimensionality=1536` 설정으로 **기존 Pinecone 인덱스에서 바로 테스트 가능** (별도 인덱스 불필요)
2. **8,192 토큰 입력**: OpenAI(8,191)과 거의 동일 — truncation 로직 호환
3. **기존 인프라 재활용**: `GEMINI_API_KEY` + `google-genai` SDK 이미 존재
4. **멀티모달 확장성**: 향후 이미지 문서(반도체 공정 사진, MSDS 이미지) 직접 임베딩 가능
5. **최신 모델**: 2026-03-10 출시, 최신 학습 데이터 반영
6. **Task Type 지원**: `RETRIEVAL_QUERY` / `RETRIEVAL_DOCUMENT` 구분으로 비대칭 검색 최적화

### 1.4 목적

1. `EmbeddingGenerator`에 Gemini Embedding 2 지원 추가
2. `output_dimensionality=1536`으로 **기존 인덱스 내 별도 네임스페이스**에서 테스트
3. 기존 OpenAI 임베딩과 **동일 쿼리셋**으로 검색 품질 A/B 비교
4. 비용, 속도, 정확도 3개 축에서 정량 평가
5. 결과에 따라 기본 임베딩 모델 전환 여부 결정

---

## 2. 범위 (Scope)

### 2.1 In Scope

- [ ] `EmbeddingGenerator` 클래스에 Gemini Embedding 2 지원 추가 (google-genai SDK 활용)
- [ ] Pinecone 기존 인덱스 내 테스트 네임스페이스 생성 (`semiconductor-v2-gemini`)
- [ ] semiconductor-v2 문서 청크를 Gemini Embedding 2(1536D)로 재인제스트
- [ ] 동일 쿼리셋(25개)으로 검색 품질 A/B 비교 스크립트 작성
- [ ] admin 설정의 embedding_model 선택지에 Gemini 모델 추가
- [ ] Task Type(`RETRIEVAL_QUERY` vs `RETRIEVAL_DOCUMENT`) 활용 테스트
- [ ] 비교 결과 리포트 작성

### 2.2 Out of Scope

- 전체 네임스페이스 마이그레이션 (테스트 결과 이후 별도 계획)
- 프로덕션 기본 모델 전환 (이번은 테스트만)
- 멀티모달 임베딩 테스트 (이미지/비디오 — 향후 별도 계획)
- 다른 임베딩 제공자 (Cohere, Voyage 등) 추가
- `gemini-embedding-001` 테스트 (입력 토큰 2,048 제한으로 제외)

---

## 3. 기술 분석

### 3.1 핵심 변경 파일

| 파일 | 변경 내용 | 영향도 |
|------|----------|--------|
| `src/embedding_generator.py` | Gemini 임베딩 API 호출 로직 추가, MODELS dict 확장, provider 분기 | 높음 |
| `services/singletons.py` | `get_gemini_client()` 재활용 (변경 불필요 가능) | 낮음 |
| `services/settings.py` | DEFAULTS에 `gemini-embedding-2-preview` 추가 | 낮음 |
| `api/v1/admin.py` | embedding_model 선택지에 Gemini 모델 추가 | 낮음 |
| `src/agent.py` | embedding_model provider에 따른 클라이언트 분기 | 중간 |

### 3.2 주요 고려사항

#### 3.2.1 Dimension 호환성 (해결됨)
- Gemini Embedding 2는 **MRL(Matryoshka Representation Learning)** 지원
- `output_dimensionality=1536` 설정으로 기존 Pinecone 1536D 인덱스와 **완전 호환**
- 별도 인덱스 생성 불필요 → 동일 인덱스 내 별도 네임스페이스로 테스트

#### 3.2.2 API 호출 방식
```python
from google import genai
from google.genai import types

client = genai.Client(api_key="...")

# 문서 임베딩 (인제스트 시)
result = client.models.embed_content(
    model="gemini-embedding-2-preview",
    contents="문서 텍스트...",
    config=types.EmbedContentConfig(
        task_type="RETRIEVAL_DOCUMENT",
        output_dimensionality=1536
    )
)

# 쿼리 임베딩 (검색 시)
result = client.models.embed_content(
    model="gemini-embedding-2-preview",
    contents="사용자 쿼리...",
    config=types.EmbedContentConfig(
        task_type="RETRIEVAL_QUERY",
        output_dimensionality=1536
    )
)
```

- `get_gemini_client()` (singletons.py) 재활용 가능
- **Task Type 분리**: 인제스트 시 `RETRIEVAL_DOCUMENT`, 검색 시 `RETRIEVAL_QUERY` → 비대칭 임베딩으로 검색 품질 향상 기대
- 배치 처리: `contents`에 리스트 전달로 배치 임베딩 가능

#### 3.2.3 토크나이저
- 현재: `tiktoken` (`cl100k_base`) — OpenAI 전용
- Gemini Embedding 2: 8,192 토큰 제한 (OpenAI 8,191과 거의 동일)
- 한국어 기준 대략 `len(text) // 2` ~ `len(text) // 3` 정도로 추정
- **접근**: 기존 `_truncate()` 로직을 provider별 분기하거나, 보수적으로 문자 수 기반 truncation 적용

#### 3.2.4 임베딩 공간 비호환성
- OpenAI와 Gemini 임베딩은 **서로 다른 벡터 공간** → 직접 벡터 비교 불가
- 반드시 **동일 모델**로 인제스트한 문서에 대해 동일 모델로 쿼리 해야 함
- 네임스페이스 분리로 해결: `semiconductor-v2` (OpenAI) vs `semiconductor-v2-gemini` (Gemini)

### 3.3 테스트 계획

#### 비교 쿼리셋 (도메인별 5개씩, 총 25개)

| 도메인 | 예시 쿼리 |
|--------|----------|
| semiconductor-v2 | "웨이퍼 세정 공정의 화학물질 안전 관리", "반도체 클린룸 정전기 방지 조치" |
| laborlaw | "연장근로 수당 계산 방법", "부당해고 구제 신청 절차" |
| field-training | "밀폐공간 작업 안전 수칙", "고소작업 안전장비 착용 기준" |
| kosha | "화학물질 취급 시 개인보호구 종류", "산업재해 예방 계획서 작성 방법" |
| msds | "톨루엔 노출 기준 및 응급조치", "황산 취급 시 주의사항" |

#### 평가 기준

| 메트릭 | 측정 방법 |
|--------|----------|
| **Recall@5** | Top-5 결과 중 관련 문서 비율 |
| **MRR** (Mean Reciprocal Rank) | 첫 번째 관련 문서의 순위 역수 평균 |
| **Latency** | 쿼리 → 임베딩 생성 → 검색 결과까지 시간 (ms) |
| **Cost** | 1000 쿼리당 API 비용 비교 ($0.02 vs $0.20/1M tokens) |
| **한국어 의미 유사도** | 동의어/유사 표현 쿼리 간 코사인 유사도 |
| **Task Type 효과** | RETRIEVAL_QUERY 사용 시 vs 미사용 시 검색 품질 차이 |

---

## 4. 구현 순서

### Phase 1: 인프라 준비 (Day 1)
1. `EmbeddingGenerator`에 Gemini provider 분기 추가
2. `google-genai` SDK `embed_content()` 호출 래퍼 구현
3. `output_dimensionality=1536` + Task Type 설정
4. 단일 텍스트 임베딩 생성으로 기본 동작 검증

### Phase 2: 데이터 준비 (Day 2)
1. semiconductor-v2 네임스페이스의 기존 청크 텍스트 추출
2. `semiconductor-v2-gemini` 네임스페이스에 Gemini 임베딩으로 인제스트
3. 인제스트 성능 측정 (속도, 에러율, 토큰 사용량)

### Phase 3: 비교 테스트 (Day 3)
1. 25개 비교 쿼리셋 확정
2. OpenAI(semiconductor-v2) / Gemini(semiconductor-v2-gemini) 양쪽 검색 실행
3. Task Type(RETRIEVAL_QUERY) 효과 추가 비교
4. 결과 수집 및 정량 평가

### Phase 4: 분석 및 정리 (Day 4-5)
1. 비교 결과 분석 리포트 작성
2. admin UI에 Gemini 모델 선택지 추가
3. 향후 전체 마이그레이션 / 멀티모달 확장 가이드 초안

---

## 5. 리스크 및 대응

| 리스크 | 확률 | 영향 | 대응 |
|--------|------|------|------|
| Preview 모델 불안정 / API 변경 | 중 | 중 | Preview 상태 모니터링, GA 전환 시 재테스트 계획 |
| 비용 10배 차이 ($0.02 → $0.20) | 높 | 중 | 무료 티어로 테스트 충분, 프로덕션 전환 시 비용 분석 필수 |
| Gemini 한국어 임베딩 품질 부족 | 중 | 중 | 정량 비교로 확인, 부족 시 OpenAI 유지 |
| API rate limit (무료 티어) | 중 | 낮 | 배치 처리 + 지수 백오프, 인제스트 시 속도 조절 |
| MRL 1536D 품질 손실 | 낮 | 중 | 3072D 네이티브와 1536D MRL 비교 테스트 추가 |

---

## 6. 성공 기준

- [ ] Gemini Embedding 2(1536D)로 semiconductor-v2 인제스트 완료
- [ ] 25개 쿼리셋으로 양쪽 모델 검색 결과 수집 완료
- [ ] Recall@5, MRR, Latency, Cost 4개 메트릭 정량 비교 완료
- [ ] Task Type(RETRIEVAL_QUERY/DOCUMENT) 효과 측정 완료
- [ ] 비교 결과에 기반한 권장 사항 도출 (전환 / 유지 / 혼합)

---

## 7. 향후 확장 가능성

Gemini Embedding 2 테스트 성공 시 고려할 후속 작업:

1. **멀티모달 임베딩**: 반도체 공정 이미지, MSDS 문서 이미지를 직접 벡터화 (현재 Vision API로 텍스트 변환 후 임베딩하는 2단계 → 1단계로 단축)
2. **3072D 인덱스**: 기존 1536D 대신 네이티브 3072D로 인덱스 마이그레이션 (품질 향상 기대)
3. **비대칭 검색 최적화**: Task Type을 활용한 query/document 비대칭 임베딩 파이프라인
4. **비용 최적화**: Batch API($0.10/1M tokens, 50% 할인) 활용
