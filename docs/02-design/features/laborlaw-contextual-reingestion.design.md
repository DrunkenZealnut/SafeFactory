# laborlaw-contextual-reingestion Design Document

> **Summary**: 38개 노동법 문서를 Contextual Retrieval로 재인제스천하여 새 네임스페이스에 업로드
>
> **Project**: SafeFactory
> **Date**: 2026-03-11
> **Status**: Draft
> **Planning Doc**: [laborlaw-contextual-reingestion.plan.md](../../01-plan/features/laborlaw-contextual-reingestion.plan.md)

---

## 1. Overview

### 1.1 Design Goals

1. **코드 변경 최소화**: 이미 구현된 ContextGenerator + CLI `--contextual` 플래그를 그대로 활용
2. **무중단 전환**: 새 네임스페이스(`laborlaw-v2`)에 업로드 후, 검증 완료 후 매핑 전환
3. **롤백 안전성**: 기존 `laborlaw` 네임스페이스 보존으로 즉시 롤백 가능

### 1.2 Design Principles

- **기존 인프라 재사용**: 새로운 코드 없이 CLI 실행 + 설정 변경만으로 달성
- **점진적 전환**: 인제스천 → 검증 → 전환 → 정리의 안전한 단계별 진행
- **비용 효율**: Prompt Caching으로 ~86% 비용 절감

---

## 2. Architecture

### 2.1 현재 데이터 흐름 (정적 프리픽스)

```
documents/laborlaw/laws/*.md
    │
    ▼
FileLoader (마크다운 파싱)
    │
    ▼
SemanticChunker._split_by_structure()
    │
    ▼
_merge_small_segments() → _add_overlap()
    │
    ▼
_add_contextual_prefix()  ← 정적: [문서: X | 섹션: Y]
    │
    ▼
EmbeddingGenerator (text-embedding-3-small, 1536dim)
    │
    ▼
PineconeUploader → namespace: "laborlaw"
```

### 2.2 변경된 데이터 흐름 (Contextual Retrieval)

```
documents/laborlaw/laws/*.md
    │
    ▼
FileLoader (마크다운 파싱)
    │
    ▼
SemanticChunker._split_by_structure()
    │
    ▼
_merge_small_segments() → _add_overlap()
    │
    ▼
ContextGenerator.generate_batch()  ← LLM: 법률명+조항+맥락 프리픽스
    │  (Claude Haiku 4.5 + Prompt Caching)
    │  (SQLite 캐시: instance/context_cache.db)
    ▼
프리픽스 + 원문 결합: f"{contextual_prefix}\n\n{segment}"
    │
    ▼
EmbeddingGenerator (text-embedding-3-small, 1536dim)
    │
    ▼
PineconeUploader → namespace: "laborlaw-v2"  ← 새 네임스페이스
```

### 2.3 핵심 차이점

| 항목 | Before (laborlaw) | After (laborlaw-v2) |
|------|-------------------|---------------------|
| 프리픽스 | `[문서: 근로기준법 \| 섹션: 제3장]` | `이 청크는 근로기준법 제60조(연차유급휴가)에 관한 내용으로, 1년간 80% 이상 출근한 근로자에게 15일의 유급휴가를 부여하는 규정과 가산휴가, 미사용 수당 관련 조항을 다루고 있습니다.` |
| 프리픽스 생성 | 정적 (룰 기반) | LLM 생성 (Claude Haiku) |
| BM25 키워드 | 원문만 | 원문 + 법률명/조항번호/맥락 키워드 |
| 임베딩 | 정적 프리픽스 포함 | 맥락 프리픽스 포함 (더 풍부한 의미) |

### 2.4 컴포넌트 의존성

| Component | File | Role | 수정 필요 |
|-----------|------|------|----------|
| CLI | `main.py` | `--contextual --namespace laborlaw-v2` | 없음 |
| ContextGenerator | `src/context_generator.py` | LLM 맥락 생성 + 캐시 | 없음 |
| SemanticChunker | `src/semantic_chunker.py` | 청킹 + 프리픽스 결합 | 없음 |
| EmbeddingGenerator | `src/embedding_generator.py` | 벡터 생성 | 없음 |
| PineconeUploader | `src/pinecone_uploader.py` | 벡터 업로드 | 없음 |
| DomainConfig | `services/domain_config.py` | 네임스페이스 매핑 | **변경 필요** |

---

## 3. Data Model

### 3.1 Pinecone 벡터 구조 (변경 없음)

```python
{
    "id": "md5(source_file + chunk_index + content_preview)",
    "values": [float] * 1536,  # text-embedding-3-small
    "metadata": {
        "content": str,          # contextual prefix + 원문 (max 1000 chars)
        "source_file": str,      # 파일 경로
        "chunk_index": int,      # 청크 순서
        "document_title": str,   # 문서 제목
        "section_title": str,    # 섹션 제목
        "domain": "laborlaw",    # 도메인
        # ... 기타 메타데이터
    }
}
```

### 3.2 ContextGenerator SQLite 캐시 구조

```sql
-- instance/context_cache.db
CREATE TABLE context_cache (
    cache_key TEXT PRIMARY KEY,   -- MD5(doc_hash:chunk_hash:domain)
    context TEXT NOT NULL,         -- 생성된 contextual prefix
    model TEXT NOT NULL,           -- claude-haiku-4-5-20251001
    domain TEXT DEFAULT '',        -- laborlaw
    created_at TEXT,               -- 생성 시각
    tokens_in INTEGER DEFAULT 0,   -- 입력 토큰 수
    tokens_out INTEGER DEFAULT 0   -- 출력 토큰 수
);
```

### 3.3 네임스페이스 매핑 변경

```python
# services/domain_config.py — Before
DIRECTORY_NAMESPACE_MAP = {
    'laborlaw': 'laborlaw',       # ← 기존
}
DOMAIN_CONFIG = {
    'laborlaw': {
        'namespace': 'laborlaw',  # ← 기존
    },
}

# services/domain_config.py — After
DIRECTORY_NAMESPACE_MAP = {
    'laborlaw': 'laborlaw-v2',    # ← 변경
}
DOMAIN_CONFIG = {
    'laborlaw': {
        'namespace': 'laborlaw-v2',  # ← 변경
    },
}
```

---

## 4. Contextual Prefix 생성 사양

### 4.1 laborlaw 도메인 프롬프트

```python
# src/context_generator.py:24-28 (이미 구현)
DOMAIN_CONTEXT_PROMPTS = {
    'laborlaw': (
        "이 청크가 어떤 법률/시행령/판례의 어떤 조항이나 쟁점에 해당하며, "
        "문서 전체에서 어떤 맥락인지 검색 품질 향상을 위해 간결하게 설명해주세요. "
        "법률명과 조항 번호를 반드시 포함하세요."
    ),
}
```

### 4.2 Prompt Caching 메커니즘

```
Call 1 (문서의 첫 번째 청크):
  [cache_control: ephemeral]
  <document>근로기준법 전문 (22K tokens)</document>  ← Cache Creation ($1.00/MTok)
  <chunk>제60조 내용...</chunk>                       ← 일반 입력

Call 2~N (같은 문서의 나머지 청크):
  [cache_control: ephemeral]
  <document>근로기준법 전문 (22K tokens)</document>  ← Cache Read ($0.08/MTok, 92% 할인)
  <chunk>제61조 내용...</chunk>                       ← 일반 입력
```

### 4.3 예상 출력 예시

**입력 청크** (근로기준법 제60조):
```
① 사용자는 1년간 80퍼센트 이상 출근한 근로자에게 15일의 유급휴가를 주어야 한다.
② 사용자는 계속하여 근로한 기간이 1년 미만인 근로자 또는 1년간 80퍼센트 미만 출근한 근로자에게 1개월 개근 시 1일의 유급휴가를 주어야 한다.
③ 삭제
④ 사용자는 3년 이상 계속하여 근로한 근로자에게는 제1항에 따른 휴가에 최초 1년을 초과하는 계속 근로 연수 매 2년에 대하여 1일을 가산한 유급휴가를 주어야 한다. 이 경우 가산휴가를 포함한 총 휴가 일수는 25일을 한도로 한다.
```

**생성되는 Contextual Prefix**:
```
이 청크는 근로기준법 제60조(연차유급휴가)에 관한 내용으로, 1년간 80% 이상 출근한 근로자의 연차휴가 15일 부여 기준, 1년 미만 근로자의 월 1일 휴가, 3년 이상 장기근속자의 가산휴가(최대 25일) 규정을 포함하고 있습니다.
```

---

## 5. 실행 절차 상세

### 5.1 Phase 1: 사전 점검

```bash
# 1. 환경변수 확인
grep ANTHROPIC_API_KEY .env

# 2. 현재 laborlaw 네임스페이스 벡터 수 확인
python main.py stats
# → laborlaw: N vectors (baseline 기록)

# 3. 문서 파일 확인
ls documents/laborlaw/laws/ | wc -l  # → 38 (디렉토리 포함 39-1)
find documents/laborlaw/laws/ -name "*.md" | wc -l  # → 38
```

### 5.2 Phase 2: 재인제스천 실행

```bash
python main.py process ./documents/laborlaw/laws \
  --namespace laborlaw-v2 \
  --contextual \
  --force \
  --batch-size 50
```

**CLI 파라미터 설명**:

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `process` | `./documents/laborlaw/laws` | 대상 디렉토리 |
| `--namespace` | `laborlaw-v2` | 새 Pinecone 네임스페이스 (자동 생성) |
| `--contextual` | (flag) | ContextGenerator 활성화 |
| `--force` | (flag) | 모든 파일 강제 재처리 (변경 여부 무시) |
| `--batch-size` | `50` | 벡터 업로드 배치 크기 |

**예상 실행 시간**: ~30~60분 (38문서, ~1,700 LLM 호출)

**예상 출력**:
```
🧠 Contextual Retrieval 활성화 (모델: claude-haiku-4-5-20251001)
📂 Processing: documents/laborlaw/laws/
...
✅ 처리 완료: 38 files, 1,700 chunks, 0 errors

🧠 Contextual Retrieval 통계:
  LLM 호출: 1,700회
  캐시 히트: 0회
  예상 비용: $4.27
```

### 5.3 Phase 3: 품질 검증

#### 검증 쿼리 세트

| # | 테스트 쿼리 | 기대 결과 (Top-5 포함) |
|---|-----------|----------------------|
| 1 | "연차휴가 미사용 수당" | 근로기준법 제60조, 제61조 |
| 2 | "해고 사유와 절차" | 근로기준법 제23조, 제27조 |
| 3 | "최저임금 위반 벌칙" | 최저임금법 제28조 |
| 4 | "산업재해 보상 범위" | 산업재해보상보험법 관련 조항 |
| 5 | "근로시간 상한" | 근로기준법 제50조, 제53조 |

```bash
# 새 네임스페이스 검색
python main.py search "연차휴가 미사용 수당" --namespace laborlaw-v2 --top-k 5
python main.py search "해고 사유와 절차" --namespace laborlaw-v2 --top-k 5
python main.py search "최저임금 위반 벌칙" --namespace laborlaw-v2 --top-k 5

# 기존 네임스페이스 비교
python main.py search "연차휴가 미사용 수당" --namespace laborlaw --top-k 5
```

#### 프리픽스 품질 검증

Pinecone 대시보드 또는 CLI에서 랜덤 벡터 10개 조회하여:
- contextual prefix에 **법률명** 포함 여부 (목표: 90%)
- contextual prefix에 **조항 번호** 포함 여부 (목표: 80%)
- prefix 길이 50~100 토큰 범위 여부

### 5.4 Phase 4: 네임스페이스 전환

**변경 파일**: `services/domain_config.py`

```python
# 변경 1: DIRECTORY_NAMESPACE_MAP (line 13)
'laborlaw': 'laborlaw-v2',     # 노동법 (Contextual Retrieval 적용)

# 변경 2: DOMAIN_CONFIG['laborlaw']['namespace'] (line 250)
'namespace': 'laborlaw-v2',
```

**배포**:
```bash
# 프로덕션 서버
cd ~/SafeFactory && git pull origin main
pkill -f gunicorn; sleep 2
nohup venv/bin/gunicorn web_app:app --bind 127.0.0.1:5001 --workers 2 --timeout 180 > app.log 2>&1 &
```

### 5.5 Phase 5: 정리 (2주 후)

```bash
# 기존 네임스페이스 삭제 (안정성 확인 후)
python main.py delete --namespace laborlaw --all
```

---

## 6. Error Handling

### 6.1 인제스천 중 에러 대응

| 에러 | 원인 | 대응 |
|------|------|------|
| `ANTHROPIC_API_KEY` 미설정 | `.env` 누락 | CLI가 즉시 에러 출력 후 종료 |
| API rate limit | 과도한 요청 | ContextGenerator가 자동 재시도 (exponential backoff) |
| 네트워크 에러 | 연결 끊김 | SQLite 캐시 덕분에 재실행 시 이미 처리된 청크 스킵 |
| 문서 > 100K 토큰 | 초대형 법률문서 | `_truncate_document()`가 자동으로 80%/20% 분할 |
| Prompt Caching 미적용 | API 버전 미지원 | 비용만 증가, 기능은 정상 동작 |

### 6.2 전환 후 에러 대응

| 에러 | 원인 | 대응 |
|------|------|------|
| 검색 결과 없음 | 네임스페이스명 오타 | `domain_config.py` 확인, `laborlaw-v2` 정확히 매칭 |
| 검색 품질 저하 | 프리픽스 품질 문제 | 즉시 `laborlaw`로 롤백 |
| 서버 시작 실패 | import 에러 | `NAMESPACE_DOMAIN_MAP` 충돌 확인 |

---

## 7. Implementation Guide

### 7.1 수정 대상 파일

```
services/
└── domain_config.py      ← 네임스페이스 매핑 2곳 변경
```

### 7.2 Implementation Order

1. [ ] `ANTHROPIC_API_KEY` 환경변수 확인
2. [ ] `python main.py stats` — 현재 상태 기록
3. [ ] `python main.py process` — Contextual Retrieval 재인제스천 실행
4. [ ] 처리 완료 로그 및 비용 확인
5. [ ] `python main.py search` — 5개 쿼리 비교 검증
6. [ ] `services/domain_config.py` — 네임스페이스 매핑 변경 (2곳)
7. [ ] 웹앱 재시작 및 프로덕션 검증
8. [ ] 2주 후 기존 `laborlaw` 네임스페이스 삭제

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-03-11 | Initial draft | Claude |
