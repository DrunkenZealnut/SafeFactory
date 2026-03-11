# Design: 안전보건공단(KOSHA) 데이터 Contextual Retrieval 적용 및 재인제스천

> Feature: kosha-contextual-retrieval
> Created: 2026-03-10
> Status: Design
> Plan: [kosha-contextual-retrieval.plan.md](../../01-plan/features/kosha-contextual-retrieval.plan.md)

---

## 1. 설계 개요

기존 `safeguide` 네임스페이스의 안전보건공단 데이터를 Contextual Retrieval(LLM 맥락 접두사)을 적용하여 새로운 `kosha` 네임스페이스에 재인제스천하고, 도메인 설정을 전환하는 작업.

### 변경 범위

| 구분 | 대상 | 변경 내용 |
|------|------|----------|
| 인제스천 | CLI `main.py process` | 7개 폴더 × `--contextual --namespace kosha` 실행 |
| 설정 변경 | `services/domain_config.py` | namespace `safeguide` → `kosha` (2곳) |
| 기존 유지 | `safeguide` 네임스페이스 | 삭제하지 않음 (롤백 가능) |

---

## 2. 인제스천 파이프라인 설계

### 2.1 데이터 흐름

```
documents/semiconductor/안전보건공단/{folder}/
    ├── {name}.md          ─┐
    └── merged.json         ─┤
                             ▼
                    FileLoader (MD/JSON 파싱)
                             ▼
                    SemanticChunker (구조 기반 분할)
                             ▼
                    ContextGenerator (LLM 맥락 접두사 생성)
                      ├── domain_prompt: 'safeguide'
                      ├── model: claude-haiku-4-5
                      └── cache: instance/context_cache.db
                             ▼
                    EmbeddingGenerator (text-embedding-3-small)
                             ▼
                    PineconeUploader → namespace: 'kosha'
```

### 2.2 폴더별 처리 순서

인제스천은 폴더 단위로 순차 실행. 각 폴더는 독립적으로 처리되어 실패 시 해당 폴더만 재시도 가능.

| 순서 | 폴더명 | 파일 수 | 비고 |
|------|--------|---------|------|
| 1 | `전자산업_확산_공정_설비_정비_작업_안전보건_가이드` | 21 | 확산 공정 |
| 2 | `전자산업_포토_공정_설비_정비_작업_안전보건_가이드` | 19 | 포토 공정 |
| 3 | `전자산업_산화공정_설비_정비작업_안전보건가이드` | 20 | 산화 공정 |
| 4 | `전자산업_크린룸내_세정_공정_작업_안전보건가이드` | 19 | 세정 공정 |
| 5 | `전자산업 크린룸 공정 지원 설비 정비작업 안전 보건 가이드2` | 24 | 크린룸 지원 설비 |
| 6 | `전자제품 제조공정의 정비 작업자를 위한 화학물질 건강 위험성평가` | 17 | 화학물질 위험성평가 |
| 7 | `chemical health risk assessment guide` | 16 | 영문 위험성평가 가이드 |

### 2.3 CLI 명령어 공통 옵션

```bash
python main.py process "{folder_path}" \
    --namespace kosha \          # 새 네임스페이스
    --contextual \               # Contextual Retrieval 활성화
    --skip-images \              # 이미지 파일 건너뜀 (MD/JSON만 처리)
    --domain safeguide \         # 메타데이터 domain 태그
    --batch-size 50              # Pinecone upsert 배치 크기
```

### 2.4 Contextual Retrieval 설정

이미 `src/context_generator.py`에 `safeguide` 도메인 프롬프트가 정의되어 있음:

```python
'safeguide': (
    "이 청크가 어떤 안전보건 가이드의 어떤 주제에 해당하며, "
    "문서 전체에서 어떤 맥락인지 검색 품질 향상을 위해 간결하게 설명해주세요. "
    "안전 규정명 또는 관련 법조항을 포함하세요."
),
```

- **모델**: `claude-haiku-4-5-20251001` (기본값)
- **캐시**: `instance/context_cache.db` (SQLite, 동일 청크 재처리 방지)
- **프롬프트 캐싱**: Anthropic API 프롬프트 캐싱으로 동일 문서 내 청크 비용 ~90% 절감

---

## 3. 설정 변경 설계

### 3.1 `services/domain_config.py` 변경사항

#### 변경 1: DIRECTORY_NAMESPACE_MAP (line 15)

```python
# Before
'안전보건공단': 'safeguide',

# After
'안전보건공단': 'kosha',
```

#### 변경 2: DOMAIN_CONFIGS > safeguide > namespace (line 269)

```python
# Before
'safeguide': {
    ...
    'namespace': 'safeguide',
    ...
}

# After
'safeguide': {
    ...
    'namespace': 'kosha',
    ...
}
```

### 3.2 변경하지 않는 항목

| 항목 | 이유 |
|------|------|
| 도메인 키 `safeguide` | 웹 앱 라우팅, URL, 시스템 프롬프트 등 전체에서 사용중. 변경 시 파급효과 큼 |
| `DOMAIN_PROMPTS['safeguide']` | 이미 최적화된 safeguide 전용 시스템 프롬프트 유지 |
| `DOMAIN_CONTEXT_PROMPTS['safeguide']` | Contextual Retrieval 프롬프트 유지 (이미 safeguide용 존재) |
| `DOMAIN_CHAIN_PROMPTS['safeguide']` | 위험평가 프레임워크 유지 |

---

## 4. 검증 설계

### 4.1 인제스천 검증

| 체크포인트 | 검증 방법 |
|-----------|----------|
| 각 폴더 처리 완료 | CLI 출력: `처리된 파일`, `생성된 청크`, `업로드된 벡터` 확인 |
| 에러 0개 | CLI 출력: `실패한 업로드: 0` 확인 |
| Contextual Retrieval 비용 | CLI 출력: `🧠 Contextual Retrieval 통계` 확인 |
| `kosha` 네임스페이스 벡터 수 | `python main.py stats` 또는 Pinecone 콘솔 확인 |

### 4.2 검색 품질 검증

인제스천 완료 후 `domain_config.py` 전환하고 아래 샘플 질문으로 검증:

| # | 질문 | 기대 결과 |
|---|------|----------|
| 1 | 확산 공정 설비 정비 시 안전 주의사항은? | 확산 공정 가이드에서 소스 인용, 상세 답변 |
| 2 | 포토 공정 유해물질 취급 방법은? | 포토 공정 가이드 소스, 보호구 정보 포함 |
| 3 | 크린룸 세정 작업 안전수칙 알려줘 | 세정 공정 가이드 소스, 화학물질 안전 정보 |

### 4.3 A/B 비교 (선택)

시간 허용 시 기존 `safeguide` vs 새 `kosha` 네임스페이스로 동일 질문 비교:
- 답변 길이, 소스 수, 응답 성공률 비교

---

## 5. 구현 순서

```
Step 1: 7개 폴더 인제스천 (CLI × 7회)
   ↓
Step 2: Pinecone 벡터 수 확인
   ↓
Step 3: domain_config.py 네임스페이스 전환 (2곳)
   ↓
Step 4: 웹 앱에서 샘플 질문 3개 검증
   ↓
Step 5: (선택) A/B 비교
```

---

## 6. 롤백 계획

| 상황 | 롤백 방법 |
|------|----------|
| 인제스천 실패 | 해당 폴더만 재실행 (idempotent: 동일 벡터 ID = 덮어쓰기) |
| 검색 품질 저하 | `domain_config.py`에서 namespace를 `safeguide`로 복원 (1줄 변경) |
| 전체 롤백 | `kosha` 네임스페이스 삭제, namespace 설정 원복 |
