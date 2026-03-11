# Plan: 노동법 문서 Contextual Retrieval 재인제스천

## Executive Summary

| 관점 | 내용 |
|------|------|
| **Problem** | 현재 `laborlaw` 네임스페이스는 정적 프리픽스(`[문서: X \| 섹션: Y]`)만 적용된 청크로 구성되어, 법률 조항이 문서 맥락에서 분리될 때 검색 정확도가 크게 저하됨 (예: "제23조" → 어떤 법률의 해고 제한 조항인지 매칭 실패) |
| **Solution** | 이미 구현된 `ContextGenerator`(Anthropic Contextual Retrieval)를 활용하여 38개 노동법 문서를 재인제스천하고, 새로운 네임스페이스 `laborlaw-v2`에 업로드. LLM이 각 청크에 법률명+조항번호+맥락 설명 프리픽스를 자동 생성 |
| **Function UX Effect** | 노동법 도메인 검색 실패율 49~67% 감소 예상. "연차휴가 미사용 수당" 같은 질문에서 근로기준법 제60조 관련 청크를 정확히 매칭, 법률 상담 답변 품질 대폭 향상 |
| **Core Value** | 기존 코드 변경 없이 CLI 실행만으로 달성 가능한 즉시 품질 향상. 새 네임스페이스 전략으로 롤백 안전성 확보 |

---

## 1. Background & Motivation

### 1.1 현재 상태

- **네임스페이스**: `laborlaw` (기존 정적 프리픽스 적용)
- **문서 수**: 38개 노동법 마크다운 파일 (`documents/laborlaw/laws/`)
- **총 용량**: ~2.55MB
- **주요 법률**: 근로기준법, 산업안전보건법, 고용보험법, 노동조합법, 최저임금법 등

### 1.2 이미 구현된 인프라

| 컴포넌트 | 상태 | 파일 |
|----------|------|------|
| `ContextGenerator` 클래스 | ✅ 구현 완료 | `src/context_generator.py` |
| laborlaw 도메인 프롬프트 | ✅ 설정 완료 | `src/context_generator.py:24-28` |
| CLI `--contextual` 플래그 | ✅ 구현 완료 | `main.py:57-58` |
| Prompt Caching (`cache_control: ephemeral`) | ✅ 적용 완료 | `src/context_generator.py:197` |
| SQLite 결과 캐시 | ✅ 구현 완료 | `src/context_generator.py:110-131` |

### 1.3 왜 지금 실행하는가?

1. **코드 준비 완료**: ContextGenerator, 도메인 프롬프트, CLI 플래그 모두 구현 완료 상태
2. **NCS 선행 사례**: `semiconductor-v2` 네임스페이스가 이미 Contextual Retrieval로 성공 적용
3. **비용 효율**: Haiku + Prompt Caching으로 38개 문서 전체 재인제스천 예상 비용 $1~3
4. **롤백 안전**: 새 네임스페이스(`laborlaw-v2`)에 업로드하므로 기존 데이터 보존

---

## 2. Scope

### 2.1 In Scope

- 38개 노동법 마크다운 문서를 Contextual Retrieval로 재처리
- 새 Pinecone 네임스페이스 `laborlaw-v2` 생성 및 업로드
- `domain_config.py`에서 네임스페이스 매핑 전환
- 전환 후 검색 품질 검증

### 2.2 Out of Scope

- ContextGenerator 코드 수정 (이미 완료)
- 다른 도메인 재인제스천 (별도 PDCA)
- 청크 크기 변경 (800토큰 전환은 `contextual-retrieval-redesign` Plan 참조)
- top_k 파라미터 변경 (별도 Plan)
- 평가 파이프라인(eval) 구축 (별도 Plan)

---

## 3. Execution Plan

### Phase 1: 사전 준비

| # | Task | Description | Priority |
|---|------|-------------|----------|
| 1-1 | `ANTHROPIC_API_KEY` 환경변수 확인 | `.env` 파일에 키 설정 여부 확인 | P0 |
| 1-2 | 기존 `laborlaw` 네임스페이스 벡터 수 확인 | Pinecone stats로 baseline 기록 | P0 |
| 1-3 | 문서 접근성 확인 | 38개 .md 파일 모두 읽기 가능한지 확인 | P0 |

### Phase 2: Contextual Retrieval 재인제스천

| # | Task | Description | Priority |
|---|------|-------------|----------|
| 2-1 | 새 네임스페이스로 처리 실행 | `python main.py process ./documents/laborlaw/laws --namespace laborlaw-v2 --contextual --force` | P0 |
| 2-2 | 처리 결과 로그 확인 | 청크 수, LLM 호출 수, 비용, 에러 확인 | P0 |
| 2-3 | Pinecone 벡터 수 확인 | `laborlaw-v2` 네임스페이스 stats 확인 | P0 |

### Phase 3: 품질 검증

| # | Task | Description | Priority |
|---|------|-------------|----------|
| 3-1 | 수동 검색 비교 | 동일 쿼리로 `laborlaw` vs `laborlaw-v2` 검색 결과 비교 (5~10개 쿼리) | P0 |
| 3-2 | 컨텍스트 프리픽스 품질 확인 | 랜덤 청크 10개의 contextual prefix가 법률명+조항번호를 포함하는지 확인 | P0 |
| 3-3 | BM25 키워드 매칭 검증 | contextual prefix 키워드가 하이브리드 검색에 기여하는지 확인 | P1 |

### Phase 4: 네임스페이스 전환

| # | Task | Description | Priority |
|---|------|-------------|----------|
| 4-1 | `domain_config.py` 업데이트 | `'laborlaw': 'laborlaw'` → `'laborlaw': 'laborlaw-v2'` | P0 |
| 4-2 | DOMAIN_CONFIG 매핑 업데이트 | `DOMAIN_CONFIG['laborlaw']['namespace']` 변경 | P0 |
| 4-3 | 웹앱 재시작 | gunicorn 재시작으로 설정 반영 | P0 |
| 4-4 | 프로덕션 검증 | 실서비스에서 노동법 질문 정상 응답 확인 | P0 |

### Phase 5: 정리 (선택)

| # | Task | Description | Priority |
|---|------|-------------|----------|
| 5-1 | 기존 `laborlaw` 네임스페이스 보존 | 롤백 대비 최소 2주 보존 | P1 |
| 5-2 | 기존 네임스페이스 삭제 | 안정성 확인 후 기존 벡터 삭제하여 비용 절감 | P2 |

---

## 4. Cost Estimation

### 4.1 Anthropic API 비용

| 항목 | 수치 |
|------|------|
| 문서 수 | 38개 |
| 총 문서 크기 | ~2.55MB |
| 예상 청크 수 (500토큰 기준) | ~1,700개 |
| 평균 문서 크기 | ~67KB (~22K 토큰) |
| 청크당 입력 (문서+프롬프트+청크) | ~23K 토큰 |
| 모델 | Claude Haiku 4.5 |

**비용 계산**:

| 항목 | 단가 | 수량 | 비용 |
|------|------|------|------|
| 첫 번째 청크 (cache creation) | $1.00/MTok | 38문서 * 22K = 836K tok | ~$0.84 |
| 이후 청크 (cache read) | $0.08/MTok | ~1,660 * 22K = 36.5M tok | ~$2.92 |
| 출력 (context prefix) | $4.00/MTok | ~1,700 * 75tok = 127K tok | ~$0.51 |
| **합계** | | | **~$4.27** |

> Prompt Caching 덕분에 캐시 없이 하면 ~$31 → 캐시 적용 시 ~$4.27 (**86% 절감**)

### 4.2 Pinecone 비용

- 추가 네임스페이스 비용: Serverless 요금제에서 벡터 수 기준 과금
- ~1,700 벡터 추가는 무시할 수 있는 수준

---

## 5. CLI 실행 커맨드

```bash
# 1. 환경 준비
source venv/bin/activate
export ANTHROPIC_API_KEY="your-key-here"  # .env에 이미 설정되어 있으면 생략

# 2. 기존 상태 확인
python main.py stats

# 3. Contextual Retrieval 재인제스천 (새 네임스페이스)
python main.py process ./documents/laborlaw/laws \
  --namespace laborlaw-v2 \
  --contextual \
  --force \
  --batch-size 50

# 4. 결과 확인
python main.py stats
python main.py search "연차휴가 미사용 수당" --namespace laborlaw-v2 --top-k 5
python main.py search "해고 제한 사유" --namespace laborlaw-v2 --top-k 5

# 5. 네임스페이스 전환 (domain_config.py 수정 후)
# services/domain_config.py:
#   'laborlaw': 'laborlaw-v2'
```

---

## 6. Risk Matrix

| Risk | Probability | Impact | Mitigation |
|------|:-----------:|:------:|------------|
| ANTHROPIC_API_KEY 미설정 | Low | High | 실행 전 `.env` 확인, 에러 시 명확한 메시지 출력 |
| LLM 환각 (잘못된 법률명/조항) | Low | Medium | laborlaw 도메인 프롬프트가 "법률명+조항 번호 반드시 포함" 지시 |
| 처리 중단 (네트워크/API 에러) | Medium | Low | SQLite 캐시 덕분에 재실행 시 이미 처리된 청크 스킵 |
| 청크 수 급증/감소 | Low | Low | `--force` 플래그로 전체 재처리, stats로 확인 |
| 검색 품질 저하 | Very Low | Medium | 기존 `laborlaw` 네임스페이스 보존으로 즉시 롤백 가능 |
| API 비용 초과 | Low | Low | 예상 $4.27, 최대 $10 이내. 캐시로 재실행 비용 $0 |

---

## 7. Success Criteria

| Metric | 기준 | 측정 방법 |
|--------|------|----------|
| 인제스천 완료 | 38개 문서 전체 처리, 에러 0 | CLI 로그 확인 |
| 벡터 수 | `laborlaw-v2`에 ~1,500~2,000 벡터 | Pinecone stats |
| Contextual prefix 품질 | 랜덤 10개 청크 중 90%가 법률명+조항번호 포함 | 수동 검증 |
| 검색 정확도 향상 | 5개 테스트 쿼리에서 관련 청크 Top-5 매칭률 향상 | 수동 비교 |
| 비용 | < $10 USD | ContextGenerator.get_stats() |
| 전환 안정성 | 네임스페이스 전환 후 웹앱 정상 동작 | 프로덕션 검증 |

---

## 8. Dependencies

| Dependency | Status | Required For |
|-----------|--------|-------------|
| `ANTHROPIC_API_KEY` | 확인 필요 | Phase 2 (LLM 호출) |
| `src/context_generator.py` | ✅ 구현 완료 | Phase 2 |
| `main.py --contextual` 플래그 | ✅ 구현 완료 | Phase 2 |
| Pinecone 인덱스 | ✅ 운영 중 | Phase 2 (새 네임스페이스 자동 생성) |
| `services/domain_config.py` | ✅ 존재 | Phase 4 (네임스페이스 매핑 전환) |

---

## 9. Related Plans

| Plan | 관계 |
|------|------|
| `contextual-retrieval-chunking.plan.md` | ContextGenerator 구현 → **완료** (이 Plan의 전제조건) |
| `contextual-retrieval-redesign.plan.md` | 청크 크기 800 + top_k=20 재설계 → **별도 실행** |
| `ncs-data-reingestion.plan.md` | NCS(반도체) 도메인 재인제스천 선행 사례 |

---

## 10. Rollback Plan

1. `domain_config.py`에서 네임스페이스를 `laborlaw`로 원복
2. gunicorn 재시작
3. `laborlaw-v2` 네임스페이스는 Pinecone에서 필요 시 삭제

> 기존 `laborlaw` 네임스페이스는 최소 2주간 보존하여 안전한 롤백 보장
