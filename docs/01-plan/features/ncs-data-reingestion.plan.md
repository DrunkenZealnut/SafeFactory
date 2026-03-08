# Plan: NCS 반도체 데이터 Contextual Retrieval 재인제스천

> Feature: ncs-data-reingestion
> Created: 2026-03-08
> Status: Plan

---

## Executive Summary

| 관점 | 설명 |
|------|------|
| **Problem** | 기존 NCS 반도체 데이터가 500토큰 청크, 맥락 없는 임베딩으로 인제스천되어 검색 품질(Recall)이 낮음 |
| **Solution** | Anthropic Contextual Retrieval 파이프라인(800토큰 청크 + LLM 맥락 접두사)으로 전체 재인제스천 |
| **Function UX Effect** | 반도체 공정 질문에 대해 더 정확한 청크가 검색되고, 답변 품질이 체감 향상됨 |
| **Core Value** | Retrieval Failure Rate 49~67% 감소 (Anthropic 벤치마크 기준), 검색 정확도 대폭 개선 |

---

## 1. Background

### 1.1 현재 상태

- `semiconductor` 네임스페이스에 기존 인제스천 데이터 존재
- 500토큰 청크, contextual prefix 없음, top_k=10 기반 검색
- BM25 하이브리드 검색은 동작하지만 맥락 없는 청크로 인해 정확도 제한

### 1.2 변경 동기

`contextual-retrieval-redesign` PDCA에서 코드 변경 완료 (Match Rate 100%):
- 청크 크기: 500 → **800 토큰**
- top_k: 10 → **20**
- Contextual prefix 생성 파이프라인 구현 완료
- Eval 파이프라인 Recall@20 + failure rate 메트릭 추가

**코드는 준비되었으나, 기존 데이터에 적용되지 않은 상태**. 재인제스천 필요.

---

## 2. Scope

### 2.1 대상 데이터

```
documents/semiconductor/ncs/data/
├── report/         (4 md,      3 images,  2.1MB md)
├── 반도체개발/     (32 md, 1,299 images,  6.8MB md)
├── 반도체장비/     (19 md,   779 images,  3.2MB md)
├── 반도체재료/     (24 md,   826 images,  4.8MB md)
└── 반도체제조/     (13 md,   696 images,  2.4MB md)
```

**총계**: 92 markdown 파일 (~19.3MB), 3,603 이미지, 365 JSON 메타파일

### 2.2 예상 청크 수

- 총 텍스트 토큰: ~2.2M (Korean 3chars/token)
- 800토큰 청크: **~2,778개**
- 문서당 평균: ~24K 토큰, ~30 청크

### 2.3 대상 네임스페이스

- **새 네임스페이스**: `semiconductor-v2` (Contextual Retrieval 적용)
- **기존 네임스페이스**: `semiconductor` (813 records, 보존 — A/B 비교용)
- `--force` 옵션으로 전체 재처리

---

## 3. Execution Plan

### Phase A: Before 기준 데이터 수집

| Step | 작업 | 명령어 |
|------|------|--------|
| A-1 | 현재 상태 eval 실행 (기준선 확보) | `python -m scripts.eval.eval_pipeline --top-k 20 --output results_before.json` |
| A-2 | golden_dataset.json에 semiconductor 쿼리 확인 | 수동 확인 |

### Phase B: 재인제스천 실행

| Step | 작업 | 명령어 |
|------|------|--------|
| B-1 | Markdown 전용 재인제스천 (이미지 제외) | `python main.py process ./documents/semiconductor/ncs/data --namespace semiconductor-v2 --contextual --force --max-chunk-tokens 800 --skip-images` |
| B-2 | 이미지 재인제스천 (필요시) | `python main.py process ./documents/semiconductor/ncs/data --namespace semiconductor-v2 --force --max-chunk-tokens 800` |
| B-3 | 인제스천 통계 확인 | `python main.py stats` |

> **Phase B-1 권장**: 이미지는 Vision API로 설명 생성 후 단일 청크가 되므로 contextual prefix 효과가 미미. `--skip-images`로 마크다운만 먼저 처리하면 비용 절감.

### Phase C: After 평가 및 비교

| Step | 작업 | 명령어 |
|------|------|--------|
| C-1 | 재인제스천 후 eval 실행 | `python -m scripts.eval.eval_pipeline --top-k 20 --output results_after.json` |
| C-2 | Before/After A/B 비교 | `python -m scripts.eval.compare_results results_before.json results_after.json` |
| C-3 | Failure Rate 감소율 확인 | 비교 리포트에서 자동 계산 |

---

## 4. Cost Estimation

### 4.1 Contextual Retrieval (Claude Haiku 4.5)

| 항목 | 토큰 수 | 단가 | 비용 |
|------|---------|------|------|
| Cache 생성 (92문서 × 24K) | 2.2M | $1.00/MTok | $2.22 |
| Cache 읽기 (2,686콜 × 24K) | 64.9M | $0.08/MTok | $5.19 |
| 일반 입력 (2,778콜 × 900) | 2.5M | $0.80/MTok | $2.00 |
| 출력 (2,778콜 × 75) | 0.2M | $4.00/MTok | $0.83 |
| **Haiku 소계** | | | **$10.24** |

### 4.2 기타 비용

| 항목 | 비용 |
|------|------|
| OpenAI 임베딩 (text-embedding-3-small) | ~$0.05 |
| 이미지 Vision API (선택, 3,603장) | ~$0.72 |
| **총 비용 (markdown만)** | **~$10.29 (~14,000원)** |
| **총 비용 (이미지 포함)** | **~$11.01 (~15,000원)** |

### 4.3 캐싱 효과

- 캐싱 없이 full input: ~$55 → 캐싱으로 **~80% 절감**
- SQLite 캐시: 한번 생성된 prefix는 재실행 시 API 호출 없이 재사용

---

## 5. Risk & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Pinecone 기존 벡터 손상 | 검색 불가 | `--force`는 동일 vector ID 덮어쓰기 (idempotent) |
| Haiku API rate limit | 처리 지연 | Sequential 호출 + prompt caching으로 자연스러운 속도 조절 |
| 청크 수 예상 초과 | 비용 증가 | `ContextGenerator.get_stats()`로 실시간 모니터링 |
| Eval golden dataset 부족 | 비교 불가 | Phase A에서 기준선 확보 후 진행 |

---

## 6. Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| 인제스천 완료율 | 100% (92 md 파일) | `main.py stats` |
| Recall@20 향상 | Before 대비 +20% 이상 | `compare_results.py` |
| Failure Rate 감소 | -30% 이상 (Anthropic 벤치마크: -49~67%) | `compare_results.py` |
| 비용 | < $15 | `ContextGenerator.get_stats()` |
| 처리 시간 | < 2시간 | wall clock |

---

## 7. Dependencies

- [x] `contextual-retrieval-redesign` 코드 구현 완료 (Match Rate 100%)
- [x] `--contextual` CLI 플래그 구현 완료
- [x] `scripts/eval/eval_pipeline.py` Recall@20 지원
- [x] `scripts/eval/compare_results.py` A/B 비교 스크립트
- [ ] `ANTHROPIC_API_KEY` 환경변수 설정
- [ ] `golden_dataset.json`에 semiconductor 쿼리 존재 확인
