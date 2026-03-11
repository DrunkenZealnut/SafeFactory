# Plan: 안전보건공단(KOSHA) 데이터 Contextual Retrieval 적용 및 재인제스천

> Feature: kosha-contextual-retrieval
> Created: 2026-03-10
> Status: Plan

---

## Executive Summary

| 관점 | 설명 |
|------|------|
| **Problem** | 안전보건공단 데이터(7개 가이드, 135개 파일)가 기존 `safeguide` 네임스페이스에 Contextual Retrieval 없이 단순 청킹으로 저장되어 있어 검색 품질이 낮음 |
| **Solution** | `--contextual` 플래그를 사용하여 LLM 기반 맥락 접두사를 생성하고 새로운 `kosha` 네임스페이스에 업로드, `domain_config.py`를 `kosha`로 전환 |
| **Function UX Effect** | 안전보건 가이드 질문 시 청크의 문서 맥락이 보존되어 더 정확하고 상세한 답변 생성 (semiconductor-v2 적용 시 답변 길이 82%↑, 소스 93%↑ 개선 사례) |
| **Core Value** | Contextual Retrieval 적용 완료 도메인을 2개(반도체+KOSHA)로 확대, 전체 시스템 검색 품질 일관성 확보 |

---

## 1. 현재 상태 분석

### 데이터 현황
- **위치**: `documents/semiconductor/안전보건공단/`
- **문서 폴더**: 7개
  1. `chemical health risk assessment guide` — 화학물질 건강 위험성평가 가이드 (영문명)
  2. `전자산업_확산_공정_설비_정비_작업_안전보건_가이드`
  3. `전자산업_포토_공정_설비_정비_작업_안전보건_가이드`
  4. `전자산업_산화공정_설비_정비작업_안전보건가이드`
  5. `전자산업_크린룸내_세정_공정_작업_안전보건가이드`
  6. `전자산업 크린룸 공정 지원 설비 정비작업 안전 보건 가이드2`
  7. `전자제품 제조공정의 정비 작업자를 위한 화학물질 건강 위험성평가`
- **총 파일**: 135개 (MD + JSON + 이미지)
- **주요 컨텐츠 파일**: 7개 MD + 7개 merged.json

### 기존 설정
- **현재 네임스페이스**: `safeguide`
- **도메인 키**: `safeguide`
- **Contextual Retrieval 프롬프트**: `src/context_generator.py`에 `safeguide` 도메인 프롬프트 이미 존재

### 선행 사례 (semiconductor-v2)
- NCS 데이터 Contextual Retrieval 적용: 813 → 9,744 벡터
- A/B 비교 결과: 성공률 60% → 100%, 답변 길이 82%↑, 소스 수 93%↑

---

## 2. 구현 전략

### 2.1 인제스천 명령어

각 폴더를 개별적으로 `--contextual` 플래그와 함께 처리:

```bash
# 1. 기본 4개 (언더스코어 파일명)
python main.py process "./documents/semiconductor/안전보건공단/전자산업_확산_공정_설비_정비_작업_안전보건_가이드" \
    --namespace kosha --contextual --skip-images --domain safeguide --batch-size 50

python main.py process "./documents/semiconductor/안전보건공단/전자산업_포토_공정_설비_정비_작업_안전보건_가이드" \
    --namespace kosha --contextual --skip-images --domain safeguide --batch-size 50

python main.py process "./documents/semiconductor/안전보건공단/전자산업_산화공정_설비_정비작업_안전보건가이드" \
    --namespace kosha --contextual --skip-images --domain safeguide --batch-size 50

python main.py process "./documents/semiconductor/안전보건공단/전자산업_크린룸내_세정_공정_작업_안전보건가이드" \
    --namespace kosha --contextual --skip-images --domain safeguide --batch-size 50

# 2. 공백 포함 폴더명
python main.py process "./documents/semiconductor/안전보건공단/전자산업 크린룸 공정 지원 설비 정비작업 안전 보건 가이드2" \
    --namespace kosha --contextual --skip-images --domain safeguide --batch-size 50

python main.py process "./documents/semiconductor/안전보건공단/전자제품 제조공정의 정비 작업자를 위한 화학물질 건강 위험성평가" \
    --namespace kosha --contextual --skip-images --domain safeguide --batch-size 50

python main.py process "./documents/semiconductor/안전보건공단/chemical health risk assessment guide" \
    --namespace kosha --contextual --skip-images --domain safeguide --batch-size 50
```

### 2.2 도메인 설정 변경

**파일**: `services/domain_config.py`

```python
# 변경 전
'안전보건공단': 'safeguide',

# 변경 후
'안전보건공단': 'kosha',
```

```python
# DOMAIN_CONFIGS 내 namespace 변경
'safeguide': {
    'namespace': 'kosha',  # 변경: safeguide → kosha
    ...
}
```

### 2.3 변경 파일 목록

| 파일 | 변경 내용 |
|------|----------|
| `services/domain_config.py` | `DIRECTORY_NAMESPACE_MAP`과 `DOMAIN_CONFIGS`에서 namespace를 `kosha`로 변경 |

---

## 3. 예상 비용 및 시간

### Contextual Retrieval LLM 비용

| 항목 | 추정 |
|------|------|
| 문서 수 | 7개 |
| 예상 청크 수 | ~200-400개 (7문서 × 30-60 청크) |
| 모델 | claude-haiku-4-5 (프롬프트 캐싱 적용) |
| 예상 비용 | ~$0.05-0.15 (캐싱 90% 할인 적용) |
| 처리 시간 | ~10-20분 |

### OpenAI Embedding 비용

| 항목 | 추정 |
|------|------|
| 모델 | text-embedding-3-small |
| 예상 토큰 | ~100K-200K |
| 예상 비용 | ~$0.002-0.004 |

---

## 4. Success Criteria

| Metric | Target |
|--------|--------|
| 7개 가이드 전체 인제스천 성공 | 에러 0개 |
| `kosha` 네임스페이스 벡터 수 | 기존 `safeguide` 대비 유사하거나 증가 |
| 도메인 페이지 검색 정상 동작 | safeguide 도메인에서 kosha 네임스페이스 사용 |
| 샘플 질문 3개 이상 정상 응답 | 답변 생성 + 소스 인용 확인 |

---

## 5. 리스크 및 대응

| 리스크 | 대응 |
|--------|------|
| 한글 공백 포함 폴더명으로 CLI 실패 | 따옴표로 경로 감싸기 |
| Contextual Retrieval LLM 호출 타임아웃 | 개별 폴더 단위 처리로 격리 |
| 기존 `safeguide` 데이터 유실 | 기존 네임스페이스 삭제 안 함, 새 `kosha`에만 업로드 |
| Reranker 1024 토큰 제한 초과 | 이미 수정 완료된 동적 truncation 로직 적용됨 |
