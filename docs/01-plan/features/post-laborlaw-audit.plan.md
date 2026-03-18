# 주석 처리 이후 전체 검색 모듈 점검 Plan

## Executive Summary

| 항목 | 내용 |
|------|------|
| Feature | laborlaw 주석 처리 이후 검색 모듈 무결성 점검 |
| 작성일 | 2026-03-18 |
| 예상 소요 | 30분 |
| 난이도 | Low |

### Value Delivered

| 관점 | 설명 |
|------|------|
| **Problem** | laborlaw 비활성화 후 잔존 dead code, 스텁 함수, 방어 로직 부재 등 6건의 클린업 필요 항목 식별 |
| **Solution** | dead code 제거 + 방어 guard 추가로 코드 품질 확보. `[LABORLAW_DISABLED]` 마커 기반 체계적 정리 |
| **Function UX Effect** | laborlaw namespace 직접 요청 시 명확한 에러 대신 기본 도메인 폴백 처리 |
| **Core Value** | 불필요 코드 제거로 유지보수성 향상, 런타임 안정성 확보 |

---

## 1. 코드 분석 결과 (code-analyzer)

### 1.1 품질 점수: 92/100

| 카테고리 | 결과 |
|----------|------|
| Critical Issues | **0건** — 런타임 에러 없음 |
| Warnings | **6건** — dead code 클린업 필요 |
| Verified Safe | **10건** — 모든 폴백 경로 안전 확인 |

### 1.2 확인된 안전 항목 (변경 불필요)

| 모듈 | 확인 결과 |
|------|-----------|
| `domain_config.py` | `NAMESPACE_DOMAIN_MAP.get('laborlaw', 'semiconductor')` → 안전 폴백 |
| `query_router.py` | `classify_domain()` / `classify_query_type()` 정상 동작 |
| `filters.py` | `build_domain_filter('laborlaw')` → NCS 필터로 폴백 (None → 무필터) |
| `search.py` | `_resolve_llm('laborlaw')` → 글로벌 설정으로 폴백 |
| `graph_config.py` | `get_graph_config('laborlaw')` → `{'enabled': False}` 기본값 |
| `major_config.py` | MAJOR_CONFIG 순회 시 laborlaw 미포함 |
| 기타 src/ 모듈 | `.get()` 패턴으로 모두 기본값 폴백 |

---

## 2. 수정 필요 항목 (6건)

| # | 파일 | 작업 | 유형 |
|---|------|------|------|
| 1 | `services/domain_config.py` L240 | 주석 내 stale laborlaw 예시 제거 | 주석 정리 |
| 2 | `templates/admin.html` L1046 | `FOLDER_ICONS`에서 laborlaw 키 제거 | dead code |
| 3 | `services/rag_pipeline.py` L1103-1107 | laborlaw 메타데이터 키 순회 — 무해하지만 dead iteration | dead code |
| 4 | `src/semantic_chunker.py` L439-444 | `LABORLAW_SECTION_PATTERNS` 빈 속성 + `_split_by_laborlaw_structure` 스텁 | dead code |
| 5 | `src/semantic_chunker.py` L664-704 | `_extract_laborlaw_metadata`, `_classify_laborlaw_category` 스텁 함수 | dead code |
| 6 | `src/semantic_chunker.py` L919 | `laborlaw_metadata = {}` 미사용 할당 | dead code |

---

## 3. 구현 순서

1. [ ] `templates/admin.html` — FOLDER_ICONS에서 `laborlaw` 키 주석 처리
2. [ ] `src/semantic_chunker.py` — dead 스텁 함수/속성/변수 정리
3. [ ] `services/rag_pipeline.py` — laborlaw 메타데이터 키 순회 제거
4. [ ] `services/domain_config.py` — stale 주석 정리
5. [ ] 전체 앱 기동 테스트 확인

---

## 4. 방어 로직 시나리오

> **시나리오**: 사용자가 API에 `namespace='laborlaw'` 직접 전송

현재 동작:
1. Pinecone에서 기존 laborlaw 벡터 검색 → 결과 반환 가능
2. `DOMAIN_PROMPTS`에 laborlaw 없음 → 기본 반도체 프롬프트 사용
3. 답변 품질 낮지만 **에러 없음**

→ 현재 폴백 동작으로 충분. 명시적 차단은 불필요.
