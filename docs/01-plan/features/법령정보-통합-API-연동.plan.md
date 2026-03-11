# 법령정보 통합 API 연동 Planning Document

> **Summary**: law.go.kr DRF Open API를 통해 법령/판례/행정해석/행정규칙 4개 소스를 통합 연동하여 노동법 RAG 답변 품질을 대폭 향상
>
> **Project**: SafeFactory
> **Author**: Claude Code
> **Date**: 2026-03-11
> **Status**: Draft

---

## Executive Summary

| Perspective | Content |
|-------------|---------|
| **Problem** | 현재 법령 조문만 RAG에 주입되며, 행정해석/판례/훈령 등 실무적으로 중요한 법적 근거가 누락되어 AI 답변의 신뢰성과 깊이가 부족함 |
| **Solution** | law.go.kr DRF Open API 단일 게이트웨이를 통해 5개 target(law, prec, moelCgmExpc, expc, admrul)을 통합 연동하고, 질문 유형별 자동 소스 선택 로직 구현 |
| **Function/UX Effect** | 노동법 질문 시 관련 법령 조문 + 대법원 판례 요지 + 고용노동부 행정해석 + 관련 고시/지침이 함께 제공되어 원스톱 법률 상담 경험 구현 |
| **Core Value** | 법령-판례-해석-행정규칙 4중 근거 기반 답변으로 "AI 노무사" 수준의 신뢰성 확보 |

---

## 1. Overview

### 1.1 Purpose

현재 SafeFactory의 노동법(laborlaw) 도메인은 법제처 law.go.kr AJAX 스크래핑과 odcloud API만으로 **법령 조문 텍스트**만 RAG 프롬프트에 주입하고 있다.

그러나 실무에서 노동법 질의응답은 다음 4가지 소스가 모두 필요하다:

1. **법령 조문** — 법적 근거의 기본 (현재 구현됨, 그러나 AJAX 스크래핑 방식)
2. **행정해석(질의회신)** — 고용노동부의 유권해석으로, 법 조문의 실무적 적용 기준
3. **법원 판례** — 대법원/하급심 판결로, 법 해석의 최종 권위
4. **훈령/예규/고시/지침** — 최저임금 고시, 근로감독 지침 등 행정 실무 규범

이 4개 소스를 **law.go.kr DRF Open API** 단일 게이트웨이로 통합 연동한다.

### 1.2 Background

- 현재 `services/law_api.py`는 law.go.kr AJAX(`lsInfoR.do`)를 스크래핑하여 조문을 가져옴
- AJAX 엔드포인트는 비공식이며, HTML 파싱에 의존하여 깨지기 쉬움
- law.go.kr DRF Open API는 공식 API로 XML/JSON 구조화된 응답 제공
- 동일한 DRF 게이트웨이에서 `target` 파라미터만 변경하면 법령/판례/해석/행정규칙 모두 조회 가능
- 인증: `OC` 파라미터(이메일 ID) + 서버 IP 등록 필요 (API 키 불필요)

### 1.3 Related Documents

- 현재 구현: `services/law_api.py` (law.go.kr AJAX + odcloud API)
- API 가이드: `https://open.law.go.kr/LSO/openApi/guideList.do`
- 판례 API 가이드: `https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=precListGuide`
- 행정해석 API 가이드: `https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=cgmExpcMoelListGuide`
- 행정규칙 API 가이드: `https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=admrulListGuide`

---

## 2. Scope

### 2.1 In Scope

- [x] **API 게이트웨이 통합**: law.go.kr DRF 단일 클라이언트로 5개 target 지원
- [x] **법령 조문 업그레이드**: AJAX 스크래핑 → DRF Open API(`target=law`) 전환
- [x] **판례 검색 연동**: `target=prec` (대법원/하급심 노동 판례)
- [x] **행정해석 연동**: `target=moelCgmExpc` (고용노동부 질의회신)
- [x] **행정규칙 연동**: `target=admrul` (훈령/예규/고시/지침)
- [x] **질문-소스 자동 매칭**: 질문 키워드/유형별 적절한 소스 자동 선택
- [x] **RAG 프롬프트 통합 포맷**: 4개 소스를 통합하여 LLM 프롬프트에 주입
- [x] **캐싱 및 서킷브레이커**: 기존 패턴 확장 적용

### 2.2 Out of Scope

- 별도 검색 UI 페이지 (향후 별도 feature로 분리)
- 법령/판례 데이터의 Pinecone 벡터 인덱싱 (별도 인제스트 파이프라인)
- 헌법재판소 결정례(`target=detc`) 연동
- 타 부처(기재부, 국세청 등) 행정해석 연동

---

## 3. Requirements

### 3.1 Functional Requirements

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| FR-01 | law.go.kr DRF 통합 클라이언트 (`LawDrfClient`) 구현 | High | Pending |
| FR-02 | 법령 조문 조회를 DRF API(`target=law`)로 전환 | High | Pending |
| FR-03 | 판례 검색 + 본문 조회(`target=prec`) 구현 | High | Pending |
| FR-04 | 고용노동부 행정해석 검색 + 본문 조회(`target=moelCgmExpc`) 구현 | High | Pending |
| FR-05 | 행정규칙 검색 + 본문 조회(`target=admrul`, knd=1~5 필터) 구현 | Medium | Pending |
| FR-06 | 질문 키워드 → 소스 자동 선택 로직 (법령/판례/해석/규칙) | High | Pending |
| FR-07 | 4개 소스 통합 포맷터 (`format_legal_references()`) | High | Pending |
| FR-08 | `rag_pipeline.py` 통합 — laborlaw 도메인에서 4개 소스 자동 주입 | High | Pending |
| FR-09 | 기존 `_LAW_REGISTRY` 하드코딩 → DRF API 동적 조회로 점진 전환 | Medium | Pending |
| FR-10 | 환경변수 `LAW_OC` (DRF 인증용 이메일 ID) 추가 | High | Pending |

### 3.2 Non-Functional Requirements

| Category | Criteria | Measurement Method |
|----------|----------|-------------------|
| Performance | 법령 API 응답 2초 이내, 전체 4소스 병렬 조회 5초 이내 | 로그 타이밍 |
| Reliability | 서킷브레이커 5회 실패 → 비활성화, 5분 후 재시도 | 기존 패턴 유지 |
| Caching | TTL 5분 캐싱 (동일 쿼리 재사용) | 캐시 히트율 모니터링 |
| Fallback | DRF API 실패 시 기존 AJAX 스크래핑으로 폴백 | 로그 확인 |
| Token Budget | 4개 소스 합산 프롬프트 주입 토큰 최대 3000자 이내 | 문자 수 제한 |

---

## 4. API Specification

### 4.1 law.go.kr DRF API 엔드포인트

모든 API는 동일한 베이스 URL에서 `target` 파라미터로 구분:

| 소스 | target 값 | 목록 검색 | 본문 조회 |
|------|-----------|-----------|-----------|
| 법령 | `law` | `lawSearch.do?target=law` | `lawService.do?target=law` |
| 판례 | `prec` | `lawSearch.do?target=prec` | `lawService.do?target=prec` |
| 고용노동부 행정해석 | `moelCgmExpc` | `lawSearch.do?target=moelCgmExpc` | `lawService.do?target=moelCgmExpc` |
| 일반 법령해석례 | `expc` | `lawSearch.do?target=expc` | `lawService.do?target=expc` |
| 행정규칙 | `admrul` | `lawSearch.do?target=admrul` | `lawService.do?target=admrul` |

**Base URLs**:
- 목록: `http://www.law.go.kr/DRF/lawSearch.do`
- 본문: `http://www.law.go.kr/DRF/lawService.do`

**공통 인증 파라미터**:
- `OC`: 이메일 ID (예: `safefactory` from `safefactory@example.com`)
- `type`: `XML` 또는 `JSON`

### 4.2 각 target별 주요 파라미터

#### 법령 (`target=law`)
```
lawSearch.do?OC={oc}&target=law&type=JSON&query={검색어}&display=20&org=1440000 (고용노동부)
lawService.do?OC={oc}&target=law&type=JSON&MST={법령명}
```
- `org=1440000`: 고용노동부 소관 법령 필터
- `search=2`: 본문 전문 검색

#### 판례 (`target=prec`)
```
lawSearch.do?OC={oc}&target=prec&type=JSON&query={검색어}&org=400201 (대법원)
lawService.do?OC={oc}&target=prec&type=JSON&ID={판례일련번호}
```
- 응답 필드: 사건명, 사건번호, 선고일자, 판결유형, **판시사항**, **판결요지**, 참조조문
- `org=400201`: 대법원, `org=400202`: 하급심

#### 고용노동부 행정해석 (`target=moelCgmExpc`)
```
lawSearch.do?OC={oc}&target=moelCgmExpc&type=JSON&query={검색어}
lawService.do?OC={oc}&target=moelCgmExpc&type=JSON&ID={법령해석일련번호}
```
- 응답 필드: 안건명, 안건번호, **질의요지**, **회답**, **이유**, 관련법령

#### 행정규칙 (`target=admrul`)
```
lawSearch.do?OC={oc}&target=admrul&type=JSON&query={검색어}&knd={종류}&org=1440000
lawService.do?OC={oc}&target=admrul&type=JSON&ID={일련번호}
```
- `knd`: 1=훈령, 2=예규, 3=고시, 4=공고, 5=지침
- 응답 필드: 행정규칙명, 행정규칙종류, 조문내용

### 4.3 인증 등록 절차

1. `open.law.go.kr` 회원가입 (이메일 기반)
2. OpenAPI 활용 신청 → 서버 IP/도메인 등록
3. 이메일 ID가 `OC` 파라미터로 사용됨 (별도 API 키 불필요)
4. 문의: 02-2109-6446

---

## 5. Architecture Design

### 5.1 모듈 구조

```
services/
├── law_api.py              ← 기존 파일 (호환성 유지, 내부적으로 DRF 클라이언트 사용)
├── law_drf_client.py       ← [NEW] law.go.kr DRF 통합 클라이언트
├── legal_source_router.py  ← [NEW] 질문→소스 자동 선택 라우터
└── rag_pipeline.py         ← 수정: 4소스 통합 주입
```

### 5.2 클래스 설계

```python
# law_drf_client.py
class LawDrfClient:
    """law.go.kr DRF Open API 통합 클라이언트"""
    BASE_SEARCH = 'http://www.law.go.kr/DRF/lawSearch.do'
    BASE_SERVICE = 'http://www.law.go.kr/DRF/lawService.do'

    def search(self, target: str, query: str, **params) -> list[dict]
    def get_detail(self, target: str, item_id: str) -> dict

    # 편의 메서드
    def search_laws(self, query: str) -> list[dict]           # target=law
    def search_precedents(self, query: str) -> list[dict]     # target=prec
    def search_interpretations(self, query: str) -> list[dict] # target=moelCgmExpc
    def search_admin_rules(self, query: str, knd: str = None) -> list[dict]  # target=admrul

    def get_law_text(self, law_id: str) -> dict
    def get_precedent_text(self, prec_id: str) -> dict
    def get_interpretation_text(self, interp_id: str) -> dict
    def get_admin_rule_text(self, rule_id: str) -> dict
```

```python
# legal_source_router.py
class LegalSourceRouter:
    """질문 분석 → 적절한 법적 소스 자동 선택"""

    def route(self, query: str, classification: dict | None) -> list[SourceRequest]:
        """질문에 필요한 소스 목록 반환
        Returns: [SourceRequest(target='law', query=...), SourceRequest(target='prec', ...)]
        """

    def search_all(self, query: str, classification: dict | None) -> LegalContext:
        """4개 소스 병렬 검색 → 통합 결과"""
```

### 5.3 RAG 파이프라인 통합 흐름

```
사용자 질문 (laborlaw 도메인)
    │
    ├─ 기존: QueryEnhancer → Vector+BM25 검색 → Reranker → Context (문서 기반)
    │
    └─ [NEW] LegalSourceRouter.search_all(query)
         ├─ LawDrfClient.search_laws()           → 법령 조문
         ├─ LawDrfClient.search_precedents()      → 판례 요지    ← 병렬 실행
         ├─ LawDrfClient.search_interpretations() → 행정해석 회답
         └─ LawDrfClient.search_admin_rules()     → 관련 고시/지침
              │
              ▼
         format_legal_references()  → 통합 포맷
              │
              ▼
         LLM 프롬프트 주입 (## 관련 법적 근거)
```

### 5.4 소스 라우팅 규칙

| 질문 유형 | 법령 | 판례 | 행정해석 | 행정규칙 | 예시 |
|-----------|:----:|:----:|:-------:|:-------:|------|
| 법조문 해석 | O | - | O | - | "연차휴가 산정 방법은?" |
| 분쟁/다툼 | O | O | - | - | "부당해고 구제 가능한가요?" |
| 실무 적용 | O | - | O | O | "5인 미만 사업장에도 적용되나요?" |
| 금액/기준 | O | - | - | O | "2026년 최저임금은 얼마?" |
| 판례 직접 질문 | - | O | - | - | "최근 직장내 괴롭힘 판례는?" |
| 종합 질문 | O | O | O | O | "해고예고수당 관련 법령과 판례" |

---

## 6. Success Criteria

### 6.1 Definition of Done

- [ ] `LawDrfClient` 구현 완료 (5개 target 지원)
- [ ] `LegalSourceRouter` 질문→소스 자동 선택 구현
- [ ] RAG 파이프라인에 4소스 통합 주입
- [ ] 기존 law_api.py 하위 호환성 유지
- [ ] 환경변수 `LAW_OC` 없으면 기존 AJAX 방식으로 폴백
- [ ] 캐싱/서킷브레이커/에러 핸들링 적용
- [ ] 프로덕션 서버 IP open.law.go.kr 등록 완료

### 6.2 Quality Criteria

- [ ] 4개 소스 병렬 조회 5초 이내
- [ ] 프롬프트 주입 토큰 3000자 이내 (소스당 ~750자)
- [ ] DRF API 실패 시 기존 AJAX 폴백 정상 동작
- [ ] 10개 노동법 대표 질문에 대해 관련 소스 정확히 매칭

---

## 7. Risks and Mitigation

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| law.go.kr DRF API IP 등록 거부/지연 | High | Low | 기존 AJAX 스크래핑을 폴백으로 유지, OC 테스트 계정 선 확보 |
| API 응답 지연 (정부 서버 특성) | Medium | Medium | 병렬 요청 + 5초 타임아웃 + TTL 캐시(5분) |
| DRF API 응답 포맷 변경 | Medium | Low | XML/JSON 파서 방어적 구현, 필드 없으면 graceful skip |
| 프롬프트 토큰 과다 (4소스 합산) | High | Medium | 소스별 최대 문자 제한(750자), 우선순위 기반 truncation |
| 판례/해석 결과의 관련성 낮음 | Medium | Medium | LLM 기반 relevance filtering, 키워드 매칭 정밀도 개선 |

---

## 8. Architecture Considerations

### 8.1 Project Level

| Level | Selected |
|-------|:--------:|
| **Dynamic** | O |

### 8.2 Key Architectural Decisions

| Decision | Options | Selected | Rationale |
|----------|---------|----------|-----------|
| API 응답 포맷 | XML / JSON | JSON | 파이썬 처리 용이, 기존 코드와 일관성 |
| 병렬 요청 | threading / asyncio / concurrent.futures | `concurrent.futures.ThreadPoolExecutor` | Flask 동기 환경 호환, 기존 threading 패턴 일관 |
| 새 모듈 vs 기존 확장 | law_api.py 확장 / 새 모듈 분리 | 새 모듈 분리 + 기존 호환 | SRP 준수, 기존 코드 영향 최소화 |
| 폴백 전략 | DRF only / DRF + AJAX fallback | DRF + AJAX fallback | IP 등록 전까지 기존 기능 유지 |

---

## 9. Convention Prerequisites

### 9.1 환경변수 추가

| Variable | Purpose | Scope | Required |
|----------|---------|-------|:--------:|
| `LAW_OC` | law.go.kr DRF 인증용 이메일 ID | Server | Yes (DRF 사용 시) |
| `LAW_API_KEY` | 기존 odcloud API 키 (폴백용) | Server | No (기존 유지) |

### 9.2 기존 컨벤션 준수

- 싱글톤 패턴: `services/singletons.py`에 `get_law_drf_client()` 추가
- 스레드 안전: `threading.RLock()` 사용
- 캐싱: 기존 `_cache` + `_CACHE_TTL` 패턴 확장
- 로깅: `[LawDRF]` 프리픽스

---

## 10. Implementation Order

### Phase 1: DRF 클라이언트 기반 (FR-01, FR-10)
1. `services/law_drf_client.py` — `LawDrfClient` 통합 클라이언트
2. `.env` + `singletons.py` — `LAW_OC` 환경변수, 싱글톤 등록

### Phase 2: 법령 전환 (FR-02, FR-09)
3. 법령 조문 조회를 DRF API로 전환 (기존 AJAX → DRF, 폴백 유지)

### Phase 3: 신규 소스 연동 (FR-03, FR-04, FR-05)
4. 판례 검색 + 본문 조회
5. 행정해석 검색 + 본문 조회
6. 행정규칙 검색 + 본문 조회

### Phase 4: 통합 (FR-06, FR-07, FR-08)
7. `LegalSourceRouter` — 질문→소스 자동 선택
8. 통합 포맷터 `format_legal_references()`
9. `rag_pipeline.py` 연동

---

## 11. Next Steps

1. [ ] Design 문서 작성 (`법령정보-통합-API-연동.design.md`)
2. [ ] open.law.go.kr 계정 생성 및 API 활용 신청
3. [ ] 프로덕션 서버 IP 등록
4. [ ] 구현 시작

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-03-11 | Initial draft | Claude Code |
