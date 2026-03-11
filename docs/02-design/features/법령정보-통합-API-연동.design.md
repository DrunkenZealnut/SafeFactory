# 법령정보 통합 API 연동 Design Document

> **Summary**: law.go.kr DRF Open API 통합 클라이언트 + 질문-소스 자동 라우터 + RAG 프롬프트 4소스 통합 주입
>
> **Project**: SafeFactory
> **Author**: Claude Code
> **Date**: 2026-03-11
> **Status**: Draft
> **Planning Doc**: [법령정보-통합-API-연동.plan.md](../01-plan/features/법령정보-통합-API-연동.plan.md)

---

## 1. Overview

### 1.1 Design Goals

1. law.go.kr DRF Open API 단일 클라이언트로 5개 target(law, prec, moelCgmExpc, expc, admrul) 통합
2. 질문 유형별 법령/판례/행정해석/행정규칙 자동 소스 선택 로직
3. 기존 `law_api.py`의 하위 호환성 유지 + DRF 미등록 시 AJAX 폴백
4. 4개 소스 병렬 조회 + 통합 포맷 후 RAG 프롬프트 주입

### 1.2 Design Principles

- **단일 책임**: DRF 통신(`law_drf_client.py`) / 소스 라우팅(`legal_source_router.py`) / 기존 호환(`law_api.py`) 분리
- **점진적 전환**: `LAW_OC` 환경변수 유무로 DRF/AJAX 자동 전환, 기존 기능 절대 깨지지 않음
- **방어적 설계**: 각 소스 독립 실패 허용, 하나가 실패해도 나머지 소스 정상 반환

---

## 2. Architecture

### 2.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│  rag_pipeline.py                                                    │
│  run_rag_pipeline(query, namespace='laborlaw')                      │
│                                                                     │
│  ┌─────────────────┐   ┌──────────────────────────────────────────┐ │
│  │ Vector+BM25     │   │ LegalSourceRouter (NEW)                  │ │
│  │ search_pipeline │   │                                          │ │
│  │   (기존 유지)    │   │  route(query) → [SourceRequest, ...]    │ │
│  └────────┬────────┘   │  search_all(query) → LegalContext        │ │
│           │             │                                          │ │
│           │             │  ┌─────────────────────────────────┐    │ │
│           │             │  │ LawDrfClient (NEW)               │    │ │
│           │             │  │                                   │    │ │
│           │             │  │  search(target, query)           │    │ │
│           │             │  │  get_detail(target, id)          │    │ │
│           │             │  │                                   │    │ │
│           │             │  │  ┌──────────────────────────┐    │    │ │
│           │             │  │  │ law.go.kr DRF Open API   │    │    │ │
│           │             │  │  │ lawSearch.do/lawService.do│    │    │ │
│           │             │  │  └──────────────────────────┘    │    │ │
│           │             │  └─────────────────────────────────┘    │ │
│           │             └──────────────────────────────────────────┘ │
│           │                            │                             │
│           ▼                            ▼                             │
│  ┌──────────────────────────────────────────────┐                   │
│  │ build_llm_prompts()                           │                   │
│  │   ## 참고 문서 (Vector/BM25)                   │                   │
│  │   ## 관련 법적 근거 (4소스 통합) ← NEW          │                   │
│  └──────────────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────────┘
                                │
                  ┌─────────────┴──────────────┐
                  │ law_api.py (기존 호환 래퍼)   │
                  │                             │
                  │ search_labor_laws()          │
                  │   → DRF 사용 가능? → DRF     │
                  │   → 불가능? → 기존 AJAX      │
                  │                             │
                  │ format_law_references()      │
                  │   (기존 유지)                 │
                  └─────────────────────────────┘
```

### 2.2 Data Flow

```
사용자 질문 (laborlaw 도메인)
    │
    ├─ [기존] Vector+BM25 검색 → Reranker → Context
    │
    └─ [NEW] LegalSourceRouter.search_all(query, classification)
         │
         ├─ 1) route(query) → 질문 분석 → 필요 소스 결정
         │     예: "부당해고 구제" → [law, prec, moelCgmExpc]
         │
         ├─ 2) ThreadPoolExecutor 병렬 실행
         │     ├─ LawDrfClient.search('law', '해고') → 법령 조문
         │     ├─ LawDrfClient.search('prec', '부당해고') → 판례 요지
         │     └─ LawDrfClient.search('moelCgmExpc', '해고') → 행정해석
         │
         ├─ 3) 각 결과에서 상위 항목 상세 조회 (get_detail)
         │     ├─ get_detail('prec', prec_id) → 판결요지 텍스트
         │     └─ get_detail('moelCgmExpc', interp_id) → 회답 텍스트
         │
         └─ 4) LegalContext 반환
               ├─ law_articles: [{name, article, text}, ...]
               ├─ precedents: [{case_name, case_no, summary}, ...]
               ├─ interpretations: [{title, answer, reason}, ...]
               └─ admin_rules: [{name, type, content}, ...]
                     │
                     ▼
              format_legal_context() → 통합 마크다운
                     │
                     ▼
              build_llm_prompts() → "## 관련 법적 근거" 섹션
```

### 2.3 Dependencies

| Component | Depends On | Purpose |
|-----------|-----------|---------|
| `LawDrfClient` | `requests`, `os.environ['LAW_OC']` | DRF API 통신 |
| `LegalSourceRouter` | `LawDrfClient`, `concurrent.futures` | 소스 라우팅 + 병렬 조회 |
| `law_api.py` | `LawDrfClient` (optional), 기존 AJAX 로직 | 하위 호환 래퍼 |
| `rag_pipeline.py` | `LegalSourceRouter` | RAG 프롬프트 주입 |
| `singletons.py` | `LawDrfClient` | 싱글톤 관리 |

---

## 3. Data Model

### 3.1 Core Data Structures

```python
from dataclasses import dataclass, field

@dataclass
class SourceRequest:
    """질문 분석 후 필요한 소스 요청"""
    target: str          # 'law', 'prec', 'moelCgmExpc', 'admrul'
    query: str           # 해당 소스에 보낼 검색어
    params: dict = field(default_factory=dict)  # 추가 파라미터 (org, knd 등)
    max_results: int = 5


@dataclass
class LawArticle:
    """법령 조문"""
    law_name: str        # 근로기준법
    article: str         # 제23조(해고 등의 제한)
    text: str            # 조문 본문 (truncated)
    source: str = 'drf'  # 'drf' or 'ajax' (폴백)


@dataclass
class Precedent:
    """법원 판례"""
    case_name: str       # 부당해고구제재심판정취소
    case_no: str         # 2023다12345
    court: str           # 대법원
    date: str            # 2024.05.15
    ruling_type: str     # 판결
    summary: str         # 판결요지 (truncated)
    ref_articles: str    # 참조조문


@dataclass
class Interpretation:
    """행정해석 (질의회신)"""
    title: str           # 안건명
    case_no: str         # 안건번호
    date: str            # 해석일자
    question: str        # 질의요지 (truncated)
    answer: str          # 회답 (truncated)
    reason: str          # 이유 (truncated)
    ref_laws: str        # 관련법령


@dataclass
class AdminRule:
    """행정규칙 (훈령/예규/고시/지침)"""
    name: str            # 행정규칙명
    rule_type: str       # 훈령/예규/고시/지침
    date: str            # 발령일자
    org: str             # 소관부처
    content: str         # 조문내용 (truncated)


@dataclass
class LegalContext:
    """4개 소스 통합 결과"""
    law_articles: list[LawArticle] = field(default_factory=list)
    precedents: list[Precedent] = field(default_factory=list)
    interpretations: list[Interpretation] = field(default_factory=list)
    admin_rules: list[AdminRule] = field(default_factory=list)

    @property
    def has_content(self) -> bool:
        return bool(self.law_articles or self.precedents
                    or self.interpretations or self.admin_rules)

    @property
    def source_count(self) -> int:
        return (len(self.law_articles) + len(self.precedents)
                + len(self.interpretations) + len(self.admin_rules))
```

---

## 4. Detailed API Specification

### 4.1 LawDrfClient — law.go.kr DRF 통합 클라이언트

**File**: `services/law_drf_client.py`

```python
class LawDrfClient:
    """law.go.kr DRF Open API 통합 클라이언트.

    모든 target(law, prec, moelCgmExpc, expc, admrul)을 단일 인터페이스로 제공.
    인증: OC 파라미터 (이메일 ID) + 서버 IP 등록 필요.
    """

    BASE_SEARCH = 'http://www.law.go.kr/DRF/lawSearch.do'
    BASE_SERVICE = 'http://www.law.go.kr/DRF/lawService.do'
    TIMEOUT = 10          # seconds (정부 서버 특성상 넉넉하게)
    MAX_DISPLAY = 20      # 목록 검색 최대 결과 수
    MAX_TEXT_CHARS = 800   # 상세 텍스트 최대 문자 수

    # 서킷브레이커 설정
    MAX_FAILURES = 5
    CIRCUIT_RESET_SEC = 300  # 5분

    def __init__(self, oc: str):
        """
        Args:
            oc: OC 파라미터 (이메일 ID, 예: 'safefactory')
        """

    # ── 범용 메서드 ──
    def search(self, target: str, query: str, **params) -> list[dict]:
        """목록 검색 (lawSearch.do).

        Args:
            target: 'law', 'prec', 'moelCgmExpc', 'expc', 'admrul'
            query: 검색어
            **params: display, page, org, knd, sort 등 추가 파라미터

        Returns:
            [{'법령명': ..., '법령일련번호': ..., ...}, ...]
            또는 target별 필드 (사건명, 안건명, 행정규칙명 등)
        """

    def get_detail(self, target: str, item_id: str, **params) -> dict:
        """본문 상세 조회 (lawService.do).

        Args:
            target: 검색 대상
            item_id: ID (법령일련번호, 판례일련번호, 해석일련번호 등)

        Returns:
            상세 정보 dict (조문, 판결요지, 회답 등)
        """

    # ── 편의 메서드 (target별 특화) ──
    def search_laws(self, query: str, org: str = '1440000',
                    display: int = 10) -> list[dict]:
        """법령 검색 (target=law, 기본 고용노동부 소관)."""

    def search_precedents(self, query: str, court: str = '400201',
                          display: int = 5) -> list[dict]:
        """판례 검색 (target=prec, 기본 대법원)."""

    def search_interpretations(self, query: str,
                               display: int = 5) -> list[dict]:
        """고용노동부 행정해석 검색 (target=moelCgmExpc)."""

    def search_admin_rules(self, query: str, knd: str | None = None,
                           org: str = '1440000',
                           display: int = 5) -> list[dict]:
        """행정규칙 검색 (target=admrul).
        knd: 1=훈령, 2=예규, 3=고시, 4=공고, 5=지침
        """

    def get_law_articles(self, law_name: str) -> list[LawArticle]:
        """법령명으로 조문 목록 조회. 기존 LawTextFetcher 대체."""

    def get_precedent_detail(self, prec_id: str) -> Precedent | None:
        """판례 상세 조회 → Precedent 객체 반환."""

    def get_interpretation_detail(self, interp_id: str) -> Interpretation | None:
        """행정해석 상세 조회 → Interpretation 객체 반환."""

    def get_admin_rule_detail(self, rule_id: str) -> AdminRule | None:
        """행정규칙 상세 조회 → AdminRule 객체 반환."""

    # ── 내부 메서드 ──
    def _request(self, base_url: str, params: dict) -> dict | list:
        """HTTP 요청 + 캐싱 + 서킷브레이커 + 에러 핸들링."""

    def _check_circuit(self) -> bool:
        """서킷브레이커 상태 확인. False면 요청 차단."""

    @property
    def available(self) -> bool:
        """DRF API 사용 가능 여부 (OC 설정 + 서킷브레이커 활성)."""
```

#### DRF API 응답 파싱 규칙

| target | 목록 검색 주요 필드 | 상세 조회 주요 필드 |
|--------|---------------------|---------------------|
| `law` | 법령명한글, 법령일련번호, 시행일자, 소관부처명 | 조문 (XML 내 조문키, 조문제목, 조문내용) |
| `prec` | 사건명, 사건번호, 선고일자, 법원명, 판례일련번호 | 판시사항, 판결요지, 참조조문, 판례내용 |
| `moelCgmExpc` | 안건명, 안건번호, 회신일자, 법령해석례일련번호 | 질의요지, 회답, 이유, 관련법령 |
| `admrul` | 행정규칙명, 행정규칙종류, 발령일자, 행정규칙일련번호 | 조문내용, 별표/서식 |

#### JSON 응답 구조 (예상)

```json
// lawSearch.do?target=prec&type=JSON 응답
{
  "PrecSearch": {
    "totalCnt": 150,
    "page": 1,
    "prec": [
      {
        "사건명": "부당해고구제재심판정취소",
        "사건번호": "2023두12345",
        "선고일자": "20240515",
        "법원명": "대법원",
        "판례일련번호": "228541",
        "판례상세링크": "/DRF/lawService.do?OC=...&ID=228541"
      }
    ]
  }
}
```

```json
// lawService.do?target=prec&type=JSON&ID=228541 응답
{
  "PrecService": {
    "사건명": "부당해고구제재심판정취소",
    "사건번호": "2023두12345",
    "선고일자": "20240515",
    "판시사항": "...",
    "판결요지": "...",
    "참조조문": "근로기준법 제23조, 제27조",
    "판례내용": "..."
  }
}
```

> **Note**: DRF API는 XML이 기본이며, JSON 응답 포맷은 target에 따라 다를 수 있음.
> 구현 시 실제 응답을 확인하고 파서를 조정해야 함.
> XML 폴백 파서도 준비할 것.

---

### 4.2 LegalSourceRouter — 소스 자동 선택 라우터

**File**: `services/legal_source_router.py`

```python
class LegalSourceRouter:
    """질문 분석 → 적절한 법적 소스 자동 선택 + 병렬 검색.

    기존 law_api.py의 _QUESTION_TO_ARTICLES 키워드 매핑을 확장하여,
    질문 유형에 따라 법령 외에 판례/행정해석/행정규칙도 자동 검색.
    """

    # 소스 선택 규칙: 키워드 → 필요 소스
    # 기존 _QUESTION_TO_ARTICLES의 키워드를 재활용
    SOURCE_RULES: dict[str, list[str]]  # keyword → ['law', 'prec', ...]

    def __init__(self, drf_client: LawDrfClient):
        self.drf = drf_client

    def route(self, query: str, classification: dict | None = None) -> list[SourceRequest]:
        """질문 분석 → 필요 소스 결정.

        로직:
        1. 키워드 매칭 (빠름, 기존 매핑 재활용)
        2. classification의 type 기반 (legal → 법령+해석, hybrid → 전체)
        3. 명시적 키워드 감지 ('판례' → prec, '행정해석' → moelCgmExpc)
        4. 기본값: 법령(law)은 항상 포함

        Returns:
            [SourceRequest(target='law', query='해고'), ...]
        """

    def search_all(self, query: str,
                   classification: dict | None = None) -> LegalContext:
        """4개 소스 병렬 검색 → LegalContext 반환.

        흐름:
        1. route() → 필요 소스 목록
        2. ThreadPoolExecutor로 목록 검색 병렬 실행
        3. 각 결과의 상위 항목 상세 조회
        4. LegalContext 조립

        타임아웃: 전체 5초 (개별 요청 TIMEOUT보다 짧을 수 있음)
        """

    def _search_source(self, req: SourceRequest) -> list:
        """단일 소스 검색 + 상세 조회. 실패 시 빈 리스트 반환."""

    def format_context(self, ctx: LegalContext,
                       start_index: int = 1) -> str:
        """LegalContext → 마크다운 포맷 (LLM 프롬프트 주입용).

        포맷:
        ### 관련 법령
        [N] **근로기준법** 제23조(해고 등의 제한)
        > 조문 텍스트...

        ### 관련 판례
        [N+1] **대법원 2024.05.15 선고 2023두12345**
        > 판결요지...

        ### 행정해석
        [N+2] **고용노동부 행정해석** (근로기준과-1234)
        > 회답...

        ### 관련 고시/지침
        [N+3] **2026년 최저임금 고시**
        > 내용...

        제한: 전체 3000자, 소스별 우선순위(법령 > 판례 > 해석 > 규칙)
        """
```

#### 소스 선택 상세 규칙

```python
# 키워드 → 추가 소스 매핑 (법령은 항상 포함)
_KEYWORD_TO_SOURCES = {
    # 판례가 필요한 키워드
    '해고': ['prec', 'moelCgmExpc'],
    '부당해고': ['prec', 'moelCgmExpc'],
    '괴롭힘': ['prec', 'moelCgmExpc'],
    '직장내 괴롭힘': ['prec'],
    '성희롱': ['prec', 'moelCgmExpc'],
    '산재': ['prec'],
    '업무상재해': ['prec'],
    '중대재해': ['prec'],

    # 행정해석이 필요한 키워드
    '연차': ['moelCgmExpc'],
    '휴가': ['moelCgmExpc'],
    '임금': ['moelCgmExpc'],
    '통상임금': ['prec', 'moelCgmExpc'],
    '퇴직금': ['moelCgmExpc'],
    '육아휴직': ['moelCgmExpc'],
    '근로시간': ['moelCgmExpc'],
    '연장근로': ['moelCgmExpc'],

    # 행정규칙이 필요한 키워드
    '최저임금': ['admrul'],      # 최저임금 고시
    '최저시급': ['admrul'],
    '안전교육': ['admrul'],      # 안전보건교육 규정
    '위험성평가': ['admrul'],    # 위험성평가 지침
    '안전보건': ['admrul'],
    '근로감독': ['admrul'],      # 근로감독관 집무규정

    # 명시적 소스 요청
    '판례': ['prec'],
    '판결': ['prec'],
    '행정해석': ['moelCgmExpc'],
    '질의회신': ['moelCgmExpc'],
    '고시': ['admrul'],
    '지침': ['admrul'],
    '훈령': ['admrul'],
    '예규': ['admrul'],
}

# classification type별 기본 소스
_TYPE_DEFAULT_SOURCES = {
    'legal': ['law', 'moelCgmExpc'],        # 법률 해석 → 법령 + 행정해석
    'calculation': ['law'],                   # 계산 → 법령만
    'hybrid': ['law', 'moelCgmExpc'],        # 복합 → 법령 + 행정해석
}
```

---

### 4.3 law_api.py 수정 — 하위 호환 래퍼

**변경 방식**: 기존 `search_labor_laws()` 내부에서 DRF 클라이언트 우선 사용, 실패/미설정 시 기존 AJAX 폴백.

```python
# law_api.py 수정 (기존 함수 시그니처 유지)

def search_labor_laws(query: str, classification: dict | None = None) -> list[dict]:
    """기존 인터페이스 유지. DRF 사용 가능 시 DRF, 아니면 기존 AJAX."""

    # Phase 0: DRF 사용 가능 여부 확인
    drf_client = _get_drf_client_if_available()

    if drf_client:
        # DRF 경로: LegalSourceRouter 사용
        try:
            router = LegalSourceRouter(drf_client)
            ctx = router.search_all(query, classification)
            if ctx.has_content:
                return _legal_context_to_legacy_format(ctx)
        except Exception as e:
            logger.warning("[LawAPI] DRF search failed, falling back to AJAX: %s", e)

    # Fallback: 기존 AJAX 경로 (현재 코드 그대로)
    return _search_via_ajax(query, classification)


def _get_drf_client_if_available() -> LawDrfClient | None:
    """LAW_OC 환경변수가 설정되어 있으면 DRF 클라이언트 반환."""
    oc = os.environ.get('LAW_OC')
    if not oc:
        return None
    from services.singletons import get_law_drf_client
    client = get_law_drf_client()
    return client if client.available else None


def _legal_context_to_legacy_format(ctx: LegalContext) -> list[dict]:
    """LegalContext → 기존 list[dict] 포맷 변환 (하위 호환)."""
    results = []
    for art in ctx.law_articles:
        results.append({
            'name': art.law_name,
            'article': art.article,
            'article_text': art.text,
            'source': art.source,
        })
    for prec in ctx.precedents:
        results.append({
            'name': f'{prec.court} {prec.date}',
            'article': f'{prec.case_name} ({prec.case_no})',
            'article_text': prec.summary,
            'source': 'precedent',
        })
    for interp in ctx.interpretations:
        results.append({
            'name': '고용노동부 행정해석',
            'article': f'{interp.title} ({interp.case_no})',
            'article_text': interp.answer,
            'source': 'interpretation',
        })
    for rule in ctx.admin_rules:
        results.append({
            'name': rule.name,
            'article': rule.rule_type,
            'article_text': rule.content,
            'source': 'admin_rule',
        })
    return results[:15]  # 최대 15개
```

---

### 4.4 rag_pipeline.py 수정

**변경 지점**: `run_rag_pipeline()` 내 laborlaw 법령 검색 부분 + `build_llm_prompts()` 프롬프트 포맷.

#### run_rag_pipeline() 변경

```python
# 기존 (line 631~641):
# from services.law_api import search_labor_laws, format_law_references
# law_refs = search_labor_laws(query, classification)

# 변경 후:
from services.law_api import search_labor_laws, format_law_references
law_refs = search_labor_laws(query, classification)  # 내부적으로 DRF/AJAX 자동 선택
if law_refs:
    result['law_references'] = law_refs
    source_count = len(result.get('sources', []))

    # 소스 유형별 분리 포맷 (NEW)
    has_multi_source = any(r.get('source') in ('precedent', 'interpretation', 'admin_rule')
                          for r in law_refs)
    if has_multi_source:
        result['law_references_formatted'] = _format_multi_source_refs(
            law_refs, start_index=source_count + 1)
    else:
        result['law_references_formatted'] = format_law_references(
            law_refs, start_index=source_count + 1)
```

#### build_llm_prompts() 프롬프트 변경

```python
# 기존: "## 관련 법령 정보"
# 변경: "## 관련 법적 근거" (법령 외 판례/해석/규칙 포함 시)

if law_references:
    section_title = "관련 법적 근거" if has_multi_source else "관련 법령 정보"
    user_prompt += f"""

## {section_title}
{law_references}

**인용 필수 규칙 (반드시 준수):**
- 위에 제공된 법적 근거 각각을 답변 본문에서 최소 1회 이상 인용하세요.
- 법령 조문은 조문번호와 함께, 판례는 사건번호와 함께, 행정해석은 해석번호와 함께 인용하세요.
- 모든 근거 번호가 답변에 빠짐없이 등장해야 합니다."""
```

---

### 4.5 singletons.py 추가

```python
_law_drf_client = None

def get_law_drf_client():
    """Get or create LawDrfClient singleton."""
    global _law_drf_client
    instance = _law_drf_client
    if instance is None:
        with _lock:
            if _law_drf_client is None:
                oc = os.getenv('LAW_OC', '')
                if oc:
                    from services.law_drf_client import LawDrfClient
                    _law_drf_client = LawDrfClient(oc=oc)
                else:
                    return None
            instance = _law_drf_client
    return instance
```

---

## 5. Error Handling

### 5.1 에러 처리 전략

| 상황 | 처리 방식 | 폴백 |
|------|----------|------|
| `LAW_OC` 미설정 | DRF 클라이언트 생성 안 함 | 기존 AJAX 스크래핑 |
| DRF API 타임아웃 (10초) | 해당 소스 빈 결과 반환 | 다른 소스 정상 반환 |
| DRF API 인증 실패 (IP 미등록) | 로그 경고 + 서킷 카운트 증가 | 기존 AJAX |
| DRF API 연속 5회 실패 | 서킷브레이커 Open (5분간 차단) | 기존 AJAX |
| 병렬 검색 전체 타임아웃 (5초) | 이미 완료된 소스만 사용 | 부분 결과 |
| JSON 파싱 에러 | XML 파싱 시도 후 실패 시 빈 결과 | graceful skip |
| 프롬프트 토큰 초과 | 우선순위 기반 truncation | 법령 > 판례 > 해석 > 규칙 |

### 5.2 서킷브레이커 설계

```python
class CircuitBreaker:
    """기존 law_api.py 패턴과 동일한 서킷브레이커."""
    MAX_FAILURES = 5
    RESET_INTERVAL = 300  # 5분

    def __init__(self):
        self._available = True
        self._failure_count = 0
        self._disabled_at: float | None = None

    def record_success(self):
        self._failure_count = 0

    def record_failure(self):
        self._failure_count += 1
        if self._failure_count >= self.MAX_FAILURES:
            self._available = False
            self._disabled_at = time.time()

    def is_available(self) -> bool:
        if not self._available and self._disabled_at:
            if time.time() - self._disabled_at > self.RESET_INTERVAL:
                self._available = True
                self._failure_count = 0
                self._disabled_at = None
        return self._available
```

### 5.3 로깅 규칙

| 모듈 | 프리픽스 | 레벨 |
|------|---------|------|
| `LawDrfClient` | `[LawDRF]` | INFO: 검색 결과 수, WARNING: 실패, DEBUG: 응답 상세 |
| `LegalSourceRouter` | `[LegalRouter]` | INFO: 선택된 소스, WARNING: 병렬 타임아웃 |
| `law_api.py` (수정) | `[LawAPI]` | INFO: DRF/AJAX 경로 선택 |

---

## 6. Security Considerations

- [x] 입력 검증: query 길이 제한 (200자), 특수문자 이스케이프
- [x] OC 파라미터 노출 방지: 로그에 OC 값 마스킹
- [x] HTTPS 미지원: law.go.kr DRF는 HTTP만 지원. 내부 서버 통신이므로 수용 가능
- [x] Rate Limiting: 기존 Flask rate limiter 적용 (API 엔드포인트)
- [x] 응답 검증: DRF 응답 크기 제한 (1MB 이상 거부)

---

## 7. Test Plan

### 7.1 테스트 범위

| Type | Target | Method |
|------|--------|--------|
| Unit | `LawDrfClient._request()` 응답 파싱 | mock responses |
| Unit | `LegalSourceRouter.route()` 키워드 매칭 | 키워드별 예상 소스 검증 |
| Integration | `LawDrfClient.search()` 실제 API 호출 | `LAW_OC` 설정된 환경에서만 실행 |
| Integration | `search_labor_laws()` DRF/AJAX 전환 | 환경변수 토글 |
| E2E | RAG 파이프라인 전체 | 대표 질문 10개 → 답변 품질 비교 |

### 7.2 핵심 테스트 케이스

- [ ] Happy path: "부당해고 구제 방법" → 법령 + 판례 + 행정해석 반환
- [ ] Happy path: "2026년 최저임금" → 법령 + 고시(행정규칙) 반환
- [ ] 폴백: `LAW_OC` 미설정 → 기존 AJAX 동작 확인
- [ ] 폴백: DRF 서버 타임아웃 → 기존 AJAX 동작 확인
- [ ] 서킷브레이커: 5회 연속 실패 → DRF 비활성화 → 5분 후 재시도
- [ ] 병렬 조회: 4개 소스 중 1개 실패 → 나머지 3개 정상 반환
- [ ] 토큰 제한: 4소스 합산 3000자 초과 시 우선순위 truncation
- [ ] Edge: 빈 쿼리 → 빈 결과 반환 (에러 없음)
- [ ] Edge: 매칭 키워드 없는 질문 → 법령(law)만 검색

---

## 8. Implementation Guide

### 8.1 File Structure

```
services/
├── law_drf_client.py       [NEW] LawDrfClient + 데이터 클래스
├── legal_source_router.py  [NEW] LegalSourceRouter + 포맷터
├── law_api.py              [MODIFY] DRF/AJAX 자동 전환 래퍼
├── singletons.py           [MODIFY] get_law_drf_client() 추가
└── rag_pipeline.py         [MODIFY] 다중 소스 포맷 주입
.env.example                [MODIFY] LAW_OC 추가
```

### 8.2 Implementation Order

#### Step 1: 데이터 클래스 + DRF 클라이언트 (FR-01, FR-10)
1. [ ] `services/law_drf_client.py` 생성
   - 데이터 클래스: `SourceRequest`, `LawArticle`, `Precedent`, `Interpretation`, `AdminRule`, `LegalContext`
   - `LawDrfClient` 클래스: `search()`, `get_detail()`, 서킷브레이커, 캐싱
2. [ ] `services/singletons.py` — `get_law_drf_client()` 추가
3. [ ] `.env.example` — `LAW_OC` 환경변수 문서화

#### Step 2: 법령 DRF 전환 (FR-02)
4. [ ] `LawDrfClient.search_laws()`, `get_law_articles()` 구현
5. [ ] `law_api.py` — `_get_drf_client_if_available()` + DRF 우선 경로 추가

#### Step 3: 판례/행정해석/행정규칙 (FR-03, FR-04, FR-05)
6. [ ] `LawDrfClient.search_precedents()`, `get_precedent_detail()` 구현
7. [ ] `LawDrfClient.search_interpretations()`, `get_interpretation_detail()` 구현
8. [ ] `LawDrfClient.search_admin_rules()`, `get_admin_rule_detail()` 구현

#### Step 4: 소스 라우터 + 통합 (FR-06, FR-07, FR-08)
9. [ ] `services/legal_source_router.py` 생성
   - `LegalSourceRouter`: `route()`, `search_all()`, `format_context()`
   - `_KEYWORD_TO_SOURCES` 매핑
10. [ ] `law_api.py` — `search_labor_laws()` DRF 통합 경로 완성
11. [ ] `rag_pipeline.py` — 다중 소스 포맷 분기 + 프롬프트 섹션 제목 변경

#### Step 5: 검증 + 정리 (FR-09)
12. [ ] 10개 대표 질문 E2E 테스트
13. [ ] 프로덕션 서버 IP `open.law.go.kr` 등록

### 8.3 Coding Conventions

| Item | Convention |
|------|-----------|
| 파일 명명 | snake_case (`law_drf_client.py`) |
| 클래스 명명 | PascalCase (`LawDrfClient`) |
| 상수 | UPPER_SNAKE_CASE (`BASE_SEARCH`, `MAX_FAILURES`) |
| 로깅 프리픽스 | `[LawDRF]`, `[LegalRouter]` |
| 싱글톤 | `services/singletons.py` double-checked locking |
| 캐싱 | `_cache` dict + `_CACHE_TTL` (기존 패턴) |
| 에러 핸들링 | try/except → 로그 + 빈 결과 반환 (절대 raise 안 함) |
| 임포트 | 지연 임포트 (함수 내 `from ... import`) — 기존 패턴 유지 |

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-03-11 | Initial draft | Claude Code |
