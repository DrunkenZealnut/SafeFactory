# 직업계고 전공별 재구성 — Design Document

> **Summary**: SafeFactory를 전공(Major) 중심 플랫폼으로 전환하기 위한 상세 설계
>
> **Project**: SafeFactory
> **Author**: zealnutkim
> **Date**: 2026-03-14
> **Status**: Draft
> **Plan Reference**: `docs/01-plan/features/vocational-major-restructuring.plan.md`

---

## 1. Design Overview

### 1.1 Design Goals

1. **전공 레이어 추가**: 기존 도메인(namespace) 체계 위에 전공(Major) 레이어를 비파괴적으로 추가
2. **마이페이지 전공 선택**: 사용자가 마이페이지에서 전공을 선택/변경하면 전체 플랫폼이 해당 전공에 맞게 동작
3. **하위 호환**: 기존 Pinecone 벡터 데이터, API 엔드포인트, 라우트 구조 유지
4. **플러그인 확장**: 새 전공 추가 시 `services/major_config.py`에 딕셔너리 엔트리만 추가

### 1.2 Design Principles

- **Additive, Not Destructive**: 기존 domain_config.py는 유지하고 major_config.py를 추가
- **Convention over Configuration**: 전공 설정의 표준 스키마로 일관성 보장
- **Graceful Fallback**: 전공 미선택 사용자는 기본 전공(semiconductor)으로 동작

### 1.3 Plan 대비 주요 변경사항

| Plan 항목 | Plan 원안 | Design 변경 | 이유 |
|-----------|-----------|-------------|------|
| FR-02 전공 선택 UI | 홈 화면 카드형 선택 | **마이페이지에서 전공 선택** | 사용자 요구. 전공은 자주 바뀌지 않으므로 프로필 설정이 적합 |
| 홈 화면 | 전공 선택 카드 | **사용자 전공 기반 통합 학습환경** 바로 진입 | 전공이 프로필에 저장되므로 홈에서 바로 학습 시작 |
| 네비게이션 | 전공별 탭 전환 | **현재 전공 표시 + 마이페이지 변경 링크** | 빈번한 전공 전환은 비현실적, 심플한 UI 유지 |

---

## 2. Architecture

### 2.1 전체 아키텍처 변경 개요

```
[AS-IS: 토픽 기반 다중 라우트]
──────────────────────────────────────────────────────
web_app.py  ──┬── /semiconductor  ──→ domain.html (semiconductor config)
              ├── /field-training ──→ domain.html (field-training config)
              ├── /safeguide      ──→ domain.html (safeguide config)
              ├── /msds           ──→ domain.html (msds config)
              └── /search         ──→ domain.html (all config)

각 라우트가 독립적 도메인 설정을 domain.html에 전달

[TO-BE: 전공 중심 단일 진입점]
──────────────────────────────────────────────────────
web_app.py  ──┬── /                ──→ home.html (전공 기반 대시보드)
              ├── /learn           ──→ domain.html (사용자 전공 자동 적용)
              ├── /msds            ──→ msds.html (유지 — 전공 무관 공통도구)
              ├── /mypage          ──→ mypage.html (전공 선택 UI 추가)
              └── /questions       ──→ questions.html (유지)

핵심 변경:
- 기존 5개 도메인 라우트 → /learn 단일 라우트로 통합
- 사용자 전공(User.major)에 따라 동적으로 설정 적용
- MSDS는 전공 공통 도구로 독립 유지
```

### 2.2 컴포넌트 다이어그램

```
┌──────────────────────────────────────────────────────────┐
│                     Frontend Layer                       │
│                                                          │
│  home.html ─── domain.html(/learn) ─── mypage.html      │
│      │              │                      │             │
│      │              │ (major 설정 적용)     │ (전공선택)  │
│      └──────────────┼──────────────────────┘             │
│                     ↓                                    │
├──────────────────────────────────────────────────────────┤
│                     API Layer                            │
│                                                          │
│  api/v1/search.py ── api/v1/auth.py ── api/v1/admin.py  │
│      │ (major 파라미터 수신)                              │
│      ↓                                                   │
├──────────────────────────────────────────────────────────┤
│                  Service Layer (변경 핵심)                │
│                                                          │
│  ┌─────────────────┐  ┌──────────────────┐               │
│  │ major_config.py  │  │ domain_config.py │               │
│  │ (신규: 전공 정의)│←→│ (기존: 유지)     │               │
│  └────────┬────────┘  └──────────────────┘               │
│           ↓                                              │
│  ┌─────────────────┐  ┌──────────────────┐               │
│  │ query_router.py  │  │ rag_pipeline.py  │               │
│  │ (전공→NS 라우팅) │→│ (다중NS 검색)    │               │
│  └─────────────────┘  └──────────────────┘               │
│                                                          │
├──────────────────────────────────────────────────────────┤
│                  Data Layer                               │
│                                                          │
│  User.major (DB) ─── Pinecone (namespace 유지)           │
│  Session (비로그인 fallback)                              │
└──────────────────────────────────────────────────────────┘
```

### 2.3 데이터 흐름

```
사용자 요청 흐름:

1. 마이페이지에서 전공 선택 → User.major = "semiconductor" (DB 저장)
2. /learn 접속 → web_app.py가 User.major 조회 → major_config에서 설정 로드
3. 검색/질문 → API에 major 파라미터 전달
4. rag_pipeline:
   a. major_config에서 관련 네임스페이스 목록 조회
      예: semiconductor → ["semiconductor-v2", "kosha", "field-training"]
   b. primary NS(semiconductor-v2)에서 학습자료 검색
   c. safety NS(kosha)에서 전공 관련 안전자료 검색 (safety_keywords 필터)
   d. 공통 NS(laborlaw, msds)는 쿼리 관련성 있을 때 자동 크로스서치
   e. 결과 RRF 병합 → 전공 특화 프롬프트로 답변 생성
```

---

## 3. Data Model

### 3.1 User 모델 변경

**파일**: `models.py` — User 클래스 (line ~84-117)

```python
# 추가할 컬럼
class User(db.Model):
    # ... 기존 필드 유지 ...
    major = db.Column(db.String(50), nullable=True, default='semiconductor')
    # 'semiconductor', 'electrical', 'mechanical', 'chemical' 등
```

**마이그레이션**: Flask-Migrate 없이 직접 ALTER TABLE (기존 패턴 따름)

```python
# web_app.py 또는 별도 마이그레이션 스크립트
with app.app_context():
    db.engine.execute(
        "ALTER TABLE user ADD COLUMN major VARCHAR(50) DEFAULT 'semiconductor'"
    )
```

### 3.2 전공 설정 데이터 구조

**파일**: `services/major_config.py` (신규 생성)

```python
"""
전공(Major) 중심 설정 모듈.
기존 domain_config.py 위에 전공 레이어를 추가.
"""

# ──────────────────────────────────────────────
# 전공 정의 (MAJOR_CONFIG)
# ──────────────────────────────────────────────
# 새 전공 추가 시 이 딕셔너리에 엔트리만 추가하면 됨

MAJOR_CONFIG = {
    "semiconductor": {
        "name": "반도체과",
        "short_name": "반도체",
        "icon": "cpu",             # Lucide icon name
        "color": "#6366f1",        # 인디고
        "gradient": "from-indigo-500 to-purple-600",
        "description": "반도체 제조공정, 장비운용, 품질관리 학습 및 안전",

        # 이 전공이 검색할 네임스페이스들
        "namespaces": {
            "primary": "semiconductor-v2",  # 전공 핵심 학습자료 (NCS)
            "safety": "kosha",              # 전공 관련 안전자료
            "training": "field-training",   # 현장실습 가이드
        },

        # 안전 네임스페이스(kosha) 검색 시 전공 맥락 필터
        "safety_keywords": [
            "반도체", "클린룸", "화학물질", "정전기", "CVD",
            "에칭", "가스", "진공", "웨이퍼", "포토레지스트"
        ],

        # LLM 프롬프트 템플릿 변수
        "prompt_vars": {
            "specialty": "반도체 제조공정",
            "role": "반도체 공정 전문가이자 직업계고 교육 조교",
            "core_topics": [
                "CVD/PVD 박막 증착", "리소그래피/노광",
                "에칭(건식/습식)", "이온주입", "CMP",
                "패키징/본딩", "품질관리/수율"
            ],
            "safety_focus": [
                "화학물질(가스/약품) 취급 안전",
                "클린룸 환경 안전수칙",
                "정전기 방지(ESD)",
                "진공장비 안전"
            ],
        },

        # 전공 페이지 샘플 질문
        "sample_questions": [
            "CVD 공정의 원리와 종류를 설명해주세요",
            "클린룸에서 지켜야 할 안전수칙은 무엇인가요?",
            "반도체 현장실습 시 주의사항을 알려주세요",
            "PECVD와 LPCVD의 차이점은?",
            "웨이퍼 세정 공정에서 사용하는 화학물질의 위험성은?",
        ],

        # 자동 라우팅 키워드 (query_router 연동)
        "routing_keywords": {
            "high": [
                "반도체", "웨이퍼", "CVD", "PVD", "에칭", "리소그래피",
                "PECVD", "LPCVD", "CMP", "포토레지스트", "이온주입",
                "FAB", "클린룸", "패키징", "다이본딩", "와이어본딩"
            ],
            "low": [
                "공정", "제조", "팹", "증착", "노광", "현상",
                "박막", "수율", "디펙트", "파티클"
            ],
        },
    },

    # ──────────────────────────────────────────
    # 향후 전공 추가 템플릿 (주석)
    # ──────────────────────────────────────────
    # "electrical": {
    #     "name": "전기전자과",
    #     "short_name": "전기전자",
    #     "icon": "zap",
    #     "color": "#f59e0b",
    #     "gradient": "from-amber-500 to-orange-600",
    #     "description": "전기회로, 전자부품, 제어시스템 학습 및 안전",
    #     "namespaces": {
    #         "primary": "electrical-v1",
    #         "safety": "kosha",
    #         "training": "field-training",
    #     },
    #     "safety_keywords": ["전기", "감전", "누전", "접지", "고압", "배전"],
    #     "prompt_vars": { ... },
    #     "sample_questions": [ ... ],
    #     "routing_keywords": { "high": [...], "low": [...] },
    # },
}


# ──────────────────────────────────────────────
# 전공 공통 자료 (모든 전공에서 접근 가능)
# ──────────────────────────────────────────────
COMMON_RESOURCES = {
    "laborlaw": {
        "name": "노동법/근로기준",
        "namespace": "laborlaw",
        "description": "근로기준법, 4대보험, 최저임금, 현장실습 근로조건",
        "auto_crosssearch": True,  # 관련 쿼리 시 자동 크로스서치
    },
    "msds": {
        "name": "화학물질 안전(MSDS)",
        "namespace": "msds",
        "description": "MSDS, GHS 분류, 화학물질 취급 안전",
        "auto_crosssearch": False,  # 별도 도구로 제공 (/msds 페이지)
    },
}


# ──────────────────────────────────────────────
# 기본값
# ──────────────────────────────────────────────
DEFAULT_MAJOR = "semiconductor"


# ──────────────────────────────────────────────
# 헬퍼 함수
# ──────────────────────────────────────────────
def get_major_config(major_key: str) -> dict:
    """전공 키로 설정 조회. 없으면 기본 전공 반환."""
    return MAJOR_CONFIG.get(major_key, MAJOR_CONFIG[DEFAULT_MAJOR])


def get_major_namespaces(major_key: str) -> list[str]:
    """전공의 모든 검색 대상 네임스페이스 목록 반환."""
    config = get_major_config(major_key)
    return list(config["namespaces"].values())


def get_primary_namespace(major_key: str) -> str:
    """전공의 핵심(primary) 네임스페이스 반환."""
    config = get_major_config(major_key)
    return config["namespaces"]["primary"]


def get_all_major_keys() -> list[str]:
    """등록된 모든 전공 키 목록."""
    return list(MAJOR_CONFIG.keys())


def get_major_for_namespace(namespace: str) -> str | None:
    """네임스페이스로 전공 역조회. 매칭되는 전공이 없으면 None."""
    for major_key, config in MAJOR_CONFIG.items():
        if namespace in config["namespaces"].values():
            return major_key
    return None
```

### 3.3 기존 domain_config.py와의 관계

```
major_config.py (신규)          domain_config.py (기존 유지)
─────────────────────          ──────────────────────────
MAJOR_CONFIG                   DOMAIN_CONFIG (UI 색상/아이콘 등)
  └── namespaces               DOMAIN_PROMPTS (LLM 프롬프트)
       └── "semiconductor-v2"  NAMESPACE_DOMAIN_MAP (역매핑)
            ↕ 참조              DOMAIN_COT_INSTRUCTIONS
                               DIRECTORY_NAMESPACE_MAP

변경 전략:
1. major_config.py가 "어떤 전공이 어떤 NS를 사용하는가" 정의 (상위 레이어)
2. domain_config.py는 "각 NS의 프롬프트/설정" 유지 (하위 레이어)
3. DOMAIN_CONFIG의 UI 설정(색상, 아이콘 등)은 major_config.py로 이관
   → domain_config.py의 DOMAIN_CONFIG는 deprecated하되 삭제하지 않음
```

---

## 4. API Specification

### 4.1 변경 엔드포인트

#### POST `/api/v1/ask` (기존 변경)

**변경 사항**: `namespace` 대신 `major` 파라미터 우선 처리

```python
# 현재 (AS-IS)
{
    "query": "CVD 공정 원리",
    "namespace": "semiconductor-v2",   # 단일 네임스페이스
    "category": "",
    "subcategory": ""
}

# 변경 후 (TO-BE)
{
    "query": "CVD 공정 원리",
    "major": "semiconductor",          # 전공 키 (신규, 우선)
    "namespace": "semiconductor-v2",   # 하위호환용 (major 없을 때 fallback)
}
```

**처리 로직**:
```python
def resolve_search_context(data, user):
    """요청에서 검색 컨텍스트(전공/네임스페이스) 결정"""
    # 1순위: 요청에 major 명시
    major = data.get('major')
    if major and major in MAJOR_CONFIG:
        return major, get_primary_namespace(major)

    # 2순위: 요청에 namespace 명시 (하위호환)
    namespace = data.get('namespace', '')
    if namespace:
        major = get_major_for_namespace(namespace)
        return major or DEFAULT_MAJOR, namespace

    # 3순위: 로그인 사용자의 저장된 전공
    if user and user.major:
        return user.major, get_primary_namespace(user.major)

    # 4순위: 기본값
    return DEFAULT_MAJOR, get_primary_namespace(DEFAULT_MAJOR)
```

#### POST `/api/v1/user/major` (신규)

**목적**: 마이페이지에서 사용자 전공 선택/변경

```
POST /api/v1/user/major
Content-Type: application/json

Request:
{
    "major": "semiconductor"
}

Response (200):
{
    "status": "ok",
    "data": {
        "major": "semiconductor",
        "major_name": "반도체과"
    }
}

Response (400):
{
    "status": "error",
    "message": "Invalid major key: 'invalid'"
}
```

#### GET `/api/v1/user/major` (신규)

**목적**: 현재 사용자 전공 조회

```
GET /api/v1/user/major

Response (200):
{
    "status": "ok",
    "data": {
        "major": "semiconductor",
        "major_name": "반도체과",
        "major_config": {
            "icon": "cpu",
            "color": "#6366f1",
            "description": "반도체 제조공정, 장비운용, 품질관리 학습 및 안전"
        }
    }
}
```

#### GET `/api/v1/majors` (신규)

**목적**: 사용 가능한 전공 목록 (마이페이지 전공 선택 UI용)

```
GET /api/v1/majors

Response (200):
{
    "status": "ok",
    "data": {
        "majors": [
            {
                "key": "semiconductor",
                "name": "반도체과",
                "short_name": "반도체",
                "icon": "cpu",
                "color": "#6366f1",
                "description": "반도체 제조공정, 장비운용, 품질관리 학습 및 안전"
            }
        ],
        "default": "semiconductor"
    }
}
```

### 4.2 기존 API 하위호환

| 엔드포인트 | 현재 파라미터 | 변경 후 | 하위호환 |
|-----------|-------------|---------|---------|
| POST /ask | namespace | major (우선) + namespace (fallback) | O |
| POST /ask/stream | namespace | major (우선) + namespace (fallback) | O |
| POST /search | namespace | major (우선) + namespace (fallback) | O |
| GET /questions | namespace | major (우선) + namespace (fallback) | O |

---

## 5. UI/UX Design

### 5.1 사용자 흐름 (User Flow)

```
[최초 방문 / 비로그인]
──────────────────────────────────────────
홈(/) → 기본 전공(semiconductor) 학습환경 자동 적용
      → 로그인 유도 배너: "전공을 설정하면 맞춤 학습이 가능합니다"
      → /learn 페이지에서 기본 전공으로 검색/질문

[로그인 후 전공 미설정]
──────────────────────────────────────────
홈(/) → 전공 선택 모달/배너 표시: "전공을 선택해주세요"
      → 마이페이지(/mypage)로 이동하여 전공 선택
      → 선택 완료 → /learn으로 리다이렉트

[전공 설정 완료 사용자 (일반 흐름)]
──────────────────────────────────────────
홈(/) → 내 전공 대시보드 (반도체과)
      → "학습하기" 클릭 → /learn (전공 자동 적용)
      → 검색/질문 → 전공 관련 NS 통합 검색 → AI 답변

[전공 변경]
──────────────────────────────────────────
마이페이지(/mypage) → 전공 설정 섹션 → 전공 카드 선택
→ POST /api/v1/user/major → 저장 → 페이지 새로고침
```

### 5.2 화면 설계

#### 5.2.1 마이페이지 — 전공 선택 섹션

```
┌──────────────────────────────────────────────────┐
│ 마이페이지                                        │
├──────────────────────────────────────────────────┤
│                                                  │
│ 👤 프로필 정보                                    │
│ ├─ 이름: 김OO                                    │
│ ├─ 이메일: xxx@gmail.com                         │
│ └─ 로그인: Google                                │
│                                                  │
│ ─────────────────────────────────────────────── │
│                                                  │
│ 🎓 내 전공 선택                                   │
│                                                  │
│ ┌─────────────┐  ┌─────────────┐                │
│ │  💻 반도체과  │  │  ⚡ 전기전자과│                │
│ │  [선택됨 ✓]  │  │  [준비 중]   │                │
│ │ #6366f1     │  │  비활성      │                │
│ └─────────────┘  └─────────────┘                │
│ ┌─────────────┐  ┌─────────────┐                │
│ │  ⚙️ 기계과   │  │  🧪 화공과   │                │
│ │  [준비 중]   │  │  [준비 중]   │                │
│ │  비활성      │  │  비활성      │                │
│ └─────────────┘  └─────────────┘                │
│                                                  │
│ ─────────────────────────────────────────────── │
│                                                  │
│ 📚 내 학습 기록 / 북마크 / 검색 히스토리           │
│ └─ (기존 기능 유지)                               │
│                                                  │
└──────────────────────────────────────────────────┘
```

#### 5.2.2 홈 화면 — 전공 대시보드

```
┌──────────────────────────────────────────────────┐
│ SafeFactory              [반도체과] 마이페이지 ▾  │
├──────────────────────────────────────────────────┤
│                                                  │
│ 🎓 반도체과 학습 및 안전자료                       │
│                                                  │
│ ┌──────────────────────────────────────────────┐ │
│ │ 💡 오늘의 추천 질문                           │ │
│ │ • CVD 공정의 원리와 종류를 설명해주세요         │ │
│ │ • 클린룸에서 지켜야 할 안전수칙은?             │ │
│ │ • PECVD와 LPCVD의 차이점은?                  │ │
│ └──────────────────────────────────────────────┘ │
│                                                  │
│ [학습하기 →]  [공유 질문 →]  [MSDS 검색 →]       │
│                                                  │
│ 📊 인기 키워드 (반도체과)                         │
│ ┌──────────────────────────────────────────────┐ │
│ │ CVD  PVD  에칭  클린룸  웨이퍼  안전  ...     │ │
│ └──────────────────────────────────────────────┘ │
│                                                  │
└──────────────────────────────────────────────────┘
```

#### 5.2.3 /learn 페이지 (기존 domain.html 전환)

```
기존 domain.html을 그대로 사용하되:
- 도메인 설정 → 사용자 전공의 major_config로 교체
- namespace → 전공의 primary namespace
- 색상/아이콘 → major_config의 color/icon
- 샘플 질문 → major_config의 sample_questions
- 자동 크로스서치 → 전공의 safety/training NS + 공통 NS
```

### 5.3 네비게이션 변경

```
[AS-IS]
──────────────────────────────────────────────
홈 | 반도체 | 현장실습 | 안전가이드 | MSDS | 통합검색 | 공유질문 | ...

[TO-BE]
──────────────────────────────────────────────
홈 | 학습하기 | MSDS | 공유질문 | 마이페이지
      ↓
  (전공 자동 적용)
```

**변경 포인트**:
- 기존 5개 도메인 탭 → "학습하기" 단일 탭으로 통합
- MSDS는 전공 무관 공통 도구로 독립 유지
- 공유질문, 마이페이지 유지
- 헤더에 현재 전공 뱃지 표시 (예: `[반도체과]`)

---

## 6. RAG Pipeline 변경 설계

### 6.1 전공 기반 다중 네임스페이스 검색

**파일**: `services/rag_pipeline.py` — `run_rag_pipeline()` 함수 변경

```python
# 현재 (AS-IS): 단일 namespace 검색
async def run_rag_pipeline(query, namespace, ...):
    # Phase 0: classify_domain → 단일 NS 결정
    # Phase 2: Pinecone search(namespace=namespace) → 단일 NS 검색

# 변경 후 (TO-BE): 전공 기반 다중 NS 검색
async def run_rag_pipeline(query, namespace, major=None, ...):
    # Phase 0: 전공 컨텍스트 결정
    if major:
        major_config = get_major_config(major)
        search_namespaces = get_major_namespaces(major)
        # primary NS를 기본으로, 나머지는 보조 검색
    else:
        # 하위호환: namespace만 있으면 기존 로직
        search_namespaces = [namespace] if namespace else ['']

    # Phase 2: 다중 NS 검색 (순차 실행 후 RRF 병합)
    all_results = []
    for ns in search_namespaces:
        results = pinecone_search(query, namespace=ns, top_k=adjusted_k)
        all_results.extend(results)

    # Phase 3: RRF 병합 (기존 hybrid searcher 활용)
    merged = rrf_fusion(all_results)
```

### 6.2 검색 가중치 전략

```python
# 전공 내 네임스페이스별 검색 가중치
NAMESPACE_WEIGHTS = {
    "primary": 1.0,    # 전공 핵심 학습자료 — 최고 가중치
    "safety": 0.7,     # 전공 관련 안전자료 — 보조
    "training": 0.6,   # 현장실습 — 보조
}

# 검색 시 top_k 조정
# primary: top_k * 1.0 (기본)
# safety: top_k * 0.5 (적은 수)
# training: top_k * 0.5 (적은 수)
```

### 6.3 프롬프트 템플릿 시스템

**기존**: `DOMAIN_PROMPTS`에 도메인별 전체 프롬프트 하드코딩 (50-230줄)

**변경**: 프롬프트 템플릿 + 전공 변수 주입

```python
# services/major_config.py 내 또는 별도 함수

MAJOR_PROMPT_TEMPLATE = """당신은 직업계고 {role}입니다.

## 전문 분야
{specialty} 분야의 전문 지식을 바탕으로 학생들의 학습을 돕습니다.

## 핵심 주제
{core_topics_formatted}

## 안전 중점 사항
{safety_focus_formatted}

## 답변 원칙
1. 직업계고 학생 수준에 맞게 쉽고 명확하게 설명합니다
2. 이론과 실무를 연결하여 설명합니다
3. 안전 관련 내용은 반드시 강조합니다
4. 제공된 참고자료를 근거로 답변합니다
5. 참고자료에 없는 내용은 일반 지식임을 명시합니다

## 답변 형식
- 핵심 요약으로 시작
- 단계별 또는 항목별로 구조화
- 안전 주의사항은 ⚠️로 강조
- 관련 참고자료 번호를 [1], [2] 형태로 인용
"""

def build_major_prompt(major_key: str) -> str:
    """전공 변수를 주입하여 시스템 프롬프트 생성."""
    config = get_major_config(major_key)
    pv = config["prompt_vars"]
    return MAJOR_PROMPT_TEMPLATE.format(
        role=pv["role"],
        specialty=pv["specialty"],
        core_topics_formatted="\n".join(f"- {t}" for t in pv["core_topics"]),
        safety_focus_formatted="\n".join(f"- {s}" for s in pv["safety_focus"]),
    )
```

### 6.4 기존 Safety Cross-Search 통합

**현재**: `rag_pipeline.py`의 `SafetyCrossSearch` — semiconductor 도메인일 때만 kosha NS에서 자동 보충 검색

**변경**: 전공별 safety NS로 일반화

```python
# 현재 (하드코딩)
if domain_key == 'semiconductor':
    safety_results = search_kosha(query)

# 변경 후 (전공 설정 기반)
if major:
    safety_ns = major_config["namespaces"].get("safety")
    if safety_ns:
        safety_results = search_namespace(
            query, namespace=safety_ns,
            filter_keywords=major_config["safety_keywords"]
        )
```

---

## 7. 라우트 변경 설계

### 7.1 web_app.py 라우트 변경

| 현재 라우트 | 변경 후 | 동작 |
|------------|---------|------|
| `GET /` | `GET /` (유지) | 전공 대시보드 or 전공 선택 유도 |
| `GET /semiconductor` | **삭제** → `/learn`으로 통합 | — |
| `GET /field-training` | **삭제** → `/learn`으로 통합 | — |
| `GET /safeguide` | **삭제** → `/learn`으로 통합 | — |
| `GET /search` (통합검색) | **삭제** → `/learn`으로 통합 | — |
| (없음) | `GET /learn` (신규) | 전공 기반 통합 학습환경 |
| `GET /msds` | `GET /msds` (유지) | 전공 무관 공통 도구 |
| `GET /questions` | `GET /questions` (유지) | 전공 무관 공유 질문 |
| (없음) | 마이페이지에 전공 선택 추가 | 기존 마이페이지 확장 |

### 7.2 하위호환 리다이렉트

```python
# 기존 URL 접근 시 리다이렉트
@app.route('/semiconductor')
@app.route('/field-training')
@app.route('/safeguide')
@app.route('/search')
def legacy_domain_redirect():
    return redirect(url_for('learn'))
```

### 7.3 /learn 라우트 구현

```python
@app.route('/learn')
def learn():
    """전공 기반 통합 학습환경"""
    # 1. 사용자 전공 결정
    if current_user.is_authenticated and current_user.major:
        major_key = current_user.major
    else:
        major_key = DEFAULT_MAJOR

    # 2. 전공 설정 로드
    config = get_major_config(major_key)

    # 3. 기존 domain.html 템플릿 재활용
    #    domain_config 형식으로 변환하여 전달
    domain_style_config = {
        'title': config['name'],
        'namespace': config['namespaces']['primary'],
        'major': major_key,
        'icon': config['icon'],
        'color': config['color'],
        'gradient': config['gradient'],
        'description': config['description'],
        'sample_questions': config['sample_questions'],
    }

    return render_template(
        'domain.html',
        domain='learn',
        config=domain_style_config,
        major=major_key,
    )
```

---

## 8. Error Handling

### 8.1 전공 관련 에러 코드

| 코드 | 상황 | 응답 |
|------|------|------|
| 400 | 유효하지 않은 major 키 | `{"status": "error", "message": "Invalid major key"}` |
| 401 | 비로그인 상태에서 전공 변경 시도 | `{"status": "error", "message": "Login required"}` |
| 200 | 전공에 해당하는 데이터 없음 | 정상 응답 + 빈 결과 + "해당 전공 자료가 아직 준비 중입니다" 메시지 |

### 8.2 Graceful Fallback

```python
# 전공 미설정 / 알 수 없는 전공 → 기본값으로 자동 전환
major = user.major or DEFAULT_MAJOR
if major not in MAJOR_CONFIG:
    major = DEFAULT_MAJOR
    # 로그 기록
    logger.warning(f"Unknown major '{user.major}' for user {user.id}, falling back to {DEFAULT_MAJOR}")
```

---

## 9. Security Considerations

### 9.1 전공 선택 API

- [x] 로그인 필수 (전공 변경은 인증된 사용자만)
- [x] major 키 화이트리스트 검증 (`major in MAJOR_CONFIG`)
- [x] CSRF 면제 (기존 v1_bp 패턴 따름 — API는 CSRF exempt)
- [x] Rate limiting 적용 (기존 `@rate_limit()` 데코레이터)

### 9.2 기존 보안 유지

- OAuth 인증 체계 변경 없음
- Fernet 토큰 암호화 유지
- XSS 방지 (DOMPurify) 유지

---

## 10. Test Plan

### 10.1 핵심 검증 항목

| 테스트 | 방법 | 기대 결과 |
|--------|------|-----------|
| 전공 설정 API | POST /api/v1/user/major {"major":"semiconductor"} | 200 OK, DB 저장 확인 |
| 유효하지 않은 전공 | POST /api/v1/user/major {"major":"invalid"} | 400 Error |
| /learn 전공 적용 | 전공 설정 후 /learn 접속 | 전공 설정(색상, 샘플질문 등) 반영 확인 |
| 전공 기반 검색 | /learn에서 "CVD 공정" 검색 | semiconductor-v2 + kosha + field-training 결과 통합 |
| 기존 URL 리다이렉트 | /semiconductor 접속 | /learn으로 302 리다이렉트 |
| 비로그인 사용자 | /learn 접속 | 기본 전공(semiconductor)으로 동작 |
| 하위호환 API | POST /ask {"namespace":"semiconductor-v2"} (major 없이) | 기존과 동일하게 동작 |
| 검색 품질 | "CVD PVD 차이" 검색 (전환 전후 비교) | 동일 품질 유지 |

### 10.2 회귀 테스트

- [ ] 커뮤니티 기능 정상 작동
- [ ] 공유 질문 페이지 정상 작동
- [ ] MSDS 검색 정상 작동
- [ ] 북마크 기능 정상 작동
- [ ] 스트리밍 응답 정상 작동
- [ ] 관리자 패널 정상 작동

---

## 11. Implementation Order

### Phase 1: 전공 설정 레이어 (Backend Core)

| 순서 | 파일 | 작업 | 의존성 |
|------|------|------|--------|
| 1-1 | `services/major_config.py` | 신규 생성 — MAJOR_CONFIG, 헬퍼 함수 | 없음 |
| 1-2 | `models.py` | User 모델에 `major` 컬럼 추가 | 없음 |
| 1-3 | `web_app.py` | DB 마이그레이션 (ALTER TABLE) | 1-2 |

### Phase 2: API 확장

| 순서 | 파일 | 작업 | 의존성 |
|------|------|------|--------|
| 2-1 | `api/v1/search.py` | major 파라미터 처리 + resolve_search_context() | 1-1 |
| 2-2 | `api/v1/auth.py` 또는 신규 `api/v1/user.py` | 전공 선택/조회 API | 1-1, 1-2 |
| 2-3 | `api/v1/__init__.py` | 신규 블루프린트 등록 | 2-2 |

### Phase 3: RAG 파이프라인 전공 지원

| 순서 | 파일 | 작업 | 의존성 |
|------|------|------|--------|
| 3-1 | `services/rag_pipeline.py` | major 파라미터 수신, 다중 NS 검색, 프롬프트 템플릿 | 1-1 |
| 3-2 | `services/query_router.py` | classify_domain()에 전공 키워드 통합 | 1-1 |

### Phase 4: Frontend 전환

| 순서 | 파일 | 작업 | 의존성 |
|------|------|------|--------|
| 4-1 | `templates/base.html` | 네비게이션 변경 (도메인탭 → 학습하기) | 없음 |
| 4-2 | `web_app.py` | /learn 라우트 추가, 레거시 리다이렉트 | 1-1 |
| 4-3 | `templates/home.html` | 전공 대시보드 UI | 1-1, 2-2 |
| 4-4 | 마이페이지 템플릿 | 전공 선택 섹션 추가 | 2-2 |
| 4-5 | `templates/domain.html` | major 파라미터 연동 (JavaScript 수정) | 2-1 |

### Phase 5: 정리 및 검증

| 순서 | 파일 | 작업 | 의존성 |
|------|------|------|--------|
| 5-1 | 전체 | 기존 도메인 라우트 제거 및 리다이렉트 확인 | 4-2 |
| 5-2 | 전체 | 회귀 테스트 (커뮤니티, MSDS, 질문 등) | 전체 |
| 5-3 | `CLAUDE.md` | 프로젝트 문서 업데이트 | 전체 |

---

## 12. File Change Summary

| 파일 | 변경 유형 | 변경 규모 |
|------|-----------|-----------|
| `services/major_config.py` | **신규 생성** | ~150줄 |
| `models.py` | 수정 (User.major 추가) | ~3줄 |
| `web_app.py` | 수정 (라우트 변경, 마이그레이션) | ~40줄 |
| `api/v1/search.py` | 수정 (major 파라미터 처리) | ~20줄 |
| `api/v1/user.py` | **신규 생성** (전공 API) | ~50줄 |
| `api/v1/__init__.py` | 수정 (블루프린트 등록) | ~3줄 |
| `services/rag_pipeline.py` | 수정 (다중 NS 검색, 프롬프트) | ~50줄 |
| `services/query_router.py` | 수정 (전공 키워드 통합) | ~20줄 |
| `templates/base.html` | 수정 (네비게이션) | ~30줄 |
| `templates/home.html` | 수정 (전공 대시보드) | ~50줄 |
| `templates/domain.html` | 수정 (major 파라미터 JS) | ~15줄 |
| 마이페이지 템플릿 | 수정 (전공 선택 UI) | ~40줄 |
| `CLAUDE.md` | 수정 (문서 업데이트) | ~20줄 |

**총 예상**: 신규 ~200줄 + 수정 ~250줄 = ~450줄

---

## Version History

| 버전 | 날짜 | 변경 사항 | 작성자 |
|------|------|-----------|--------|
| 0.1 | 2026-03-14 | 초안 — Plan 기반 + "전공선택은 마이페이지" 반영 | zealnutkim |
