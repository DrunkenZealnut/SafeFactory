"""
전공(Major) 중심 설정 모듈.
기존 domain_config.py 위에 전공 레이어를 추가.
새 전공 추가 시 MAJOR_CONFIG에 엔트리만 추가하면 됨.
"""

# ──────────────────────────────────────────────
# 전공 정의 (MAJOR_CONFIG)
# ──────────────────────────────────────────────

MAJOR_CONFIG = {
    "semiconductor": {
        "name": "반도체과",
        "short_name": "반도체",
        "icon": "💻",
        "color": "#6366f1",
        "gradient": "from-indigo-500 to-purple-600",
        "description": "반도체 제조공정, 장비운용, 품질관리 학습 및 안전",

        "namespaces": {
            "primary": "semiconductor-v2",
            "safety": "kosha",
            "training": "field-training",
        },

        "safety_keywords": [
            "반도체", "클린룸", "화학물질", "정전기", "CVD",
            "에칭", "가스", "진공", "웨이퍼", "포토레지스트",
        ],

        "prompt_vars": {
            "specialty": "반도체 제조공정",
            "role": "반도체 공정 전문가이자 직업계고 교육 조교",
            "core_topics": [
                "CVD/PVD 박막 증착", "리소그래피/노광",
                "에칭(건식/습식)", "이온주입", "CMP",
                "패키징/본딩", "품질관리/수율",
            ],
            "safety_focus": [
                "화학물질(가스/약품) 취급 안전",
                "클린룸 환경 안전수칙",
                "정전기 방지(ESD)",
                "진공장비 안전",
            ],
        },

        "sample_questions": [
            "CVD 공정의 원리와 종류를 설명해주세요",
            "클린룸에서 지켜야 할 안전수칙은 무엇인가요?",
            "반도체 현장실습 시 주의사항을 알려주세요",
            "PECVD와 LPCVD의 차이점은?",
            "웨이퍼 세정 공정에서 사용하는 화학물질의 위험성은?",
        ],

        "routing_keywords": {
            "high": [
                "반도체", "웨이퍼", "CVD", "PVD", "에칭", "리소그래피",
                "PECVD", "LPCVD", "CMP", "포토레지스트", "이온주입",
                "FAB", "클린룸", "패키징", "다이본딩", "와이어본딩",
            ],
            "low": [
                "공정", "제조", "팹", "증착", "노광", "현상",
                "박막", "수율", "디펙트", "파티클",
            ],
        },
    },

    # 향후 전공 추가 예시 (주석)
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
    # [LABORLAW_DISABLED] "laborlaw": {
    #     "name": "노동법/근로기준",
    #     "namespace": "laborlaw",
    #     "description": "근로기준법, 4대보험, 최저임금, 현장실습 근로조건",
    #     "auto_crosssearch": True,
    # },
    "msds": {
        "name": "화학물질 안전(MSDS)",
        "namespace": "msds",
        "description": "MSDS, GHS 분류, 화학물질 취급 안전",
        "auto_crosssearch": False,
    },
}


# ──────────────────────────────────────────────
# 기본값
# ──────────────────────────────────────────────
DEFAULT_MAJOR = "semiconductor"

# 네임스페이스별 검색 가중치
NAMESPACE_WEIGHTS = {
    "primary": 1.0,
    "safety": 0.7,
    "training": 0.6,
}


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


def resolve_search_context(data, user=None):
    """요청에서 검색 컨텍스트(전공/네임스페이스) 결정.

    우선순위: request.major → request.namespace (하위호환) → user.major → DEFAULT_MAJOR
    Returns: (major_key, primary_namespace)
    """
    # 1순위: 요청에 major 명시
    major = data.get('major')
    if major and major in MAJOR_CONFIG:
        return major, get_primary_namespace(major)

    # 2순위: 요청에 namespace 명시 (하위호환)
    namespace = data.get('namespace', '')
    if namespace:
        major_from_ns = get_major_for_namespace(namespace)
        return major_from_ns or DEFAULT_MAJOR, namespace

    # 3순위: 로그인 사용자의 저장된 전공
    if user and hasattr(user, 'major') and user.major:
        return user.major, get_primary_namespace(user.major)

    # 4순위: 기본값
    return DEFAULT_MAJOR, get_primary_namespace(DEFAULT_MAJOR)


# ──────────────────────────────────────────────
# LLM 프롬프트 템플릿 시스템
# ──────────────────────────────────────────────
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
