"""Query type classification, domain routing, and pipeline configuration.

Classifies queries into types (factual, procedural, comparison, calculation)
and routes queries to the optimal domain namespace based on keyword analysis.
"""

import logging
import re

# Per-type pipeline parameter overrides.
# These are applied in run_rag_pipeline() to tune each Phase.
QUERY_TYPE_CONFIG = {
    'factual': {
        'top_k_mult': 3,
        'use_hyde': False,
        'use_multi_query': True,
        'rerank_weight': 0.80,
    },
    'procedural': {
        'top_k_mult': 4,
        'use_hyde': True,
        'use_multi_query': True,
        'rerank_weight': 0.70,
    },
    'comparison': {
        'top_k_mult': 5,
        'use_hyde': True,
        'use_multi_query': True,
        'rerank_weight': 0.65,
    },
    'calculation': {
        'top_k_mult': 2,
        'use_hyde': False,
        'use_multi_query': False,
        'rerank_weight': 0.85,
    },
}

# Rule-based classification patterns (ordered by specificity).
_CALCULATION_PATTERNS = [
    re.compile(r'\d+.*(?:만원|원|시간|일|개월|년)'),
    re.compile(r'(?:계산|산출|산정|몇|얼마)'),
    re.compile(r'(?:4대보험|보험료|세금|수당|퇴직금|연봉).*\d'),
]
_COMPARISON_PATTERNS = [
    re.compile(r'(?:차이|비교|다른|구분|vs|versus)', re.IGNORECASE),
    re.compile(r'\S+[와과]\s*\S+.*(?:차이|다른|비교)'),
]
_PROCEDURAL_PATTERNS = [
    re.compile(r'(?:방법|절차|순서|과정|단계|어떻게)'),
    re.compile(r'(?:하는\s*법|하려면|해야|신청|청구|신고)'),
]


def classify_query_type(query: str) -> str:
    """Classify a query into one of four types using regex patterns.

    Returns 'factual' when no other pattern matches.
    """
    for pattern in _CALCULATION_PATTERNS:
        if pattern.search(query):
            return 'calculation'
    for pattern in _COMPARISON_PATTERNS:
        if pattern.search(query):
            return 'comparison'
    for pattern in _PROCEDURAL_PATTERNS:
        if pattern.search(query):
            return 'procedural'
    return 'factual'


# ---------------------------------------------------------------------------
# Domain classification: keyword-based namespace routing
# ---------------------------------------------------------------------------

# {namespace: {'high': [domain-specific keywords], 'low': [shared keywords]}}
DOMAIN_KEYWORDS = {
    'semiconductor-v2': {
        'high': [
            'CVD', 'PVD', '에칭', '리소그래피', '웨이퍼', '패키징', '반도체',
            'NCS', '포토공정', 'CMP', '이온주입', '스퍼터링', '증착', '식각',
            '도핑', 'PECVD', 'LPCVD', '확산', '세정', '산화',
            '디퓨전', '박막', '전사', '노광', '현상', '포토레지스트',
            '다이싱', '와이어본딩', '플립칩', '언더필', '몰딩',
            '클린룸', 'FAB', 'EDS', 'PKG',
        ],
        'low': ['공정', '제조', '장비', '품질'],
    },
    'laborlaw': {
        'high': [
            '근로기준법', '최저임금', '퇴직금', '4대보험', '연차', '부당해고',
            '임금', '연봉', '실수령', '세후', '주휴수당', '해고', '노동법',
            '근로계약', '야간근로', '연장근로', '휴일근로', '통상임금',
            '주52시간', '퇴직연금', '고용보험', '국민연금', '건강보험',
            '산재보험', '근로감독', '체불', '해고예고', '정규직', '비정규직',
            '파견', '기간제', '수습', '인턴',
        ],
        'low': ['법률', '법적', '위반', '규정'],
    },
    'field-training': {
        'high': ['현장실습', '카드북', '실습생', '직업계고', '현장실습생'],
        'low': ['안전수칙', '재해', '보호구'],
    },
    'kosha': {
        'high': [
            '안전보건공단', 'KOSHA', '산업재해', '안전관리', '작업환경측정',
            '위험성평가', '안전보건교육', '관리감독자', '안전인증',
            '자율안전확인',
        ],
        'low': ['안전', '보건', '위험', '보호구', '유해물질'],
    },
    'msds': {
        'high': [
            'MSDS', 'CAS', '물질안전보건자료', 'GHS', 'H코드', 'P코드',
            '벤젠', '톨루엔', '아세톤', '에탄올', '메탄올', '포름알데히드',
            '불산', '염산', '황산', '질산', '암모니아', '시안화',
            '트리클로로에틸렌', '이소프로필알코올', 'IPA', 'NMP', 'PGMEA',
        ],
        'low': ['화학물질', '유해성', '독성'],
    },
}

_NAMESPACE_LABELS = {
    'semiconductor-v2': '반도체',
    'laborlaw': '노동법',
    'field-training': '현장실습',
    'kosha': '안전보건',
    'msds': 'MSDS',
}

_HIGH_KEYWORD_WEIGHT = 1.0
_LOW_KEYWORD_WEIGHT = 0.3
_DEFAULT_NS_BONUS = 0.5
_CONFIDENCE_THRESHOLD = 0.3


def classify_domain(query: str, default_namespace: str = '') -> tuple:
    """Classify query into the best-matching domain namespace.

    Uses keyword matching with weighted scoring. The current page's namespace
    receives a bonus to respect page context.

    Args:
        query: User question text.
        default_namespace: Namespace of the current page (e.g. 'semiconductor-v2').

    Returns:
        (namespace, confidence, domain_label) tuple.
    """
    query_lower = query.lower()

    scores = {}
    for ns, kw_dict in DOMAIN_KEYWORDS.items():
        score = 0.0
        for kw in kw_dict['high']:
            if kw.lower() in query_lower:
                score += _HIGH_KEYWORD_WEIGHT
        for kw in kw_dict['low']:
            if kw.lower() in query_lower:
                score += _LOW_KEYWORD_WEIGHT
        scores[ns] = score

    # Apply bonus to default namespace
    if default_namespace in scores:
        scores[default_namespace] += _DEFAULT_NS_BONUS

    # Find best namespace
    best_ns = max(scores, key=scores.get)
    best_score = scores[best_ns]

    # Normalize confidence to 0-1 range (cap at 5.0 for normalization)
    confidence = min(best_score / 5.0, 1.0)

    # If no strong signal, fall back to default
    if best_score < _CONFIDENCE_THRESHOLD or best_score <= _DEFAULT_NS_BONUS:
        final_ns = default_namespace or ''
        label = _NAMESPACE_LABELS.get(final_ns, '전체')
        logging.info("[DomainRouter] Low confidence (%.2f) → default: %s", best_score, final_ns)
        return final_ns, 0.0, label

    label = _NAMESPACE_LABELS.get(best_ns, '전체')
    logging.info("[DomainRouter] %s → %s (score=%.2f, confidence=%.2f)", query[:30], best_ns, best_score, confidence)
    return best_ns, confidence, label
