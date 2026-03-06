"""Query type classification and routing configuration.

Classifies queries into types (factual, procedural, comparison, calculation)
and provides per-type pipeline parameter overrides.
"""

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
