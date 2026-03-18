"""Extract keywords from shared question texts for word cloud and graph visualization."""

import re
from collections import Counter
from itertools import combinations

# Korean stopwords: particles, conjunctions, common verbs/adjectives
STOPWORDS_KO = frozenset({
    '하는', '있는', '없는', '되는', '하고', '하면', '하여', '해서', '합니다',
    '입니다', '있습니다', '없습니다', '됩니다', '합니까', '있나요', '없나요',
    '인가요', '건가요', '한가요', '인지', '에서', '에게', '으로', '부터',
    '까지', '에는', '에도', '이나', '이고', '하지', '이란', '대한', '위한',
    '통한', '관한', '따른', '같은', '다른', '어떤', '어떻게', '무엇',
    '그리고', '그래서', '하지만', '그런데', '그러나', '또한', '그것',
    '이것', '저것', '여기', '거기', '무엇이', '어디', '언제', '왜',
    '어떤', '모든', '각각', '얼마', '얼마나', '정도', '때문', '경우',
    '사용', '필요', '가능', '여부', '방법', '문의', '질문', '궁금',
    '알고', '싶은', '싶습니다', '알려', '주세요', '알려주세요', '해주세요',
    '부탁', '드립니다', '감사', '답변', '내용', '관련', '사항', '해야',
    '하나요', '되나요', '할까요', '건지', '인데', '인데요', '거든요',
})

# English stopwords
STOPWORDS_EN = frozenset({
    'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
    'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been', 'some',
    'them', 'than', 'its', 'over', 'said', 'each', 'which', 'their',
    'will', 'other', 'about', 'many', 'then', 'what', 'when', 'where',
    'how', 'who', 'this', 'that', 'with', 'from', 'were', 'they', 'been',
    'would', 'there', 'into', 'more', 'also', 'very', 'just', 'does',
})

# Patterns: Korean 2+ chars, English 3+ chars, common abbreviations
_RE_KO = re.compile(r'[가-힣]{2,}')
_RE_EN = re.compile(r'[a-zA-Z]{3,}')


def extract_keywords(questions, limit=80):
    """Extract keyword frequencies from a list of (query_text, like_count) tuples.

    Args:
        questions: list of (query: str, like_count: int) tuples
        limit: max number of keywords to return

    Returns:
        list of {"text": str, "weight": int} sorted by weight desc
    """
    counter = Counter()

    for query, like_count in questions:
        weight = 1 + (like_count or 0)
        tokens = set()

        # Extract Korean tokens
        for match in _RE_KO.findall(query):
            if match not in STOPWORDS_KO and len(match) >= 2:
                tokens.add(match)

        # Extract English tokens (uppercase for consistency)
        for match in _RE_EN.findall(query):
            upper = match.upper()
            if match.lower() not in STOPWORDS_EN and len(match) >= 3:
                tokens.add(upper)

        for token in tokens:
            counter[token] += weight

    # Filter out low-frequency terms and return top N
    result = [
        {"text": text, "weight": weight}
        for text, weight in counter.most_common(limit)
        if weight >= 2
    ]

    return result


def _extract_tokens(query):
    """Extract keyword tokens from a single query string."""
    tokens = set()
    for match in _RE_KO.findall(query):
        if match not in STOPWORDS_KO and len(match) >= 2:
            tokens.add(match)
    for match in _RE_EN.findall(query):
        upper = match.upper()
        if match.lower() not in STOPWORDS_EN and len(match) >= 3:
            tokens.add(upper)
    return tokens


def extract_keyword_graph(questions, node_limit=80, min_edge_weight=2):
    """Extract keyword co-occurrence graph from shared questions.

    Args:
        questions: list of (query: str, like_count: int, namespace: str) tuples
        node_limit: max number of keyword nodes
        min_edge_weight: minimum co-occurrence count to include an edge

    Returns:
        dict with 'nodes' and 'edges' lists
    """
    counter = Counter()
    domain_map = {}
    per_question_keywords = []

    for query, like_count, namespace in questions:
        weight = 1 + (like_count or 0)
        tokens = _extract_tokens(query)

        for token in tokens:
            counter[token] += weight
            if token not in domain_map:
                domain_map[token] = set()
            if namespace:
                domain_map[token].add(namespace)

        if len(tokens) >= 2:
            per_question_keywords.append((tokens, weight))

    # Top N nodes by weight
    top_keywords = {kw for kw, _ in counter.most_common(node_limit) if counter[kw] >= 2}

    nodes = [
        {
            "id": kw,
            "text": kw,
            "weight": counter[kw],
            "domains": sorted(domain_map.get(kw, set())),
        }
        for kw in top_keywords
    ]
    nodes.sort(key=lambda n: n["weight"], reverse=True)

    # Build edges from co-occurrence
    edge_counter = Counter()
    for tokens, weight in per_question_keywords:
        relevant = tokens & top_keywords
        for a, b in combinations(sorted(relevant), 2):
            edge_counter[(a, b)] += weight

    edges = [
        {"source": a, "target": b, "weight": w}
        for (a, b), w in edge_counter.most_common()
        if w >= min_edge_weight
    ]

    return {"nodes": nodes, "edges": edges}
