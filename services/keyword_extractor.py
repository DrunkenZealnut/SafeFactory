"""Extract keywords from shared question texts for word cloud visualization."""

import re
from collections import Counter

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
