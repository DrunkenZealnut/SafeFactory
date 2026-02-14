"""
Shared NCS (National Competency Standards) utilities.

Provides category lists, section classification patterns, and metadata
extraction used by both the ingestion pipeline and patch scripts.
"""

import re
import unicodedata
from typing import Dict, Optional, Tuple


NCS_CATEGORIES = ['반도체개발', '반도체장비', '반도체재료', '반도체제조']

NCS_SECTION_PATTERNS = [
    (r'필요\s*지식', 'required_knowledge'),
    (r'수행\s*순서', 'performance_procedure'),
    (r'수행\s*내용', 'performance_content'),
    (r'수행\s*tip', 'performance_tip'),
    (r'학습\s*목표', 'learning_objective'),
    (r'학습모듈의\s*목표', 'learning_objective'),
    (r'평가\s*준거', 'evaluation_criteria'),
    (r'평가\s*방법', 'evaluation_method'),
    (r'평가자\s*체크리스트', 'evaluation_checklist'),
    (r'평가자\s*질문', 'evaluation_question'),
    (r'서술형\s*시험', 'evaluation_written'),
    (r'논술형\s*시험', 'evaluation_essay'),
    (r'구두\s*발표', 'evaluation_oral'),
    (r'피드백', 'feedback'),
    (r'교수\s*방법', 'teaching_method'),
    (r'학습\s*방법', 'learning_method'),
    (r'안전\s*유의\s*사항', 'safety_notes'),
    (r'안전\s*유의사항', 'safety_notes'),
    (r'핵심\s*용어', 'key_terms'),
    (r'선수\s*학습|선수학습', 'prerequisite'),
    (r'기기\s*\(?\s*장비', 'equipment'),
    (r'재료\s*자료', 'materials'),
    (r'학습모듈의\s*내용\s*체계', 'module_structure'),
    (r'NCS\s*학습모듈의\s*위치', 'module_position'),
    (r'NCS\s*학습모듈이란', 'module_intro'),
    (r'학습\s+(\d+)', 'learning_unit'),
]


def classify_section(title: str) -> Tuple[str, Optional[int]]:
    """Classify NCS section by type based on title patterns.

    Strips markdown bold markers and normalizes special characters
    before matching.

    Returns:
        (section_type, learning_unit_number or None)
    """
    if not title:
        return 'general', None

    # Strip markdown bold and extra whitespace
    clean = re.sub(r'\*+', '', title).strip()
    # Normalize middle-dot variants (・, ‧, ·) to space
    clean = re.sub(r'[・‧·]', ' ', clean)
    clean = unicodedata.normalize('NFC', clean)

    for pattern, stype in NCS_SECTION_PATTERNS:
        m = re.search(pattern, clean, re.IGNORECASE)
        if m:
            lu = int(m.group(1)) if stype == 'learning_unit' and m.lastindex else None
            return stype, lu
    return 'general', None


def extract_ncs_metadata(source_file: str) -> Dict:
    """Extract NCS metadata (code, category, title) from file path.

    Handles both versioned (LM1903060101_23v6) and unversioned
    (LM1903060107) NCS codes.
    """
    normalized = unicodedata.normalize('NFC', source_file)
    meta = {}

    code_match = re.search(r'(LM\d{10}(?:_\d+v\d+)?)', normalized)
    if code_match:
        meta['ncs_code'] = code_match.group(1)

    for cat in NCS_CATEGORIES:
        if cat in normalized:
            meta['ncs_category'] = cat
            break

    title_match = re.search(r'LM\d{10}(?:_\d+v\d+)?_(.+?)(?:/|$)', normalized)
    if title_match:
        meta['ncs_document_title'] = title_match.group(1).replace('_', ' ')

    return meta
