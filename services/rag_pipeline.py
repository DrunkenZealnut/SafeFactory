"""Shared RAG pipeline, LLM message builder, and image discovery."""

import concurrent.futures
import hashlib
import logging
import os
import re
import time
import unicodedata
from datetime import datetime

from services.domain_config import (
    DOCUMENTS_PATH,
    DOMAIN_PROMPTS,
    DEFAULT_SYSTEM_PROMPT,
    COT_INSTRUCTIONS,
    DOMAIN_COT_INSTRUCTIONS,
    VISUAL_GUIDELINES,
    NAMESPACE_DOMAIN_MAP,
)
from services.filters import build_domain_filter
from services.query_router import classify_query_type, classify_domain, QUERY_TYPE_CONFIG
from services.major_config import (
    resolve_search_context,
    get_major_config,
    build_major_prompt,
    DEFAULT_MAJOR,
    MAJOR_CONFIG,
)
from services.singletons import (
    get_agent,
    get_query_enhancer,
    get_context_optimizer,
    get_reranker_instance,
    get_hybrid_searcher_instance,
    get_uploader,
)
from src.query_enhancer import EnhancementResult

MIN_RELEVANCE_SCORE = float(os.environ.get('MIN_RELEVANCE_SCORE', '0.2'))
MIN_TOKEN_COUNT = int(os.environ.get('MIN_TOKEN_COUNT', '30'))
TOP_K_NO_BM25_MULT = int(os.environ.get('TOP_K_NO_BM25_MULT', '4'))
TOP_K_DEFAULT_MULT = int(os.environ.get('TOP_K_DEFAULT_MULT', '3'))


def post_process_answer(answer: str, source_count: int) -> str:
    """Post-process LLM answer to clean up citation issues and formatting.

    Args:
        answer: Raw LLM answer text.
        source_count: Number of sources provided to the LLM.

    Returns:
        Cleaned answer text.
    """
    import re as _re

    if not answer:
        return answer

    # 1. Remove invalid citation numbers (exceeding source count)
    # Matches [N] where N > source_count, but not inside law reference blocks
    def _replace_invalid_citation(match):
        num = int(match.group(1))
        if num > source_count and num <= source_count:
            return match.group(0)
        if num > source_count:
            return ''
        return match.group(0)

    if source_count > 0:
        answer = _re.sub(r'\[(\d+)\]', _replace_invalid_citation, answer)

    # 2. Normalize excessive consecutive newlines (3+ → 2)
    answer = _re.sub(r'\n{3,}', '\n\n', answer)

    # 3. Clean up empty citation artifacts (e.g., trailing spaces from removed citations)
    answer = _re.sub(r'\s+\n', '\n', answer)

    return answer.strip()


def compute_answer_confidence(answer: str, sources: list, context: str) -> dict:
    """Compute a confidence indicator for the generated answer.

    Factors considered:
    - Citation coverage: how many sources are actually cited in the answer
    - Source quality: average relevance score of sources
    - Answer length: very short answers may indicate insufficient context

    Args:
        answer: Generated answer text.
        sources: List of source dicts with 'score' fields.
        context: Context text provided to the LLM.

    Returns:
        dict with 'score' (0.0-1.0), 'level' ('high'|'medium'|'low'), 'factors'.
    """
    import re as _re

    if not answer or not sources:
        return {'score': 0.0, 'level': 'low', 'factors': {}}

    # Factor 1: Citation coverage (0-1)
    cited_nums = set(int(n) for n in _re.findall(r'\[(\d+)\]', answer))
    valid_cited = len([n for n in cited_nums if 1 <= n <= len(sources)])
    citation_coverage = valid_cited / len(sources) if sources else 0

    # Factor 2: Source quality — average score (0-1)
    scores = [s.get('score', 0) for s in sources if isinstance(s.get('score'), (int, float))]
    avg_source_score = sum(scores) / len(scores) if scores else 0

    # Factor 3: Answer substantiveness (0-1)
    answer_len = len(answer)
    if answer_len > 500:
        substantiveness = 1.0
    elif answer_len > 200:
        substantiveness = 0.7
    elif answer_len > 50:
        substantiveness = 0.4
    else:
        substantiveness = 0.2

    # Weighted combination
    confidence = (
        citation_coverage * 0.4 +
        avg_source_score * 0.35 +
        substantiveness * 0.25
    )
    confidence = min(1.0, max(0.0, confidence))

    if confidence >= 0.7:
        level = 'high'
    elif confidence >= 0.4:
        level = 'medium'
    else:
        level = 'low'

    return {
        'score': round(confidence, 3),
        'level': level,
        'factors': {
            'citation_coverage': round(citation_coverage, 3),
            'avg_source_score': round(avg_source_score, 3),
            'substantiveness': round(substantiveness, 3),
        }
    }



def _run_legal_analysis_pass(query, classification, law_refs_text, context):
    """First LLM pass: produce a structured legal analysis for complex labor law questions.

    Uses a lightweight model (gemini-2.0-flash) to analyze the legal aspects
    before the main generation pass. Only called for legal/hybrid questions.

    Returns:
        str: Structured analysis text, or empty string on failure.
    """
    from services.singletons import get_gemini_client
    from google.genai import types as genai_types

    q_type = (classification or {}).get('type', 'legal')
    calc_type = (classification or {}).get('calc_type', '')

    # Build a concise context snippet (first 1500 chars to stay within budget)
    context_snippet = ''
    if law_refs_text:
        context_snippet += f"[법령 참조]\n{law_refs_text[:800]}\n\n"
    if context:
        context_snippet += f"[검색된 문서 발췌]\n{context[:700]}"

    analysis_prompt = f"""당신은 한국 노동법 분석가입니다. 다음 질문을 분석하고 구조화된 분석을 작성하세요.

질문: {query}
분류: {q_type}, 계산유형: {calc_type or '없음'}

{context_snippet}

아래 5개 항목을 각각 1-2문장으로 간결하게 분석하세요:

1. **적용 법률**: 이 질문에 적용되는 핵심 법률과 조문
2. **쟁점 분석**: 사용자 상황에서의 핵심 쟁점(논점)
3. **예외/특례**: 고려해야 할 예외 조항이나 특례
4. **위반 가능성**: 법 위반 가능성 여부와 근거
5. **행동 지침**: 사용자에게 안내해야 할 실질적 행동 (구제방법, 신고, 기한)"""

    client = get_gemini_client()
    config = genai_types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=1000,
    )
    resp = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=analysis_prompt,
        config=config,
    )
    return resp.text or ''


# ---------------------------------------------------------------------------
# Cross-domain safety search (semiconductor → kosha)
# ---------------------------------------------------------------------------
SAFETY_CROSS_SEARCH_TOP_K = 5
SAFETY_CROSS_SEARCH_MIN_SCORE = 0.3
SAFETY_CROSS_SEARCH_MAX_RESULTS = 3
SAFETY_CROSS_SEARCH_MAX_CHARS = 1500
SAFETY_CROSS_SEARCH_NAMESPACE = 'kosha'


def _search_safety_context(query: str) -> str:
    """Cross-search kosha namespace for safety/health content related to semiconductor query.

    Args:
        query: Search query (after mention parsing).

    Returns:
        Formatted safety references string for prompt injection.
        Empty string if no relevant results found.
    """
    try:
        agent = get_agent()
        results = agent.search(
            query=query,
            top_k=SAFETY_CROSS_SEARCH_TOP_K,
            namespace=SAFETY_CROSS_SEARCH_NAMESPACE,
        )

        # Filter by minimum relevance score
        results = [r for r in results if r.get('score', 0) >= SAFETY_CROSS_SEARCH_MIN_SCORE]

        if not results:
            return ''

        # Take top N results
        results = results[:SAFETY_CROSS_SEARCH_MAX_RESULTS]

        # Format as safety references
        parts = []
        total_chars = 0
        for i, r in enumerate(results):
            metadata = r.get('metadata', {})
            content = metadata.get('content', '')
            source_file = metadata.get('source_file', 'Unknown')

            # Truncate content to stay within budget
            remaining = SAFETY_CROSS_SEARCH_MAX_CHARS - total_chars
            if remaining <= 0:
                break
            if len(content) > remaining:
                content = content[:remaining] + '...'

            parts.append(f"[S{i+1}] (출처: {source_file})\n{content}")
            total_chars += len(content) + len(source_file) + 20  # overhead

        return '\n\n'.join(parts)

    except Exception as e:
        logging.warning("[SafetyCross] Cross-search failed: %s", e)
        return ''


# ---------------------------------------------------------------------------
# MSDS Cross-Search (auto-link hazardous substances to MSDS data)
# ---------------------------------------------------------------------------
MSDS_CROSS_SEARCH_MAX_CHEMICALS = 3
MSDS_CROSS_SEARCH_NAMESPACES = {'', 'semiconductor-v2', 'field-training', 'kosha', 'all'}

# Trigger keywords indicating hazardous substance discussion
_CHEMICAL_TRIGGER_RE = re.compile(
    r'유해물질|화학물질|유기용제|유해가스|유해인자|독성|발암물질|'
    r'취급주의|노출기준|허용농도|보호구|화학적 인자|화학적 유해|'
    r'MSDS|GHS|CAS|SDS|물질안전보건'
)

# Common industrial chemicals (Korean names)
_KNOWN_CHEMICALS = [
    # Solvents
    '벤젠', '톨루엔', '자일렌', '스티렌', '에틸벤젠',
    '아세톤', '메탄올', '에탄올', '이소프로판올', '이소프로필알코올',
    '포름알데히드', '아세트알데히드', '글루타르알데히드',
    '트리클로로에틸렌', '테트라클로로에틸렌', '디클로로메탄', '클로로포름',
    '이황화탄소', '사염화탄소', '노말헥산',
    '에틸렌글리콜', '프로필렌글리콜',
    # Acids & bases
    '염산', '황산', '질산', '불산', '인산', '초산', '아세트산',
    '수산화나트륨', '수산화칼륨', '가성소다',
    # Gases
    '암모니아', '일산화탄소', '이산화질소', '이산화황', '황화수소',
    '아르신', '포스핀', '디보란', '실란', '포스겐', '시안화수소',
    '에틸렌옥사이드', '산화에틸렌',
    # Semiconductor-specific
    'NMP', 'HMDS', 'TMAH', 'BOE', 'PGMEA', 'PGME',
    # Oxidizers
    '과산화수소', '차아염소산나트륨',
    # Metals (2+ chars only to avoid false positives)
    '수은', '카드뮴', '크롬', '비소', '니켈', '망간', '코발트', '베릴륨',
    # Fibers & dusts
    '석면', '실리카',
    # Organics
    '페놀', '크레졸', '나프탈렌', '이소시아네이트',
    '아크릴로니트릴', '아크릴산',
    # Halogens
    '염소', '브롬', '불소',
    # Common names
    '시너', '신나',
]

# CAS number pattern
_CAS_NUMBER_RE = re.compile(r'\b(\d{2,7}-\d{2}-\d)\b')

# Semiconductor process → representative hazardous chemicals mapping
# When a process keyword is detected, its chemicals become MSDS lookup candidates
_PROCESS_CHEMICAL_MAP = {
    # CMP (Chemical Mechanical Planarization/Polishing)
    'CMP': ['실리카', '과산화수소', '수산화칼륨'],
    '슬러리': ['실리카', '과산화수소', '수산화칼륨'],
    'slurry': ['실리카', '과산화수소', '수산화칼륨'],
    '연마': ['실리카', '과산화수소'],
    # Etching
    '식각': ['불산', '인산', '질산'],
    '에칭': ['불산', '인산', '질산'],
    '습식식각': ['불산', '황산', '과산화수소'],
    '건식식각': ['염소', '불소', '아르신'],
    'wet etch': ['불산', '황산', '과산화수소'],
    'dry etch': ['염소', '불소'],
    # Cleaning
    '세정': ['황산', '과산화수소', '불산'],
    'cleaning': ['황산', '과산화수소', '불산'],
    'SC-1': ['암모니아', '과산화수소'],
    'SC-2': ['염산', '과산화수소'],
    'SPM': ['황산', '과산화수소'],
    'DHF': ['불산'],
    'BOE': ['불산'],
    'piranha': ['황산', '과산화수소'],
    # Photolithography
    '포토': ['PGMEA', 'PGME', 'TMAH'],
    '리소그래피': ['PGMEA', 'PGME', 'TMAH'],
    '현상': ['TMAH'],
    '감광액': ['PGMEA', 'PGME'],
    'PR': ['PGMEA', 'PGME'],
    'photoresist': ['PGMEA', 'PGME', 'TMAH'],
    # CVD / Deposition
    'CVD': ['실란', '포스핀', '디보란', '암모니아'],
    'PECVD': ['실란', '암모니아'],
    'LPCVD': ['실란', '디보란'],
    'ALD': ['TMAH', '암모니아'],
    # Diffusion / Ion Implantation
    '확산': ['포스핀', '디보란', '아르신'],
    '이온주입': ['아르신', '포스핀', '디보란'],
    # SOD (Spin-On Dielectric)
    'SOD': ['NMP', 'PGMEA'],
    'spin on': ['NMP', 'PGMEA'],
}


def _extract_chemical_names(query: str, context: str) -> list:
    """Extract chemical substance names from query and context.

    Returns up to MSDS_CROSS_SEARCH_MAX_CHEMICALS chemical names.
    Uses three sources: direct chemical mentions, CAS numbers, and
    process-to-chemical mapping (e.g. "CMP" → silica, H2O2, KOH).
    """
    text = f"{query} {context[:5000]}"
    text_lower = text.lower()

    # Check for chemical trigger keywords, known chemicals, CAS numbers,
    # or process keywords in query+context
    has_trigger = bool(_CHEMICAL_TRIGGER_RE.search(text))
    query_has_chemical = any(chem in query for chem in _KNOWN_CHEMICALS)
    query_has_cas = bool(_CAS_NUMBER_RE.search(query))
    has_process = any(
        proc.lower() in text_lower or proc in text
        for proc in _PROCESS_CHEMICAL_MAP
    )
    if not has_trigger and not query_has_chemical and not query_has_cas and not has_process:
        return []

    found = []
    seen = set()
    max_chems = MSDS_CROSS_SEARCH_MAX_CHEMICALS

    # Priority 1: chemicals explicitly in the query
    for chem in _KNOWN_CHEMICALS:
        if chem in query and chem not in seen:
            found.append(chem)
            seen.add(chem)

    # Priority 2: process-keyword mapping (query first, then context)
    for source in (query, text):
        if len(found) >= max_chems:
            break
        source_lower = source.lower()
        for proc, chems in _PROCESS_CHEMICAL_MAP.items():
            if len(found) >= max_chems:
                break
            if proc.lower() in source_lower or proc in source:
                for chem in chems:
                    if len(found) >= max_chems:
                        break
                    if chem not in seen:
                        found.append(chem)
                        seen.add(chem)

    # Priority 3: CAS numbers in query
    if len(found) < max_chems:
        for m in _CAS_NUMBER_RE.finditer(query):
            cas = m.group(1)
            if cas not in seen:
                found.append(cas)
                seen.add(cas)

    # Priority 4: chemicals in context
    if len(found) < max_chems:
        for chem in _KNOWN_CHEMICALS:
            if len(found) >= max_chems:
                break
            if chem in text and chem not in seen:
                found.append(chem)
                seen.add(chem)

    # Priority 5: CAS numbers in context
    if len(found) < max_chems:
        for m in _CAS_NUMBER_RE.finditer(text):
            if len(found) >= max_chems:
                break
            cas = m.group(1)
            if cas not in seen:
                found.append(cas)
                seen.add(cas)

    return found[:max_chems]


def _format_msds_items(items: list) -> str:
    """Format MSDS detail items into readable text."""
    lines = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get('msdsItemNameKor', '')
        detail = item.get('itemDetail', '')
        if not detail or not detail.strip():
            continue
        detail = detail.strip()[:300]
        if name:
            lines.append(f"  - {name}: {detail}")
        else:
            lines.append(f"  - {detail}")
    return '\n'.join(lines[:8])


def _fetch_single_msds_summary(client, chem_name: str):
    """Fetch MSDS summary for a single chemical.

    Returns (formatted_text, chem_info_dict) or (None, None).
    """
    try:
        is_cas = bool(_CAS_NUMBER_RE.match(chem_name))
        result = client.search_chemicals(
            search_word=chem_name,
            search_type=1 if is_cas else 0,
            num_of_rows=1,
        )
        if not result.get('success') or not result.get('items'):
            return None, None

        item = result['items'][0]
        chem_id = item.get('chemId', '')
        if not chem_id:
            return None, None

        name_kr = item.get('chemNameKor', chem_name)
        cas_no = item.get('casNo', '')

        header = f"### {name_kr}"
        if cas_no:
            header += f" (CAS: {cas_no})"

        sections = [header]
        section_labels = {
            '02': '유해성·위험성',
            '04': '응급조치요령',
            '08': '노출방지·개인보호구',
        }

        # Fetch key sections in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
            futs = {
                ex.submit(client.get_chemical_detail, chem_id, code): label
                for code, label in section_labels.items()
            }
            for fut in concurrent.futures.as_completed(futs, timeout=8):
                label = futs[fut]
                try:
                    detail = fut.result()
                    if detail.get('success') and detail.get('items'):
                        fmt = _format_msds_items(detail['items'])
                        if fmt:
                            sections.append(f"**{label}:**\n{fmt}")
                except Exception:
                    pass

        if len(sections) <= 1:
            return None, None

        chem_info = {'name': name_kr, 'cas_no': cas_no, 'chem_id': chem_id}
        return '\n'.join(sections), chem_info

    except Exception as e:
        logging.debug("[MSDSCross] Fetch failed for '%s': %s", chem_name, e)
        return None, None


def _search_msds_context(chemical_names: list) -> tuple:
    """Search MSDS API for detected chemicals and return formatted summary.

    Args:
        chemical_names: List of chemical names or CAS numbers to search.

    Returns:
        Tuple of (formatted_text, chemicals_info_list).
        formatted_text: MSDS summary string for LLM prompt injection.
        chemicals_info_list: List of dicts with name/cas_no/chem_id for frontend.
    """
    try:
        from msds_client import MsdsApiClient
        client = MsdsApiClient()

        if not client.API_KEY:
            logging.debug("[MSDSCross] Skipped — no MSDS_API_KEY")
            return '', []

        parts = []
        chemicals_info = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=MSDS_CROSS_SEARCH_MAX_CHEMICALS
        ) as ex:
            futs = {
                ex.submit(_fetch_single_msds_summary, client, name): name
                for name in chemical_names
            }
            for fut in concurrent.futures.as_completed(futs, timeout=15):
                try:
                    text, info = fut.result()
                    if text and info:
                        parts.append(text)
                        chemicals_info.append(info)
                except Exception:
                    pass

        return '\n\n'.join(parts), chemicals_info

    except Exception as e:
        logging.warning("[MSDSCross] MSDS cross-search failed: %s", e)
        return '', []


def find_related_images(source_file: str) -> list:
    """Find related images from the same document folder."""
    images = []
    try:
        # Normalize unicode for consistent path matching
        normalized_file = unicodedata.normalize('NFC', source_file)
        # Extract the folder path from source_file
        # e.g., "ncs/반도체개발/LM1903060102_23v5_반도체_아키텍처_설계/..." -> folder path
        parts = normalized_file.split('/')
        if len(parts) >= 3:
            folder_path = DOCUMENTS_PATH / parts[0] / parts[1] / parts[2]
            # Try NFD form if NFC path doesn't exist (macOS filesystem)
            if not folder_path.exists():
                nfd_file = unicodedata.normalize('NFD', source_file)
                nfd_parts = nfd_file.split('/')
                folder_path = DOCUMENTS_PATH / nfd_parts[0] / nfd_parts[1] / nfd_parts[2]
            if folder_path.exists():
                # Find all image files in the folder
                for ext in ['*.jpeg', '*.jpg', '*.png', '*.JPEG', '*.JPG', '*.PNG']:
                    for img_file in folder_path.glob(ext):
                        rel_path = img_file.relative_to(DOCUMENTS_PATH)
                        # Use POSIX path format for URL and normalize to NFC
                        url_path = unicodedata.normalize('NFC', rel_path.as_posix())
                        images.append({
                            'path': f'/documents/{url_path}',
                            'name': img_file.name
                        })
    except Exception as e:
        logging.warning("Error finding images: %s", e)
    return images[:10]  # Limit to 10 images


# ---------------------------------------------------------------------------
# Phase 1 & 2 helper functions (extracted from run_rag_pipeline)
# ---------------------------------------------------------------------------

def _content_hash(content: str) -> str:
    """Content hash for search result deduplication."""
    return hashlib.sha256(content[:5000].encode()).hexdigest()


def _enhance_query(
    search_query: str,
    namespace: str,
    route_cfg: dict,
    use_enhancement: bool,
) -> EnhancementResult:
    """Phase 1: Query enhancement.

    Delegates synonym expansion, multi-query, HyDE, and keyword extraction
    to QueryEnhancer.enhance_query().

    Args:
        search_query: Original search query.
        namespace: Pinecone namespace.
        route_cfg: Routing config from QUERY_TYPE_CONFIG.
        use_enhancement: Whether enhancement is enabled.

    Returns:
        EnhancementResult (original-only when enhancement is disabled).
    """
    if not use_enhancement:
        return EnhancementResult(
            original=search_query,
            variations=[search_query],
        )

    try:
        query_enhancer = get_query_enhancer()
        domain = NAMESPACE_DOMAIN_MAP.get(namespace, 'semiconductor')

        return query_enhancer.enhance_query(
            query=search_query,
            domain=domain,
            use_multi_query=route_cfg.get('use_multi_query', True),
            use_hyde=route_cfg.get('use_hyde', False),
            use_keywords=True,
        )
    except Exception as e:
        logging.warning("[Query Enhancement] Failed: %s, using original query", e)
        return EnhancementResult(
            original=search_query,
            variations=[search_query],
        )


def _search_single_query(
    agent,
    query: str,
    namespace: str,
    top_k: int,
    domain_filter: dict,
    is_all_namespace: bool,
) -> list:
    """Execute a single vector search with one retry on failure.

    Args:
        agent: PineconeAgent instance.
        query: Search query text.
        namespace: Pinecone namespace.
        top_k: Max results to return.
        domain_filter: Metadata filter.
        is_all_namespace: Whether to search all namespaces.

    Returns:
        Search results list (empty list on failure).
    """
    for attempt in range(2):
        try:
            if is_all_namespace:
                try:
                    uploader = get_uploader()
                    stats = uploader.get_stats()
                    ns_list = [ns for ns in stats.get('namespaces', {}).keys() if ns]
                except Exception:
                    ns_list = ['semiconductor', 'laborlaw', 'field-training']
                return agent.search_all_namespaces(
                    query=query,
                    namespaces=ns_list,
                    top_k=top_k,
                    filter=domain_filter,
                )
            else:
                return agent.search(
                    query=query,
                    top_k=top_k,
                    namespace=namespace,
                    filter=domain_filter,
                )
        except Exception as e:
            if attempt == 0:
                logging.warning(
                    "[Search] Attempt 1 failed for '%.30s...': %s — retrying",
                    query, e,
                )
                time.sleep(0.5)
            else:
                logging.warning(
                    "[Search] Attempt 2 failed for '%.30s...': %s",
                    query, e,
                )
    return []


def _search_with_variations(
    agent,
    enhancement: EnhancementResult,
    namespace: str,
    domain_filter: dict,
    search_top_k: int,
) -> list:
    """Phase 2: Multi-query search with dedup merging.

    Searches with regular queries (original + variations + synonym expansion)
    and HyDE queries separately, deduplicating by content hash.

    Args:
        agent: PineconeAgent instance.
        enhancement: Phase 1 EnhancementResult.
        namespace: Pinecone namespace.
        domain_filter: Metadata filter.
        search_top_k: Top-k for each Pinecone search call.

    Returns:
        Deduplicated search results list.
    """
    is_all_ns = (namespace == 'all')
    all_results = []
    seen_ids = set()

    def _collect(query: str):
        results = _search_single_query(
            agent, query, namespace, search_top_k, domain_filter, is_all_ns,
        )
        for r in results:
            content = r.get('metadata', {}).get('content', '')
            cid = _content_hash(content)
            if cid not in seen_ids:
                seen_ids.add(cid)
                all_results.append(r)

    # Regular queries (original + multi-query variations + synonym expansion)
    for eq in enhancement.search_queries:
        _collect(eq)

    # HyDE queries (separate pool)
    for hq in enhancement.hyde_queries:
        _collect(hq)

    logging.info(
        "[Search] Retrieved %d unique documents from %d queries",
        len(all_results),
        len(enhancement.search_queries) + len(enhancement.hyde_queries),
    )
    return all_results


def run_rag_pipeline(data):
    """Shared RAG search pipeline (Phase 1-7). Returns dict with context, sources, messages, tools, etc.
    Raises ValueError if query is empty. Returns early dict with 'answer' key if no results found."""
    query = data.get('query', '').strip()
    try:
        top_k = max(1, min(int(data.get('top_k', 20)), 100))
    except (ValueError, TypeError):
        logging.warning("[RAG] Invalid top_k value: %s, using default 20", data.get('top_k'))
        top_k = 20
    use_enhancement = data.get('use_enhancement', True)

    # Resolve major/namespace from request (supports both major and legacy namespace params)
    from flask_login import current_user
    _user = current_user if hasattr(current_user, 'major') and current_user.is_authenticated else None
    major_key, namespace = resolve_search_context(data, _user)
    logging.info("[API/ask] Query: %.50s..., Major: %s, Namespace: %s, TopK: %d, Enhanced: %s",
                 query, major_key, namespace, top_k, use_enhancement)

    if not query:
        raise ValueError('질문을 입력해주세요.')

    skip_bm25 = os.environ.get('SKIP_BM25_HYBRID', '').lower() in ('true', '1', 'yes')

    search_query = query
    agent = get_agent()
    _timings = {}

    # ========================================
    # Phase 0: Domain Classification (auto-routing)
    # ========================================
    detected_namespace = None
    detected_domain_label = None
    if namespace != 'all':
        detected_namespace, domain_confidence, detected_domain_label = classify_domain(query, namespace)
        if detected_namespace and detected_namespace != namespace and domain_confidence > 0:
            logging.info("[DomainRouter] Override namespace: %s → %s (label=%s)",
                         namespace, detected_namespace, detected_domain_label)
            namespace = detected_namespace

    domain_key = NAMESPACE_DOMAIN_MAP.get(namespace, 'semiconductor')

    # Query type classification for routing
    query_type = classify_query_type(search_query)
    route_cfg = QUERY_TYPE_CONFIG.get(query_type, QUERY_TYPE_CONFIG['factual'])
    logging.info("[Query Router] Type: %s", query_type)

    # ========================================
    # Phase 1: Query Enhancement
    # ========================================
    _t0 = time.perf_counter()
    enhancement = _enhance_query(search_query, namespace, route_cfg, use_enhancement)
    enhanced_queries = enhancement.all_queries
    keywords = enhancement.keywords
    _timings['phase1_enhancement_ms'] = round((time.perf_counter() - _t0) * 1000)

    # ========================================
    # Phase 2: Multi-Query Search (with domain metadata filtering)
    # ========================================
    domain_filter = build_domain_filter(search_query, namespace)

    _t0 = time.perf_counter()
    top_k_mult = route_cfg.get('top_k_mult', TOP_K_DEFAULT_MULT)
    if skip_bm25:
        search_top_k = top_k * TOP_K_NO_BM25_MULT
    else:
        search_top_k = top_k * top_k_mult
    results = _search_with_variations(
        agent, enhancement, namespace, domain_filter, search_top_k,
    )
    _timings['phase2_search_ms'] = round((time.perf_counter() - _t0) * 1000)

    if not results and namespace != 'laborlaw':
        return {
            'early_response': True,
            'data': {
                'answer': '관련 문서를 찾을 수 없습니다. 다른 검색어로 시도해주세요.',
                'sources': [],
                'enhancement_used': use_enhancement
            }
        }

    # ========================================
    # Phase 4: Hybrid Search (BM25 + Vector)
    # ========================================
    _t0 = time.perf_counter()
    if skip_bm25:
        logging.info("[Hybrid Search] Skipped (SKIP_BM25_HYBRID=true)")
    elif use_enhancement and len(results) > 3:
        try:
            hybrid_searcher = get_hybrid_searcher_instance()
            domain_key = NAMESPACE_DOMAIN_MAP.get(namespace, 'semiconductor')
            results = hybrid_searcher.search_with_keyword_boost(
                query=search_query,
                vector_results=results,
                keywords=keywords,
                top_k=min(len(results), top_k * 2),
                domain=domain_key,
            )
            logging.info("[Hybrid Search] Applied BM25 + keyword boosting")
        except Exception as e:
            logging.warning("[Hybrid Search] Failed: %s, using vector results only", e)

    _timings['phase4_hybrid_ms'] = round((time.perf_counter() - _t0) * 1000)

    # ========================================
    # Phase 5: Reranking (hybrid: cross-encoder 70% + original score 30%)
    # ========================================
    _t0 = time.perf_counter()
    if use_enhancement and len(results) > 3:
        try:
            from src.reranker import DOMAIN_RERANK_CONFIG
            reranker = get_reranker_instance()
            domain_key = NAMESPACE_DOMAIN_MAP.get(namespace, 'semiconductor')
            rerank_cfg = DOMAIN_RERANK_CONFIG.get(domain_key, {})
            rerank_kwargs = {}
            if rerank_cfg:
                rerank_kwargs['rerank_weight'] = rerank_cfg['rerank_weight']
                rerank_kwargs['original_weight'] = rerank_cfg['original_weight']
            else:
                # Fallback to query-type-based rerank weight
                rw = route_cfg.get('rerank_weight', 0.70)
                rerank_kwargs['rerank_weight'] = rw
                rerank_kwargs['original_weight'] = round(1.0 - rw, 2)
            results = reranker.hybrid_rerank(
                query=search_query,
                docs=results,
                top_k=min(len(results), top_k * 2),  # Keep extra for context optimization
                **rerank_kwargs,
            )
            logging.info("[Reranking] Hybrid-reranked to %d documents (domain=%s)", len(results), domain_key)
        except Exception as e:
            logging.warning("[Reranking] Failed: %s, using original order", e)

    _timings['phase5_rerank_ms'] = round((time.perf_counter() - _t0) * 1000)

    # ========================================
    # Phase 6: Filtering, Context Optimization, and Reordering
    # ========================================
    _t0 = time.perf_counter()
    # Filter by minimum relevance score BEFORE reordering (to preserve Lost-in-Middle intent)
    try:
        min_score = float(data.get('min_score', MIN_RELEVANCE_SCORE))
        min_score = max(0.0, min(1.0, min_score))
    except (ValueError, TypeError):
        min_score = MIN_RELEVANCE_SCORE
    results = [r for r in results if r.get('rerank_score', r.get('combined_score', r.get('rrf_score', r.get('score', 0)))) >= min_score]

    # Filter out noise chunks that are too short (token_count below threshold)
    results = [r for r in results
               if r.get('metadata', {}).get('token_count', MIN_TOKEN_COUNT + 1) >= MIN_TOKEN_COUNT]

    # Limit to top_k before reordering
    results = results[:top_k]

    # Early return if all results were filtered out by min_score (laborlaw bypasses this)
    if not results and namespace != 'laborlaw':
        return {
            'early_response': True,
            'data': {
                'answer': '관련도가 충분한 문서를 찾지 못했습니다. 더 구체적인 키워드로 다시 질문해주세요.',
                'sources': [],
                'enhancement_used': use_enhancement
            }
        }

    if use_enhancement:
        try:
            context_optimizer = get_context_optimizer()

            # Deduplicate similar content (domain-specific thresholds)
            results = context_optimizer.deduplicate(results, domain=domain_key)

            # Reorder for LLM attention (Lost in the Middle prevention)
            results = context_optimizer.reorder_for_llm(results, strategy="lost_in_middle")

            logging.info("[Context Optimization] Final context: %d documents", len(results))
        except Exception as e:
            logging.warning("[Context Optimization] Failed: %s", e)

    _timings['phase6_optimize_ms'] = round((time.perf_counter() - _t0) * 1000)

    # ========================================
    # Phase 7: Build Context and Sources
    # ========================================
    _t0 = time.perf_counter()
    context_parts = []
    sources = []

    for i, r in enumerate(results):
        metadata = r.get('metadata', {})
        content = metadata.get('content', '')
        source_file = metadata.get('source_file', 'Unknown')
        file_type = metadata.get('file_type', 'unknown')
        score = r.get('rerank_score', r.get('rrf_score', r.get('score', 0)))

        if content:
            context_parts.append(f"[{i+1}] (출처: {source_file})\n{content}")

            # Build image URL for image files (normalize unicode for macOS)
            image_url = None
            if file_type == 'image' or source_file.lower().endswith(('.jpeg', '.jpg', '.png', '.gif')):
                image_url = f'/documents/{unicodedata.normalize("NFC", source_file)}'

            source_entry = {
                'source_file': source_file,
                'file_type': file_type,
                'score': round(score, 4) if isinstance(score, float) else score,
                'content_preview': content[:200] + '...' if len(content) > 200 else content,
                'content_text': content[:3000],
                'image_url': image_url,
                'ncs_category': metadata.get('ncs_category', ''),
                'ncs_document_title': metadata.get('ncs_document_title', ''),
                'ncs_section_type': metadata.get('ncs_section_type', ''),
                'ncs_code': metadata.get('ncs_code', ''),
                'page_id': metadata.get('page_id'),
            }
            # Laborlaw domain metadata
            for key in ('content_type', 'law_name', 'law_number', 'law_date',
                        'law_category', 'article_number', 'case_collection'):
                if metadata.get(key):
                    source_entry[key] = metadata[key]
            # Field-training domain metadata
            for key in ('training_type', 'cardbook_number', 'equipment_type',
                        'ft_section_type', 'hazard_category'):
                if metadata.get(key):
                    source_entry[key] = metadata[key]
            sources.append(source_entry)

    context = "\n\n---\n\n".join(context_parts)

    _timings['phase7_context_ms'] = round((time.perf_counter() - _t0) * 1000)
    _timings['total_pipeline_ms'] = sum(_timings.values())

    result = {
        'early_response': False,
        'query': query,
        'namespace': namespace,
        'search_query': search_query,
        'context': context,
        'sources': sources,
        'context_parts': context_parts,
        'enhanced_queries': enhanced_queries,
        'keywords': keywords,
        'use_enhancement': use_enhancement,
        'query_type': query_type,
        'detected_namespace': detected_namespace,
        'detected_domain_label': detected_domain_label,
        'latencies': _timings,
    }

    # ========================================
    # Phase 7.5: Safety Cross-Search (major-config based)
    # ========================================
    # Search safety namespace if the major defines one and current NS is the primary
    major_cfg = get_major_config(major_key) if major_key else None
    safety_ns = major_cfg['namespaces'].get('safety') if major_cfg else None
    primary_ns = major_cfg['namespaces'].get('primary') if major_cfg else None
    if safety_ns and namespace in ('', primary_ns):
        try:
            safety_refs = _search_safety_context(search_query)
            if safety_refs:
                result['safety_references'] = safety_refs
                logging.info("[SafetyCross] Found safety context (%d chars)", len(safety_refs))
        except Exception as e:
            logging.warning("[SafetyCross] Failed: %s", e)

    # ========================================
    # Phase 7.6: MSDS Cross-Search (auto-link hazardous substances)
    # ========================================
    if namespace in MSDS_CROSS_SEARCH_NAMESPACES:
        try:
            chemical_names = _extract_chemical_names(search_query, context)
            if chemical_names:
                msds_refs, msds_chems = _search_msds_context(chemical_names)
                if msds_refs:
                    result['msds_references'] = msds_refs
                    result['msds_chemicals'] = msds_chems
                    logging.info("[MSDSCross] Found MSDS data for %d chemicals: %s",
                                 len(msds_chems), [c['name'] for c in msds_chems])
        except Exception as e:
            logging.warning("[MSDSCross] Failed: %s", e)

    # Laborlaw: search law API and run analysis pass
    if namespace == 'laborlaw':
        # Search for relevant law references via public API
        try:
            from services.law_api import search_labor_laws, format_law_references, has_multi_source
            law_refs = search_labor_laws(query)
            if law_refs:
                result['law_references'] = law_refs
                source_count = len(result.get('sources', []))
                result['law_references_formatted'] = format_law_references(
                    law_refs, start_index=source_count + 1)
                result['law_multi_source'] = has_multi_source(law_refs)
        except Exception as e:
            logging.warning("[LaborLaw] Law API search failed: %s", e)

        # Multi-step reasoning: run analysis pass for substantive questions
        if len(query) > 20:
            try:
                analysis = _run_legal_analysis_pass(
                    query, None,
                    result.get('law_references_formatted', ''),
                    result.get('context', ''))
                if analysis:
                    result['legal_analysis'] = analysis
                    logging.info("[LaborLaw] Analysis pass completed (%d chars)", len(analysis))
            except Exception as e:
                logging.warning("[LaborLaw] Analysis pass failed: %s", e)

    return result


def build_llm_prompts(query, sources, context, namespace, calc_result=None,
                      law_references=None, labor_classification=None,
                      legal_analysis=None, safety_references=None,
                      msds_references=None):
    """Build separate system and user prompts for LLM call.

    Args:
        calc_result: Unused (kept for API compatibility), always None.
        law_references: Optional formatted string of relevant law references from the Law API.
        labor_classification: Unused (kept for API compatibility), always None.
        legal_analysis: Optional string from _run_legal_analysis_pass() with structured analysis.
        safety_references: Optional formatted string of related safety/health content from kosha namespace.
        msds_references: Optional formatted string of MSDS data for hazardous substances detected in context.

    Returns:
        tuple: (system_prompt, user_prompt)
    """
    base_prompt = DOMAIN_PROMPTS.get(namespace, DEFAULT_SYSTEM_PROMPT)
    domain_key = NAMESPACE_DOMAIN_MAP.get(namespace, 'semiconductor')
    domain_cot = DOMAIN_COT_INSTRUCTIONS.get(domain_key, '')
    system_prompt = base_prompt + COT_INSTRUCTIONS + domain_cot + VISUAL_GUIDELINES

    user_prompt = f"""## 질문
{query}"""

    # Place reference documents right after the question (Lost in the Middle optimization)
    if sources:
        user_prompt += f"""

## 참고 문서 ({len(sources)}개)
{context}

**인용 규칙**: 위 참고 문서의 번호 [1], [2], ... 을 답변 본문에서 해당 정보 뒤에 인용하세요."""

    # Inject relevant law references from Law API
    if law_references:
        # Extract citation numbers from formatted law references
        import re as _re
        _law_nums = _re.findall(r'^\[(\d+)\]', law_references, _re.MULTILINE)
        _law_range = ', '.join(f'[{n}]' for n in _law_nums) if _law_nums else ''

        # 판례/행정해석/행정규칙이 포함된 multi-source 여부 감지
        _is_multi = any(h in law_references for h in (
            '### 관련 판례', '### 행정해석', '### 관련 고시/지침'))

        _section_title = '관련 법적 근거' if _is_multi else '관련 법령 정보'

        user_prompt += f"""

## {_section_title}
{law_references}

**인용 필수 규칙 (반드시 준수):**
- 위에 제공된 근거 {_law_range} 각각을 답변 본문에서 최소 1회 이상 인용하세요.
- 각 근거의 핵심 내용을 답변에 포함하고 해당 인용 번호를 표기하세요.
- 예: "근로기준법 제23조에 따르면 ... [6]", "대법원 판례에 의하면 ... [7]" 등
- 모든 인용 번호({_law_range})가 답변에 빠짐없이 등장해야 합니다."""

    # Inject multi-step analysis result for legal/hybrid questions
    if legal_analysis:
        user_prompt += f"""

## 사전 법률 분석 (1차 분석 결과)
{legal_analysis}

위 분석을 참고하여 종합적이고 실용적인 답변을 작성하세요."""

    # Inject safety/health cross-search results for semiconductor domain
    if safety_references:
        user_prompt += f"""

## 관련 안전보건 참고자료
{safety_references}

위 안전보건 자료를 참고하여, 답변 말미에 "⚠️ 관련 안전보건 정보" 섹션을 추가하세요.
해당 공정/물질의 유해물질, 보호구, 안전수칙, 응급조치 등을 간결하게 안내하세요.
관련성이 낮으면 이 섹션을 생략하세요."""

    # Inject MSDS cross-search results for hazardous substances
    if msds_references:
        user_prompt += f"""

## 관련 MSDS (물질안전보건자료) 요약
{msds_references}

위 MSDS 정보를 참고하여, 답변에서 해당 유해물질이 언급될 때 "🧪 MSDS 요약" 섹션을 추가하세요.
유해성·위험성, 응급조치, 보호구 정보를 간결하게 안내하세요.
관련성이 낮으면 이 섹션을 생략하세요."""

    # Inject legal analysis mode instructions for laborlaw
    if namespace == 'laborlaw':
        user_prompt += """

## 분석 모드: 법률 해석
이 질문은 법률 해석이 필요한 질문입니다. 다음을 수행하세요:
1. 관련 법조항이 이 상황에 어떻게 적용되는지 분석하세요
2. 예외 조항이나 특례가 적용될 수 있는지 검토하세요
3. 실제로 어떻게 행동해야 하는지 구체적 조언을 포함하세요 (신고처, 기한, 서류 등)
4. 위반 사항이 감지되면 적극적으로 경고하세요
5. 관련될 수 있는 다른 법률도 언급하세요"""

    user_prompt += """

위 정보를 참고하여 질문에 대해 종합적으로 답변해주세요.

**중요 지침:**
1. 각 정보의 출처를 [1], [2] 등의 인용 번호로 표시하세요
2. **질문을 확장해석하여** 관련된 개념, 공정, 위험요인, 주의사항 등 사용자에게 유용한 정보를 적극적으로 포함하세요. 직접 묻지 않아도 알아두어야 할 관련 정보라면 포함하세요
3. 문서에서 확인되지 않는 사실은 추측하지 마세요. 단, 문서에서 확인된 관련 정보는 충분히 활용하세요
4. 핵심 답변을 먼저 1-2문장으로 제시하고, 상세 설명을 이어가세요
5. 기술 용어는 한글과 영문을 병기하세요
6. 질문에 부분적으로만 답변 가능한 경우, 답변 가능한 부분을 먼저 제시하고 문서에서 확인된 관련 추가 정보도 포함하세요"""

    return system_prompt, user_prompt


def build_llm_messages(query, sources, context, namespace, calc_result=None,
                       law_references=None, labor_classification=None,
                       legal_analysis=None, safety_references=None,
                       msds_references=None):
    """Build OpenAI-format messages for LLM call (kept for compatibility)."""
    system_prompt, user_prompt = build_llm_prompts(
        query, sources, context, namespace, calc_result, law_references,
        labor_classification, legal_analysis, safety_references,
        msds_references,
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
