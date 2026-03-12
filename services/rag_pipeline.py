"""Shared RAG pipeline, LLM message builder, and image discovery."""

import concurrent.futures
import hashlib
import logging
import os
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
from services.filters import parse_mentions, build_domain_filter
from services.query_router import classify_query_type, QUERY_TYPE_CONFIG
from services.singletons import (
    get_agent,
    get_query_enhancer,
    get_context_optimizer,
    get_reranker_instance,
    get_hybrid_searcher_instance,
    get_uploader,
)

MIN_RELEVANCE_SCORE = float(os.environ.get('MIN_RELEVANCE_SCORE', '0.2'))
MIN_TOKEN_COUNT = int(os.environ.get('MIN_TOKEN_COUNT', '30'))
TOP_K_MENTION_MULT = int(os.environ.get('TOP_K_MENTION_MULT', '3'))
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


def _get_num_variations(query: str) -> int:
    """쿼리 길이에 따라 멀티쿼리 변형 수를 동적으로 결정."""
    length = len(query)
    if length < 15:
        return 1   # 짧은 키워드 쿼리 — 변형 불필요
    elif length < 40:
        return 2   # 일반 쿼리
    else:
        return 3   # 긴 복잡한 쿼리


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


def run_rag_pipeline(data):
    """Shared RAG search pipeline (Phase 1-7). Returns dict with context, sources, messages, tools, etc.
    Raises ValueError if query is empty. Returns early dict with 'answer' key if no results found."""
    query = data.get('query', '').strip()
    namespace = data.get('namespace', '')
    try:
        top_k = max(1, min(int(data.get('top_k', 20)), 100))
    except (ValueError, TypeError):
        logging.warning("[RAG] Invalid top_k value: %s, using default 20", data.get('top_k'))
        top_k = 20
    use_enhancement = data.get('use_enhancement', True)

    logging.info("[API/ask] Query: %.50s..., Namespace: %s, TopK: %d, Enhanced: %s", query, namespace, top_k, use_enhancement)

    if not query:
        raise ValueError('질문을 입력해주세요.')

    skip_bm25 = os.environ.get('SKIP_BM25_HYBRID', '').lower() in ('true', '1', 'yes')

    # Parse @mentions for source filtering
    clean_query, mention_filters = parse_mentions(query)

    # Build search query
    if mention_filters:
        filter_keywords = ' '.join([f['value'].replace('_', ' ') for f in mention_filters])
        if clean_query and len(clean_query) >= 3:
            search_query = f"{filter_keywords} {clean_query}"
        else:
            search_query = filter_keywords
    else:
        search_query = clean_query if clean_query else query

    agent = get_agent()
    _timings = {}
    domain_key = NAMESPACE_DOMAIN_MAP.get(namespace, 'semiconductor')

    # Query type classification for routing
    query_type = classify_query_type(search_query)
    route_cfg = QUERY_TYPE_CONFIG.get(query_type, QUERY_TYPE_CONFIG['factual'])
    logging.info("[Query Router] Type: %s", query_type)

    # ========================================
    # Phase 1: Query Enhancement
    # ========================================
    _t0 = time.perf_counter()
    enhanced_queries = [search_query]
    keywords = []

    use_multi_query = route_cfg.get('use_multi_query', True)
    use_hyde = route_cfg.get('use_hyde', False)

    if use_enhancement:
        try:
            query_enhancer = get_query_enhancer()
            domain = NAMESPACE_DOMAIN_MAP.get(namespace, 'semiconductor')

            # Expand query with domain-specific synonyms before multi-query generation
            expanded_query = query_enhancer.expand_with_synonyms(search_query, domain)
            if expanded_query != search_query:
                logging.info("[Synonym Expansion] '%s' → '%s'", search_query, expanded_query)

            # Run query enhancement calls in parallel for lower latency
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                future_multi = None
                if use_multi_query:
                    num_variations = _get_num_variations(search_query)
                    future_multi = executor.submit(query_enhancer.multi_query, search_query, num_variations)

                future_hyde = None
                if use_hyde and len(search_query) >= 10:
                    future_hyde = executor.submit(query_enhancer.hyde, search_query, domain)

                future_keywords = executor.submit(query_enhancer.extract_keywords_fast, search_query)

                if future_multi:
                    try:
                        enhanced_queries = future_multi.result(timeout=10)
                    except Exception as e:
                        logging.warning("[Query Enhancement] multi_query failed: %s", e)
                        enhanced_queries = [search_query]
                    logging.info("[Query Enhancement] Generated %d query variations", len(enhanced_queries))
                else:
                    logging.info("[Query Enhancement] multi_query skipped (query_type=%s)", query_type)

                if future_hyde:
                    try:
                        hyde_doc = future_hyde.result(timeout=10)
                        if hyde_doc and hyde_doc != search_query:
                            enhanced_queries.append(hyde_doc)
                            logging.info("[HyDE] Added hypothetical document query")
                    except Exception as e:
                        logging.warning("[HyDE] Failed: %s", e)
                else:
                    logging.info("[HyDE] Skipped (query_type=%s)", query_type)

                try:
                    keywords = future_keywords.result(timeout=10)
                except Exception as e:
                    logging.warning("[Query Enhancement] keywords failed: %s", e)
                    keywords = []
                logging.info("[Query Enhancement] Keywords: %s", keywords)

                # Add synonym-expanded query as an additional variation
                if expanded_query != search_query and expanded_query not in enhanced_queries:
                    enhanced_queries.append(expanded_query)
                    logging.info("[Synonym Expansion] Added expanded query as variation")

        except Exception as e:
            logging.warning("[Query Enhancement] Failed: %s, using original query", e)
            enhanced_queries = [search_query]

    _timings['phase1_enhancement_ms'] = round((time.perf_counter() - _t0) * 1000)

    # ========================================
    # Phase 2: Multi-Query Search (with domain metadata filtering)
    # ========================================
    # Build domain-aware metadata filter from query and namespace
    domain_filter = build_domain_filter(search_query, namespace)

    _t0 = time.perf_counter()
    # Search with multiple query variations and merge results
    # When BM25 is skipped, fetch more candidates so the reranker has a wider pool
    top_k_mult = route_cfg.get('top_k_mult', TOP_K_DEFAULT_MULT)
    if mention_filters:
        search_top_k = top_k * TOP_K_MENTION_MULT
    elif skip_bm25:
        search_top_k = top_k * TOP_K_NO_BM25_MULT  # Wider recall when relying on reranker alone
    else:
        search_top_k = top_k * top_k_mult  # Route-aware fetch multiplier
    is_all_namespace = (namespace == 'all')
    all_results = []
    seen_ids = set()

    for eq in enhanced_queries:
        try:
            if is_all_namespace:
                # Multi-namespace simultaneous query via Pinecone server-side parallelism
                try:
                    uploader = get_uploader()
                    stats = uploader.get_stats()
                    ns_list = [ns for ns in stats.get('namespaces', {}).keys() if ns]
                except Exception:
                    ns_list = ['semiconductor', 'laborlaw', 'field-training']
                results = agent.search_all_namespaces(
                    query=eq,
                    namespaces=ns_list,
                    top_k=search_top_k,
                    filter=domain_filter
                )
            else:
                results = agent.search(
                    query=eq,
                    top_k=search_top_k,
                    namespace=namespace,
                    filter=domain_filter
                )
            for r in results:
                # Deduplicate by content hash
                content = r.get('metadata', {}).get('content', '')
                content_id = hashlib.sha256(content[:5000].encode()).hexdigest()
                if content_id not in seen_ids:
                    seen_ids.add(content_id)
                    all_results.append(r)
        except Exception as e:
            logging.warning("[Search] Failed for query '%.30s...': %s", eq, e)

    results = all_results
    logging.info("[Search] Retrieved %d unique documents from %d queries", len(results), len(enhanced_queries))
    _timings['phase2_search_ms'] = round((time.perf_counter() - _t0) * 1000)

    # ========================================
    # Phase 3: Apply mention filters
    # ========================================
    _t0 = time.perf_counter()
    if mention_filters and results:
        filtered_results = []
        for r in results:
            source_file = unicodedata.normalize('NFC', r.get('metadata', {}).get('source_file', ''))
            filename = unicodedata.normalize('NFC', r.get('metadata', {}).get('filename', ''))

            match = False
            for f in mention_filters:
                filter_value = unicodedata.normalize('NFC', f['value'].lower())
                if f['type'] == 'file':
                    if filter_value in filename.lower():
                        match = True
                        break
                elif f['type'] == 'folder':
                    if filter_value in source_file.lower():
                        match = True
                        break
                elif f['type'] == 'keyword':
                    if filter_value in source_file.lower():
                        match = True
                        break

            if match:
                filtered_results.append(r)

        results = filtered_results
        logging.info("[Filter] After mention filtering: %d documents", len(results))

    if not results and namespace != 'laborlaw':
        return {
            'early_response': True,
            'data': {
                'answer': '관련 문서를 찾을 수 없습니다. 다른 검색어로 시도해주세요.',
                'sources': [],
                'enhancement_used': use_enhancement
            }
        }

    _timings['phase3_filter_ms'] = round((time.perf_counter() - _t0) * 1000)

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
        'latencies': _timings,
    }

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
                      legal_analysis=None):
    """Build separate system and user prompts for LLM call.

    Args:
        calc_result: Unused (kept for API compatibility), always None.
        law_references: Optional formatted string of relevant law references from the Law API.
        labor_classification: Unused (kept for API compatibility), always None.
        legal_analysis: Optional string from _run_legal_analysis_pass() with structured analysis.

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
                       legal_analysis=None):
    """Build OpenAI-format messages for LLM call (kept for compatibility)."""
    system_prompt, user_prompt = build_llm_prompts(
        query, sources, context, namespace, calc_result, law_references,
        labor_classification, legal_analysis
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
