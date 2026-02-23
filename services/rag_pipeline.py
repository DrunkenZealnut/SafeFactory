"""Shared RAG pipeline, LLM message builder, and image discovery."""

import concurrent.futures
import hashlib
import logging
import os
import unicodedata
from datetime import datetime

from services.domain_config import (
    DOCUMENTS_PATH,
    DOMAIN_PROMPTS,
    DEFAULT_SYSTEM_PROMPT,
    COT_INSTRUCTIONS,
    VISUAL_GUIDELINES,
    NAMESPACE_DOMAIN_MAP,
)
from services.filters import parse_mentions, build_domain_filter
from services.singletons import (
    get_agent,
    get_query_enhancer,
    get_context_optimizer,
    get_reranker_instance,
    get_hybrid_searcher_instance,
    get_uploader,
)

MIN_RELEVANCE_SCORE = float(os.environ.get('MIN_RELEVANCE_SCORE', '0.3'))


def _run_legal_analysis_pass(query, classification, law_refs_text, context):
    """First LLM pass: produce a structured legal analysis for complex labor law questions.

    Uses a lightweight model (gemini-2.0-flash) to analyze the legal aspects
    before the main generation pass. Only called for legal/hybrid questions.

    Returns:
        str: Structured analysis text, or empty string on failure.
    """
    from services.singletons import get_gemini_client
    from google.genai import types as genai_types

    q_type = classification.get('type', 'legal')
    calc_type = classification.get('calc_type', '')

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
        max_output_tokens=600,
    )
    resp = client.models.generate_content(
        model='gemini-2.0-flash',
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
        top_k = max(1, min(int(data.get('top_k', 10)), 100))
    except (ValueError, TypeError):
        logging.warning("[RAG] Invalid top_k value: %s, using default 10", data.get('top_k'))
        top_k = 10
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

    # ========================================
    # Phase 1: Query Enhancement
    # ========================================
    enhanced_queries = [search_query]
    keywords = []

    if use_enhancement:
        try:
            query_enhancer = get_query_enhancer()
            domain = NAMESPACE_DOMAIN_MAP.get(namespace, 'semiconductor')

            # Run query enhancement calls in parallel for lower latency
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                future_multi = executor.submit(query_enhancer.multi_query, search_query, 2)
                future_hyde = (
                    executor.submit(query_enhancer.hyde, search_query, domain)
                    if len(search_query) >= 30 else None
                )
                future_keywords = executor.submit(query_enhancer.extract_keywords_fast, search_query)

                try:
                    enhanced_queries = future_multi.result(timeout=10)
                except Exception as e:
                    logging.warning("[Query Enhancement] multi_query failed: %s", e)
                    enhanced_queries = [search_query]
                logging.info("[Query Enhancement] Generated %d query variations", len(enhanced_queries))

                if future_hyde:
                    try:
                        hyde_doc = future_hyde.result(timeout=10)
                        if hyde_doc and hyde_doc != search_query:
                            enhanced_queries.append(hyde_doc)
                            logging.info("[HyDE] Added hypothetical document query")
                    except Exception as e:
                        logging.warning("[HyDE] Failed: %s", e)

                try:
                    keywords = future_keywords.result(timeout=10)
                except Exception as e:
                    logging.warning("[Query Enhancement] keywords failed: %s", e)
                    keywords = []
                logging.info("[Query Enhancement] Keywords: %s", keywords)

        except Exception as e:
            logging.warning("[Query Enhancement] Failed: %s, using original query", e)
            enhanced_queries = [search_query]

    # ========================================
    # Phase 2: Multi-Query Search (with domain metadata filtering)
    # ========================================
    # Build domain-aware metadata filter from query and namespace
    domain_filter = build_domain_filter(search_query, namespace)

    # Search with multiple query variations and merge results
    # When BM25 is skipped, fetch more candidates so the reranker has a wider pool
    if mention_filters:
        search_top_k = top_k * 3
    elif skip_bm25:
        search_top_k = top_k * 4  # Wider recall when relying on reranker alone
    else:
        search_top_k = top_k * 2  # Fetch more for filtering/reranking
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

    # ========================================
    # Phase 3: Apply mention filters
    # ========================================
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

    # ========================================
    # Phase 4: Hybrid Search (BM25 + Vector)
    # ========================================
    if skip_bm25:
        logging.info("[Hybrid Search] Skipped (SKIP_BM25_HYBRID=true)")
    elif use_enhancement and len(results) > 3:
        try:
            hybrid_searcher = get_hybrid_searcher_instance()
            results = hybrid_searcher.search_with_keyword_boost(
                query=search_query,
                vector_results=results,
                keywords=keywords,
                top_k=min(len(results), top_k * 2)
            )
            logging.info("[Hybrid Search] Applied BM25 + keyword boosting")
        except Exception as e:
            logging.warning("[Hybrid Search] Failed: %s, using vector results only", e)

    # ========================================
    # Phase 5: Reranking (hybrid: cross-encoder 70% + original score 30%)
    # ========================================
    if use_enhancement and len(results) > 3:
        try:
            reranker = get_reranker_instance()
            results = reranker.hybrid_rerank(
                query=search_query,
                docs=results,
                top_k=min(len(results), top_k + 5)  # Keep extra for context optimization
            )
            logging.info("[Reranking] Hybrid-reranked to %d documents", len(results))
        except Exception as e:
            logging.warning("[Reranking] Failed: %s, using original order", e)

    # ========================================
    # Phase 6: Filtering, Context Optimization, and Reordering
    # ========================================
    # Filter by minimum relevance score BEFORE reordering (to preserve Lost-in-Middle intent)
    try:
        min_score = float(data.get('min_score', MIN_RELEVANCE_SCORE))
        min_score = max(0.0, min(1.0, min_score))
    except (ValueError, TypeError):
        min_score = MIN_RELEVANCE_SCORE
    results = [r for r in results if r.get('rerank_score', r.get('combined_score', r.get('rrf_score', r.get('score', 0)))) >= min_score]

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

            # Deduplicate similar content
            results = context_optimizer.deduplicate(results)

            # Reorder for LLM attention (Lost in the Middle prevention)
            results = context_optimizer.reorder_for_llm(results, strategy="lost_in_middle")

            logging.info("[Context Optimization] Final context: %d documents", len(results))
        except Exception as e:
            logging.warning("[Context Optimization] Failed: %s", e)

    # ========================================
    # Phase 7: Build Context and Sources
    # ========================================
    context_parts = []
    sources = []

    for i, r in enumerate(results):
        metadata = r.get('metadata', {})
        content = metadata.get('content', '')
        source_file = metadata.get('source_file', 'Unknown')
        file_type = metadata.get('file_type', 'unknown')
        score = r.get('rerank_score', r.get('rrf_score', r.get('score', 0)))

        if content:
            context_parts.append(f"[문서 {i+1}] (출처: {source_file})\n{content}")

            # Build image URL for image files (normalize unicode for macOS)
            image_url = None
            if file_type == 'image' or source_file.lower().endswith(('.jpeg', '.jpg', '.png', '.gif')):
                image_url = f'/documents/{unicodedata.normalize("NFC", source_file)}'

            source_entry = {
                'source_file': source_file,
                'file_type': file_type,
                'score': round(score, 4) if isinstance(score, float) else score,
                'content_preview': content[:200] + '...' if len(content) > 200 else content,
                'image_url': image_url,
                'ncs_category': metadata.get('ncs_category', ''),
                'ncs_document_title': metadata.get('ncs_document_title', ''),
                'ncs_section_type': metadata.get('ncs_section_type', ''),
                'ncs_code': metadata.get('ncs_code', ''),
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
    }

    # Laborlaw: classify question, run calculator, search law API, and run analysis pass
    if namespace == 'laborlaw':
        classification = None
        try:
            from services.labor_classifier import classify_labor_question
            from services.labor_calculator import run_labor_calculation
            classification = classify_labor_question(query)
            result['labor_classification'] = classification
            if classification['type'] in ('calculation', 'hybrid'):
                calc_result = run_labor_calculation(classification)
                if calc_result:
                    result['labor_calc_result'] = calc_result
        except Exception as e:
            logging.warning("[LaborLaw] Classification/calculation failed: %s", e)

        # Search for relevant law references via public API
        try:
            from services.law_api import search_labor_laws, format_law_references
            law_refs = search_labor_laws(query, classification)
            if law_refs:
                result['law_references'] = law_refs
                source_count = len(result.get('sources', []))
                result['law_references_formatted'] = format_law_references(
                    law_refs, start_index=source_count + 1)
        except Exception as e:
            logging.warning("[LaborLaw] Law API search failed: %s", e)

        # Multi-step reasoning: run analysis pass for legal/hybrid questions
        if (classification and classification.get('type') in ('legal', 'hybrid')
                and len(query) > 20):
            try:
                analysis = _run_legal_analysis_pass(
                    query, classification,
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
        calc_result: Optional dict from run_labor_calculation() with 'formatted' key.
        law_references: Optional formatted string of relevant law references from the Law API.
        labor_classification: Optional dict from classify_labor_question() with 'type' and 'calc_type'.
        legal_analysis: Optional string from _run_legal_analysis_pass() with structured analysis.

    Returns:
        tuple: (system_prompt, user_prompt)
    """
    base_prompt = DOMAIN_PROMPTS.get(namespace, DEFAULT_SYSTEM_PROMPT)
    system_prompt = base_prompt + COT_INSTRUCTIONS + VISUAL_GUIDELINES

    user_prompt = f"""## 질문
{query}"""

    # Inject precise calculation results for laborlaw
    if calc_result and calc_result.get('formatted'):
        user_prompt += f"""

## 시스템 계산 결과 (정확한 수치)
{calc_result['formatted']}

위 계산 결과는 {datetime.now().year}년 기준 공식 요율로 정밀 계산된 수치입니다.
이 수치를 답변에 그대로 사용하고, 직접 재계산하지 마세요."""

    # Inject compliance warnings for detected violations
    if calc_result:
        result_data = calc_result.get('result') or {}
        violation = str(result_data.get('위반_여부', ''))
        if '위반' in violation:
            user_prompt += """

## ⚠️ 법 위반 감지
시스템 분석 결과 법 위반 가능성이 감지되었습니다.
답변에 반드시 다음을 포함하세요:
1. 위반 사실과 법적 근거를 명확히 설명
2. 사업주의 벌칙/제재 내용 (형사처벌, 과태료 등)
3. 근로자가 취할 수 있는 구제 방법 (고용노동부 신고 1350, 노동위원회 등)
4. 관련 소멸시효 및 기한 안내"""

    # Inject relevant law references from Law API
    if law_references:
        # Extract citation numbers from formatted law references
        import re as _re
        _law_nums = _re.findall(r'^\[(\d+)\]', law_references, _re.MULTILINE)
        _law_range = ', '.join(f'[{n}]' for n in _law_nums) if _law_nums else ''

        user_prompt += f"""

## 관련 법령 정보
{law_references}

**법령 인용 필수 규칙 (반드시 준수):**
- 위에 제공된 법령 {_law_range} 각각을 답변 본문에서 최소 1회 이상 인용하세요.
- 각 법령 조문의 핵심 내용을 답변에 포함하고 해당 인용 번호를 표기하세요.
- 예: "최저임금의 정의는 ... [6]", "최저임금 결정기준은 ... [7]" 등
- 모든 법령 번호({_law_range})가 답변에 빠짐없이 등장해야 합니다."""

    # Inject multi-step analysis result for legal/hybrid questions
    if legal_analysis:
        user_prompt += f"""

## 사전 법률 분석 (1차 분석 결과)
{legal_analysis}

위 분석을 참고하여 종합적이고 실용적인 답변을 작성하세요."""

    if sources:
        user_prompt += f"""

## 참고 문서 ({len(sources)}개)
{context}"""

    # Inject classification-aware analysis mode instructions for laborlaw
    if namespace == 'laborlaw' and labor_classification:
        q_type = labor_classification.get('type', 'legal')

        if q_type == 'legal':
            user_prompt += """

## 분석 모드: 법률 해석
이 질문은 법률 해석이 필요한 질문입니다. 다음을 수행하세요:
1. 관련 법조항이 이 상황에 어떻게 적용되는지 분석하세요
2. 예외 조항이나 특례가 적용될 수 있는지 검토하세요
3. 실제로 어떻게 행동해야 하는지 구체적 조언을 포함하세요 (신고처, 기한, 서류 등)
4. 위반 사항이 감지되면 적극적으로 경고하세요
5. 관련될 수 있는 다른 법률도 언급하세요"""

        elif q_type == 'calculation':
            user_prompt += """

## 분석 모드: 계산 분석
시스템 계산 결과를 기반으로 다음도 분석하세요:
1. 계산 결과의 의미를 쉽게 설명하세요
2. 최저임금 위반, 초과근로 등 법적 이슈가 있는지 체크하세요
3. 절세나 최적화할 수 있는 팁이 있으면 제안하세요
4. 비과세 한도, 보험료 상한 등 사용자에게 유리한 정보를 안내하세요"""

        elif q_type == 'hybrid':
            user_prompt += """

## 분석 모드: 복합 분석
계산과 법률 해석이 모두 필요한 질문입니다:
1. 시스템 계산 결과를 먼저 제시하세요
2. 해당 계산 결과의 법적 적합성을 분석하세요
3. 위반 사항이 있으면 정확한 법적 근거와 함께 경고하세요
4. 구체적인 조치 방법을 안내하세요"""

    user_prompt += """

위 정보를 참고하여 질문에 대해 종합적으로 답변해주세요.

**중요 지침:**
1. 각 정보의 출처를 [1], [2] 등의 인용 번호로 표시하세요
2. 문서에 없는 내용은 추측하지 말고, "제공된 문서에서 해당 정보를 찾을 수 없습니다"라고 명시하세요
3. 핵심 답변을 먼저 1-2문장으로 제시하고, 상세 설명을 이어가세요
4. 기술 용어는 한글과 영문을 병기하세요
5. 질문에 부분적으로만 답변 가능한 경우, 답변 가능한 부분을 먼저 제시하고 부족한 부분을 명시하세요"""

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
