"""Shared RAG pipeline, LLM message builder, and image discovery."""

import hashlib
import logging
import os
import unicodedata

from services.domain_config import (
    DOCUMENTS_PATH,
    DOMAIN_PROMPTS,
    DEFAULT_SYSTEM_PROMPT,
    COT_INSTRUCTIONS,
    VISUAL_GUIDELINES,
)
from services.filters import parse_mentions, build_domain_filter
from services.singletons import (
    get_agent,
    get_openai_client,
    get_query_enhancer,
    get_context_optimizer,
    get_reranker_instance,
    get_hybrid_searcher_instance,
    get_uploader,
)

MIN_RELEVANCE_SCORE = 0.3


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
        logging.warning(f"Error finding images: {e}")
    return images[:10]  # Limit to 10 images


def run_rag_pipeline(data):
    """Shared RAG search pipeline (Phase 1-7). Returns dict with context, sources, messages, tools, etc.
    Raises ValueError if query is empty. Returns early dict with 'answer' key if no results found."""
    query = data.get('query', '').strip()
    namespace = data.get('namespace', '')
    try:
        top_k = max(1, min(int(data.get('top_k', 10)), 100))
    except (ValueError, TypeError):
        logging.warning(f"[RAG] Invalid top_k value: {data.get('top_k')}, using default 10")
        top_k = 10
    use_enhancement = data.get('use_enhancement', True)

    logging.info(f"[API/ask] Query: {query[:50]}..., Namespace: {namespace}, TopK: {top_k}, Enhanced: {use_enhancement}")

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
    client = get_openai_client()

    # ========================================
    # Phase 1: Query Enhancement
    # ========================================
    enhanced_queries = [search_query]
    keywords = []

    if use_enhancement:
        try:
            query_enhancer = get_query_enhancer()

            # Generate query variations for broader retrieval
            enhanced_queries = query_enhancer.multi_query(search_query, num_variations=2)
            logging.info(f"[Query Enhancement] Generated {len(enhanced_queries)} query variations")

            # HyDE: generate hypothetical document for longer queries
            if len(search_query) >= 30:
                try:
                    domain = 'laborlaw' if namespace == 'laborlaw' else 'semiconductor'
                    hyde_doc = query_enhancer.hyde(search_query, domain=domain)
                    if hyde_doc and hyde_doc != search_query:
                        enhanced_queries.append(hyde_doc)
                        logging.info(f"[HyDE] Added hypothetical document query")
                except Exception as e:
                    logging.warning(f"[HyDE] Failed: {e}")

            # Extract keywords for potential filtering
            keywords = query_enhancer.extract_keywords(search_query)
            logging.info(f"[Query Enhancement] Keywords: {keywords}")

        except Exception as e:
            logging.warning(f"[Query Enhancement] Failed: {e}, using original query")
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
                content_id = hashlib.md5(content[:200].encode()).hexdigest()
                if content_id not in seen_ids:
                    seen_ids.add(content_id)
                    all_results.append(r)
        except Exception as e:
            logging.warning(f"[Search] Failed for query '{eq[:30]}...': {e}")

    results = all_results
    logging.info(f"[Search] Retrieved {len(results)} unique documents from {len(enhanced_queries)} queries")

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
        logging.info(f"[Filter] After mention filtering: {len(results)} documents")

    if not results:
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
            logging.info(f"[Hybrid Search] Applied BM25 + keyword boosting")
        except Exception as e:
            logging.warning(f"[Hybrid Search] Failed: {e}, using vector results only")

    # ========================================
    # Phase 5: Reranking
    # ========================================
    if use_enhancement and len(results) > 3:
        try:
            reranker = get_reranker_instance()
            results = reranker.rerank(
                query=search_query,
                docs=results,
                top_k=min(len(results), top_k + 5)  # Keep extra for context optimization
            )
            logging.info(f"[Reranking] Reranked to {len(results)} documents")
        except Exception as e:
            logging.warning(f"[Reranking] Failed: {e}, using original order")

    # ========================================
    # Phase 6: Context Optimization
    # ========================================
    if use_enhancement:
        try:
            context_optimizer = get_context_optimizer()

            # Deduplicate similar content
            results = context_optimizer.deduplicate(results)

            # Reorder for LLM attention (Lost in the Middle prevention)
            results = context_optimizer.reorder_for_llm(results, strategy="lost_in_middle")

            logging.info(f"[Context Optimization] Final context: {len(results)} documents")
        except Exception as e:
            logging.warning(f"[Context Optimization] Failed: {e}")

    # Filter by minimum relevance score
    results = [r for r in results if r.get('rerank_score', r.get('rrf_score', r.get('score', 0))) >= MIN_RELEVANCE_SCORE]

    # Limit to top_k after all processing
    results = results[:top_k]

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

    return {
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
        'client': client,
    }


def build_llm_messages(query, sources, context, namespace):
    """Build system prompt and user message for LLM call."""
    base_prompt = DOMAIN_PROMPTS.get(namespace, DEFAULT_SYSTEM_PROMPT)
    system_prompt = base_prompt + COT_INSTRUCTIONS + VISUAL_GUIDELINES

    user_prompt = f"""## 질문
{query}

## 참고 문서 ({len(sources)}개)
{context}

위 문서들을 참고하여 질문에 대해 종합적으로 답변해주세요.

**중요 지침:**
1. 각 정보의 출처를 [1], [2] 등의 인용 번호로 표시하세요
2. 문서에 없는 내용은 추측하지 마세요
3. 핵심 답변을 먼저 제시하고, 상세 설명을 이어가세요
4. 기술 용어는 한글과 영문을 병기하세요"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
