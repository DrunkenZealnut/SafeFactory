"""Search, ask, ask/stream (SSE), and PDF resolve endpoints."""

import json
import logging
import re
import unicodedata
from urllib.parse import quote
from flask import request, Response, stream_with_context

from api.v1 import v1_bp
from api.response import success_response, error_response
from services.singletons import get_agent, get_hybrid_searcher_instance
from services.rag_pipeline import run_rag_pipeline, build_llm_messages, find_related_images
from services.calculator import CALCULATOR_FUNCTIONS, handle_tool_calls, calculate_wage, calculate_insurance
from services.domain_config import DOCUMENTS_PATH, DOMAIN_PROMPTS, DEFAULT_SYSTEM_PROMPT


def _collect_related_images(sources, max_images=12):
    """Collect related images from source documents."""
    related_images = []
    seen_folders = set()
    for source in sources:
        source_file = source.get('source_file', '')
        folder_key = '/'.join(source_file.split('/')[:3])
        if folder_key not in seen_folders:
            seen_folders.add(folder_key)
            images = find_related_images(source_file)
            related_images.extend(images)
    return related_images[:max_images]


@v1_bp.route('/search', methods=['POST'])
def api_search():
    """Search for similar content with optional hybrid/keyword modes."""
    try:
        data = request.get_json()
        if not data:
            return error_response('요청 데이터가 없습니다.', 400)
        query = data.get('query', '').strip()
        try:
            top_k = max(1, min(int(data.get('top_k', 5)), 100))
        except (ValueError, TypeError):
            top_k = 5
        namespace = data.get('namespace', '')
        file_type = data.get('file_type', '')
        search_mode = data.get('search_mode', 'vector')  # vector | hybrid | keyword

        # 3-level metadata filters
        domain = data.get('domain', '')
        category = data.get('category', '')
        subcategory = data.get('subcategory', '')

        if not query:
            return error_response('검색어를 입력해주세요.', 400)

        agent = get_agent()

        # Build filter from file_type + metadata levels
        filter_dict = {}
        if file_type:
            filter_dict["file_type"] = file_type
        if domain:
            filter_dict["domain"] = domain
        if category:
            filter_dict["category"] = category
        if subcategory:
            filter_dict["subcategory"] = subcategory
        filter_dict = filter_dict or None

        # Always start with vector search to get candidate documents
        vector_results = agent.search(
            query=query,
            top_k=max(top_k, 20) if search_mode != 'vector' else top_k,
            namespace=namespace,
            filter=filter_dict
        )

        if search_mode == 'vector':
            # Pure vector search - existing behavior
            formatted_results = []
            for r in vector_results:
                metadata = r.get('metadata', {})
                formatted_results.append({
                    'score': round(r.get('score', 0), 4),
                    'source_file': metadata.get('source_file', 'N/A'),
                    'file_type': metadata.get('file_type', 'N/A'),
                    'content': metadata.get('content_preview', metadata.get('content', '')[:500]),
                    'filename': metadata.get('filename', ''),
                    'relative_path': metadata.get('relative_path', ''),
                    'ncs_category': metadata.get('ncs_category', ''),
                    'ncs_document_title': metadata.get('ncs_document_title', ''),
                    'ncs_section_type': metadata.get('ncs_section_type', ''),
                    'ncs_code': metadata.get('ncs_code', ''),
                })
        else:
            # hybrid or keyword mode - use HybridSearcher
            hybrid_searcher = get_hybrid_searcher_instance()
            hybrid_results = hybrid_searcher.search(
                query=query,
                vector_results=vector_results,
                top_k=top_k,
                build_index=True
            )

            if search_mode == 'keyword':
                # Re-sort by BM25 rank (keyword relevance)
                hybrid_results.sort(
                    key=lambda x: x.get('bm25_rank') or 9999
                )
                hybrid_results = hybrid_results[:top_k]

            # Pre-compute BM25 scores via public API
            bm25_score_map = hybrid_searcher.get_bm25_scores(query)

            formatted_results = []
            for r in hybrid_results:
                metadata = r.get('metadata', {})
                bm25_score_val = bm25_score_map.get(r.get('id'))

                formatted_results.append({
                    'score': round(r.get('score', 0), 4),
                    'rrf_score': round(r.get('rrf_score', 0), 6),
                    'vector_rank': r.get('vector_rank'),
                    'bm25_rank': r.get('bm25_rank'),
                    'bm25_score': bm25_score_val,
                    'source_file': metadata.get('source_file', 'N/A'),
                    'file_type': metadata.get('file_type', 'N/A'),
                    'content': metadata.get('content_preview', metadata.get('content', '')[:500]),
                    'filename': metadata.get('filename', ''),
                    'relative_path': metadata.get('relative_path', ''),
                    'ncs_category': metadata.get('ncs_category', ''),
                    'ncs_document_title': metadata.get('ncs_document_title', ''),
                    'ncs_section_type': metadata.get('ncs_section_type', ''),
                    'ncs_code': metadata.get('ncs_code', ''),
                })

        return success_response(data={
            'query': query,
            'search_mode': search_mode,
            'count': len(formatted_results),
            'results': formatted_results
        })
    except Exception as e:
        return error_response(str(e), 500)


@v1_bp.route('/ask', methods=['POST'])
def api_ask():
    """RAG endpoint - search and generate comprehensive answer with enhanced retrieval."""
    try:
        data = request.get_json()
        if not data:
            return error_response('요청 데이터가 없습니다.', 400)
        pipeline = run_rag_pipeline(data)

        if pipeline.get('early_response'):
            return success_response(data=pipeline['data'])

        query = pipeline['query']
        namespace = pipeline['namespace']
        context = pipeline['context']
        sources = pipeline['sources']
        enhanced_queries = pipeline['enhanced_queries']
        keywords = pipeline['keywords']
        use_enhancement = pipeline['use_enhancement']
        client = pipeline['client']

        # ========================================
        # Phase 8: Generate Answer with Improved Prompt
        # ========================================
        messages = build_llm_messages(query, sources, context, namespace)

        # Enable calculator functions only for laborlaw namespace
        tools = CALCULATOR_FUNCTIONS if namespace == 'laborlaw' else None

        # Initial GPT call (with function calling if tools available)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto" if tools else None,
            temperature=0.3,
            max_tokens=2500
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        # Handle function calls
        calculation_results = []
        if tool_calls:
            calculation_results, messages = handle_tool_calls(messages, response_message)

            # Get final response with function results
            second_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.3,
                max_tokens=2500
            )

            answer = second_response.choices[0].message.content
        else:
            answer = response_message.content

        related_images = _collect_related_images(sources)

        return success_response(data={
            'query': query,
            'answer': answer,
            'sources': sources,
            'source_count': len(sources),
            'images': related_images,
            'calculations': calculation_results if calculation_results else None,
            'enhancement_used': use_enhancement,
            'query_variations': enhanced_queries if use_enhancement else None,
            'keywords_extracted': keywords if use_enhancement else None
        })

    except Exception as e:
        logging.exception(f"[API/ask] Error: {str(e)}")
        return error_response(str(e), 500)


@v1_bp.route('/ask/stream', methods=['POST'])
def api_ask_stream():
    """SSE streaming RAG endpoint - sends metadata first, then streams LLM answer token by token."""
    try:
        data = request.get_json()
        if not data:
            return error_response('요청 데이터가 없습니다.', 400)
        pipeline = run_rag_pipeline(data)
    except ValueError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logging.exception(f"[API/ask/stream] Pipeline error: {str(e)}")
        return error_response(str(e), 500)

    if pipeline.get('early_response'):
        return success_response(data=pipeline['data'])

    query = pipeline['query']
    namespace = pipeline['namespace']
    context = pipeline['context']
    sources = pipeline['sources']
    enhanced_queries = pipeline['enhanced_queries']
    keywords = pipeline['keywords']
    use_enhancement = pipeline['use_enhancement']
    client = pipeline['client']

    messages = build_llm_messages(query, sources, context, namespace)
    tools = CALCULATOR_FUNCTIONS if namespace == 'laborlaw' else None

    def generate():
        try:
            calculation_results = []

            related_images = _collect_related_images(sources)

            # Send metadata event first (sources, images, query info)
            metadata_event = json.dumps({
                'type': 'metadata',
                'data': {
                    'query': query,
                    'sources': sources,
                    'source_count': len(sources),
                    'images': related_images,
                    'enhancement_used': use_enhancement,
                    'query_variations': enhanced_queries if use_enhancement else None,
                    'keywords_extracted': keywords if use_enhancement else None
                }
            }, ensure_ascii=False)
            yield f"data: {metadata_event}\n\n"

            # If tools available, first call is non-streaming to check for tool_calls
            if tools:
                first_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.3,
                    max_tokens=2500
                )
                response_message = first_response.choices[0].message

                if response_message.tool_calls:
                    calc_results, messages_updated = handle_tool_calls(messages, response_message)
                    calculation_results = calc_results

                    # Send calculation results event
                    calc_event = json.dumps({
                        'type': 'calculations',
                        'data': calculation_results
                    }, ensure_ascii=False)
                    yield f"data: {calc_event}\n\n"

                    # Stream the final response after tool calls
                    stream = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=messages_updated,
                        temperature=0.3,
                        max_tokens=2500,
                        stream=True
                    )
                    for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            token = chunk.choices[0].delta.content
                            token_event = json.dumps({'type': 'token', 'data': token}, ensure_ascii=False)
                            yield f"data: {token_event}\n\n"
                else:
                    # No tool calls - reuse the already-received response
                    content = response_message.content or ""
                    if content:
                        token_event = json.dumps({'type': 'token', 'data': content}, ensure_ascii=False)
                        yield f"data: {token_event}\n\n"
            else:
                # No tools - stream directly
                stream = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.3,
                    max_tokens=2500,
                    stream=True
                )
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        token = chunk.choices[0].delta.content
                        token_event = json.dumps({'type': 'token', 'data': token}, ensure_ascii=False)
                        yield f"data: {token_event}\n\n"

            # Send done event
            done_event = json.dumps({
                'type': 'done',
                'data': {'calculations': calculation_results if calculation_results else None}
            }, ensure_ascii=False)
            yield f"data: {done_event}\n\n"

        except Exception as e:
            logging.exception(f"[API/ask/stream] Streaming error: {str(e)}")
            error_event = json.dumps({'type': 'error', 'data': str(e)}, ensure_ascii=False)
            yield f"data: {error_event}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive'
        }
    )


@v1_bp.route('/pdf/resolve', methods=['POST'])
def api_pdf_resolve():
    """Resolve a source_file path to its corresponding PDF URL.

    Extracts the LM code from the source markdown path and searches
    the ``documents/ncs/pdfs/`` directory tree for a matching PDF.
    """
    try:
        data = request.get_json()
        if not data:
            return error_response('요청 데이터가 없습니다.', 400)

        source_file = data.get('source_file', '').strip()
        if not source_file:
            return error_response('source_file이 필요합니다.', 400)

        # Extract LM code (e.g. LM1903060409)
        lm_match = re.search(r'(LM\d+)', source_file)
        if not lm_match:
            return success_response(data={'found': False, 'pdf_url': None})

        lm_code = lm_match.group(1)

        # Extract category from source path: ncs/data/<category>/LM.../...
        parts = source_file.replace('\\', '/').split('/')
        category = None
        for i, part in enumerate(parts):
            if part == 'data' and i + 1 < len(parts):
                category = parts[i + 1]
                break

        pdfs_dir = DOCUMENTS_PATH / 'ncs' / 'pdfs'
        found_pdf = None

        # Search category subdirectory first (both NFC and NFD normalizations)
        if category:
            for norm_form in ('NFC', 'NFD'):
                category_dir = pdfs_dir / unicodedata.normalize(norm_form, category)
                try:
                    if category_dir.exists():
                        for pdf_file in category_dir.iterdir():
                            if pdf_file.suffix.lower() == '.pdf' and lm_code in pdf_file.name:
                                found_pdf = pdf_file
                                break
                except PermissionError:
                    logging.warning(f"[API/pdf/resolve] Permission denied: {category_dir}")
                    continue
                if found_pdf:
                    break

        # Fallback: search all pdfs subdirectories
        if not found_pdf and pdfs_dir.exists():
            try:
                for pdf_file in pdfs_dir.rglob('*.pdf'):
                    if lm_code in pdf_file.name:
                        found_pdf = pdf_file
                        break
            except PermissionError:
                logging.warning(f"[API/pdf/resolve] Permission denied during rglob: {pdfs_dir}")

        if found_pdf:
            relative_path = found_pdf.relative_to(DOCUMENTS_PATH)
            # Encode each path component individually to handle special characters (#, ?, &, spaces)
            encoded_parts = '/'.join(quote(part, safe='') for part in relative_path.parts)
            pdf_url = f'/documents/{encoded_parts}'
            return success_response(data={
                'found': True,
                'pdf_url': pdf_url,
                'filename': found_pdf.name,
            })

        return success_response(data={'found': False, 'pdf_url': None})

    except Exception as e:
        logging.exception(f"[API/pdf/resolve] Error: {str(e)}")
        return error_response(str(e), 500)
