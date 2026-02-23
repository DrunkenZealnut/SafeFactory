"""Search, ask, ask/stream (SSE), and PDF resolve endpoints."""

import json
import logging
import re
import threading
import time
import unicodedata
from urllib.parse import quote
from flask import request, Response, stream_with_context

from api.v1 import v1_bp
from api.response import success_response, error_response
from api import rate_limit
from services.settings import get_setting
from services.singletons import get_agent, get_anthropic_client, get_gemini_client, get_hybrid_searcher_instance, get_openai_client
from services.rag_pipeline import run_rag_pipeline, build_llm_messages, find_related_images
from services.domain_config import DOCUMENTS_PATH

# ---------------------------------------------------------------------------
# PDF index cache – avoids repeated rglob over large directories
# ---------------------------------------------------------------------------
_pdf_index = {}          # lm_code -> Path
_pdf_index_lock = threading.Lock()
_pdf_index_ts = 0.0
_PDF_INDEX_TTL = 300     # seconds
_HEARTBEAT_INTERVAL = 15  # SSE heartbeat interval (seconds)


def _get_answer_max_tokens(namespace: str, source_count: int) -> int:
    """Return LLM max_tokens budget based on domain and source count."""
    if namespace == 'laborlaw':
        return min(6000, 2000 + source_count * 250)
    return min(4000, 1500 + source_count * 200)


def _get_pdf_index():
    """Return a dict mapping LM codes to PDF paths, rebuilt every TTL seconds."""
    global _pdf_index, _pdf_index_ts
    now = time.monotonic()
    if _pdf_index and (now - _pdf_index_ts) < _PDF_INDEX_TTL:
        return _pdf_index
    with _pdf_index_lock:
        # Double-check after acquiring lock (recapture time to avoid stale value)
        if _pdf_index and (time.monotonic() - _pdf_index_ts) < _PDF_INDEX_TTL:
            return _pdf_index
        pdfs_dir = DOCUMENTS_PATH / 'ncs' / 'pdfs'
        index = {}
        if pdfs_dir.exists():
            try:
                for pdf_file in pdfs_dir.rglob('*.pdf'):
                    m = re.search(r'(LM\d+)', pdf_file.name)
                    if m:
                        index[m.group(1)] = pdf_file
            except PermissionError:
                logging.warning("[PDF index] Permission denied: %s", pdfs_dir)
        _pdf_index = index
        _pdf_index_ts = time.monotonic()
        return _pdf_index


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


_VALID_MODEL_RE = re.compile(r'^[a-zA-Z0-9._-]+$')


def _prepare_gemini_params(messages, temperature, max_tokens):
    """Extract Gemini-compatible params from OpenAI-style messages."""
    system_msg = next((m['content'] for m in messages if m['role'] == 'system'), '')
    user_msg = next((m['content'] for m in messages if m['role'] == 'user'), '')
    config = {
        'system_instruction': system_msg,
        'temperature': temperature,
        'max_output_tokens': max_tokens,
    }
    return user_msg, config


@v1_bp.route('/search', methods=['POST'])
@rate_limit("30 per minute")
def api_search():
    """Search for similar content with optional hybrid/keyword modes."""
    try:
        data = request.get_json(silent=True)
        if not data:
            return error_response('요청 데이터가 없습니다.', 400)
        query = data.get('query', '').strip()
        try:
            top_k = max(1, min(int(data.get('top_k', 5)), 100))
        except (ValueError, TypeError):
            top_k = 5
        namespace = data.get('namespace', '')
        if namespace == 'all':
            namespace = ''
        file_type = data.get('file_type', '')
        search_mode = data.get('search_mode', 'vector')  # vector | hybrid | keyword
        if search_mode not in ('vector', 'hybrid', 'keyword'):
            return error_response('search_mode 값이 올바르지 않습니다.', 400)

        # 3-level metadata filters
        domain = data.get('domain', '')
        category = data.get('category', '')
        subcategory = data.get('subcategory', '')

        if not query:
            return error_response('검색어를 입력해주세요.', 400)
        if len(query) > 500:
            return error_response('검색어는 500자 이하여야 합니다.', 400)

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
    except Exception:
        logging.exception('Search failed')
        return error_response('검색 중 오류가 발생했습니다.', 500)


@v1_bp.route('/ask', methods=['POST'])
@rate_limit("20 per minute")
def api_ask():
    """RAG endpoint - search and generate comprehensive answer with enhanced retrieval."""
    try:
        data = request.get_json(silent=True)
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

        # ========================================
        # Phase 8: Generate Answer via LLM (OpenAI or Gemini)
        # ========================================
        calc_result = pipeline.get('labor_calc_result')
        law_refs_formatted = pipeline.get('law_references_formatted')
        labor_classification = pipeline.get('labor_classification')
        legal_analysis = pipeline.get('legal_analysis')
        messages = build_llm_messages(query, sources, context, namespace,
                                      calc_result, law_refs_formatted,
                                      labor_classification, legal_analysis)
        provider = get_setting('llm_answer_provider', 'openai')
        model = get_setting('llm_answer_model', 'gpt-4o-mini')
        if not _VALID_MODEL_RE.match(model):
            logging.error('Invalid model name in settings: %s', model)
            return error_response('잘못된 모델 설정입니다. 관리자에게 문의하세요.', 500)
        try:
            temperature = float(get_setting('llm_answer_temperature', '0.3'))
        except (ValueError, TypeError):
            temperature = 0.3
        answer_max_tokens = _get_answer_max_tokens(namespace, len(sources))

        if provider == 'gemini':
            gemini = get_gemini_client()
            user_msg, config = _prepare_gemini_params(messages, temperature, answer_max_tokens)
            gemini_resp = gemini.models.generate_content(
                model=model, contents=user_msg, config=config,
            )
            answer = gemini_resp.text or ''
        elif provider == 'anthropic':
            client = get_anthropic_client()
            system_msg = next((m['content'] for m in messages if m['role'] == 'system'), '')
            user_msgs = [m for m in messages if m['role'] != 'system']
            response = client.messages.create(
                model=model,
                system=system_msg,
                messages=user_msgs,
                temperature=temperature,
                max_tokens=answer_max_tokens,
            )
            answer = response.content[0].text if response.content else ''
        else:
            client = get_openai_client()
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=answer_max_tokens,
            )
            answer = response.choices[0].message.content

        related_images = _collect_related_images(sources)
        law_refs_raw = pipeline.get('law_references')

        return success_response(data={
            'query': query,
            'answer': answer,
            'sources': sources,
            'source_count': len(sources),
            'images': related_images,
            'enhancement_used': use_enhancement,
            'query_variations': enhanced_queries if use_enhancement else None,
            'keywords_extracted': keywords if use_enhancement else None,
            'law_references': law_refs_raw if law_refs_raw else None,
        })

    except Exception:
        logging.exception('[API/ask] Error')
        return error_response('답변 생성 중 오류가 발생했습니다.', 500)


@v1_bp.route('/ask/stream', methods=['POST'])
@rate_limit("20 per minute")
def api_ask_stream():
    """SSE streaming RAG endpoint - sends metadata first, then streams LLM answer token by token."""
    try:
        data = request.get_json(silent=True)
        if not data:
            return error_response('요청 데이터가 없습니다.', 400)
        pipeline = run_rag_pipeline(data)
    except ValueError as e:
        return error_response(str(e), 400)
    except Exception:
        logging.exception('[API/ask/stream] Pipeline error')
        return error_response('스트리밍 응답 생성 중 오류가 발생했습니다.', 500)

    if pipeline.get('early_response'):
        return success_response(data=pipeline['data'])

    query = pipeline['query']
    namespace = pipeline['namespace']
    context = pipeline['context']
    sources = pipeline['sources']
    enhanced_queries = pipeline['enhanced_queries']
    keywords = pipeline['keywords']
    use_enhancement = pipeline['use_enhancement']

    calc_result = pipeline.get('labor_calc_result')
    law_refs_formatted = pipeline.get('law_references_formatted')
    law_refs_raw = pipeline.get('law_references')
    labor_classification = pipeline.get('labor_classification')
    legal_analysis = pipeline.get('legal_analysis')
    messages = build_llm_messages(query, sources, context, namespace,
                                  calc_result, law_refs_formatted,
                                  labor_classification, legal_analysis)
    provider = get_setting('llm_answer_provider', 'openai')
    model = get_setting('llm_answer_model', 'gpt-4o-mini')
    if not _VALID_MODEL_RE.match(model):
        logging.error('Invalid model name in settings: %s', model)
        return error_response('잘못된 모델 설정입니다. 관리자에게 문의하세요.', 500)
    try:
        temperature = float(get_setting('llm_answer_temperature', '0.3'))
    except (ValueError, TypeError):
        temperature = 0.3

    def generate():
        try:
            related_images = _collect_related_images(sources)

            # Laborlaw gets higher token budget for richer analysis
            # Send calculation result event if available (before metadata)
            if calc_result:
                calc_event = json.dumps({
                    'type': 'calculation',
                    'data': {
                        'calc_type': calc_result.get('calc_type'),
                        'input_summary': calc_result.get('input_summary'),
                        'formatted': calc_result.get('formatted'),
                    }
                }, ensure_ascii=False)
                yield f"data: {calc_event}\n\n"

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
                    'keywords_extracted': keywords if use_enhancement else None,
                    'law_references': law_refs_raw if law_refs_raw else None,
                }
            }, ensure_ascii=False)
            yield f"data: {metadata_event}\n\n"

            # Dynamic max_tokens: scale with source count for complex multi-doc answers
            answer_max_tokens = _get_answer_max_tokens(namespace, len(sources))
            last_heartbeat = time.monotonic()

            def _maybe_heartbeat():
                """Yield an SSE comment to keep the connection alive."""
                nonlocal last_heartbeat
                now = time.monotonic()
                if now - last_heartbeat >= _HEARTBEAT_INTERVAL:
                    last_heartbeat = now
                    return ": heartbeat\n\n"
                return None

            if provider == 'gemini':
                # Stream LLM response via Gemini
                gemini = get_gemini_client()
                user_msg, config = _prepare_gemini_params(messages, temperature, answer_max_tokens)
                stream = gemini.models.generate_content_stream(
                    model=model, contents=user_msg, config=config,
                )
                for chunk in stream:
                    if chunk.text:
                        token_event = json.dumps({'type': 'token', 'data': chunk.text}, ensure_ascii=False)
                        yield f"data: {token_event}\n\n"
                        last_heartbeat = time.monotonic()
                    else:
                        hb = _maybe_heartbeat()
                        if hb:
                            yield hb
            elif provider == 'anthropic':
                # Stream LLM response via Anthropic Claude
                client = get_anthropic_client()
                system_msg = next((m['content'] for m in messages if m['role'] == 'system'), '')
                user_msgs = [m for m in messages if m['role'] != 'system']
                with client.messages.stream(
                    model=model,
                    system=system_msg,
                    messages=user_msgs,
                    temperature=temperature,
                    max_tokens=answer_max_tokens,
                ) as stream:
                    for text in stream.text_stream:
                        token_event = json.dumps({'type': 'token', 'data': text}, ensure_ascii=False)
                        yield f"data: {token_event}\n\n"
                        last_heartbeat = time.monotonic()
            else:
                # Stream LLM response via OpenAI
                client = get_openai_client()
                stream = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=answer_max_tokens,
                    stream=True,
                )
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        token = chunk.choices[0].delta.content
                        token_event = json.dumps({'type': 'token', 'data': token}, ensure_ascii=False)
                        yield f"data: {token_event}\n\n"
                        last_heartbeat = time.monotonic()
                    else:
                        hb = _maybe_heartbeat()
                        if hb:
                            yield hb

            # Send done event
            done_event = json.dumps({
                'type': 'done',
                'data': {}
            }, ensure_ascii=False)
            yield f"data: {done_event}\n\n"

        except Exception:
            logging.exception("[API/ask/stream] Streaming error")
            error_event = json.dumps({'type': 'error', 'data': '답변 생성 중 오류가 발생했습니다.'}, ensure_ascii=False)
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
        data = request.get_json(silent=True)
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
                    logging.warning("[API/pdf/resolve] Permission denied: %s", category_dir)
                    continue
                if found_pdf:
                    break

        # Fallback: lookup from cached PDF index (avoids rglob on every request)
        if not found_pdf:
            pdf_index = _get_pdf_index()
            found_pdf = pdf_index.get(lm_code)

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

    except Exception:
        logging.exception('[API/pdf/resolve] Error')
        return error_response('PDF 조회 중 오류가 발생했습니다.', 500)
