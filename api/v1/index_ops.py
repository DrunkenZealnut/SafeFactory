"""Index operations: stats, namespaces, sources, delete."""

import logging
import os
from flask import request

from api.v1 import v1_bp
from api.v1.admin import admin_required
from api.response import success_response, error_response
from services.singletons import get_agent, get_uploader

# Fallback queries for vector-based source discovery when SQLite metadata is empty.
# Configurable for multi-language or domain-specific deployments.
FALLBACK_SOURCE_QUERIES = os.environ.get(
    'SOURCE_FALLBACK_QUERIES', '문서,정보',
).split(',')


@v1_bp.route('/stats')
def api_stats():
    """Get index statistics."""
    try:
        uploader = get_uploader()
        stats = uploader.get_stats()

        # Format namespaces for frontend
        namespaces = []
        if stats.get('namespaces'):
            for ns_name, ns_info in stats['namespaces'].items():
                namespaces.append({
                    'name': ns_name if ns_name else '(기본)',
                    'vector_count': ns_info.get('vector_count', 0)
                })

        return success_response(data={
            'index_name': os.getenv("PINECONE_INDEX_NAME", "document-index"),
            'dimension': stats.get('dimension', 0),
            'total_vectors': stats.get('total_vector_count', 0),
            'namespaces': namespaces
        })
    except Exception:
        logging.exception('Internal error')
        return error_response('서버 내부 오류가 발생했습니다.', 500)


@v1_bp.route('/namespaces')
def api_namespaces():
    """Get list of namespaces."""
    try:
        uploader = get_uploader()
        stats = uploader.get_stats()

        namespaces = []
        if stats.get('namespaces'):
            for ns_name in stats['namespaces'].keys():
                namespaces.append(ns_name if ns_name else '(기본)')

        return success_response(data=namespaces)
    except Exception:
        logging.exception('Internal error')
        return error_response('서버 내부 오류가 발생했습니다.', 500)


@v1_bp.route('/sources')
def api_sources():
    """Get list of available source files and folders for autocomplete."""
    try:
        namespace = request.args.get('namespace', '')
        agent = get_agent()

        folders = set()
        files = set()

        # Primary: use SQLite metadata for a complete source list
        if agent.metadata_manager:
            records = agent.metadata_manager.get_all_metadata(
                namespace=namespace or None,
            )
            for rec in records:
                source_file = rec.get('source_file', '')
                if source_file:
                    # Extract filename from path
                    fname = source_file.rsplit('/', 1)[-1] if '/' in source_file else source_file
                    if fname:
                        files.add(fname)
                    parts = source_file.split('/')
                    for i in range(1, len(parts)):
                        folder = '/'.join(parts[:i])
                        if folder:
                            folders.add(folder)

        # Fallback: vector search if metadata unavailable or returned nothing
        if not files:
            all_results = []
            for query in FALLBACK_SOURCE_QUERIES:
                results = agent.search(query=query, top_k=50, namespace=namespace)
                all_results.extend(results)

            seen = set()
            for r in all_results:
                source_file = r.get('metadata', {}).get('source_file', '')
                if source_file and source_file not in seen:
                    seen.add(source_file)
                    filename = r.get('metadata', {}).get('filename', '')
                    if filename:
                        files.add(filename)
                    parts = source_file.split('/')
                    for i in range(1, len(parts)):
                        folder = '/'.join(parts[:i])
                        if folder:
                            folders.add(folder)

        return success_response(data={
            'folders': sorted(folders),
            'files': sorted(files)
        })
    except Exception:
        logging.exception('Internal error')
        return error_response('서버 내부 오류가 발생했습니다.', 500)


@v1_bp.route('/delete', methods=['POST'])
@admin_required
def api_delete():
    """Delete vectors."""
    try:
        data = request.get_json(silent=True)
        if not data:
            return error_response('요청 데이터가 없습니다.', 400)
        namespace = data.get('namespace', '')
        delete_all = data.get('delete_all', False)
        confirm = data.get('confirm', False)
        source_file = data.get('source_file', '')

        uploader = get_uploader()
        if not getattr(uploader, 'index', None):
            return error_response('인덱스에 연결할 수 없습니다.', 500)

        if delete_all:
            if not confirm:
                return error_response('전체 삭제를 확인하려면 confirm=true를 포함해주세요.', 400)
            uploader.index.delete(delete_all=True, namespace=namespace)
            return success_response(message=f"네임스페이스 '{namespace or '(기본)'}' 의 모든 벡터가 삭제되었습니다.")
        elif source_file:
            success = uploader.delete_by_filter(
                filter={"source_file": source_file},
                namespace=namespace
            )
            if success:
                return success_response(message=f"'{source_file}'의 벡터가 삭제되었습니다.")
            else:
                return error_response('삭제 실패', 500)
        else:
            return error_response('삭제할 대상을 지정해주세요.', 400)

    except Exception:
        logging.exception('Internal error')
        return error_response('서버 내부 오류가 발생했습니다.', 500)
