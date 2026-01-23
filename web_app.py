#!/usr/bin/env python3
"""
Pinecone Agent Web Interface
Flask-based web UI for Pinecone vector database operations with RAG support.
"""

import os
import re
import json
import unicodedata
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)

# Global instances (lazy initialization)
_agent = None
_openai_client = None

def get_openai_client():
    """Get or create OpenAI client."""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client

def get_agent():
    """Get or create the PineconeAgent instance."""
    global _agent
    if _agent is None:
        from src.agent import PineconeAgent
        _agent = PineconeAgent(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            pinecone_api_key=os.getenv("PINECONE_API_KEY"),
            pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "document-index"),
            create_index_if_not_exists=False
        )
    return _agent

def get_uploader():
    """Get PineconeUploader for stats."""
    from src.pinecone_uploader import PineconeUploader
    return PineconeUploader(
        api_key=os.getenv("PINECONE_API_KEY"),
        index_name=os.getenv("PINECONE_INDEX_NAME", "document-index"),
        create_if_not_exists=False
    )


def parse_mentions(query):
    """Parse @mentions from query and return (clean_query, filters).

    Supports:
    - @파일명.md - specific file
    - @폴더명/ - folder path (ends with /)
    - @키워드 - partial match on source_file
    """
    mentions = re.findall(r'@([^\s@]+)', query)
    clean_query = re.sub(r'@[^\s@]+', '', query).strip()

    filters = []
    for mention in mentions:
        if mention.endswith('/'):
            # Folder filter
            filters.append({'type': 'folder', 'value': mention.rstrip('/')})
        elif '.' in mention:
            # File filter (has extension)
            filters.append({'type': 'file', 'value': mention})
        else:
            # Keyword filter
            filters.append({'type': 'keyword', 'value': mention})

    return clean_query, filters


def build_source_filter(filters):
    """Build Pinecone filter from parsed mentions.

    Note: Pinecone doesn't support substring matching directly,
    so we'll filter results post-query.
    """
    # For now, return None and do post-filtering
    # Pinecone filter would require exact match or $in operator
    return None


@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')


@app.route('/api/stats')
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
                    'vector_count': ns_info.vector_count
                })

        return jsonify({
            'success': True,
            'data': {
                'index_name': os.getenv("PINECONE_INDEX_NAME", "document-index"),
                'dimension': stats.get('dimension', 0),
                'total_vectors': stats.get('total_vector_count', 0),
                'namespaces': namespaces
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/search', methods=['POST'])
def api_search():
    """Search for similar content."""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        top_k = int(data.get('top_k', 5))
        namespace = data.get('namespace', '')
        file_type = data.get('file_type', '')

        if not query:
            return jsonify({'success': False, 'error': '검색어를 입력해주세요.'})

        agent = get_agent()

        # Build filter
        filter_dict = None
        if file_type:
            filter_dict = {"file_type": file_type}

        results = agent.search(
            query=query,
            top_k=top_k,
            namespace=namespace,
            filter=filter_dict
        )

        # Format results for frontend
        formatted_results = []
        for r in results:
            metadata = r.get('metadata', {})
            formatted_results.append({
                'score': round(r.get('score', 0), 4),
                'source_file': metadata.get('source_file', 'N/A'),
                'file_type': metadata.get('file_type', 'N/A'),
                'content': metadata.get('content', '')[:500],
                'filename': metadata.get('filename', ''),
                'relative_path': metadata.get('relative_path', '')
            })

        return jsonify({
            'success': True,
            'data': {
                'query': query,
                'count': len(formatted_results),
                'results': formatted_results
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/ask', methods=['POST'])
def api_ask():
    """RAG endpoint - search and generate comprehensive answer."""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        namespace = data.get('namespace', '')
        top_k = int(data.get('top_k', 10))  # More documents for better context

        if not query:
            return jsonify({'success': False, 'error': '질문을 입력해주세요.'})

        # Parse @mentions for source filtering
        clean_query, mention_filters = parse_mentions(query)

        # Build search query: include filter keywords to improve relevance
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

        # Step 1: Search for relevant documents (fetch more if filtering)
        # When filtering, we need to fetch significantly more results
        search_top_k = top_k * 5 if mention_filters else top_k
        results = agent.search(
            query=search_query,
            top_k=search_top_k,
            namespace=namespace
        )

        # Step 1.5: Apply mention filters (post-query filtering)
        # Note: Use Unicode NFC normalization to handle Korean character encoding differences
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

            results = filtered_results[:top_k]

        if not results:
            return jsonify({
                'success': True,
                'data': {
                    'answer': '관련 문서를 찾을 수 없습니다. 다른 검색어로 시도해주세요.',
                    'sources': []
                }
            })

        # Step 2: Build context from search results
        context_parts = []
        sources = []

        for i, r in enumerate(results):
            metadata = r.get('metadata', {})
            content = metadata.get('content', '')
            source_file = metadata.get('source_file', 'Unknown')
            file_type = metadata.get('file_type', 'unknown')
            score = r.get('score', 0)

            if content:
                context_parts.append(f"[문서 {i+1}] (출처: {source_file})\n{content}")
                sources.append({
                    'source_file': source_file,
                    'file_type': file_type,
                    'score': round(score, 4),
                    'content_preview': content[:200] + '...' if len(content) > 200 else content
                })

        context = "\n\n---\n\n".join(context_parts)

        # Step 3: Generate comprehensive answer using GPT
        system_prompt = """당신은 반도체 기술 전문가입니다.
제공된 문서들을 바탕으로 사용자의 질문에 대해 종합적이고 정확한 답변을 제공합니다.

답변 작성 지침:
1. 제공된 문서 내용을 기반으로 답변하세요
2. **중요**: 각 정보의 출처를 반드시 인용 번호로 표시하세요. 예: "CVD 공정은 화학 기상 증착 방식입니다[1]."
3. 인용 형식: 문장 끝에 [1], [2] 등의 번호를 붙여 어떤 문서에서 가져온 정보인지 명시하세요
4. 여러 문서의 내용을 종합할 때는 [1][3]처럼 복수 인용도 가능합니다
5. 기술 용어는 한글과 영문을 병기하세요
6. 핵심 포인트를 명확하게 구분하세요
7. 문서에 없는 내용은 추측하지 마세요
8. 마크다운 형식으로 가독성 있게 작성하세요"""

        user_prompt = f"""## 질문
{query}

## 참고 문서
{context}

위 문서들을 참고하여 질문에 대해 종합적으로 답변해주세요.
**반드시 각 정보의 출처를 [1], [2] 등의 인용 번호로 표시하세요.**"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )

        answer = response.choices[0].message.content

        return jsonify({
            'success': True,
            'data': {
                'query': query,
                'answer': answer,
                'sources': sources,
                'source_count': len(sources)
            }
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/namespaces')
def api_namespaces():
    """Get list of namespaces."""
    try:
        uploader = get_uploader()
        stats = uploader.get_stats()

        namespaces = []
        if stats.get('namespaces'):
            for ns_name in stats['namespaces'].keys():
                namespaces.append(ns_name if ns_name else '')

        return jsonify({
            'success': True,
            'data': namespaces
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/sources')
def api_sources():
    """Get list of available source files and folders for autocomplete."""
    try:
        namespace = request.args.get('namespace', '')
        agent = get_agent()

        # Search with a generic query to get sample of documents
        results = agent.search(
            query="반도체 공정 기술",
            top_k=100,
            namespace=namespace
        )

        folders = set()
        files = set()

        for r in results:
            metadata = r.get('metadata', {})
            source_file = metadata.get('source_file', '')
            filename = metadata.get('filename', '')

            if filename:
                files.add(filename)

            if source_file:
                # Extract folder paths
                parts = source_file.split('/')
                for i in range(1, len(parts)):
                    folder = '/'.join(parts[:i])
                    if folder:
                        folders.add(folder)

        return jsonify({
            'success': True,
            'data': {
                'folders': sorted(folders),
                'files': sorted(files)
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/delete', methods=['POST'])
def api_delete():
    """Delete vectors."""
    try:
        data = request.get_json()
        namespace = data.get('namespace', '')
        delete_all = data.get('delete_all', False)
        source_file = data.get('source_file', '')

        uploader = get_uploader()

        if delete_all:
            uploader.index.delete(delete_all=True, namespace=namespace)
            return jsonify({
                'success': True,
                'message': f"네임스페이스 '{namespace or '(기본)'}' 의 모든 벡터가 삭제되었습니다."
            })
        elif source_file:
            success = uploader.delete_by_filter(
                filter={"source_file": source_file},
                namespace=namespace
            )
            if success:
                return jsonify({
                    'success': True,
                    'message': f"'{source_file}'의 벡터가 삭제되었습니다."
                })
            else:
                return jsonify({'success': False, 'error': '삭제 실패'})
        else:
            return jsonify({'success': False, 'error': '삭제할 대상을 지정해주세요.'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    # Check environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        exit(1)
    if not os.getenv("PINECONE_API_KEY"):
        print("Error: PINECONE_API_KEY not set")
        exit(1)

    print("🚀 Pinecone Agent Web Interface")
    print("=" * 40)
    print(f"Index: {os.getenv('PINECONE_INDEX_NAME', 'document-index')}")
    print("=" * 40)
    print("\n🌐 http://localhost:5001 에서 접속하세요\n")

    app.run(debug=True, host='0.0.0.0', port=5001)
