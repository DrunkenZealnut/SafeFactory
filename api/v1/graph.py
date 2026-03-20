"""Knowledge graph visualization API — AI-generated nodes/edges."""

import json
import logging
import re

from flask import request

from api.response import error_response, success_response
from api.v1 import v1_bp
from models import GraphEdge, GraphNode, db

# ── Public: graph data for homepage ──


@v1_bp.route('/graph/data', methods=['GET'])
def api_graph_data():
    """Return graph nodes and edges for the homepage visualization."""
    ns = request.args.get('namespace', 'default')
    nodes = GraphNode.query.filter_by(namespace=ns).all()
    edges = GraphEdge.query.filter_by(namespace=ns).all()

    if not nodes:
        return success_response(data={'nodes': [], 'edges': [], 'empty': True})

    return success_response(data={
        'nodes': [
            {
                'id': n.node_id, 'l': n.label, 'd': n.description,
                't': n.node_type, 'c': n.color, 'r': n.radius,
                'o': n.order_num, 'p': n.parent_node_id,
            }
            for n in nodes
        ],
        'edges': [
            {'s': e.source_id, 't': e.target_id, 'flow': e.is_flow}
            for e in edges
        ],
    })


# ── Admin: AI graph generation ──


def _admin_required_check():
    """Quick admin check — returns error response or None."""
    from flask_login import current_user
    if not current_user.is_authenticated or current_user.role != 'admin':
        return error_response('관리자 권한이 필요합니다.', 403)
    return None


@v1_bp.route('/admin/graph/generate', methods=['POST'])
def api_graph_generate():
    """Generate graph nodes/edges from a seed term using Gemini."""
    err = _admin_required_check()
    if err:
        return err

    body = request.get_json(silent=True) or {}
    seed = (body.get('seed') or '').strip()
    ns = body.get('namespace', 'default').strip()

    if not seed:
        return error_response('시드 단어를 입력해주세요.', 400)

    # Gemini call
    try:
        from services.singletons import get_gemini_client
        client = get_gemini_client()

        prompt = f"""당신은 교육용 지식 그래프 생성 전문가입니다.
주제: "{seed}"

이 주제에 대한 지식 그래프 데이터를 JSON으로 생성해주세요.

규칙:
1. 주요 카테고리 (type: "m") — 6~10개, 핵심 분류
2. 세부 항목 (type: "s") — 각 주요 카테고리 하위에 2~4개
3. 상세 기술 (type: "d") — 각 세부 항목 하위에 2~3개
4. 공유 기술 (type: "sh") — 여러 카테고리에 걸치는 공통 개념 3~5개
5. id는 영문 소문자+숫자, 2~8자
6. color: 주요(#cba6f7), 세부(#89b4fa), 상세(#f9e2af), 공유(#a6e3a1)
7. radius: 주요(24), 세부(14), 상세(9), 공유(15)
8. 주요 카테고리 간 flow edge 연결 (순서대로)
9. 부모-자식 관계 edge 추가
10. 공유 기술은 관련 주요 카테고리와 edge 연결

JSON 형식 (반드시 이 형식만 출력):
```json
{{
  "nodes": [
    {{"id": "abc", "label": "이름", "desc": "설명", "type": "m", "color": "#cba6f7", "radius": 24, "order": 1, "parent": null}},
    {{"id": "def", "label": "이름", "desc": "설명", "type": "s", "color": "#89b4fa", "radius": 14, "order": null, "parent": "abc"}}
  ],
  "edges": [
    {{"source": "abc", "target": "ghi", "flow": true}},
    {{"source": "abc", "target": "def", "flow": false}}
  ]
}}
```

JSON만 출력하세요. 설명이나 마크다운 없이 순수 JSON만."""

        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
        )
        raw = response.text.strip()

        # Extract JSON from possible markdown code block
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw)
        if json_match:
            raw = json_match.group(1).strip()

        graph_data = json.loads(raw)
        gen_nodes = graph_data.get('nodes', [])
        gen_edges = graph_data.get('edges', [])

        if not gen_nodes:
            return error_response('AI가 유효한 그래프를 생성하지 못했습니다.', 500)

    except json.JSONDecodeError as e:
        logging.exception('Graph AI JSON parse failed')
        return error_response(f'AI 응답 파싱 실패: {e}', 500)
    except Exception as e:
        logging.exception('Graph AI generation failed')
        return error_response(f'AI 그래프 생성 실패: {e}', 500)

    # Clear existing data for this namespace
    GraphEdge.query.filter_by(namespace=ns).delete()
    GraphNode.query.filter_by(namespace=ns).delete()

    # Insert new nodes
    for n in gen_nodes:
        node = GraphNode(
            node_id=n['id'],
            label=n['label'],
            description=n.get('desc', ''),
            node_type=n.get('type', 's'),
            color=n.get('color', '#89b4fa'),
            radius=n.get('radius', 14),
            order_num=n.get('order'),
            parent_node_id=n.get('parent'),
            namespace=ns,
        )
        db.session.add(node)

    # Insert new edges
    for e in gen_edges:
        edge = GraphEdge(
            source_id=e['source'],
            target_id=e['target'],
            is_flow=bool(e.get('flow', False)),
            namespace=ns,
        )
        db.session.add(edge)

    try:
        db.session.commit()
    except Exception:
        db.session.rollback()
        logging.exception('Graph save failed')
        return error_response('그래프 저장 실패', 500)

    return success_response(
        data={'nodes': len(gen_nodes), 'edges': len(gen_edges)},
        message=f'"{seed}" 주제로 {len(gen_nodes)}개 노드, {len(gen_edges)}개 엣지 생성 완료',
    )


@v1_bp.route('/admin/graph/clear', methods=['POST'])
def api_graph_clear():
    """Clear all graph data for a namespace."""
    err = _admin_required_check()
    if err:
        return err

    body = request.get_json(silent=True) or {}
    ns = body.get('namespace', 'default').strip()

    GraphEdge.query.filter_by(namespace=ns).delete()
    GraphNode.query.filter_by(namespace=ns).delete()

    try:
        db.session.commit()
    except Exception:
        db.session.rollback()
        return error_response('삭제 실패', 500)

    return success_response(message=f'네임스페이스 "{ns}" 그래프 초기화 완료')
