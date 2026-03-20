"""NCS category browser API — list categories and modules."""

import json
import os
import re
import time

from flask import request

from api.response import success_response
from api.v1 import v1_bp

# Cache
_cache = {'data': None, 'ts': 0}
_CACHE_TTL = 300  # 5 minutes

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NCS_DATA_DIR = os.path.join(_BASE_DIR, 'documents', 'semiconductor', 'ncs', 'data')
NCS_JSON_FALLBACK = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ncs_modules.json')

NCS_CATEGORIES = ['반도체개발', '반도체장비', '반도체재료', '반도체제조']

CATEGORY_META = {
    '반도체개발': {'icon': '💡', 'desc': '소자 설계, 회로 설계, 검증 등'},
    '반도체장비': {'icon': '⚙️', 'desc': '장비 설계, 제어, 유지보수 등'},
    '반도체재료': {'icon': '🧪', 'desc': '웨이퍼, 가스, 화학재료 제조 등'},
    '반도체제조': {'icon': '🏭', 'desc': 'Photo, Etch, 증착, 배선 공정 등'},
}


def _parse_module_name(dirname):
    """Extract human-readable title from NCS directory name.

    Example: 'LM1903060101_23v6_반도체_제품_기획' -> ('LM1903060101', '반도체 제품 기획')
    """
    # Extract LM code
    code_match = re.match(r'(LM\d+)', dirname)
    code = code_match.group(1) if code_match else ''

    # Extract title part (after version info like _23v6_)
    title_match = re.search(r'_\d+v\d+_(.*)', dirname)
    if title_match:
        title = title_match.group(1).replace('_', ' ')
    else:
        # Fallback: remove LM code prefix
        title = re.sub(r'^LM\d+_?\d*v?\d*_?', '', dirname).replace('_', ' ')

    return code, title.strip() or dirname


def _scan_categories():
    """Scan NCS data directory or load from JSON fallback."""
    now = time.time()
    if _cache['data'] and now - _cache['ts'] < _CACHE_TTL:
        return _cache['data']

    categories = []

    # Try filesystem first
    if os.path.isdir(NCS_DATA_DIR):
        for cat_name in NCS_CATEGORIES:
            cat_path = os.path.join(NCS_DATA_DIR, cat_name)
            if not os.path.isdir(cat_path):
                continue

            meta = CATEGORY_META.get(cat_name, {})
            modules = []
            try:
                for entry in sorted(os.listdir(cat_path)):
                    entry_path = os.path.join(cat_path, entry)
                    if os.path.isdir(entry_path):
                        code, title = _parse_module_name(entry)
                        modules.append({
                            'code': code,
                            'title': title,
                            'dirname': entry,
                        })
            except OSError:
                pass

            categories.append({
                'name': cat_name,
                'icon': meta.get('icon', '📁'),
                'desc': meta.get('desc', ''),
                'module_count': len(modules),
                'modules': modules,
            })

    # Fallback: load from bundled JSON
    if not categories and os.path.isfile(NCS_JSON_FALLBACK):
        try:
            with open(NCS_JSON_FALLBACK, encoding='utf-8') as f:
                raw = json.load(f)
            for cat in raw:
                meta = CATEGORY_META.get(cat['name'], {})
                categories.append({
                    'name': cat['name'],
                    'icon': meta.get('icon', '📁'),
                    'desc': meta.get('desc', ''),
                    'module_count': len(cat.get('modules', [])),
                    'modules': cat.get('modules', []),
                })
        except Exception:
            pass

    _cache['data'] = categories
    _cache['ts'] = now
    return categories


@v1_bp.route('/ncs/categories', methods=['GET'])
def api_ncs_categories():
    """List NCS categories and their modules."""
    categories = _scan_categories()
    total = sum(c['module_count'] for c in categories)
    return success_response(data={
        'categories': categories,
        'total_modules': total,
    })
