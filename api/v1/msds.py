"""MSDS (Material Safety Data Sheets) endpoints."""

import json
import logging
import threading
from flask import request

from api.v1 import v1_bp
from api.response import success_response, error_response
from services.singletons import get_openai_client
from msds_client import MsdsApiClient

_msds_client = None
_msds_lock = threading.Lock()


def _get_msds_client():
    global _msds_client
    if _msds_client is None:
        with _msds_lock:
            if _msds_client is None:
                _msds_client = MsdsApiClient()
    return _msds_client


@v1_bp.route('/msds/search', methods=['POST'])
def msds_search():
    """Search chemicals in KOSHA MSDS database."""
    try:
        data = request.json
        if not data:
            return error_response('요청 데이터가 없습니다.', 400)
        search_word = data.get('search_word', '')
        try:
            search_type = int(data.get('search_type', 0))  # 0=국문명, 1=CAS, 2=UN, 3=KE, 4=EN
            page_no = int(data.get('page_no', 1))
            num_of_rows = int(data.get('num_of_rows', 10))
        except (ValueError, TypeError):
            return error_response('잘못된 숫자 형식입니다.', 400)

        if not search_word:
            return error_response('검색어를 입력해주세요.', 400)

        msds_client = _get_msds_client()
        result = msds_client.search_chemicals(
            search_word=search_word,
            search_type=search_type,
            page_no=page_no,
            num_of_rows=num_of_rows
        )

        return success_response(data=result)

    except Exception:
        logging.exception('MSDS API error')
        return error_response('MSDS 조회 중 오류가 발생했습니다.', 500)


@v1_bp.route('/msds/detail', methods=['POST'])
def msds_detail():
    """Get detailed chemical information from KOSHA MSDS."""
    try:
        data = request.json
        if not data:
            return error_response('요청 데이터가 없습니다.', 400)
        chem_id = data.get('chem_id', '')
        section = data.get('section', '')  # Optional: specific section (01-16)

        if not chem_id:
            return error_response('화학물질 ID를 입력해주세요.', 400)

        msds_client = _get_msds_client()

        # If section is specified, get that section only
        if section:
            result = msds_client.get_chemical_detail(chem_id, section)
        else:
            # Get all sections
            result = msds_client.get_full_chemical_detail(chem_id)

        return success_response(data=result)

    except Exception:
        logging.exception('MSDS API error')
        return error_response('MSDS 조회 중 오류가 발생했습니다.', 500)


@v1_bp.route('/msds/identify', methods=['POST'])
def msds_identify():
    """Identify chemical substance from image using OpenAI Vision API."""
    result_text = None
    try:
        data = request.json
        if not data:
            return error_response('요청 데이터가 없습니다.', 400)
        image_data = data.get('image', '')  # Base64 encoded image

        if not image_data:
            return error_response('이미지 데이터가 없습니다.', 400)

        # Extract MIME type and remove data URL prefix if present
        mime_type = "image/jpeg"
        if ',' in image_data:
            prefix, image_data = image_data.split(',', 1)
            if 'png' in prefix:
                mime_type = "image/png"
            elif 'gif' in prefix:
                mime_type = "image/gif"
            elif 'webp' in prefix:
                mime_type = "image/webp"

        # Use OpenAI Vision API to identify chemical
        client = get_openai_client()

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """이미지에서 화학물질 정보를 찾아주세요.

다음 정보를 추출해주세요:
1. 화학물질명 (국문명 또는 영문명)
2. CAS 번호 (있는 경우)
3. 제품명 또는 상품명 (있는 경우)

응답 JSON 형식:
- 정보를 찾은 경우: {"chemical_name": "화학물질명", "cas_no": "CAS번호 또는 null", "product_name": "제품명 또는 null"}
- 찾을 수 없는 경우: {"chemical_name": null, "cas_no": null, "product_name": null, "error": "화학물질 정보를 찾을 수 없습니다."}"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=500
        )

        # Parse response - guaranteed JSON by response_format
        result_text = response.choices[0].message.content.strip()
        result = json.loads(result_text)

        msds_client = _get_msds_client()

        # If chemical name found, search for it
        if result.get('chemical_name'):
            search_word = result['chemical_name']
            search_type = 0  # Default to Korean name

            # If CAS number is available, use that for search
            if result.get('cas_no'):
                search_word = result['cas_no']
                search_type = 1  # CAS number search

            # Search in MSDS database
            search_result = msds_client.search_chemicals(
                search_word=search_word,
                search_type=search_type,
                page_no=1,
                num_of_rows=10
            )

            return success_response(data={
                'identified': result,
                'search_result': search_result
            })
        else:
            return error_response(
                result.get('error', '화학물질 정보를 찾을 수 없습니다.'), 404,
                details={'identified': result}
            )

    except json.JSONDecodeError as e:
        logging.error("[API/msds/identify] JSON decode error: %s, raw: %.500s", e, result_text)
        return error_response('응답 파싱 실패', 422)
    except Exception:
        logging.exception('MSDS API error')
        return error_response('MSDS 조회 중 오류가 발생했습니다.', 500)
