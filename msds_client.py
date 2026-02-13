"""
MSDS API Client (Python)
산업안전보건공단 물질안전보건자료 API 클라이언트
"""

import os
import requests
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional


class MsdsApiClient:
    """KOSHA MSDS API Client"""

    # API 설정
    ENDPOINT = 'https://msds.kosha.or.kr/openapi/service/msdschem'
    API_KEY = os.environ.get('MSDS_API_KEY', '')

    # 검색 조건
    SEARCH_BY_NAME = 0  # 국문명
    SEARCH_BY_CAS = 1   # CAS No
    SEARCH_BY_UN = 2    # UN No
    SEARCH_BY_KE = 3    # KE No
    SEARCH_BY_EN = 4    # EN No

    # 상세정보 섹션
    DETAIL_SECTIONS = {
        '01': '화학제품과 회사에 관한 정보',
        '02': '유해성·위험성',
        '03': '구성성분의 명칭 및 함유량',
        '04': '응급조치요령',
        '05': '폭발·화재시 대처방법',
        '06': '누출사고시 대처방법',
        '07': '취급 및 저장방법',
        '08': '노출방지 및 개인보호구',
        '09': '물리화학적 특성',
        '10': '안정성 및 반응성',
        '11': '독성에 관한 정보',
        '12': '환경에 미치는 영향',
        '13': '폐기시 주의사항',
        '14': '운송에 필요한 정보',
        '15': '법적 규제현황',
        '16': '그 밖의 참고사항'
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def _parse_xml_to_dict(self, element: ET.Element) -> Dict:
        """XML 요소를 딕셔너리로 변환"""
        result = {}

        # 텍스트 내용
        if element.text and element.text.strip():
            return element.text.strip()

        # 자식 요소 처리
        for child in element:
            child_data = self._parse_xml_to_dict(child)

            if child.tag in result:
                # 이미 존재하면 리스트로 변환
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_data)
            else:
                result[child.tag] = child_data

        return result

    def _request(self, path: str, params: Dict = None) -> Optional[Dict]:
        """API 요청 실행"""
        if params is None:
            params = {}

        params['serviceKey'] = self.API_KEY
        url = f"{self.ENDPOINT}{path}"

        try:
            # SSL 인증서 검증 활성화 - 보안을 위해 필수
            response = self.session.get(url, params=params, timeout=30, verify=True)
            response.raise_for_status()

            # XML 파싱
            root = ET.fromstring(response.content)
            return self._parse_xml_to_dict(root)

        except requests.exceptions.RequestException as e:
            print(f"MSDS API Request Error: {e}")
            return None
        except ET.ParseError as e:
            print(f"XML Parse Error: {e}")
            return None

    def search_chemicals(
        self,
        search_word: str,
        search_type: int = 0,
        page_no: int = 1,
        num_of_rows: int = 10
    ) -> Dict:
        """
        화학물질 검색

        Args:
            search_word: 검색어
            search_type: 검색 조건 (0: 국문명, 1: CAS No, 2: UN No, 3: KE No, 4: EN No)
            page_no: 페이지 번호
            num_of_rows: 페이지당 결과 수

        Returns:
            검색 결과 딕셔너리
        """
        params = {
            'searchWrd': search_word,
            'searchCnd': search_type,
            'pageNo': page_no,
            'numOfRows': num_of_rows
        }

        response = self._request('/chemlist', params)

        if not response:
            return {
                'success': False,
                'message': 'API 요청 실패',
                'items': [],
                'totalCount': 0,
                'pageNo': page_no,
                'numOfRows': num_of_rows
            }

        # 응답 파싱
        header = response.get('header', {})
        result_code = header.get('resultCode', '')

        if result_code != '00':
            return {
                'success': False,
                'message': header.get('resultMsg', 'API 오류'),
                'items': [],
                'totalCount': 0,
                'pageNo': page_no,
                'numOfRows': num_of_rows
            }

        body = response.get('body', {})
        items_data = body.get('items', {})
        items = items_data.get('item', [])

        # 단일 결과인 경우 리스트로 변환
        if isinstance(items, dict):
            items = [items]
        elif items is None:
            items = []

        return {
            'success': True,
            'items': items,
            'totalCount': int(body.get('totalCount', 0)),
            'pageNo': int(body.get('pageNo', page_no)),
            'numOfRows': int(body.get('numOfRows', num_of_rows))
        }

    def get_chemical_detail(self, chem_id: str, section: str) -> Dict:
        """
        화학물질 상세정보 조회

        Args:
            chem_id: 화학물질 ID
            section: 섹션 번호 (01-16)

        Returns:
            상세정보 딕셔너리
        """
        response = self._request(f'/chemdetail{section}', {'chemId': chem_id})

        if not response:
            return {
                'success': False,
                'message': 'API 요청 실패',
                'items': []
            }

        header = response.get('header', {})
        result_code = header.get('resultCode', '')

        if result_code != '00':
            return {
                'success': False,
                'message': header.get('resultMsg', 'API 오류'),
                'items': []
            }

        body = response.get('body', {})
        items_data = body.get('items', {})
        items = items_data.get('item', [])

        # 단일 결과인 경우 리스트로 변환
        if isinstance(items, dict):
            items = [items]
        elif items is None:
            items = []

        return {
            'success': True,
            'section': section,
            'section_name': self.DETAIL_SECTIONS.get(section, ''),
            'items': items
        }

    def get_full_chemical_detail(self, chem_id: str) -> Dict:
        """
        화학물질 전체 상세정보 조회 (모든 섹션)

        Args:
            chem_id: 화학물질 ID

        Returns:
            전체 상세정보 딕셔너리
        """
        result = {
            'success': True,
            'chem_id': chem_id,
            'sections': {}
        }

        for section_code in self.DETAIL_SECTIONS.keys():
            detail = self.get_chemical_detail(chem_id, section_code)
            if detail['success']:
                result['sections'][section_code] = {
                    'name': detail['section_name'],
                    'items': detail['items']
                }

        return result

    def organize_detail_items(self, items: List[Dict]) -> List[Dict]:
        """상세정보 아이템을 계층 구조로 정리"""
        organized = []

        for item in items:
            organized.append({
                'level': int(item.get('lev', 1)),
                'code': item.get('msdsItemCode', ''),
                'name': item.get('msdsItemNameKor', ''),
                'detail': item.get('itemDetail', '')
            })

        return organized
