#!/usr/bin/env python3
"""법령 API 검증 스크립트: 36개 법률 + 시행령 fetching 테스트.

검증 항목:
1. _LAW_REGISTRY에 모든 법률이 등록되어 있는지
2. law.go.kr에서 법률 본문을 가져올 수 있는지
3. law.go.kr에서 시행령 본문을 가져올 수 있는지
4. 조문 파싱이 정상 동작하는지

Usage:
    python test_law_verification.py
    python test_law_verification.py --fetch   # 실제 API 호출 포함
"""

import re
import sys
import time
import logging
import argparse
import requests
import xml.etree.ElementTree as ET

sys.path.insert(0, '.')
from services.law_api import _LAW_REGISTRY

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 검증 대상: 36개 법률 목록 (filename_translator_agent.py의 TRANSLATIONS)
# ---------------------------------------------------------------------------
TARGET_LAWS = [
    "근로기준법",
    "노동조합 및 노동관계조정법",
    "산업안전보건법",
    "산업재해보상보험법",
    "고용보험법",
    "최저임금법",
    "남녀고용평등과 일ㆍ가정 양립 지원에 관한 법률",
    "직업안정법",
    "고용정책 기본법",
    "근로자퇴직급여 보장법",
    "근로복지기본법",
    "근로자참여 및 협력증진에 관한 법률",
    "노동위원회법",
    "파견근로자 보호 등에 관한 법률",
    "기간제 및 단시간근로자 보호 등에 관한 법률",
    "외국인근로자의 고용 등에 관한 법률",
    "장애인고용촉진 및 직업재활법",
    "고용상 연령차별금지 및 고령자고용촉진에 관한 법률",
    "중대재해 처벌 등에 관한 법률",
    "진폐의 예방과 진폐근로자의 보호 등에 관한 법률",
    "공무원의 노동조합 설립 및 운영 등에 관한 법률",
    "교원의 노동조합 설립 및 운영 등에 관한 법률",
    "구직자 취업촉진 및 생활안정지원에 관한 법률",
    "임금채권보장법",
    "채용절차의 공정화에 관한 법률",
    "사회적기업 육성법",
    "숙련기술장려법",
    "산업현장 일학습병행 지원에 관한 법률",
    "고용보험 및 산업재해보상보험의 보험료징수 등에 관한 법률",
    "선원법",
    "어선원 및 어선 재해보상보험법",
    "공무원 재해보상법",
    "공무원연금법",
    "국가인권위원회법",
    "영화 및 비디오물의 진흥에 관한 법률",
    "대중문화예술산업발전법",
    "전공의의 수련환경 개선 및 지위 향상을 위한 법률",
]

# ---------------------------------------------------------------------------
# services/law_api.py의 현재 _LAW_REGISTRY (비교용)
# ---------------------------------------------------------------------------
CURRENT_LAW_REGISTRY = {
    '근로기준법': {'lsiSeq': '265959', 'efYd': '20251023'},
    '근로자퇴직급여 보장법': {'lsiSeq': '266139', 'efYd': '20251111'},
    '남녀고용평등과 일ㆍ가정 양립 지원에 관한 법률': {
        'lsiSeq': '276851', 'efYd': '20251001',
    },
    '고용보험법': {'lsiSeq': '266295', 'efYd': '20251119'},
    '최저임금법': {'lsiSeq': '218303', 'efYd': '20200526'},
    '산업안전보건법': {'lsiSeq': '276853', 'efYd': '20251001'},
    '산업재해보상보험법': {'lsiSeq': '265977', 'efYd': '20250101'},
    '기간제 및 단시간근로자 보호 등에 관한 법률': {
        'lsiSeq': '232201', 'efYd': '20210518',
    },
    '파견근로자 보호 등에 관한 법률': {
        'lsiSeq': '223983', 'efYd': '20201208',
    },
    '직업안정법': {'lsiSeq': '259231', 'efYd': '20240724'},
}


# ---------------------------------------------------------------------------
# law.go.kr Open API를 사용한 법령 검색
# ---------------------------------------------------------------------------
class LawGoKrVerifier:
    """law.go.kr 법령 Open API + AJAX API를 사용한 법령 정보 검증."""

    # 법제처 법령 검색 Open API (인증 불필요)
    SEARCH_URL = 'http://www.law.go.kr/DRF/lawSearch.do'
    # 법제처 법령 상세 조회 Open API
    SERVICE_URL = 'http://www.law.go.kr/DRF/lawService.do'
    # 법제처 법령 본문 AJAX (lsInfoR.do)
    INFO_URL = 'https://www.law.go.kr/LSW/lsInfoR.do'

    TIMEOUT = 25

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                           'AppleWebKit/537.36 (KHTML, like Gecko)'),
        })
        self.results = {}

    def search_law(self, law_name: str, law_type: str = '법률') -> dict | None:
        """법제처 Open API로 법령을 검색하여 lsiSeq, efYd 등을 가져온다.

        law_type: '법률' or '시행령' or '시행규칙'
        """
        # 시행령인 경우 법명에 '시행령' 붙이기
        search_query = law_name
        if law_type == '시행령':
            search_query = law_name + ' 시행령'
        elif law_type == '시행규칙':
            search_query = law_name + ' 시행규칙'

        params = {
            'OC': 'test',  # Open API 사용자 ID (공개)
            'target': 'law',
            'type': 'XML',
            'query': search_query,
            'display': 5,
            'sort': 'lawNm',
        }

        try:
            resp = self.session.get(self.SEARCH_URL, params=params, timeout=self.TIMEOUT)
            resp.raise_for_status()

            root = ET.fromstring(resp.content)
            total = root.findtext('.//totalCnt', '0')

            for law_elem in root.findall('.//law'):
                found_name = law_elem.findtext('법령명한글', '')
                lsi_seq = law_elem.findtext('법령일련번호', '')
                ef_yd = law_elem.findtext('시행일자', '')
                law_id = law_elem.findtext('법령ID', '')
                promulgation_date = law_elem.findtext('공포일자', '')
                law_num = law_elem.findtext('법령구분명', '')  # 법률, 대통령령, 총리령 등

                # 정확히 매칭되는지 확인
                if law_type == '법률':
                    if found_name == law_name and law_num in ('법률', ''):
                        return {
                            'name': found_name,
                            'lsiSeq': lsi_seq,
                            'efYd': ef_yd,
                            'lawId': law_id,
                            'promulgation_date': promulgation_date,
                            'law_type': law_num or '법률',
                            'total_results': total,
                        }
                elif law_type == '시행령':
                    expected_name = law_name + ' 시행령'
                    if found_name == expected_name and law_num in ('대통령령', ''):
                        return {
                            'name': found_name,
                            'lsiSeq': lsi_seq,
                            'efYd': ef_yd,
                            'lawId': law_id,
                            'promulgation_date': promulgation_date,
                            'law_type': law_num or '대통령령',
                            'total_results': total,
                        }

            # 정확 매칭 실패시 부분 매칭 시도
            for law_elem in root.findall('.//law'):
                found_name = law_elem.findtext('법령명한글', '')
                lsi_seq = law_elem.findtext('법령일련번호', '')
                ef_yd = law_elem.findtext('시행일자', '')
                law_id = law_elem.findtext('법령ID', '')
                law_num = law_elem.findtext('법령구분명', '')

                if law_type == '법률' and law_name in found_name:
                    return {
                        'name': found_name,
                        'lsiSeq': lsi_seq,
                        'efYd': ef_yd,
                        'lawId': law_id,
                        'law_type': law_num or '법률',
                        'total_results': total,
                        'partial_match': True,
                    }
                elif law_type == '시행령' and '시행령' in found_name and law_name.split()[0] in found_name:
                    return {
                        'name': found_name,
                        'lsiSeq': lsi_seq,
                        'efYd': ef_yd,
                        'lawId': law_id,
                        'law_type': law_num or '대통령령',
                        'total_results': total,
                        'partial_match': True,
                    }

            return None

        except Exception as e:
            logger.warning("  검색 실패 [%s %s]: %s", law_name, law_type, e)
            return None

    def fetch_articles_by_lsi(self, lsi_seq: str, ef_yd: str) -> dict[str, str]:
        """lsiSeq와 efYd를 사용하여 law.go.kr AJAX API에서 조문을 가져온다."""
        params = {
            'lsiSeq': lsi_seq,
            'lsId': '',
            'efYd': ef_yd,
            'chrClsCd': '010202',
            'urlMode': 'lsInfoR',
            'viewCls': 'lsInfoR',
            'ancYnChk': '0',
        }

        try:
            resp = self.session.get(
                self.INFO_URL, params=params,
                timeout=self.TIMEOUT,
                headers={
                    'X-Requested-With': 'XMLHttpRequest',
                    'Referer': 'https://www.law.go.kr/',
                }
            )
            resp.raise_for_status()
            return self._parse_articles(resp.text)
        except Exception as e:
            logger.warning("  조문 가져오기 실패 (lsiSeq=%s): %s", lsi_seq, e)
            return {}

    @staticmethod
    def _parse_articles(html: str) -> dict[str, str]:
        """HTML에서 조문 파싱 (law_api.py의 LawTextFetcher._parse_articles와 동일)."""
        # 부칙 제거
        for marker in ('<!-- 부칙 영역', '<div id="arDivArea">', '<a name="arArea">'):
            idx = html.find(marker)
            if idx > 0:
                html = html[:idx]
                break

        text = re.sub(r'<br\s*/?>', '\n', html)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'&nbsp;', ' ', text)
        text = re.sub(r'&amp;', '&', text)
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)
        text = re.sub(r'&[a-z]+;', '', text)
        text = re.sub(r'[ \t]+', ' ', text)

        article_pattern = r'(제\d+조(?:의\d+)?)\s*[\(（]([^)）]+)[\)）]'
        matches = list(re.finditer(article_pattern, text))

        articles: dict[str, str] = {}
        for i, match in enumerate(matches):
            article_num = match.group(1)
            article_title = match.group(2).strip()
            start = match.start()
            end = (matches[i + 1].start()
                   if i + 1 < len(matches) else start + 3000)
            content = text[start:end].strip()
            content = content.strip()
            key = f'{article_num}({article_title})'
            if key not in articles:
                articles[key] = content[:800]

        return articles


def check_registry_coverage():
    """검증 1: _LAW_REGISTRY 등록 여부 확인."""
    print("\n" + "=" * 80)
    print("검증 1: _LAW_REGISTRY 등록 여부 확인")
    print("=" * 80)

    registered = []
    missing = []
    decree_registered = []
    decree_missing = []

    for law_name in TARGET_LAWS:
        if law_name in _LAW_REGISTRY:
            registered.append(law_name)
        else:
            missing.append(law_name)

        decree_key = f"{law_name} 시행령"
        if decree_key in _LAW_REGISTRY:
            decree_registered.append(decree_key)
        else:
            decree_missing.append(decree_key)

    print(f"\n법률 등록됨: {len(registered)}/{len(TARGET_LAWS)}")
    for name in registered:
        info = _LAW_REGISTRY[name]
        print(f"  [O] {name} (lsiSeq={info['lsiSeq']}, efYd={info['efYd']})")

    if missing:
        print(f"\n법률 미등록: {len(missing)}/{len(TARGET_LAWS)}")
        for name in missing:
            print(f"  [X] {name}")

    print(f"\n시행령 등록됨: {len(decree_registered)}/{len(TARGET_LAWS)}")
    for name in decree_registered:
        info = _LAW_REGISTRY[name]
        print(f"  [O] {name} (lsiSeq={info['lsiSeq']}, efYd={info['efYd']})")

    if decree_missing:
        print(f"\n시행령 미등록: {len(decree_missing)}/{len(TARGET_LAWS)}")
        for name in decree_missing:
            print(f"  [X] {name}")

    print(f"\n총 _LAW_REGISTRY 항목 수: {len(_LAW_REGISTRY)}")

    return registered, missing


def verify_api_fetch(do_fetch: bool = False):
    """검증 2: _LAW_REGISTRY의 lsiSeq/efYd로 AJAX API 조문 파싱 테스트."""
    if not do_fetch:
        print("\n" + "=" * 80)
        print("검증 2: 실제 API 호출 테스트 (--fetch 옵션으로 실행)")
        print("=" * 80)
        print("  --fetch 옵션을 추가하면 _LAW_REGISTRY의 lsiSeq/efYd로")
        print("  law.go.kr AJAX API에서 조문 파싱을 검증합니다.")
        return {}

    verifier = LawGoKrVerifier()
    results = {}

    # 검증 2: 법률 조문 가져오기 (_LAW_REGISTRY 사용)
    print("\n" + "=" * 80)
    print("검증 2: 법률 조문 파싱 테스트 (AJAX API, _LAW_REGISTRY 기반)")
    print("=" * 80)

    law_success = 0
    for i, law_name in enumerate(TARGET_LAWS, 1):
        info = _LAW_REGISTRY.get(law_name)
        if not info:
            print(f"\n[{i:02d}/{len(TARGET_LAWS)}] {law_name} → 미등록 (skip)")
            results[law_name] = {'law_articles': {}, 'decree_articles': {}}
            continue

        articles = verifier.fetch_articles_by_lsi(info['lsiSeq'], info['efYd'])
        article_count = len(articles)
        if article_count > 0:
            law_success += 1
            sample_keys = list(articles.keys())[:3]
            print(f"[{i:02d}/{len(TARGET_LAWS)}] {law_name} → ✅ {article_count}개 조문")
        else:
            print(f"[{i:02d}/{len(TARGET_LAWS)}] {law_name} → ❌ 파싱 실패")

        results[law_name] = {'law_articles': articles, 'decree_articles': {}}
        time.sleep(0.3)

    print(f"\n법률 조문 파싱: {law_success}/{len(TARGET_LAWS)}")

    # 검증 3: 시행령 조문 가져오기
    print("\n" + "=" * 80)
    print("검증 3: 시행령 조문 파싱 테스트 (AJAX API, _LAW_REGISTRY 기반)")
    print("=" * 80)

    decree_success = 0
    for i, law_name in enumerate(TARGET_LAWS, 1):
        decree_key = f"{law_name} 시행령"
        info = _LAW_REGISTRY.get(decree_key)
        if not info:
            print(f"[{i:02d}/{len(TARGET_LAWS)}] {decree_key} → 미등록 (skip)")
            continue

        articles = verifier.fetch_articles_by_lsi(info['lsiSeq'], info['efYd'])
        article_count = len(articles)
        if article_count > 0:
            decree_success += 1
            print(f"[{i:02d}/{len(TARGET_LAWS)}] {decree_key} → ✅ {article_count}개 조문")
        else:
            print(f"[{i:02d}/{len(TARGET_LAWS)}] {decree_key} → ❌ 파싱 실패")

        results[law_name]['decree_articles'] = articles
        time.sleep(0.3)

    print(f"\n시행령 조문 파싱: {decree_success}/{len(TARGET_LAWS)}")

    return results


def print_summary(registered, missing, results):
    """결과 요약 출력."""
    print("\n" + "=" * 80)
    print("종합 검증 결과")
    print("=" * 80)

    total = len(TARGET_LAWS)

    # _LAW_REGISTRY 현황
    decree_count = sum(1 for name in TARGET_LAWS if f"{name} 시행령" in _LAW_REGISTRY)
    print(f"\n[1] _LAW_REGISTRY 등록 현황")
    print(f"    - 법률 등록: {len(registered)}/{total}개")
    print(f"    - 법률 미등록: {len(missing)}/{total}개")
    print(f"    - 시행령 등록: {decree_count}/{total}개")
    print(f"    - 총 항목 수: {len(_LAW_REGISTRY)}개")

    if results:
        # 조문 파싱 성공률
        law_parsed = sum(1 for r in results.values()
                         if r.get('law_articles'))
        decree_parsed = sum(1 for r in results.values()
                            if r.get('decree_articles'))

        print(f"\n[2] AJAX API 조문 파싱 결과")
        print(f"    - 법률 조문 파싱 성공: {law_parsed}/{total}개")
        print(f"    - 시행령 조문 파싱 성공: {decree_parsed}/{total}개")

        # 실패 항목 표시
        law_failed = [name for name in TARGET_LAWS
                      if not results.get(name, {}).get('law_articles')]
        decree_failed = [name for name in TARGET_LAWS
                         if not results.get(name, {}).get('decree_articles')]

        if law_failed:
            print(f"\n    법률 파싱 실패:")
            for name in law_failed:
                print(f"      - {name}")

        if decree_failed:
            print(f"\n    시행령 파싱 실패:")
            for name in decree_failed:
                print(f"      - {name} 시행령")

    # 상태 요약
    print(f"\n{'=' * 80}")
    print("상태 요약")
    print("=" * 80)

    if len(missing) == 0 and decree_count == total:
        print(f"\n✅ 모든 법률({total}개) 및 시행령({decree_count}개) 등록 완료!")
    else:
        if len(missing) > 0:
            print(f"\n⚠️  {len(missing)}개 법률 미등록")
        if decree_count < total:
            print(f"\n⚠️  {total - decree_count}개 시행령 미등록")

    print(f"\n참고: lsiSeq/efYd 값은 법 개정 시 변경될 수 있음"
          f"\n  → python test_law_verification.py --fetch 로 정기 검증 권장")


def main():
    parser = argparse.ArgumentParser(description='법령 API 검증 스크립트')
    parser.add_argument('--fetch', action='store_true',
                        help='실제 law.go.kr API 호출 포함')
    args = parser.parse_args()

    print("=" * 80)
    print("법령 API 종합 검증")
    print(f"대상: {len(TARGET_LAWS)}개 법률 (법률 + 시행령)")
    print("=" * 80)

    # 검증 1: Registry 확인
    registered, missing = check_registry_coverage()

    # 검증 2-4: API 호출
    results = verify_api_fetch(do_fetch=args.fetch)

    # 종합 결과
    print_summary(registered, missing, results)


if __name__ == '__main__':
    main()
