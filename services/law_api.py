"""노동법 법령 검색 API — 사용자 질문에 관련 법률 조문을 자동으로 찾아 제공한다.

전체 흐름:
  사용자 질문
    → (1) 키워드 매칭: 질문 텍스트에서 법률 용어를 직접 찾음 (빠름, 비용 없음)
    → (2) LLM 토픽 추출: Gemini flash로 구어체/자연어에서 법률 키워드 추출 (느리지만 정확)
    → (3) 조문 가져오기: 매칭된 법률의 조문을 law.go.kr AJAX API에서 fetch
    → (4) 포맷: 번호가 매겨진 마크다운으로 포맷하여 LLM 프롬프트에 주입

데이터 소스:
  1. 법제처 법령 본문 (law.go.kr AJAX) — 전문 조문 텍스트 (주 데이터 소스)
  2. 고용노동부 법령 API (api.odcloud.kr) — 법령명 + 조문 제목 (보조/폴백)

지원 법률: 37개 법률 + 37개 시행령 = 74개 항목 (_LAW_REGISTRY)
"""

import os
import logging
import re
import time
import threading

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Calc-type → API search keywords
# ---------------------------------------------------------------------------
# NOTE: API only contains 고용노동부 소관 법령 (76 laws).
# 최저임금법, 국민연금법, 국민건강보험법, 소득세법, 산업재해보상보험법 등은 미포함.
_CALC_TYPE_API_KEYWORDS: dict[str, list[str]] = {
    'wage': ['근로기준법 임금'],
    'wage_reverse': ['근로기준법 임금'],
    'insurance': ['고용보험법 보험료', '고용보험 보험료징수'],
    'minimum_wage': ['근로기준법 임금'],  # 최저임금법 미포함 → 근로기준법 임금 참조
    'overtime': ['근로기준법 연장근로', '근로기준법 가산'],
    'weekly_holiday': ['근로기준법 휴일'],
    'severance': ['퇴직급여 퇴직금', '근로기준법 퇴직'],
    'annual_leave': ['근로기준법 연차'],
    'income_tax': ['근로기준법 임금'],  # 소득세법 미포함 → 근로기준법 참조
}

# ---------------------------------------------------------------------------
# 법제처 법령 본문 조회 (law.go.kr AJAX API)
# ---------------------------------------------------------------------------
# 법령 레지스트리: 법률명 → {lsiSeq, efYd}
# ---------------------------------------------------------------------------
# law.go.kr AJAX API(lsInfoR.do)로 조문을 가져올 때 lsiSeq(법령일련번호)와
# efYd(시행일자)가 필요하다. 37개 법률 + 37개 시행령 = 74개 항목.
# 시행령 키 형식: "법명 시행령" (예: "근로기준법 시행령")
#
# 갱신 시기: 법 개정 시 lsiSeq/efYd가 변경될 수 있음
# 갱신 방법: python test_law_verification.py --fetch 로 전체 검증
# 최종 검증: 2026-02-23 (74/74 조문 파싱 성공)
_LAW_REGISTRY: dict[str, dict[str, str]] = {
    # ── 핵심 노동법 ──
    '근로기준법': {'lsiSeq': '265959', 'efYd': '20251023'},
    '근로기준법 시행령': {'lsiSeq': '270551', 'efYd': '20251023'},
    '근로자퇴직급여 보장법': {'lsiSeq': '279829', 'efYd': '20251111'},
    '근로자퇴직급여 보장법 시행령': {'lsiSeq': '262801', 'efYd': '20240528'},
    '최저임금법': {'lsiSeq': '218303', 'efYd': '20200526'},
    '최저임금법 시행령': {'lsiSeq': '206564', 'efYd': '20190101'},
    '임금채권보장법': {'lsiSeq': '259881', 'efYd': '20240807'},
    '임금채권보장법 시행령': {'lsiSeq': '281625', 'efYd': '20260102'},
    # ── 고용/보험 ──
    '고용보험법': {'lsiSeq': '276843', 'efYd': '20251001'},
    '고용보험법 시행령': {'lsiSeq': '281219', 'efYd': '20260102'},
    '산업재해보상보험법': {'lsiSeq': '279733', 'efYd': '20260212'},
    '산업재해보상보험법 시행령': {'lsiSeq': '281227', 'efYd': '20260102'},
    '고용보험 및 산업재해보상보험의 보험료징수 등에 관한 법률': {
        'lsiSeq': '247481', 'efYd': '20240101',
    },
    '고용보험 및 산업재해보상보험의 보험료징수 등에 관한 법률 시행령': {
        'lsiSeq': '280527', 'efYd': '20251223',
    },
    '고용정책 기본법': {'lsiSeq': '276847', 'efYd': '20260102'},
    '고용정책 기본법 시행령': {'lsiSeq': '281221', 'efYd': '20260102'},
    '직업안정법': {'lsiSeq': '259231', 'efYd': '20240724'},
    '직업안정법 시행령': {'lsiSeq': '267797', 'efYd': '20241231'},
    # ── 근로조건/비정규직 ──
    '남녀고용평등과 일ㆍ가정 양립 지원에 관한 법률': {
        'lsiSeq': '276851', 'efYd': '20251001',
    },
    '남녀고용평등과 일ㆍ가정 양립 지원에 관한 법률 시행령': {
        'lsiSeq': '277437', 'efYd': '20251001',
    },
    '기간제 및 단시간근로자 보호 등에 관한 법률': {
        'lsiSeq': '232201', 'efYd': '20210518',
    },
    '기간제 및 단시간근로자 보호 등에 관한 법률 시행령': {
        'lsiSeq': '230833', 'efYd': '20210408',
    },
    '파견근로자 보호 등에 관한 법률': {
        'lsiSeq': '223983', 'efYd': '20201208',
    },
    '파견근로자 보호 등에 관한 법률 시행령': {
        'lsiSeq': '272523', 'efYd': '20250621',
    },
    '외국인근로자의 고용 등에 관한 법률': {
        'lsiSeq': '276857', 'efYd': '20251001',
    },
    '외국인근로자의 고용 등에 관한 법률 시행령': {
        'lsiSeq': '244037', 'efYd': '20230203',
    },
    '채용절차의 공정화에 관한 법률': {'lsiSeq': '218301', 'efYd': '20200526'},
    '채용절차의 공정화에 관한 법률 시행령': {'lsiSeq': '209867', 'efYd': '20190717'},
    # ── 산업안전/중대재해 ──
    '산업안전보건법': {'lsiSeq': '276853', 'efYd': '20251001'},
    '산업안전보건법 시행령': {'lsiSeq': '277411', 'efYd': '20251001'},
    '중대재해 처벌 등에 관한 법률': {'lsiSeq': '228817', 'efYd': '20220127'},
    '중대재해 처벌 등에 관한 법률 시행령': {'lsiSeq': '277417', 'efYd': '20251001'},
    '진폐의 예방과 진폐근로자의 보호 등에 관한 법률': {
        'lsiSeq': '253647', 'efYd': '20230808',
    },
    '진폐의 예방과 진폐근로자의 보호 등에 관한 법률 시행령': {
        'lsiSeq': '277419', 'efYd': '20251001',
    },
    # ── 노동조합/노사관계 ──
    '노동조합 및 노동관계조정법': {'lsiSeq': '228175', 'efYd': '20210706'},
    '노동조합 및 노동관계조정법 시행령': {'lsiSeq': '273063', 'efYd': '20250807'},
    '노동위원회법': {'lsiSeq': '232203', 'efYd': '20220519'},
    '노동위원회법 시행령': {'lsiSeq': '267225', 'efYd': '20241227'},
    '근로자참여 및 협력증진에 관한 법률': {
        'lsiSeq': '243063', 'efYd': '20221211',
    },
    '근로자참여 및 협력증진에 관한 법률 시행령': {
        'lsiSeq': '245235', 'efYd': '20221211',
    },
    '공무원의 노동조합 설립 및 운영 등에 관한 법률': {
        'lsiSeq': '243037', 'efYd': '20231211',
    },
    '공무원의 노동조합 설립 및 운영 등에 관한 법률 시행령': {
        'lsiSeq': '256437', 'efYd': '20231211',
    },
    '교원의 노동조합 설립 및 운영 등에 관한 법률': {
        'lsiSeq': '243039', 'efYd': '20231211',
    },
    '교원의 노동조합 설립 및 운영 등에 관한 법률 시행령': {
        'lsiSeq': '256439', 'efYd': '20231211',
    },
    # ── 고용촉진/복지 ──
    '장애인고용촉진 및 직업재활법': {'lsiSeq': '279737', 'efYd': '20251111'},
    '장애인고용촉진 및 직업재활법 시행령': {'lsiSeq': '281849', 'efYd': '20260102'},
    '고용상 연령차별금지 및 고령자고용촉진에 관한 법률': {
        'lsiSeq': '243057', 'efYd': '20220610',
    },
    '고용상 연령차별금지 및 고령자고용촉진에 관한 법률 시행령': {
        'lsiSeq': '267181', 'efYd': '20250101',
    },
    '구직자 취업촉진 및 생활안정지원에 관한 법률': {
        'lsiSeq': '253641', 'efYd': '20240209',
    },
    '구직자 취업촉진 및 생활안정지원에 관한 법률 시행령': {
        'lsiSeq': '260081', 'efYd': '20240209',
    },
    '근로복지기본법': {'lsiSeq': '243061', 'efYd': '20230611'},
    '근로복지기본법 시행령': {'lsiSeq': '281225', 'efYd': '20260102'},
    '사회적기업 육성법': {'lsiSeq': '122694', 'efYd': '20120802'},
    '사회적기업 육성법 시행령': {'lsiSeq': '281623', 'efYd': '20260102'},
    '숙련기술장려법': {'lsiSeq': '252725', 'efYd': '20230718'},
    '숙련기술장려법 시행령': {'lsiSeq': '230605', 'efYd': '20210401'},
    '산업현장 일학습병행 지원에 관한 법률': {
        'lsiSeq': '234773', 'efYd': '20220218',
    },
    '산업현장 일학습병행 지원에 관한 법률 시행령': {
        'lsiSeq': '282967', 'efYd': '20260201',
    },
    # ── 특수분야 ──
    '선원법': {'lsiSeq': '279821', 'efYd': '20251111'},
    '선원법 시행령': {'lsiSeq': '269731', 'efYd': '20250312'},
    '어선원 및 어선 재해보상보험법': {'lsiSeq': '259243', 'efYd': '20240724'},
    '어선원 및 어선 재해보상보험법 시행령': {'lsiSeq': '273825', 'efYd': '20251002'},
    '공무원 재해보상법': {'lsiSeq': '268859', 'efYd': '20260201'},
    '공무원 재해보상법 시행령': {'lsiSeq': '272695', 'efYd': '20250708'},
    '공무원연금법': {'lsiSeq': '277137', 'efYd': '20260102'},
    '공무원연금법 시행령': {'lsiSeq': '281715', 'efYd': '20260102'},
    '국가인권위원회법': {'lsiSeq': '266711', 'efYd': '20250604'},
    '국가인권위원회법 시행령': {'lsiSeq': '243639', 'efYd': '20220701'},
    # ── 문화/의료 ──
    '영화 및 비디오물의 진흥에 관한 법률': {'lsiSeq': '277357', 'efYd': '20251001'},
    '영화 및 비디오물의 진흥에 관한 법률 시행령': {
        'lsiSeq': '277829', 'efYd': '20251001',
    },
    '대중문화예술산업발전법': {'lsiSeq': '270187', 'efYd': '20250926'},
    '대중문화예술산업발전법 시행령': {'lsiSeq': '273025', 'efYd': '20250926'},
    '전공의의 수련환경 개선 및 지위 향상을 위한 법률': {
        'lsiSeq': '281943', 'efYd': '20260221',
    },
    '전공의의 수련환경 개선 및 지위 향상을 위한 법률 시행령': {
        'lsiSeq': '265707', 'efYd': '20241008',
    },
}

# ---------------------------------------------------------------------------
# LLM 기반 법률 토픽 추출 (구어체/자연어 → 법률 키워드)
# ---------------------------------------------------------------------------
# 사용자가 "나오지 말라", "잘렸어" 등 구어체를 써도 "해고" 관련 조문을 찾을 수 있도록
# Gemini flash로 질문에서 법률 키워드를 추출한다.
# _QUESTION_TO_ARTICLES의 키 목록을 프롬프트에 제공하여 할루시네이션을 방지하고,
# 키를 추가하면 프롬프트에 자동 반영된다.

_llm_topic_cache_lock = threading.Lock()
_llm_topic_cache: dict[str, list[str]] = {}  # 동일 query 중복 API 호출 방지


def _extract_legal_topics(query: str) -> list[str]:
    """Gemini flash로 자연어 질문에서 법률 토픽 키워드를 추출한다.

    예: "사장님이 나오지 말라고 합니다" → ['해고', '부당해고', '해고예고']
        "주급은 얼마인가요" → ['주휴', '주급', '임금']

    _QUESTION_TO_ARTICLES의 키워드 중에서만 선택하도록 프롬프트에 후보 목록을 제공.
    실패 시 빈 리스트를 반환하여 기존 키워드 매칭에 영향 없음.

    Returns:
        _QUESTION_TO_ARTICLES에 존재하는 키워드 리스트 (최대 ~5개)
    """
    # 캐시 확인 (같은 query에 대해 search_labor_laws 내에서 2회 호출 가능)
    with _llm_topic_cache_lock:
        if query in _llm_topic_cache:
            return _llm_topic_cache[query]

    try:
        from services.singletons import get_gemini_client
        client = get_gemini_client()

        # _QUESTION_TO_ARTICLES의 키 목록을 후보로 제공
        available_keywords = ', '.join(sorted(_QUESTION_TO_ARTICLES.keys()))

        prompt = (
            "다음 노동법 관련 질문에서, 아래 후보 키워드 중 관련된 것을 골라 "
            "쉼표로 구분하여 반환하세요. 관련 없는 키워드는 절대 포함하지 마세요.\n\n"
            f"후보 키워드: {available_keywords}\n\n"
            f"질문: {query}\n\n"
            "키워드:"
        )

        resp = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
            config={'temperature': 0.0, 'max_output_tokens': 100},
        )

        text = (resp.text or '').strip()
        keywords = [kw.strip() for kw in text.split(',') if kw.strip()]
        # _QUESTION_TO_ARTICLES에 실제로 존재하는 키워드만 필터링
        valid = [kw for kw in keywords if kw in _QUESTION_TO_ARTICLES]

        if valid:
            logger.info("[LawAPI] LLM extracted topics: %s", valid)

        # 캐시 저장 (최대 50개, 오래된 것부터 제거)
        with _llm_topic_cache_lock:
            _llm_topic_cache[query] = valid
            if len(_llm_topic_cache) > 50:
                oldest = next(iter(_llm_topic_cache))
                del _llm_topic_cache[oldest]

        return valid

    except Exception as e:
        logger.debug("[LawAPI] LLM topic extraction failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# 질문 키워드 → 관련 법령 + 조문번호 매핑
# ---------------------------------------------------------------------------
# 키(keyword)가 질문 텍스트에 포함되어 있으면 해당 법률의 조문을 가져온다.
# LLM 토픽 추출 결과도 이 매핑의 키로 사용된다.
# 새 키워드를 추가하면 LLM 프롬프트에도 자동 반영된다.
# 값: [(법률명, [조문번호, ...]), ...] — 법률명은 _LAW_REGISTRY의 키와 일치해야 함.
_QUESTION_TO_ARTICLES: dict[str, list[tuple[str, list[str]]]] = {
    # ── 급여/임금 ──
    '임금': [('근로기준법', ['제2조', '제43조', '제44조'])],
    '급여': [('근로기준법', ['제2조', '제43조', '제44조'])],
    '연봉': [('근로기준법', ['제2조', '제43조'])],
    '월급': [('근로기준법', ['제2조', '제43조'])],
    '실수령': [('근로기준법', ['제2조', '제43조'])],
    '체불': [('근로기준법', ['제36조', '제37조', '제43조', '제43조의2']),
            ('임금채권보장법', ['제7조', '제8조'])],
    '통상임금': [('근로기준법', ['제2조', '제56조']),
               ('근로기준법 시행령', ['제6조'])],
    '평균임금': [('근로기준법', ['제2조']),
               ('근로기준법 시행령', ['제2조', '제3조', '제4조'])],
    # ── 근로시간/가산수당 ──
    '연장근로': [('근로기준법', ['제50조', '제53조', '제56조'])],
    '야간': [('근로기준법', ['제56조'])],
    '가산수당': [('근로기준법', ['제56조'])],
    '근로시간': [('근로기준법', ['제50조', '제51조', '제53조', '제56조'])],
    '주휴': [('근로기준법', ['제55조'])],
    '주급': [('근로기준법', ['제2조', '제43조', '제55조', '제56조'])],
    '주휴수당': [('근로기준법', ['제55조', '제56조'])],
    '탄력근로': [('근로기준법', ['제51조', '제51조의2'])],
    '선택근로': [('근로기준법', ['제52조'])],
    '휴게': [('근로기준법', ['제54조'])],
    # ── 해고 ──
    '해고': [('근로기준법', ['제23조', '제24조', '제25조', '제26조', '제27조',
                         '제28조', '제29조', '제30조']),
            ('근로기준법 시행령', ['제10조'])],
    '부당해고': [('근로기준법', ['제23조', '제26조', '제27조', '제28조',
                            '제29조', '제30조'])],
    '정리해고': [('근로기준법', ['제24조', '제25조']),
               ('근로기준법 시행령', ['제10조'])],
    '징계': [('근로기준법', ['제23조', '제27조'])],
    '해고예고': [('근로기준법', ['제26조']),
               ('근로기준법 시행령', ['제10조'])],
    '해고통보': [('근로기준법', ['제23조', '제26조', '제27조'])],
    '즉시해고': [('근로기준법', ['제26조'])],
    '구두해고': [('근로기준법', ['제23조', '제26조', '제27조'])],
    # ── 퇴직금 ──
    '퇴직': [('근로기준법', ['제34조', '제36조']),
            ('근로자퇴직급여 보장법', ['제4조', '제8조', '제9조', '제10조'])],
    '퇴직금': [('근로자퇴직급여 보장법', ['제4조', '제8조', '제9조', '제10조']),
             ('근로자퇴직급여 보장법 시행령', ['제3조'])],
    '퇴직연금': [('근로자퇴직급여 보장법', ['제13조', '제14조', '제19조', '제20조']),
               ('근로자퇴직급여 보장법 시행령', ['제2조', '제4조', '제5조'])],
    # ── 연차/휴가 ──
    '연차': [('근로기준법', ['제60조', '제61조'])],
    '휴가': [('근로기준법', ['제54조', '제55조', '제60조', '제61조', '제74조'])],
    '휴일': [('근로기준법', ['제55조', '제56조'])],
    # ── 최저임금 ──
    '최저임금': [('최저임금법', ['제4조', '제5조', '제5조의2', '제6조', '제6조의2']),
               ('근로기준법', ['제2조']),
               ('최저임금법 시행령', ['제5조의2'])],
    '최저시급': [('최저임금법', ['제4조', '제5조', '제5조의2', '제6조']),
               ('최저임금법 시행령', ['제5조의2'])],
    # ── 근로계약 ──
    '계약': [('근로기준법', ['제2조', '제15조', '제16조', '제17조'])],
    '근로계약': [('근로기준법', ['제2조', '제15조', '제16조', '제17조']),
               ('근로기준법 시행령', ['제8조'])],
    '수습': [('근로기준법', ['제35조']),
            ('최저임금법', ['제5조']),
            ('최저임금법 시행령', ['제3조'])],
    # ── 비정규직/기간제 ──
    '비정규직': [('기간제 및 단시간근로자 보호 등에 관한 법률',
                ['제2조', '제4조', '제8조', '제9조']),
               ('기간제 및 단시간근로자 보호 등에 관한 법률 시행령', ['제3조'])],
    '기간제': [('기간제 및 단시간근로자 보호 등에 관한 법률',
              ['제2조', '제4조', '제8조', '제9조']),
             ('기간제 및 단시간근로자 보호 등에 관한 법률 시행령', ['제3조'])],
    '계약직': [('기간제 및 단시간근로자 보호 등에 관한 법률',
              ['제2조', '제4조', '제8조'])],
    '단시간': [('기간제 및 단시간근로자 보호 등에 관한 법률',
              ['제2조', '제6조', '제7조']),
             ('근로기준법 시행령', ['제9조'])],
    '파견': [('파견근로자 보호 등에 관한 법률',
            ['제2조', '제5조', '제6조의2', '제21조']),
           ('파견근로자 보호 등에 관한 법률 시행령', ['제2조'])],
    '파견근로': [('파견근로자 보호 등에 관한 법률',
               ['제2조', '제5조', '제6조의2', '제21조']),
              ('파견근로자 보호 등에 관한 법률 시행령', ['제2조'])],
    # ── 육아/출산 ──
    '육아': [('남녀고용평등과 일ㆍ가정 양립 지원에 관한 법률',
             ['제19조', '제19조의2', '제19조의3']),
            ('남녀고용평등과 일ㆍ가정 양립 지원에 관한 법률 시행령',
             ['제10조', '제11조'])],
    '육아휴직': [('남녀고용평등과 일ㆍ가정 양립 지원에 관한 법률',
                ['제19조', '제19조의2', '제19조의3']),
               ('남녀고용평등과 일ㆍ가정 양립 지원에 관한 법률 시행령',
                ['제10조', '제11조'])],
    '출산': [('근로기준법', ['제74조']),
            ('남녀고용평등과 일ㆍ가정 양립 지원에 관한 법률', ['제18조', '제18조의2'])],
    '출산휴가': [('근로기준법', ['제74조']),
              ('남녀고용평등과 일ㆍ가정 양립 지원에 관한 법률', ['제18조'])],
    '배우자출산': [('남녀고용평등과 일ㆍ가정 양립 지원에 관한 법률', ['제18조의2'])],
    # ── 성차별/직장내 괴롭힘 ──
    '성희롱': [('남녀고용평등과 일ㆍ가정 양립 지원에 관한 법률',
              ['제12조', '제13조', '제14조']),
             ('남녀고용평등과 일ㆍ가정 양립 지원에 관한 법률 시행령', ['제3조'])],
    '직장내 괴롭힘': [('근로기준법', ['제76조의2', '제76조의3'])],
    '괴롭힘': [('근로기준법', ['제76조의2', '제76조의3'])],
    # ── 보험 ──
    '보험': [('고용보험법', ['제1조', '제2조', '제10조'])],
    '4대보험': [('고용보험법', ['제1조', '제2조', '제10조']),
              ('산업재해보상보험법', ['제1조', '제5조', '제6조']),
              ('고용보험법 시행령', ['제2조', '제3조'])],
    '고용보험': [('고용보험법', ['제1조', '제2조', '제10조']),
              ('고용보험법 시행령', ['제2조', '제3조'])],
    '실업급여': [('고용보험법', ['제40조', '제41조', '제42조', '제43조', '제50조']),
              ('고용보험법 시행령', ['제68조', '제69조'])],
    '보험료': [('고용보험 및 산업재해보상보험의 보험료징수 등에 관한 법률',
              ['제13조', '제14조', '제16조의2', '제16조의3'])],
    # ── 산업재해/안전 ──
    '산재': [('산업재해보상보험법', ['제1조', '제5조', '제36조', '제37조', '제40조']),
           ('산업재해보상보험법 시행령', ['제27조', '제29조'])],
    '산업재해': [('산업재해보상보험법', ['제1조', '제5조', '제36조', '제37조']),
              ('산업재해보상보험법 시행령', ['제27조'])],
    '업무상재해': [('산업재해보상보험법', ['제5조', '제37조']),
                ('산업재해보상보험법 시행령', ['제27조', '제29조'])],
    '산업안전': [('산업안전보건법', ['제5조', '제38조', '제39조', '제51조', '제52조']),
              ('산업안전보건법 시행령', ['제2조', '제47조'])],
    '안전보건': [('산업안전보건법', ['제5조', '제38조', '제39조', '제51조']),
              ('산업안전보건법 시행령', ['제14조', '제16조'])],
    '안전교육': [('산업안전보건법', ['제29조', '제31조']),
              ('산업안전보건법 시행령', ['제26조', '제27조', '제28조'])],
    '작업중지': [('산업안전보건법', ['제51조', '제52조'])],
    '위험성평가': [('산업안전보건법', ['제36조'])],
    # ── 중대재해 ──
    '중대재해': [('중대재해 처벌 등에 관한 법률', ['제2조', '제4조', '제5조', '제6조']),
              ('중대재해 처벌 등에 관한 법률 시행령',
               ['제2조', '제4조', '제5조'])],
    '중대산업재해': [('중대재해 처벌 등에 관한 법률', ['제2조', '제4조', '제5조']),
                  ('중대재해 처벌 등에 관한 법률 시행령',
                   ['제2조', '제4조'])],
    '안전보건관리체계': [('중대재해 처벌 등에 관한 법률', ['제4조']),
                     ('중대재해 처벌 등에 관한 법률 시행령', ['제4조', '제5조'])],
    # ── 노동조합/노사관계 ──
    '노동조합': [('노동조합 및 노동관계조정법',
                ['제2조', '제5조', '제7조', '제12조'])],
    '단체교섭': [('노동조합 및 노동관계조정법',
                ['제29조', '제30조', '제33조'])],
    '쟁의': [('노동조합 및 노동관계조정법',
            ['제37조', '제38조', '제42조'])],
    '부당노동행위': [('노동조합 및 노동관계조정법',
                   ['제81조', '제82조', '제83조'])],
    '노사협의회': [('근로자참여 및 협력증진에 관한 법률',
                 ['제3조', '제4조', '제5조', '제19조', '제20조'])],
    # ── 고용촉진/차별금지 ──
    '외국인근로자': [('외국인근로자의 고용 등에 관한 법률',
                   ['제2조', '제6조', '제8조', '제22조']),
                  ('외국인근로자의 고용 등에 관한 법률 시행령', ['제2조'])],
    '장애인고용': [('장애인고용촉진 및 직업재활법',
                 ['제27조', '제28조', '제33조']),
                ('장애인고용촉진 및 직업재활법 시행령', ['제2조'])],
    '고령자': [('고용상 연령차별금지 및 고령자고용촉진에 관한 법률',
              ['제4조의4', '제4조의5', '제19조']),
             ('고용상 연령차별금지 및 고령자고용촉진에 관한 법률 시행령',
              ['제2조'])],
    '연령차별': [('고용상 연령차별금지 및 고령자고용촉진에 관한 법률',
               ['제4조의4', '제4조의5'])],
    '임금채권': [('임금채권보장법', ['제7조', '제8조']),
              ('임금채권보장법 시행령', ['제5조', '제6조', '제7조'])],
    '채용절차': [('채용절차의 공정화에 관한 법률', ['제4조', '제7조', '제11조'])],
    '구직급여': [('구직자 취업촉진 및 생활안정지원에 관한 법률',
               ['제5조', '제6조', '제8조'])],
    # ── 특수분야 ──
    '선원': [('선원법', ['제2조', '제53조', '제54조', '제55조'])],
    '진폐': [('진폐의 예방과 진폐근로자의 보호 등에 관한 법률',
            ['제2조', '제5조', '제12조'])],
    '공무원노조': [('공무원의 노동조합 설립 및 운영 등에 관한 법률',
                 ['제2조', '제6조', '제8조', '제10조'])],
    '교원노조': [('교원의 노동조합 설립 및 운영 등에 관한 법률',
               ['제2조', '제6조'])],
}

# ---------------------------------------------------------------------------
# Simple TTL cache
# ---------------------------------------------------------------------------
_cache: dict[str, tuple[float, object]] = {}
_cache_lock = threading.Lock()
_CACHE_TTL = 300  # 5 minutes


def _get_cached(key: str) -> object | None:
    with _cache_lock:
        entry = _cache.get(key)
        if entry and time.time() - entry[0] < _CACHE_TTL:
            return entry[1]
        if entry:
            del _cache[key]
    return None


def _set_cached(key: str, value: object) -> None:
    with _cache_lock:
        _cache[key] = (time.time(), value)


# ---------------------------------------------------------------------------
# LawApiClient (고용노동부 법령 API via api.odcloud.kr)
# ---------------------------------------------------------------------------
class LawApiClient:
    """고용노동부 법령 API client via api.odcloud.kr (공공데이터포털 Infuser).

    API: https://api.odcloud.kr/api/15072614/v1/uddi:95c11cda-...
    Auth: Infuser {serviceKey} header or serviceKey query param.
    Response: JSON with {data: [{법령명, 조문명, ...}], totalCount, ...}
    """

    BASE_URL = 'https://api.odcloud.kr/api'
    ENDPOINT = '/15072614/v1/uddi:95c11cda-a14d-404e-9e33-e594eae2812d'

    TIMEOUT = 5  # seconds

    def __init__(self):
        self.api_key = os.environ.get('LAW_API_KEY', '')
        self._api_available = bool(self.api_key)
        self._failure_count = 0
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'SafeFactory/1.0',
            'Accept': 'application/json',
        })

    def search_laws(self, query: str, num_results: int = 5) -> list[dict]:
        """Search 고용노동관련 법령 via odcloud API. Returns empty list on failure."""
        if not self._api_available or not self.api_key:
            return []

        cached = _get_cached(f'api:{query}')
        if cached is not None:
            return cached

        results = []
        try:
            results = self._search_odcloud(query, num_results)
            self._failure_count = 0
        except Exception as e:
            logger.debug("[LawAPI] odcloud search failed: %s", e)
            self._failure_count += 1

        # Disable API after 5 consecutive failures
        if self._failure_count >= 5:
            logger.warning("[LawAPI] Disabling external API after %d failures",
                           self._failure_count)
            self._api_available = False

        if results:
            _set_cached(f'api:{query}', results)

        return results

    def _search_odcloud(self, query: str, num_results: int) -> list[dict]:
        """Search via 고용노동부 법령 API on api.odcloud.kr.

        Uses server-side filtering: cond[법령명::LIKE] and cond[조문명::LIKE].
        """
        url = self.BASE_URL + self.ENDPOINT
        headers = {'Authorization': f'Infuser {self.api_key}'}

        # Extract the main law name keyword (e.g. "근로기준법" from "근로기준법 임금")
        parts = query.split()
        law_name = parts[0] if parts else query
        article_keyword = parts[1] if len(parts) > 1 else ''

        results = []

        params = {
            'page': 1,
            'perPage': num_results * 2,
            'returnType': 'JSON',
            'cond[법령명::LIKE]': law_name,
        }
        if article_keyword:
            params['cond[조문명::LIKE]'] = article_keyword

        resp = self.session.get(url, headers=headers, params=params, timeout=self.TIMEOUT)
        resp.raise_for_status()
        body = resp.json()

        if body.get('code') and body['code'] < 0:
            raise RuntimeError(f"API error: {body.get('msg', 'unknown')}")

        for item in body.get('data', [])[:num_results]:
            results.append({
                'name': item.get('법령명', ''),
                'article': item.get('조문명', ''),
                **{k: v for k, v in item.items()
                   if k not in ('법령명', '조문명', '번호') and v},
            })

        return results

    def fetch_all_laws(self, page: int = 1, per_page: int = 10) -> dict:
        """Fetch raw paginated data from the API. Useful for debugging."""
        if not self.api_key:
            return {}
        url = self.BASE_URL + self.ENDPOINT
        headers = {'Authorization': f'Infuser {self.api_key}'}
        params = {'page': page, 'perPage': per_page, 'returnType': 'JSON'}
        resp = self.session.get(url, headers=headers, params=params, timeout=self.TIMEOUT)
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# LawTextFetcher (법제처 법령 본문 via law.go.kr AJAX)
# ---------------------------------------------------------------------------
class LawTextFetcher:
    """Fetch actual law article text from law.go.kr internal AJAX API.

    Provides full article content for LLM prompt enrichment.
    Falls back gracefully when law.go.kr is unavailable.
    """

    BASE_URL = 'https://www.law.go.kr/LSW/lsInfoR.do'
    TIMEOUT = 20
    CIRCUIT_RESET_INTERVAL = 300  # 5분 후 재시도
    MAX_ARTICLE_CHARS = 800
    MAX_ARTICLES = 18

    def __init__(self):
        self._available = True
        self._failure_count = 0
        self._disabled_at: float | None = None
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                           'AppleWebKit/537.36 (KHTML, like Gecko)'),
            'X-Requested-With': 'XMLHttpRequest',
            'Referer': 'https://www.law.go.kr/',
        })

    def _check_circuit_reset(self):
        """Re-enable after CIRCUIT_RESET_INTERVAL seconds."""
        if not self._available and self._disabled_at:
            if time.time() - self._disabled_at > self.CIRCUIT_RESET_INTERVAL:
                self._available = True
                self._failure_count = 0
                self._disabled_at = None
                logger.info("[LawText] Circuit breaker reset — retrying law.go.kr")

    def fetch_law_articles(self, law_name: str) -> dict[str, str]:
        """Fetch all articles of a law. Returns {article_key: content}.

        Uses TTL cache. Returns empty dict on failure.
        """
        self._check_circuit_reset()
        if not self._available:
            return {}

        cache_key = f'lawtext:{law_name}'
        cached = _get_cached(cache_key)
        if cached is not None:
            return cached

        law_info = _LAW_REGISTRY.get(law_name)
        if not law_info:
            return {}

        params = {
            'lsiSeq': law_info['lsiSeq'],
            'lsId': '',
            'efYd': law_info['efYd'],
            'chrClsCd': '010202',
            'urlMode': 'lsInfoR',
            'viewCls': 'lsInfoR',
            'ancYnChk': '0',
        }

        try:
            resp = self.session.get(self.BASE_URL, params=params,
                                    timeout=self.TIMEOUT)
            resp.raise_for_status()
            articles = self._parse_articles(resp.text)
            self._failure_count = 0
            if articles:
                _set_cached(cache_key, articles)
            logger.info("[LawText] %s: %d articles parsed",
                        law_name, len(articles))
            return articles
        except Exception as e:
            logger.warning("[LawText] Failed to fetch %s: %s", law_name, e)
            self._failure_count += 1
            if self._failure_count >= 5:
                logger.warning(
                    "[LawText] Disabling law.go.kr after %d failures",
                    self._failure_count)
                self._available = False
                self._disabled_at = time.time()
            return {}

    def get_specific_articles(
        self, law_name: str, article_nums: list[str],
    ) -> list[tuple[str, str]]:
        """Get specific articles by number. Returns [(key, content), ...]."""
        all_articles = self.fetch_law_articles(law_name)
        results = []
        for key, content in all_articles.items():
            for num in article_nums:
                if key.startswith(num + '(') or key.startswith(num + ' '):
                    results.append((f"{law_name} {key}", content))
                    break
        return results

    @staticmethod
    def _parse_articles(html: str) -> dict[str, str]:
        """Parse law HTML into {article_key: content} dict.

        Stops at 부칙 (supplementary provisions) to avoid duplicate
        article numbers from amendment appendices.
        """
        # Remove 부칙 section using HTML structural markers.
        # law.go.kr wraps appendix articles in <div id="arDivArea">,
        # preceded by "<!-- 부칙 영역 끝 -->" comment.
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
            content = re.sub(
                r'\n\s*<[^>]*>\s*\d{4}\.\s*\d+\.\s*\d+\.>', '', content)
            content = content.strip()

            key = f'{article_num}({article_title})'
            if key not in articles:
                articles[key] = content[:LawTextFetcher.MAX_ARTICLE_CHARS]

        return articles


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------
_client: LawApiClient | None = None
_client_lock = threading.Lock()

_text_fetcher: LawTextFetcher | None = None
_text_fetcher_lock = threading.Lock()


def _get_client() -> LawApiClient:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = LawApiClient()
    return _client


def _get_text_fetcher() -> LawTextFetcher:
    global _text_fetcher
    if _text_fetcher is None:
        with _text_fetcher_lock:
            if _text_fetcher is None:
                _text_fetcher = LawTextFetcher()
    return _text_fetcher


def _find_relevant_articles(query: str) -> list[tuple[str, str]]:
    """질문과 관련된 법률 조문 텍스트를 찾는다.

    2단계 키워드 매칭:
      1단계: query 텍스트에 _QUESTION_TO_ARTICLES 키워드가 포함되어 있는지 직접 매칭
      2단계: Gemini flash로 구어체/자연어에서 법률 키워드를 추출하여 추가 매칭

    법률(본법)을 시행령보다 우선 처리하여, MAX_ARTICLES 한도 내에서
    본법 조문이 먼저 포함되도록 한다.

    Returns:
        [(article_key, content), ...] — 최대 MAX_ARTICLES개
    """
    fetcher = _get_text_fetcher()
    needed_laws: set[tuple[str, tuple[str, ...]]] = set()

    # 1단계: 기존 키워드 직접 매칭 (빠름, 비용 없음)
    for keyword, law_articles_list in _QUESTION_TO_ARTICLES.items():
        if keyword in query:
            for law_name, article_nums in law_articles_list:
                needed_laws.add((law_name, tuple(article_nums)))

    # 2단계: LLM 키워드 추출 (구어체/자연어 → 법률 용어)
    # "나오지 말라" → "해고", "주급" → "주휴" 등 자연어 의미 파악
    llm_topics = _extract_legal_topics(query)
    for topic in llm_topics:
        if topic in _QUESTION_TO_ARTICLES:
            for law_name, article_nums in _QUESTION_TO_ARTICLES[topic]:
                needed_laws.add((law_name, tuple(article_nums)))

    if not needed_laws:
        return []

    results: list[tuple[str, str]] = []
    # 법률을 시행령보다 먼저 처리 (법률 우선)
    sorted_laws = sorted(needed_laws, key=lambda x: (1 if '시행령' in x[0] else 0, x[0]))
    for law_name, article_nums in sorted_laws:
        articles = fetcher.get_specific_articles(law_name, list(article_nums))
        results.extend(articles)
        if len(results) >= LawTextFetcher.MAX_ARTICLES:
            break

    return results[:LawTextFetcher.MAX_ARTICLES]


def _extract_article_num(article: str) -> str:
    """Extract article number (e.g. '제23조', '제23조의2') from article string."""
    m = re.match(r'(제\d+조(?:의\d+)?)', article)
    return m.group(1) if m else article


def search_labor_laws(query: str, classification: dict | None = None) -> list[dict]:
    """질문에 관련된 노동법 조문을 검색하여 반환한다.

    RAG 파이프라인(services/rag_pipeline.py)에서 laborlaw 도메인일 때 호출된다.
    반환된 조문은 format_law_references()로 포맷되어 LLM 프롬프트에 주입된다.

    2단계 검색:
      Phase 1: law.go.kr AJAX API (주 데이터 소스) — 전문 조문 텍스트
               _find_relevant_articles() → 키워드 매칭 + LLM 토픽 추출
      Phase 2: odcloud API (보조) — Phase 1에서 결과 없을 때만 사용
               법령명 + 조문 제목만 제공 (전문 없음)

    Returns:
        [{'name': 법률명, 'article': 조문번호, 'article_text': 조문내용, ...}, ...]
    """
    # --- Phase 1: law.go.kr article text (primary, has full content) ---
    seen_nums: set[str] = set()
    results: list[dict] = []

    try:
        article_texts = _find_relevant_articles(query)
        for key, content in article_texts:
            parts = key.split(' ', 1)
            law_name = parts[0] if parts else ''
            article_part = parts[1] if len(parts) > 1 else ''
            num_key = f"{law_name}|{_extract_article_num(article_part)}"
            if num_key not in seen_nums:
                seen_nums.add(num_key)
                results.append({
                    'name': law_name,
                    'article': article_part,
                    'article_text': content,
                    'source': 'law.go.kr',
                })
    except Exception as e:
        logger.debug("[LawText] law.go.kr fetch failed: %s", e)

    # --- Phase 2: odcloud API (supplement, only if law.go.kr returned nothing) ---
    if not results:
        client = _get_client()
        if client.api_key:
            keywords = _extract_api_keywords(query, classification)
            for kw in keywords[:3]:
                for law in client.search_laws(kw, num_results=5):
                    raw_name = law.get('name', '')
                    article = law.get('article', '')
                    clean_name = (raw_name.split('[')[0].strip()
                                  if '[' in raw_name else raw_name)
                    num_key = f"{clean_name}|{_extract_article_num(article)}"
                    if num_key not in seen_nums:
                        seen_nums.add(num_key)
                        results.append(law)

    results = results[:10]

    if results:
        logger.info("[LawAPI] Found %d law references for: %s",
                    len(results), query[:50])
    return results


def format_law_references(laws: list[dict], start_index: int = 1) -> str:
    """Format law references as numbered markdown for LLM prompt injection.

    Each item is numbered [start_index], [start_index+1], ... so the AI
    uses matching citation numbers in its answer.
    """
    if not laws:
        return ''

    lines = []
    for i, law in enumerate(laws):
        idx = start_index + i
        name = law.get('name', '')
        if '[' in name:
            name = name[:name.index('[')].strip()

        article = law.get('article', '')
        article_text = law.get('article_text', '')

        header = f"[{idx}] **{name}**"
        if article:
            header += f" {article}"
        lines.append(header)

        if article_text:
            text = article_text[:800].strip()
            indented = '\n'.join(
                f'> {l}' for l in text.split('\n') if l.strip())
            lines.append(indented)
        lines.append('')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 토픽 키워드 → odcloud API 검색 키워드 매핑
# ---------------------------------------------------------------------------
# Phase 2(odcloud API)에서 질문 속 토픽을 API 검색어로 변환할 때 사용.
# Phase 1(law.go.kr)에서 결과를 못 찾았을 때의 폴백 경로이다.
_TOPIC_TO_LAW_KEYWORDS: dict[str, str] = {
    # 근로기준법
    '해고': '근로기준법 해고',
    '부당해고': '근로기준법 해고',
    '정리해고': '근로기준법 해고',
    '연차': '근로기준법 연차',
    '휴가': '근로기준법 휴가',
    '임금': '근로기준법 임금',
    '체불': '근로기준법 임금',
    '근로시간': '근로기준법 근로시간',
    '연장근로': '근로기준법 연장근로',
    '야간근로': '근로기준법 야간',
    '주휴': '근로기준법 휴일',
    '근로계약': '근로기준법 근로계약',
    '괴롭힘': '근로기준법 괴롭힘',
    # 퇴직급여
    '퇴직금': '퇴직급여 퇴직금',
    '퇴직': '퇴직급여 퇴직금',
    '퇴직연금': '퇴직급여 퇴직연금',
    # 최저임금
    '최저임금': '최저임금법',
    '최저시급': '최저임금법',
    # 남녀고용평등
    '육아휴직': '남녀고용평등 육아휴직',
    '출산휴가': '남녀고용평등 출산',
    '성희롱': '남녀고용평등 성희롱',
    # 고용보험
    '고용보험': '고용보험법',
    '실업급여': '고용보험법 실업급여',
    '4대보험': '고용보험법 보험료',
    # 산재보험
    '산재': '산업재해보상보험법',
    '산업재해': '산업재해보상보험법',
    # 산업안전보건
    '산업안전': '산업안전보건 안전조치',
    '안전보건': '산업안전보건 안전조치',
    '안전교육': '산업안전보건 교육',
    # 비정규직
    '비정규직': '기간제 단시간근로자',
    '기간제': '기간제 단시간근로자',
    '계약직': '기간제 단시간근로자',
    '파견': '파견근로자',
    # 노동조합
    '노동조합': '노동조합 노동관계',
    '단체교섭': '노동조합 단체교섭',
    '단체협약': '노동조합 단체협약',
    '쟁의행위': '노동조합 쟁의행위',
    # 중대재해
    '중대재해': '중대재해 처벌',
    '중대산업재해': '중대재해 처벌',
    # 외국인근로자
    '외국인근로자': '외국인근로자 고용',
    '외국인고용': '외국인근로자 고용',
    # 장애인고용
    '장애인고용': '장애인고용촉진 직업재활',
    '장애인의무고용': '장애인고용촉진 직업재활',
    # 연령차별
    '연령차별': '고령자고용촉진 연령차별',
    '정년': '고령자고용촉진 정년',
    # 임금채권
    '임금채권': '임금채권보장법',
    '체당금': '임금채권보장법 체당금',
    # 채용절차
    '채용절차': '채용절차 공정화',
    '채용서류': '채용절차 공정화',
    # 구직급여
    '구직급여': '구직자 취업촉진',
    '국민취업지원': '구직자 취업촉진',
    # 노사협의회
    '노사협의회': '근로자참여 협력증진',
    # 보험료징수
    '보험료': '보험료징수',
}


def _extract_api_keywords(query: str, classification: dict | None) -> list[str]:
    """classification과 질문에서 odcloud API 검색 키워드를 결정한다.

    우선순위:
      1. classification의 calc_type → _CALC_TYPE_API_KEYWORDS 매핑
      2. query에서 _TOPIC_TO_LAW_KEYWORDS 직접 매칭
      3. LLM 추출 키워드 → _TOPIC_TO_LAW_KEYWORDS 매핑 (구어체 대응)
      4. 폴백: query 원문 앞 50자
    """
    keywords = []
    if classification:
        calc_type = classification.get('calc_type')
        q_type = classification.get('type', 'legal')
        if calc_type and calc_type in _CALC_TYPE_API_KEYWORDS:
            keywords.extend(_CALC_TYPE_API_KEYWORDS[calc_type])
        if q_type in ('legal', 'hybrid'):
            # 직접 매칭: query 텍스트에 토픽 키워드가 포함되어 있는지 확인
            for topic, law_kw in _TOPIC_TO_LAW_KEYWORDS.items():
                if topic in query:
                    if law_kw not in keywords:
                        keywords.append(law_kw)
            # LLM 매칭: 직접 매칭으로 충분한 키워드를 못 찾은 경우
            # (캐시됨 — _find_relevant_articles에서 이미 호출했으면 즉시 반환)
            if len(keywords) <= 1:
                llm_topics = _extract_legal_topics(query)
                for topic in llm_topics:
                    if topic in _TOPIC_TO_LAW_KEYWORDS:
                        law_kw = _TOPIC_TO_LAW_KEYWORDS[topic]
                        if law_kw not in keywords:
                            keywords.append(law_kw)
    if not keywords:
        # 폴백: classification 없이 직접 매칭 시도
        for topic, law_kw in _TOPIC_TO_LAW_KEYWORDS.items():
            if topic in query:
                if law_kw not in keywords:
                    keywords.append(law_kw)
        if not keywords:
            keywords.append(query[:50])
    return keywords
