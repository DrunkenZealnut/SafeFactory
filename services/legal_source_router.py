"""질문 분석 → 법적 소스 자동 선택 + 병렬 검색 + LLM 프롬프트 포맷.

LawDrfClient를 사용하여 질문에 적합한 법적 소스(법령/판례/행정해석/행정규칙)를
자동 선택하고 병렬로 검색한 뒤, RAG 프롬프트에 주입할 마크다운을 생성한다.
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from services.law_drf_client import (
    AdminRule,
    Interpretation,
    LawArticle,
    LawDrfClient,
    LegalContext,
    Precedent,
    SourceRequest,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 키워드 → 추가 소스 매핑 (법령은 항상 포함)
# ---------------------------------------------------------------------------
_KEYWORD_TO_SOURCES: dict[str, list[str]] = {
    # 판례 + 행정해석이 필요한 키워드
    '해고': ['prec', 'moelCgmExpc'],
    '부당해고': ['prec', 'moelCgmExpc'],
    '정리해고': ['prec', 'moelCgmExpc'],
    '괴롭힘': ['prec', 'moelCgmExpc'],
    '직장내 괴롭힘': ['prec'],
    '성희롱': ['prec', 'moelCgmExpc'],
    '통상임금': ['prec', 'moelCgmExpc'],

    # 판례가 필요한 키워드
    '산재': ['prec'],
    '업무상재해': ['prec'],
    '중대재해': ['prec'],
    '부당노동행위': ['prec'],

    # 행정해석이 필요한 키워드
    '연차': ['moelCgmExpc'],
    '휴가': ['moelCgmExpc'],
    '임금': ['moelCgmExpc'],
    '급여': ['moelCgmExpc'],
    '퇴직금': ['moelCgmExpc'],
    '퇴직연금': ['moelCgmExpc'],
    '육아휴직': ['moelCgmExpc'],
    '출산휴가': ['moelCgmExpc'],
    '근로시간': ['moelCgmExpc'],
    '연장근로': ['moelCgmExpc'],
    '주휴': ['moelCgmExpc'],
    '근로계약': ['moelCgmExpc'],
    '수습': ['moelCgmExpc'],
    '비정규직': ['moelCgmExpc'],
    '기간제': ['moelCgmExpc'],
    '파견': ['moelCgmExpc'],
    '고용보험': ['moelCgmExpc'],
    '실업급여': ['moelCgmExpc'],

    # 행정규칙(고시/지침)이 필요한 키워드
    '최저임금': ['admrul'],
    '최저시급': ['admrul'],
    '안전교육': ['admrul'],
    '위험성평가': ['admrul'],
    '안전보건': ['admrul'],
    '근로감독': ['admrul'],

    # 명시적 소스 요청 키워드
    '판례': ['prec'],
    '판결': ['prec'],
    '대법원': ['prec'],
    '행정해석': ['moelCgmExpc'],
    '질의회신': ['moelCgmExpc'],
    '고시': ['admrul'],
    '지침': ['admrul'],
    '훈령': ['admrul'],
    '예규': ['admrul'],
}

# classification type별 기본 소스
_TYPE_DEFAULT_SOURCES: dict[str, list[str]] = {
    'legal': ['moelCgmExpc'],
    'calculation': [],
    'hybrid': ['moelCgmExpc'],
}

# 검색 쿼리 변환: 사용자 키워드 → 소스별 최적 검색어
_QUERY_REFINEMENTS: dict[str, dict[str, str]] = {
    # keyword: {target: refined_query}
    '해고': {'prec': '부당해고 구제'},
    '부당해고': {'prec': '부당해고 구제'},
    '통상임금': {'prec': '통상임금 범위'},
    '최저임금': {'admrul': '최저임금 고시'},
    '안전교육': {'admrul': '안전보건교육'},
    '위험성평가': {'admrul': '위험성평가 지침'},
    '근로감독': {'admrul': '근로감독관 집무규정'},
}


class LegalSourceRouter:
    """질문 분석 → 적절한 법적 소스 자동 선택 + 병렬 검색."""

    MAX_TOTAL_CHARS = 3000  # 프롬프트 주입 전체 제한
    SEARCH_TIMEOUT = 8      # 병렬 검색 전체 타임아웃 (초)

    def __init__(self, drf_client: LawDrfClient):
        self.drf = drf_client

    def route(self, query: str,
              classification: dict | None = None) -> list[SourceRequest]:
        """질문 분석 → 필요 소스 결정. 법령(law)은 항상 포함."""

        targets: set[str] = set()

        # 1. 키워드 매칭
        for keyword, sources in _KEYWORD_TO_SOURCES.items():
            if keyword in query:
                targets.update(sources)

        # 2. classification type 기반
        if classification:
            q_type = classification.get('type', 'legal')
            defaults = _TYPE_DEFAULT_SOURCES.get(q_type, [])
            targets.update(defaults)

        # 3. 소스 요청 생성
        requests_list: list[SourceRequest] = []

        # 법령은 항상 포함 (기존 law_api.py의 키워드 매칭으로 이미 처리되므로 여기선 추가 소스만)
        for target in sorted(targets):
            # 검색어 최적화
            search_query = query
            for keyword, refinements in _QUERY_REFINEMENTS.items():
                if keyword in query and target in refinements:
                    search_query = refinements[target]
                    break

            params = {}
            if target == 'prec':
                params['org'] = '400201'  # 대법원 우선
            elif target == 'admrul':
                params['org'] = '1440000'  # 고용노동부

            requests_list.append(SourceRequest(
                target=target,
                query=search_query,
                params=params,
                max_results=3,
            ))

        if requests_list:
            logger.info("[LegalRouter] Query '%s' → targets: %s",
                        query[:30], [r.target for r in requests_list])
        return requests_list

    def search_all(self, query: str,
                   classification: dict | None = None) -> LegalContext:
        """4개 소스 병렬 검색 → LegalContext 반환."""
        source_requests = self.route(query, classification)
        if not source_requests:
            return LegalContext()

        ctx = LegalContext()

        with ThreadPoolExecutor(max_workers=4) as executor:
            future_map = {
                executor.submit(self._search_source, req): req
                for req in source_requests
            }

            for future in as_completed(future_map, timeout=self.SEARCH_TIMEOUT):
                req = future_map[future]
                try:
                    items = future.result()
                    if req.target == 'prec':
                        ctx.precedents.extend(items)
                    elif req.target in ('moelCgmExpc', 'expc'):
                        ctx.interpretations.extend(items)
                    elif req.target == 'admrul':
                        ctx.admin_rules.extend(items)
                except Exception as e:
                    logger.warning("[LegalRouter] %s search failed: %s",
                                   req.target, e)

        if ctx.has_content:
            logger.info("[LegalRouter] Results: prec=%d, interp=%d, admrul=%d",
                        len(ctx.precedents), len(ctx.interpretations),
                        len(ctx.admin_rules))
        return ctx

    def _search_source(self, req: SourceRequest) -> list:
        """단일 소스 검색 + 상위 항목 상세 조회. 실패 시 빈 리스트."""
        try:
            if req.target == 'prec':
                return self._search_precedents(req)
            elif req.target in ('moelCgmExpc', 'expc'):
                return self._search_interpretations(req)
            elif req.target == 'admrul':
                return self._search_admin_rules(req)
        except Exception as e:
            logger.warning("[LegalRouter] _search_source(%s) error: %s",
                           req.target, e)
        return []

    def _search_precedents(self, req: SourceRequest) -> list[Precedent]:
        """판례 검색 → 상위 결과 상세 조회."""
        search_results = self.drf.search_precedents(
            req.query,
            court=req.params.get('org', ''),
            display=req.max_results,
        )
        precedents = []
        for item in search_results[:2]:  # 상위 2개만 상세 조회
            detail = self.drf.get_precedent_detail(item['seq'])
            if detail:
                precedents.append(detail)
        return precedents

    def _search_interpretations(self, req: SourceRequest) -> list[Interpretation]:
        """행정해석 검색 → 상위 결과 상세 조회."""
        search_results = self.drf.search_interpretations(
            req.query,
            display=req.max_results,
        )
        interps = []
        for item in search_results[:2]:  # 상위 2개만 상세 조회
            detail = self.drf.get_interpretation_detail(item['seq'])
            if detail:
                interps.append(detail)
        return interps

    def _search_admin_rules(self, req: SourceRequest) -> list[AdminRule]:
        """행정규칙 검색 → 상위 결과 상세 조회."""
        search_results = self.drf.search_admin_rules(
            req.query,
            knd=req.params.get('knd'),
            org=req.params.get('org', ''),
            display=req.max_results,
        )
        rules = []
        for item in search_results[:2]:
            detail = self.drf.get_admin_rule_detail(item['seq'])
            if detail:
                rules.append(detail)
        return rules

    # ── 포맷터 ──

    def format_context(self, ctx: LegalContext,
                       start_index: int = 1) -> str:
        """LegalContext → 마크다운 포맷 (LLM 프롬프트 주입용).

        소스 우선순위: 법령(기존) > 판례 > 행정해석 > 행정규칙
        전체 제한: MAX_TOTAL_CHARS
        """
        if not ctx.has_content:
            return ''

        lines: list[str] = []
        idx = start_index
        chars_used = 0

        # 판례
        if ctx.precedents:
            lines.append('### 관련 판례')
            for prec in ctx.precedents:
                header = f"[{idx}] **{prec.court} {prec.date} 선고 {prec.case_no}** ({prec.case_name})"
                body = prec.summary
                entry = header + '\n' + self._indent(body) + '\n'

                if chars_used + len(entry) > self.MAX_TOTAL_CHARS:
                    break
                lines.append(entry)
                chars_used += len(entry)
                idx += 1

        # 행정해석
        if ctx.interpretations:
            lines.append('### 행정해석 (고용노동부 질의회신)')
            for interp in ctx.interpretations:
                header = f"[{idx}] **{interp.title}** ({interp.case_no}, {interp.date})"
                body = interp.answer
                if interp.reason:
                    body += f"\n[이유] {interp.reason[:200]}"
                entry = header + '\n' + self._indent(body) + '\n'

                if chars_used + len(entry) > self.MAX_TOTAL_CHARS:
                    break
                lines.append(entry)
                chars_used += len(entry)
                idx += 1

        # 행정규칙
        if ctx.admin_rules:
            lines.append('### 관련 고시/지침')
            for rule in ctx.admin_rules:
                header = f"[{idx}] **{rule.name}** ({rule.rule_type}, {rule.date})"
                body = rule.content
                entry = header + '\n' + self._indent(body) + '\n'

                if chars_used + len(entry) > self.MAX_TOTAL_CHARS:
                    break
                lines.append(entry)
                chars_used += len(entry)
                idx += 1

        return '\n'.join(lines)

    @staticmethod
    def _indent(text: str) -> str:
        """텍스트를 블록인용(>) 형태로 들여쓰기."""
        return '\n'.join(
            f'> {line}' for line in text.split('\n') if line.strip()
        )
