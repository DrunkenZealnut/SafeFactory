"""law.go.kr DRF Open API 통합 클라이언트.

법령/판례/행정해석/행정규칙을 단일 게이트웨이로 조회한다.
인증: OC 파라미터(이메일 ID) + open.law.go.kr 서버 IP 등록 필요.

지원 target:
  - law:          법령 검색/조문 조회
  - prec:         판례 검색/본문 조회
  - moelCgmExpc:  고용노동부 행정해석 (질의회신)
  - expc:         일반 법령해석례
  - admrul:       행정규칙 (훈령/예규/고시/지침)
"""

import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from xml.etree import ElementTree

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SourceRequest:
    """질문 분석 후 필요한 소스 요청."""
    target: str
    query: str
    params: dict = field(default_factory=dict)
    max_results: int = 5


@dataclass
class LawArticle:
    """법령 조문."""
    law_name: str
    article: str
    text: str
    source: str = 'drf'


@dataclass
class Precedent:
    """법원 판례."""
    case_name: str
    case_no: str
    court: str
    date: str
    ruling_type: str
    summary: str
    ref_articles: str = ''


@dataclass
class Interpretation:
    """행정해석 (질의회신)."""
    title: str
    case_no: str
    date: str
    question: str
    answer: str
    reason: str = ''
    ref_laws: str = ''


@dataclass
class AdminRule:
    """행정규칙 (훈령/예규/고시/지침)."""
    name: str
    rule_type: str
    date: str
    org: str
    content: str


@dataclass
class LegalContext:
    """4개 소스 통합 결과."""
    law_articles: list[LawArticle] = field(default_factory=list)
    precedents: list[Precedent] = field(default_factory=list)
    interpretations: list[Interpretation] = field(default_factory=list)
    admin_rules: list[AdminRule] = field(default_factory=list)

    @property
    def has_content(self) -> bool:
        return bool(self.law_articles or self.precedents
                    or self.interpretations or self.admin_rules)

    @property
    def source_count(self) -> int:
        return (len(self.law_articles) + len(self.precedents)
                + len(self.interpretations) + len(self.admin_rules))


# ---------------------------------------------------------------------------
# TTL cache (shared with law_api.py pattern)
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
        # 캐시 크기 제한 (100개)
        if len(_cache) > 100:
            oldest = next(iter(_cache))
            del _cache[oldest]


# ---------------------------------------------------------------------------
# LawDrfClient
# ---------------------------------------------------------------------------

class LawDrfClient:
    """law.go.kr DRF Open API 통합 클라이언트.

    모든 target(law, prec, moelCgmExpc, expc, admrul)을 단일 인터페이스로 제공.
    """

    BASE_SEARCH = 'http://www.law.go.kr/DRF/lawSearch.do'
    BASE_SERVICE = 'http://www.law.go.kr/DRF/lawService.do'
    TIMEOUT = 10
    MAX_TEXT_CHARS = 800

    # 서킷브레이커
    MAX_FAILURES = 5
    CIRCUIT_RESET_SEC = 300

    def __init__(self, oc: str | None = None):
        self._oc = oc or os.environ.get('LAW_OC', '')
        self._available = bool(self._oc)
        self._failure_count = 0
        self._disabled_at: float | None = None
        self._lock = threading.Lock()

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'SafeFactory/2.0',
            'Accept': 'application/xml, application/json',
        })

    @property
    def available(self) -> bool:
        """DRF API 사용 가능 여부."""
        if not self._oc:
            return False
        self._check_circuit_reset()
        return self._available

    def _check_circuit_reset(self):
        if not self._available and self._disabled_at:
            if time.time() - self._disabled_at > self.CIRCUIT_RESET_SEC:
                with self._lock:
                    self._available = True
                    self._failure_count = 0
                    self._disabled_at = None
                logger.info("[LawDRF] Circuit breaker reset")

    def _record_success(self):
        with self._lock:
            self._failure_count = 0

    def _record_failure(self):
        with self._lock:
            self._failure_count += 1
            if self._failure_count >= self.MAX_FAILURES:
                self._available = False
                self._disabled_at = time.time()
                logger.warning("[LawDRF] Circuit breaker OPEN after %d failures",
                               self._failure_count)

    # ── 범용 요청 ──

    def _request_xml(self, base_url: str, params: dict) -> ElementTree.Element | None:
        """HTTP GET → XML 파싱. 캐싱 + 서킷브레이커 포함."""
        if not self.available:
            return None

        cache_key = f"drf:{base_url}:{sorted(params.items())}"
        cached = _get_cached(cache_key)
        if cached is not None:
            return cached

        params['OC'] = self._oc
        # DRF는 XML이 기본이고 가장 안정적
        params.setdefault('type', 'XML')

        try:
            resp = self.session.get(base_url, params=params, timeout=self.TIMEOUT)
            resp.raise_for_status()

            # 응답 크기 제한 (1MB)
            if len(resp.content) > 1_000_000:
                logger.warning("[LawDRF] Response too large: %d bytes", len(resp.content))
                return None

            root = ElementTree.fromstring(resp.content)
            self._record_success()
            _set_cached(cache_key, root)
            return root

        except requests.Timeout:
            logger.warning("[LawDRF] Timeout: %s", base_url)
            self._record_failure()
        except requests.RequestException as e:
            logger.warning("[LawDRF] Request failed: %s", e)
            self._record_failure()
        except ElementTree.ParseError as e:
            logger.warning("[LawDRF] XML parse failed: %s", e)
            self._record_failure()

        return None

    # ── 법령 (target=law) ──

    def search_laws(self, query: str, org: str = '',
                    display: int = 10) -> list[dict]:
        """법령 목록 검색."""
        params = {
            'target': 'law',
            'query': query[:100],
            'display': display,
        }
        if org:
            params['org'] = org

        root = self._request_xml(self.BASE_SEARCH, params)
        if root is None:
            return []

        results = []
        for item in root.findall('.//law') or root.findall('.//*'):
            name = self._text(item, '법령명한글') or self._text(item, '법령명')
            seq = self._text(item, '법령일련번호')
            if name and seq:
                results.append({
                    'name': name,
                    'seq': seq,
                    'ef_date': self._text(item, '시행일자', ''),
                    'org': self._text(item, '소관부처명', ''),
                })
        return results[:display]

    def get_law_articles(self, law_name: str) -> list[LawArticle]:
        """법령명으로 전체 조문 조회 → LawArticle 리스트.

        DRF lawService.do의 MST 파라미터는 법령일련번호(lsiSeq)를 요구하므로,
        먼저 lawSearch.do로 법령명을 검색하여 lsiSeq를 얻은 뒤 조문을 가져온다.
        """
        # Step 1: 법령명으로 검색 → lsiSeq 획득
        search_results = self.search_laws(law_name, display=1)
        if not search_results:
            logger.warning("[LawDRF] Law not found: %s", law_name)
            return []

        lsi_seq = search_results[0]['seq']
        params = {
            'target': 'law',
            'MST': lsi_seq,
        }
        root = self._request_xml(self.BASE_SERVICE, params)
        if root is None:
            return []

        articles = []
        for jo in root.findall('.//조문단위'):
            # 조문여부가 '전문'(장/절 제목)이면 스킵
            if self._text(jo, '조문여부', '') == '전문':
                continue

            num_raw = self._text(jo, '조문번호', '')
            title = self._text(jo, '조문제목', '')
            content = self._text(jo, '조문내용', '')

            if not num_raw:
                continue

            # 조문번호 정규화: "제23조" 형태
            article_str = f"제{num_raw}조"
            if title:
                article_str = f"제{num_raw}조({title})"

            articles.append(LawArticle(
                law_name=law_name,
                article=article_str,
                text=content[:self.MAX_TEXT_CHARS],
                source='drf',
            ))

        logger.info("[LawDRF] %s: %d articles parsed", law_name, len(articles))
        return articles

    def get_specific_law_articles(self, law_name: str,
                                  article_nums: list[str]) -> list[LawArticle]:
        """특정 조문번호만 필터링하여 반환."""
        all_articles = self.get_law_articles(law_name)
        results = []
        for art in all_articles:
            for num in article_nums:
                if art.article.startswith(num + '(') or art.article.startswith(num + ' ') or art.article == num:
                    results.append(art)
                    break
        return results

    # ── 판례 (target=prec) ──

    def search_precedents(self, query: str, court: str = '',
                          display: int = 5) -> list[dict]:
        """판례 목록 검색."""
        params = {
            'target': 'prec',
            'query': query[:100],
            'display': display,
        }
        if court:
            params['org'] = court

        root = self._request_xml(self.BASE_SEARCH, params)
        if root is None:
            return []

        results = []
        for item in root.findall('.//prec') or root.findall('.//*'):
            name = self._text(item, '사건명')
            seq = self._text(item, '판례일련번호')
            if name and seq:
                results.append({
                    'case_name': name,
                    'case_no': self._text(item, '사건번호', ''),
                    'date': self._text(item, '선고일자', ''),
                    'court': self._text(item, '법원명', ''),
                    'seq': seq,
                })
        return results[:display]

    def get_precedent_detail(self, prec_id: str) -> Precedent | None:
        """판례 상세 조회 → Precedent."""
        params = {
            'target': 'prec',
            'ID': prec_id,
        }
        root = self._request_xml(self.BASE_SERVICE, params)
        if root is None:
            return None

        summary = self._text(root, '판결요지', '') or self._text(root, '판시사항', '')
        case_name = self._text(root, '사건명', '')

        if not summary and not case_name:
            return None

        return Precedent(
            case_name=case_name,
            case_no=self._text(root, '사건번호', ''),
            court=self._text(root, '법원명', ''),
            date=self._format_date(self._text(root, '선고일자', '')),
            ruling_type=self._text(root, '판결유형', ''),
            summary=summary[:self.MAX_TEXT_CHARS],
            ref_articles=self._text(root, '참조조문', ''),
        )

    # ── 행정해석 (target=moelCgmExpc) ──

    def search_interpretations(self, query: str,
                               display: int = 5) -> list[dict]:
        """고용노동부 행정해석 목록 검색."""
        params = {
            'target': 'moelCgmExpc',
            'query': query[:100],
            'display': display,
        }

        root = self._request_xml(self.BASE_SEARCH, params)
        if root is None:
            return []

        results = []
        # DRF XML 구조: target=moelCgmExpc → 실제 태그는 <cgmExpc>
        for item in (root.findall('.//cgmExpc')
                     or root.findall('.//moelCgmExpc')
                     or root.findall('.//expc')
                     or root.findall('.//*')):
            title = (self._text(item, '안건명')
                     or self._text(item, '법령해석례안건명'))
            seq = (self._text(item, '법령해석일련번호')
                   or self._text(item, '법령해석례일련번호')
                   or self._text(item, '판례일련번호'))
            if title and seq:
                results.append({
                    'title': title,
                    'case_no': self._text(item, '안건번호', ''),
                    'date': self._text(item, '해석일자', '')
                           or self._text(item, '회신일자', ''),
                    'seq': seq,
                })
        return results[:display]

    def get_interpretation_detail(self, interp_id: str) -> Interpretation | None:
        """행정해석 상세 조회 → Interpretation."""
        params = {
            'target': 'moelCgmExpc',
            'ID': interp_id,
        }
        root = self._request_xml(self.BASE_SERVICE, params)
        if root is None:
            return None

        answer = self._text(root, '회답', '')
        title = self._text(root, '안건명', '')

        if not answer and not title:
            return None

        return Interpretation(
            title=title,
            case_no=self._text(root, '안건번호', ''),
            date=self._format_date(self._text(root, '해석일자', '')
                                   or self._text(root, '회신일자', '')),
            question=self._text(root, '질의요지', '')[:self.MAX_TEXT_CHARS],
            answer=answer[:self.MAX_TEXT_CHARS],
            reason=self._text(root, '이유', '')[:400],
            ref_laws=self._text(root, '관련법령', ''),
        )

    # ── 행정규칙 (target=admrul) ──

    def search_admin_rules(self, query: str, knd: str | None = None,
                           org: str = '', display: int = 5) -> list[dict]:
        """행정규칙 검색.
        knd: 1=훈령, 2=예규, 3=고시, 4=공고, 5=지침
        """
        params = {
            'target': 'admrul',
            'query': query[:100],
            'display': display,
        }
        if knd:
            params['knd'] = knd
        if org:
            params['org'] = org

        root = self._request_xml(self.BASE_SEARCH, params)
        if root is None:
            return []

        results = []
        for item in root.findall('.//admrul') or root.findall('.//*'):
            name = self._text(item, '행정규칙명')
            seq = (self._text(item, '행정규칙일련번호')
                   or self._text(item, '행정규칙ID'))
            if name and seq:
                results.append({
                    'name': name,
                    'rule_type': self._text(item, '행정규칙종류', ''),
                    'date': self._text(item, '발령일자', ''),
                    'org': self._text(item, '소관부처명', ''),
                    'seq': seq,
                })
        return results[:display]

    def get_admin_rule_detail(self, rule_id: str) -> AdminRule | None:
        """행정규칙 상세 조회 → AdminRule."""
        params = {
            'target': 'admrul',
            'ID': rule_id,
        }
        root = self._request_xml(self.BASE_SERVICE, params)
        if root is None:
            return None

        name = self._text(root, '행정규칙명', '')
        content = self._text(root, '조문내용', '')

        if not name:
            return None

        return AdminRule(
            name=name,
            rule_type=self._text(root, '행정규칙종류', ''),
            date=self._format_date(self._text(root, '발령일자', '')),
            org=self._text(root, '소관부처명', ''),
            content=content[:self.MAX_TEXT_CHARS],
        )

    # ── XML 유틸리티 ──

    @staticmethod
    def _text(element: ElementTree.Element, tag: str,
              default: str | None = None) -> str | None:
        """XML element에서 태그 텍스트 추출. 재귀 탐색."""
        el = element.find(f'.//{tag}')
        if el is not None and el.text:
            return el.text.strip()
        return default

    @staticmethod
    def _format_date(raw: str) -> str:
        """'20240515' → '2024.05.15' 변환."""
        raw = raw.strip()
        if len(raw) == 8 and raw.isdigit():
            return f"{raw[:4]}.{raw[4:6]}.{raw[6:8]}"
        return raw
