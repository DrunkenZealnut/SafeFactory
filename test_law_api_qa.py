"""Test: 법제처 법령 본문 → Prompt → AI direct answer (bypassing calculator).

This script tests the approach of:
1. User asks a labor law question
2. Fetch actual law article text from law.go.kr (법제처 내부 API)
3. Build a comprehensive prompt with legal text + current rates
4. Send to AI for direct computation/answer
5. Compare with calculator-based approach

Usage:
    python test_law_api_qa.py
"""

import os
import sys
import re
import time
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

import requests

# ---------------------------------------------------------------------------
# 법제처 법령 조문 크롤러 (law.go.kr 내부 AJAX API)
# ---------------------------------------------------------------------------
# 주요 노동법령 lsiSeq 매핑
LAW_REGISTRY = {
    '근로기준법': {'lsiSeq': '265959', 'efYd': '20251023'},
    '근로자퇴직급여 보장법': {'lsiSeq': '266139', 'efYd': '20251111'},
    '남녀고용평등과 일ㆍ가정 양립 지원에 관한 법률': {'lsiSeq': '264654', 'efYd': '20250223'},
    '고용보험법': {'lsiSeq': '266295', 'efYd': '20251119'},
}

# 질문 키워드 → 관련 법령 + 조문 매핑
QUESTION_TO_ARTICLES = {
    # 급여/임금 관련
    '임금': [('근로기준법', ['제2조', '제43조', '제44조'])],
    '급여': [('근로기준법', ['제2조', '제43조', '제44조'])],
    '연봉': [('근로기준법', ['제2조', '제43조'])],
    '월급': [('근로기준법', ['제2조', '제43조'])],
    '실수령': [('근로기준법', ['제2조', '제43조'])],
    '체불': [('근로기준법', ['제36조', '제37조', '제43조', '제43조의2'])],

    # 근로시간/가산수당
    '연장근로': [('근로기준법', ['제50조', '제53조', '제56조'])],
    '야간': [('근로기준법', ['제56조'])],
    '가산수당': [('근로기준법', ['제56조'])],
    '근로시간': [('근로기준법', ['제50조', '제51조', '제53조', '제56조'])],
    '주휴': [('근로기준법', ['제55조'])],

    # 해고
    '해고': [('근로기준법', ['제23조', '제24조', '제25조', '제26조', '제27조', '제28조', '제29조', '제30조'])],
    '부당해고': [('근로기준법', ['제23조', '제26조', '제27조', '제28조', '제29조', '제30조'])],

    # 퇴직금
    '퇴직': [('근로기준법', ['제34조', '제36조']), ('근로자퇴직급여 보장법', ['제4조', '제8조', '제9조', '제10조'])],
    '퇴직금': [('근로자퇴직급여 보장법', ['제4조', '제8조', '제9조', '제10조'])],

    # 연차
    '연차': [('근로기준법', ['제60조', '제61조'])],
    '휴가': [('근로기준법', ['제54조', '제55조', '제60조', '제61조', '제74조'])],

    # 최저임금
    '최저임금': [('근로기준법', ['제2조', '제6조'])],

    # 근로계약
    '계약': [('근로기준법', ['제2조', '제15조', '제16조', '제17조'])],
    '근로계약': [('근로기준법', ['제2조', '제15조', '제16조', '제17조'])],

    # 육아/출산
    '육아': [('남녀고용평등과 일ㆍ가정 양립 지원에 관한 법률', ['제19조', '제19조의2'])],
    '출산': [('근로기준법', ['제74조']), ('남녀고용평등과 일ㆍ가정 양립 지원에 관한 법률', ['제18조'])],

    # 4대보험
    '보험': [('고용보험법', ['제1조', '제2조', '제10조'])],
    '4대보험': [('고용보험법', ['제1조', '제2조', '제10조'])],
    '고용보험': [('고용보험법', ['제1조', '제2조', '제10조'])],
}


class LawTextFetcher:
    """Fetch actual law article text from law.go.kr internal AJAX API."""

    BASE_URL = 'https://www.law.go.kr/LSW/lsInfoR.do'
    TIMEOUT = 20
    _cache: dict[str, dict[str, str]] = {}

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'X-Requested-With': 'XMLHttpRequest',
            'Referer': 'https://www.law.go.kr/',
        })

    def fetch_law_articles(self, law_name: str) -> dict[str, str]:
        """Fetch all articles of a law. Returns {article_key: content}."""
        if law_name in self._cache:
            return self._cache[law_name]

        law_info = LAW_REGISTRY.get(law_name)
        if not law_info:
            logging.warning("Unknown law: %s", law_name)
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
            resp = self.session.get(self.BASE_URL, params=params, timeout=self.TIMEOUT)
            resp.raise_for_status()
            articles = self._parse_articles(resp.text)
            self._cache[law_name] = articles
            logging.info("[LawText] %s: %d articles parsed", law_name, len(articles))
            return articles
        except Exception as e:
            logging.warning("[LawText] Failed to fetch %s: %s", law_name, e)
            return {}

    def get_specific_articles(self, law_name: str, article_nums: list[str]) -> list[tuple[str, str]]:
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
        """Parse law HTML into {article_key: content} dict."""
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

        articles = {}
        for i, match in enumerate(matches):
            article_num = match.group(1)
            article_title = match.group(2).strip()
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else start + 3000
            content = text[start:end].strip()
            content = re.sub(r'\n\s*<[^>]*>\s*\d{4}\.\s*\d+\.\s*\d+\.>', '', content)
            content = content.strip()

            key = f'{article_num}({article_title})'
            if key not in articles:
                articles[key] = content[:1000]

        return articles


def find_relevant_articles(query: str, fetcher: LawTextFetcher) -> list[tuple[str, str]]:
    """Find relevant law articles for a query."""
    needed_laws = set()

    for keyword, law_articles_list in QUESTION_TO_ARTICLES.items():
        if keyword in query:
            for law_name, article_nums in law_articles_list:
                needed_laws.add((law_name, tuple(article_nums)))

    if not needed_laws:
        # Default: search 근로기준법 broadly
        needed_laws.add(('근로기준법', ('제2조', '제43조', '제56조', '제60조')))

    results = []
    for law_name, article_nums in needed_laws:
        articles = fetcher.get_specific_articles(law_name, list(article_nums))
        results.extend(articles)

    return results


# ---------------------------------------------------------------------------
# 2026년 기준 법정 요율
# ---------------------------------------------------------------------------
CURRENT_RATES_2026 = """
## 2026년 대한민국 노동법 기준 요율

### 4대보험 요율
| 보험 | 근로자 | 사업주 | 비고 |
|------|--------|--------|------|
| 국민연금 | 4.75% | 4.75% | 상한 617만원/월 |
| 건강보험 | 3.595% | 3.595% | |
| 장기요양보험 | 건강보험료×13.14% | 건강보험료×13.14% | |
| 고용보험 | 0.9% | 0.9%+ | 150인 미만 +0.25% |
| 산재보험 | - | 업종별 (기타 0.7%) | |

### 소득세 간이세액표
- 근로소득공제: ~500만 70%, ~1500만 40%+350만, ~4500만 15%+725만, ~1억 5%+1175만, 1억초과 2%+1475만
- 기본공제: 1인당 150만원
- 세율: 1400만↓6%, 5000만↓15%-126만, 8800만↓24%-576만, 1.5억↓35%-1544만
- 근로소득세액공제: 130만↓55%, 130만초과 71.5만+(초과×30%), 상한 총급여별 74만/66만/50만/20만
- 지방소득세 = 소득세×10%

### 최저임금 (2026년)
- 시급: 10,320원 / 일급(8h): 82,560원 / 월급(209h): 2,156,880원

### 주요 기준
- 월 소정근로시간 (주40h): 209시간
- 통상시급 = 월급 ÷ 209
- 가산: 연장 50%, 야간(22~06시) 50%, 휴일(8h이내) 50%, 휴일(8h초과) 100%
- 주휴수당: (주근무시간/40)×8×시급
- 퇴직금: 1일평균임금×30×재직일수/365 (1년이상)
- 연차: 1년미만 월1일(최대11), 1년이상 15일, 3년이상 매2년+1일(최대25)
- 오늘 날짜: 2026년 2월 23일
"""


def build_prompt(query: str, law_articles: list[tuple[str, str]]) -> tuple[str, str]:
    """Build system + user prompt with actual law text."""
    system_prompt = f"""당신은 대한민국 노동법 전문가입니다.

## 역할
- 노동법 질문에 실제 법률 조문을 근거로 정확하게 답변
- 계산이 필요하면 단계별로 수행하고 마크다운 표로 정리
- 반드시 관련 법조문을 인용

## 답변 원칙
1. 아래 제공된 법률 조문을 반드시 인용하세요
2. 계산은 단계별로 과정을 보여주세요
3. 불확실한 내용은 명시하세요
4. 결과는 마크다운 표로 정리하세요

{CURRENT_RATES_2026}"""

    law_text = ""
    if law_articles:
        law_text = "\n\n## 관련 법률 조문 (법제처 국가법령정보센터)\n\n"
        for key, content in law_articles:
            law_text += f"### {key}\n{content}\n\n"

    user_prompt = f"""## 질문
{query}
{law_text}
위 법률 조문과 현행 기준을 바탕으로 질문에 상세히 답변해주세요."""

    return system_prompt, user_prompt


def ask_ai(system_prompt: str, user_prompt: str) -> tuple[str, str]:
    """Send to AI. Try Gemini, fallback to OpenAI."""
    for provider in ['gemini', 'openai']:
        try:
            if provider == 'gemini':
                from google import genai
                client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
                resp = client.models.generate_content(
                    model='gemini-2.0-flash',
                    contents=user_prompt,
                    config={'system_instruction': system_prompt, 'temperature': 0.2, 'max_output_tokens': 4000},
                )
                return resp.text or '(빈 응답)', provider
            else:
                from openai import OpenAI
                client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
                resp = client.chat.completions.create(
                    model='gpt-4o-mini',
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2, max_tokens=4000,
                )
                return resp.choices[0].message.content or '(빈 응답)', provider
        except Exception as e:
            if '429' in str(e) or 'RESOURCE_EXHAUSTED' in str(e):
                logging.warning("  %s rate limited, trying next...", provider)
                continue
            return f"AI 오류 ({provider}): {e}", provider
    return "모든 AI 제공자 rate limit", 'none'


def run_calculator(query: str) -> str | None:
    """Run existing calculator for comparison."""
    try:
        from services.labor_classifier import classify_labor_question
        from services.labor_calculator import run_labor_calculation
        classification = classify_labor_question(query)
        if classification['type'] in ('calculation', 'hybrid'):
            result = run_labor_calculation(classification)
            if result:
                return result.get('formatted', '')
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
TEST_QUESTIONS = [
    "연봉 4000만원이면 실수령액이 얼마인가요?",
    "3년 근무 후 퇴직할 때 퇴직금은 어떻게 계산하나요? 월급 300만원입니다.",
    "부당해고를 당했을 때 구제 절차가 어떻게 되나요?",
    "시급 9000원으로 일하고 있는데 최저임금 위반인가요?",
    "2022년 3월 1일 입사했으면 지금까지 연차가 몇 일 발생하나요?",
]


def main():
    print("=" * 70)
    print("법제처 법령 본문 → AI 직접 답변 테스트")
    print("=" * 70)

    fetcher = LawTextFetcher()

    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n{'='*70}")
        print(f"테스트 {i}: {question}")
        print(f"{'='*70}")

        # Step 1: 법령 조문 가져오기
        print("\n[1] 법제처에서 관련 법률 조문 가져오는 중...")
        t0 = time.time()
        articles = find_relevant_articles(question, fetcher)
        api_time = time.time() - t0
        print(f"    → {len(articles)}개 관련 조문 ({api_time:.2f}초)")
        for key, _ in articles[:5]:
            print(f"    - {key}")
        if len(articles) > 5:
            print(f"    ... 외 {len(articles)-5}개")

        # Step 2: 프롬프트 구성
        system_prompt, user_prompt = build_prompt(question, articles)
        tokens_est = (len(system_prompt) + len(user_prompt)) // 3
        print(f"\n[2] 프롬프트 구성 (~{tokens_est} 토큰)")

        # Step 3: AI 답변
        print("\n[3] AI 답변 생성 중...")
        t1 = time.time()
        answer, provider = ask_ai(system_prompt, user_prompt)
        ai_time = time.time() - t1
        print(f"    → {provider} ({ai_time:.2f}초)")

        print(f"\n--- AI 답변 (법제처 법령 본문 기반, {provider}) ---")
        print(answer)

        # Step 4: 기존 계산기 비교
        print(f"\n--- 기존 계산기 결과 (비교용) ---")
        calc = run_calculator(question)
        print(calc if calc else "(계산기 해당 없음)")

        print(f"\n총 소요시간: 법령조회 {api_time:.2f}초 + AI {ai_time:.2f}초 = {api_time+ai_time:.2f}초")


if __name__ == '__main__':
    main()
