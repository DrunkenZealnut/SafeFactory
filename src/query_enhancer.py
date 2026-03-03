"""
Query Enhancer Module
Implements query expansion and enhancement techniques for improved RAG retrieval.
"""

import logging
import os
import re
import certifi
import httpx
from typing import List, Optional
from openai import OpenAI

from src import HttpClientMixin

# Set SSL certificate environment variables
os.environ.setdefault('SSL_CERT_FILE', certifi.where())
os.environ.setdefault('REQUESTS_CA_BUNDLE', certifi.where())


# Domain-specific synonym mappings for query expansion
DOMAIN_SYNONYMS = {
    'semiconductor': {
        'CVD': ['화학기상증착', 'Chemical Vapor Deposition'],
        '화학기상증착': ['CVD', 'Chemical Vapor Deposition'],
        'PVD': ['물리기상증착', 'Physical Vapor Deposition'],
        '물리기상증착': ['PVD', 'Physical Vapor Deposition'],
        '에칭': ['식각', 'etching'],
        '식각': ['에칭', 'etching'],
        'CMP': ['화학기계연마', 'Chemical Mechanical Polishing'],
        '화학기계연마': ['CMP'],
        'PECVD': ['플라즈마화학기상증착', 'Plasma Enhanced CVD'],
        'LPCVD': ['저압화학기상증착', 'Low Pressure CVD'],
        'ALD': ['원자층증착', 'Atomic Layer Deposition'],
        '원자층증착': ['ALD'],
        '포토리소그래피': ['노광', 'photolithography', '포토공정'],
        '노광': ['포토리소그래피', 'photolithography', '포토공정'],
        '이온주입': ['ion implantation', '이온임플란트'],
        '확산': ['diffusion', '열확산'],
        '웨이퍼': ['wafer', '실리콘웨이퍼'],
        '다이싱': ['dicing', '절단'],
        '본딩': ['bonding', '와이어본딩'],
        '패키징': ['packaging', '조립'],
    },
    'laborlaw': {
        '해고': ['부당해고', '정리해고', '해임', '면직'],
        '부당해고': ['해고', '부당면직'],
        '임금': ['급여', '보수', '급료', '월급'],
        '급여': ['임금', '보수', '급료', '월급'],
        '연차': ['연차유급휴가', '연차휴가', '유급휴가'],
        '연차유급휴가': ['연차', '연차휴가'],
        '퇴직금': ['퇴직급여', '퇴직연금'],
        '최저임금': ['최저시급', '법정최저임금'],
        '4대보험': ['사회보험', '국민연금', '건강보험', '고용보험', '산재보험'],
        '초과근무': ['연장근로', '초과근로', '야간근로', '휴일근로'],
        '연장근로': ['초과근무', '초과근로', '잔업'],
        '근로계약': ['고용계약', '근로계약서'],
        '산업재해': ['산재', '업무상재해'],
        '산재': ['산업재해', '업무상재해'],
    },
    'field-training': {
        '안전모': ['보호모', '머리보호구', '안전헬멧'],
        '보호구': ['PPE', '개인보호구', '안전장비'],
        '안전대': ['안전벨트', '추락방지대'],
        '방독마스크': ['방독면', '가스마스크', '호흡보호구'],
        '안전화': ['안전장화', '발보호구'],
        '보안경': ['안전안경', '보호안경', '눈보호구'],
        '안전수칙': ['작업수칙', '안전규정', '안전지침'],
    },
    'msds': {
        'MSDS': ['물질안전보건자료', 'SDS', 'Material Safety Data Sheet'],
        '물질안전보건자료': ['MSDS', 'SDS'],
        'GHS': ['화학물질분류표시', '세계조화시스템'],
        'CAS': ['CAS번호', 'CAS Number'],
        'TWA': ['시간가중평균', 'Time Weighted Average'],
        'STEL': ['단시간노출기준', 'Short Term Exposure Limit'],
        'LC50': ['반수치사농도'],
        'LD50': ['반수치사량'],
    },
}


class QueryEnhancer(HttpClientMixin):
    """
    Enhances user queries for better retrieval results.

    Techniques:
    - Multi-query: Generate multiple query variations
    - HyDE: Hypothetical Document Embedding
    - Keyword extraction: Extract key terms for filtering
    """

    def __init__(
        self,
        openai_api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        provider: str = "openai",
    ):
        """
        Initialize the QueryEnhancer.

        Args:
            openai_api_key: API key for the configured provider
            model: Model for query enhancement
            temperature: Generation temperature
            provider: LLM provider ('openai', 'gemini', 'anthropic')
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature

        if provider == 'gemini':
            from google import genai
            self._gemini = genai.Client(api_key=openai_api_key)
            self.client = None
        elif provider == 'anthropic':
            import anthropic
            self._anthropic = anthropic.Anthropic(api_key=openai_api_key)
            self.client = None
        else:
            self._http_client = httpx.Client(verify=certifi.where())
            self.client = OpenAI(api_key=openai_api_key, http_client=self._http_client, timeout=60.0)




    def _chat_complete(self, messages, temperature=None, max_tokens=500):
        """Dispatch chat completion to the configured provider."""
        temp = temperature if temperature is not None else self.temperature
        if self.provider == 'gemini':
            from google.genai import types
            system_msg = next((m['content'] for m in messages if m['role'] == 'system'), '')
            user_msg = next((m['content'] for m in messages if m['role'] == 'user'), '')
            config = types.GenerateContentConfig(
                system_instruction=system_msg or None,
                temperature=temp,
                max_output_tokens=max_tokens,
            )
            resp = self._gemini.models.generate_content(
                model=self.model, contents=user_msg, config=config,
            )
            return resp.text or ''
        elif self.provider == 'anthropic':
            system_msg = next((m['content'] for m in messages if m['role'] == 'system'), '')
            user_msgs = [m for m in messages if m['role'] != 'system']
            resp = self._anthropic.messages.create(
                model=self.model,
                system=system_msg or None,
                messages=user_msgs,
                temperature=temp,
                max_tokens=max_tokens,
            )
            return resp.content[0].text if resp.content else ''
        else:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content

    def expand_with_synonyms(self, query: str, domain: str = '') -> str:
        """Expand query with domain-specific synonyms.

        Appends synonym terms found in the query to improve retrieval recall.
        Does not modify the original query — only appends synonyms.

        Args:
            query: Original user query.
            domain: Domain key (e.g., 'semiconductor', 'laborlaw').

        Returns:
            Query with appended synonyms, or original query if no matches.
        """
        synonyms_map = DOMAIN_SYNONYMS.get(domain, {})
        if not synonyms_map:
            return query

        added = []
        query_lower = query.lower()
        for term, syns in synonyms_map.items():
            if term.lower() in query_lower:
                for syn in syns:
                    if syn.lower() not in query_lower and syn not in added:
                        added.append(syn)
                        if len(added) >= 5:  # Limit expansion to avoid noise
                            break
            if len(added) >= 5:
                break

        if added:
            return f"{query} ({' '.join(added)})"
        return query

    def multi_query(self, query: str, num_variations: int = 3) -> List[str]:
        """
        Generate multiple query variations for broader retrieval.

        Args:
            query: Original user query
            num_variations: Number of variations to generate

        Returns:
            List of query variations including the original
        """
        prompt = f"""주어진 질문을 {num_variations}가지 다른 방식으로 재구성해주세요.
각 변형은 원래 질문과 동일한 정보를 찾지만 다른 단어나 관점을 사용해야 합니다.

원래 질문: {query}

규칙:
1. 각 변형은 한 줄로 작성
2. 번호나 기호 없이 질문만 작성
3. 원래 의미를 유지하면서 다양한 표현 사용
4. 기술 용어의 경우 영문/한글 번역도 포함

변형된 질문들:"""

        try:
            content = self._chat_complete(
                [{"role": "user", "content": prompt}],
                max_tokens=500,
            ).strip()
            variations = [line.strip() for line in content.split('\n') if line.strip()]

            # Always include original query first
            result = [query]
            for v in variations[:num_variations]:
                # Clean up numbering if present
                cleaned = re.sub(r'^[\d]+[.)\-]\s*', '', v)
                if cleaned and cleaned != query:
                    result.append(cleaned)

            return result[:num_variations + 1]

        except Exception as e:
            logging.warning("Multi-query generation failed: %s", e)
            return [query]

    def hyde(self, query: str, domain: str = "general") -> str:
        """
        Generate a hypothetical document that would answer the query (HyDE).

        This technique creates a synthetic answer document, which is then
        embedded and used for similarity search. This often improves retrieval
        because the embedding of the hypothetical document is closer to actual
        relevant documents than the query embedding.

        Args:
            query: User query
            domain: Domain context (e.g., 'semiconductor', 'laborlaw')

        Returns:
            Hypothetical document text
        """
        domain_context = {
            "semiconductor": "반도체 기술 및 공정",
            "laborlaw": "한국 노동법 및 고용 관련 법률",
            "field-training": "산업안전보건 현장실습 교육",
            "safeguide": "안전보건공단 안전보건 가이드",
            "msds": "물질안전보건자료(MSDS) 화학물질 안전 정보",
            "general": "기술 문서"
        }

        context = domain_context.get(domain, domain_context["general"])

        prompt = f"""다음 질문에 대한 답변이 포함된 {context} 문서의 일부를 작성해주세요.
이 문서는 실제 존재하는 것처럼 상세하고 기술적으로 정확해야 합니다.

질문: {query}

가상 문서 내용 (200-300자):"""

        try:
            return self._chat_complete(
                [{"role": "user", "content": prompt}],
                temperature=0.7,  # Slightly higher for creative generation
                max_tokens=400,
            ).strip()

        except Exception as e:
            logging.warning("HyDE generation failed: %s", e)
            return query

    def extract_keywords_fast(self, query: str, max_keywords: int = 5) -> List[str]:
        """
        Extract keywords using regex rules without LLM calls.

        Extracts English technical terms and Korean nouns with particle stripping.
        Much faster than LLM-based extraction (~0ms vs ~500ms).

        Args:
            query: User query
            max_keywords: Maximum number of keywords to extract

        Returns:
            List of extracted keywords
        """
        # Extract English technical terms (2+ chars)
        english_terms = re.findall(r'[A-Za-z][A-Za-z0-9-]+', query)

        # Extract Korean words (2+ chars)
        korean_words = re.findall(r'[가-힣]{2,}', query)

        # Strip common Korean particles
        particles = re.compile(r'(은|는|이|가|을|를|의|에|에서|으로|로|와|과|도|만|까지|부터|에게|한테|께)$')
        korean_cleaned = []
        for w in korean_words:
            cleaned = particles.sub('', w)
            if len(cleaned) >= 2:
                korean_cleaned.append(cleaned)

        # Deduplicate while preserving order
        seen = set()
        result = []
        for term in english_terms + korean_cleaned:
            lower = term.lower()
            if lower not in seen:
                seen.add(lower)
                result.append(term)

        return result[:max_keywords]

    def extract_keywords(self, query: str, max_keywords: int = 5) -> List[str]:
        """
        Extract key terms from the query for potential filtering.

        Args:
            query: User query
            max_keywords: Maximum number of keywords to extract

        Returns:
            List of extracted keywords
        """
        prompt = f"""다음 질문에서 핵심 키워드를 추출해주세요.
기술 용어, 고유명사, 핵심 개념을 우선적으로 추출합니다.

질문: {query}

규칙:
1. 키워드만 쉼표로 구분하여 나열
2. 최대 {max_keywords}개
3. 영문 기술 용어는 그대로 유지
4. 불용어(~은, ~는, ~가, ~이) 제외

키워드:"""

        try:
            content = self._chat_complete(
                [{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=100,
            ).strip()
            keywords = [kw.strip() for kw in content.split(',') if kw.strip()]

            return keywords[:max_keywords]

        except Exception as e:
            logging.warning("Keyword extraction failed: %s", e)
            # Fallback: simple word extraction
            words = query.split()
            return [w for w in words if len(w) > 1][:max_keywords]

    def enhance_query(
        self,
        query: str,
        domain: str = "general",
        use_multi_query: bool = True,
        use_hyde: bool = True,
        use_keywords: bool = True
    ) -> dict:
        """
        Apply all enhancement techniques to a query.

        Args:
            query: Original user query
            domain: Domain context
            use_multi_query: Whether to generate query variations
            use_hyde: Whether to generate hypothetical document
            use_keywords: Whether to extract keywords

        Returns:
            Dictionary containing enhanced query components
        """
        result = {
            "original": query,
            "variations": [query],
            "hyde_doc": None,
            "keywords": []
        }

        if use_multi_query:
            result["variations"] = self.multi_query(query)

        if use_hyde:
            result["hyde_doc"] = self.hyde(query, domain)

        if use_keywords:
            result["keywords"] = self.extract_keywords(query)

        return result


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        enhancer = QueryEnhancer(api_key)

        # Test query
        test_query = "CVD 공정이란 무엇인가요?"

        print("=== Multi-Query Test ===")
        variations = enhancer.multi_query(test_query)
        for i, v in enumerate(variations):
            print(f"{i+1}. {v}")

        print("\n=== HyDE Test ===")
        hyde_doc = enhancer.hyde(test_query, domain="semiconductor")
        print(hyde_doc)

        print("\n=== Keywords Test ===")
        keywords = enhancer.extract_keywords(test_query)
        print(f"Keywords: {keywords}")

        print("\n=== Full Enhancement ===")
        enhanced = enhancer.enhance_query(test_query, domain="semiconductor")
        print(f"Original: {enhanced['original']}")
        print(f"Variations: {enhanced['variations']}")
        print(f"Keywords: {enhanced['keywords']}")
        print(f"HyDE (first 100 chars): {enhanced['hyde_doc'][:100]}...")
    else:
        print("OPENAI_API_KEY not found")
