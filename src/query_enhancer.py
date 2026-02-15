"""
Query Enhancer Module
Implements query expansion and enhancement techniques for improved RAG retrieval.
"""

import os
import re
import certifi
import httpx
from typing import List, Optional
from openai import OpenAI

# Set SSL certificate environment variables
os.environ.setdefault('SSL_CERT_FILE', certifi.where())
os.environ.setdefault('REQUESTS_CA_BUNDLE', certifi.where())


class QueryEnhancer:
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
        temperature: float = 0.3
    ):
        """
        Initialize the QueryEnhancer.

        Args:
            openai_api_key: OpenAI API key
            model: Model for query enhancement
            temperature: Generation temperature
        """
        http_client = httpx.Client(verify=certifi.where())
        self.client = OpenAI(api_key=openai_api_key, http_client=http_client, timeout=60.0)
        self.model = model
        self.temperature = temperature

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
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=500
            )

            content = response.choices[0].message.content.strip()
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
            print(f"Multi-query generation failed: {e}")
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
            "general": "기술 문서"
        }

        context = domain_context.get(domain, domain_context["general"])

        prompt = f"""다음 질문에 대한 답변이 포함된 {context} 문서의 일부를 작성해주세요.
이 문서는 실제 존재하는 것처럼 상세하고 기술적으로 정확해야 합니다.

질문: {query}

가상 문서 내용 (200-300자):"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,  # Slightly higher for creative generation
                max_tokens=400
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"HyDE generation failed: {e}")
            return query

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
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=100
            )

            content = response.choices[0].message.content.strip()
            keywords = [kw.strip() for kw in content.split(',') if kw.strip()]

            return keywords[:max_keywords]

        except Exception as e:
            print(f"Keyword extraction failed: {e}")
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
