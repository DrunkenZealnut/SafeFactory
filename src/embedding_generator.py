"""
Embedding Generator Module
Generates embeddings using OpenAI or Google Gemini embedding models.
"""

import os
import time
import certifi
import httpx
import tiktoken
from typing import List, Optional
from dataclasses import dataclass
from openai import OpenAI

from src import HttpClientMixin

# Set SSL certificate environment variables
os.environ.setdefault('SSL_CERT_FILE', certifi.where())
os.environ.setdefault('REQUESTS_CA_BUNDLE', certifi.where())


@dataclass
class EmbeddingResult:
    """Represents an embedding result with metadata."""
    text: str
    embedding: List[float]
    model: str
    dimensions: int
    token_count: Optional[int] = None


class EmbeddingGenerator(HttpClientMixin):
    """Generates embeddings for text using OpenAI or Gemini API."""

    MODELS = {
        # OpenAI
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
        # Gemini
        "gemini-embedding-2-preview": 3072,
        "gemini-embedding-001": 3072,
    }

    GEMINI_MODELS = {"gemini-embedding-2-preview", "gemini-embedding-001"}

    MAX_INPUT_TOKENS = 8191  # OpenAI embedding model token limit
    GEMINI_MAX_CHARS = 24000  # Conservative char limit for Gemini (~8192 tokens Korean)

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimensions: Optional[int] = None
    ):
        """
        Initialize the EmbeddingGenerator.

        Args:
            api_key: API key (OpenAI or Gemini depending on model)
            model: Embedding model to use
            dimensions: Custom dimensions (MRL for Gemini, native for OpenAI text-embedding-3)
        """
        self.model = model

        if model not in self.MODELS:
            raise ValueError(f"Unknown model: {model}. Available: {list(self.MODELS.keys())}")

        self._provider = "gemini" if model in self.GEMINI_MODELS else "openai"

        if self._provider == "gemini":
            from google import genai
            self._gemini_client = genai.Client(api_key=api_key)
            self.client = None
            self._http_client = None
            self._encoding = None
            # Gemini MRL: default to model's native dims, or user-specified
            if dimensions:
                self.dimensions = min(dimensions, self.MODELS[model])
            else:
                self.dimensions = self.MODELS[model]
        else:
            # OpenAI path (existing logic)
            self._gemini_client = None
            self._http_client = httpx.Client(verify=certifi.where())
            self.client = OpenAI(api_key=api_key, http_client=self._http_client, timeout=60.0)
            self._encoding = tiktoken.get_encoding("cl100k_base")
            if dimensions and model.startswith("text-embedding-3"):
                self.dimensions = min(dimensions, self.MODELS[model])
            else:
                self.dimensions = self.MODELS[model]

    def _truncate(self, text: str) -> str:
        """Truncate text to fit within embedding model token limit."""
        if self._provider == "gemini":
            if len(text) > self.GEMINI_MAX_CHARS:
                return text[:self.GEMINI_MAX_CHARS]
            return text
        else:
            tokens = self._encoding.encode(text)
            if len(tokens) <= self.MAX_INPUT_TOKENS:
                return text
            return self._encoding.decode(tokens[:self.MAX_INPUT_TOKENS])

    # ------------------------------------------------------------------
    # Gemini embedding methods
    # ------------------------------------------------------------------

    def _generate_gemini(self, text: str, task_type: Optional[str] = None) -> EmbeddingResult:
        """Generate a single embedding via Gemini API."""
        from google.genai import types

        config = types.EmbedContentConfig(
            output_dimensionality=self.dimensions,
        )
        if task_type:
            config.task_type = task_type

        response = self._gemini_client.models.embed_content(
            model=self.model,
            contents=text,
            config=config,
        )
        embedding = response.embeddings[0].values

        return EmbeddingResult(
            text=text,
            embedding=embedding,
            model=self.model,
            dimensions=len(embedding),
            token_count=None,
        )

    def _call_api_gemini(self, batch: List[str], task_type: Optional[str] = None) -> List[EmbeddingResult]:
        """Embed a batch of texts via Gemini API (one call per text)."""
        from google.genai import types

        config = types.EmbedContentConfig(
            output_dimensionality=self.dimensions,
        )
        if task_type:
            config.task_type = task_type

        results = []
        for text in batch:
            response = self._gemini_client.models.embed_content(
                model=self.model,
                contents=text,
                config=config,
            )
            results.append(EmbeddingResult(
                text=text,
                embedding=response.embeddings[0].values,
                model=self.model,
                dimensions=len(response.embeddings[0].values),
                token_count=None,
            ))
        return results

    # ------------------------------------------------------------------
    # OpenAI embedding methods
    # ------------------------------------------------------------------

    def _generate_openai(self, text: str) -> EmbeddingResult:
        """Generate a single embedding via OpenAI API."""
        if self.model.startswith("text-embedding-3") and self.dimensions != self.MODELS[self.model]:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=self.dimensions
            )
        else:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )

        embedding = response.data[0].embedding
        token_count = response.usage.total_tokens if response.usage else None

        return EmbeddingResult(
            text=text,
            embedding=embedding,
            model=self.model,
            dimensions=len(embedding),
            token_count=token_count
        )

    def _call_api_openai(self, batch: List[str]) -> List[EmbeddingResult]:
        """Single OpenAI embeddings API call for a batch."""
        if self.model.startswith("text-embedding-3") and self.dimensions != self.MODELS[self.model]:
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
                dimensions=self.dimensions
            )
        else:
            response = self.client.embeddings.create(
                model=self.model,
                input=batch
            )
        return [
            EmbeddingResult(
                text=batch[j],
                embedding=data.embedding,
                model=self.model,
                dimensions=len(data.embedding),
                token_count=None
            )
            for j, data in enumerate(response.data)
        ]

    # ------------------------------------------------------------------
    # Public API (provider-agnostic)
    # ------------------------------------------------------------------

    def generate(self, text: str, task_type: Optional[str] = None) -> EmbeddingResult:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed
            task_type: Gemini task type hint (e.g. "RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT").
                       Ignored for OpenAI models.

        Returns:
            EmbeddingResult with embedding vector
        """
        text = self._truncate(text)

        if self._provider == "gemini":
            return self._generate_gemini(text, task_type)
        else:
            return self._generate_openai(text)

    def _call_api(self, batch: List[str], task_type: Optional[str] = None) -> List[EmbeddingResult]:
        """Dispatch batch embedding to the appropriate provider."""
        batch = [self._truncate(t) for t in batch]

        if self._provider == "gemini":
            return self._call_api_gemini(batch, task_type)
        else:
            return self._call_api_openai(batch)

    def _embed_batch_with_retry(self, batch: List[str], max_retries: int = 3,
                                task_type: Optional[str] = None) -> List[EmbeddingResult]:
        """Embed a batch with exponential backoff for rate limits.
        Falls back to per-item processing on persistent or non-retryable errors."""
        for attempt in range(max_retries + 1):
            try:
                return self._call_api(batch, task_type)
            except Exception as e:
                err_str = str(e)
                is_rate_limit = '429' in err_str or 'rate_limit' in err_str.lower()

                if is_rate_limit and attempt < max_retries:
                    wait = 2 ** (attempt + 1)  # 2, 4, 8 초
                    print(f"  ⚠️ Rate limit, {wait}초 후 재시도 ({attempt + 1}/{max_retries})...")
                    time.sleep(wait)
                elif len(batch) > 1:
                    # 배치 전체 실패 시 개별 처리로 폴백 (token too long 등 대응)
                    print(f"  ⚠️ 배치 오류, 개별 처리 전환: {str(e)[:100]}")
                    return self._embed_individually(batch, task_type)
                else:
                    print(f"  ✗ 임베딩 실패 (건너뜀): {str(e)[:100]}")
                    return []
        return []

    def _embed_individually(self, batch: List[str],
                            task_type: Optional[str] = None) -> List[Optional[EmbeddingResult]]:
        """Embed items one by one. Returns None placeholder for failures (preserves alignment)."""
        results: List[Optional[EmbeddingResult]] = []
        for text in batch:
            try:
                results.append(self.generate(text, task_type))
            except Exception as e:
                print(f"  ✗ 개별 항목 임베딩 실패 (건너뜀): {str(e)[:80]}")
                results.append(None)
        return results

    def generate_batch(
        self,
        texts: List[str],
        batch_size: int = 100,
        task_type: Optional[str] = None
    ) -> List[Optional[EmbeddingResult]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts per API call
            task_type: Gemini task type hint. Ignored for OpenAI models.

        Returns:
            List of EmbeddingResult (or None for failed items). Length always equals len(texts).
        """
        results: List[Optional[EmbeddingResult]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            results.extend(self._embed_batch_with_retry(batch, task_type=task_type))
        return results

    def get_model_info(self) -> dict:
        """Get information about the current model."""
        return {
            "model": self.model,
            "provider": self._provider,
            "max_dimensions": self.MODELS[self.model],
            "current_dimensions": self.dimensions
        }


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # Test OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        generator = EmbeddingGenerator(api_key)
        print(f"[OpenAI] Model info: {generator.get_model_info()}")
        result = generator.generate("반도체 클린룸 정전기 방지 조치")
        print(f"[OpenAI] Embedding dimensions: {result.dimensions}")
        print(f"[OpenAI] First 5 values: {result.embedding[:5]}")

    # Test Gemini
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        generator = EmbeddingGenerator(gemini_key, model="gemini-embedding-2-preview", dimensions=1536)
        print(f"\n[Gemini] Model info: {generator.get_model_info()}")
        result = generator.generate("반도체 클린룸 정전기 방지 조치", task_type="RETRIEVAL_QUERY")
        print(f"[Gemini] Embedding dimensions: {result.dimensions}")
        print(f"[Gemini] First 5 values: {result.embedding[:5]}")
