"""
Embedding Generator Module
Generates embeddings using OpenAI's text-embedding models.
"""

import os
import certifi
import httpx
from typing import List, Optional
from dataclasses import dataclass
from openai import OpenAI

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


class EmbeddingGenerator:
    """Generates embeddings for text using OpenAI API."""

    MODELS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536
    }

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimensions: Optional[int] = None
    ):
        """
        Initialize the EmbeddingGenerator.

        Args:
            api_key: OpenAI API key
            model: Embedding model to use
            dimensions: Custom dimensions (only for text-embedding-3 models)
        """
        # Create httpx client with explicit SSL certificate verification
        http_client = httpx.Client(verify=certifi.where())
        self.client = OpenAI(api_key=api_key, http_client=http_client)
        self.model = model

        if model not in self.MODELS:
            raise ValueError(f"Unknown model: {model}. Available: {list(self.MODELS.keys())}")

        # Set dimensions
        if dimensions and model.startswith("text-embedding-3"):
            self.dimensions = min(dimensions, self.MODELS[model])
        else:
            self.dimensions = self.MODELS[model]

    def generate(self, text: str) -> EmbeddingResult:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            EmbeddingResult with embedding vector
        """
        # text-embedding-3 models support custom dimensions
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

    def generate_batch(
        self,
        texts: List[str],
        batch_size: int = 100
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts per API call

        Returns:
            List of EmbeddingResult objects
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

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

            for j, data in enumerate(response.data):
                results.append(EmbeddingResult(
                    text=batch[j],
                    embedding=data.embedding,
                    model=self.model,
                    dimensions=len(data.embedding),
                    token_count=None  # Batch doesn't provide per-item token count
                ))

        return results

    def get_model_info(self) -> dict:
        """Get information about the current model."""
        return {
            "model": self.model,
            "max_dimensions": self.MODELS[self.model],
            "current_dimensions": self.dimensions
        }


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        generator = EmbeddingGenerator(api_key)
        print(f"Model info: {generator.get_model_info()}")

        # Test embedding
        result = generator.generate("Hello, world!")
        print(f"Embedding dimensions: {result.dimensions}")
        print(f"First 5 values: {result.embedding[:5]}")
    else:
        print("OPENAI_API_KEY not found")
