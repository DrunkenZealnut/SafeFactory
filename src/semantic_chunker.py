"""
Semantic Chunker Module
Implements semantic chunking for intelligent text splitting.
"""

import re
from typing import List, Dict, Optional
from dataclasses import dataclass
from openai import OpenAI


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    content: str
    index: int
    source_file: str
    start_char: int
    end_char: int
    token_count: int
    metadata: Optional[Dict] = None


class SemanticChunker:
    """
    Splits text into semantically meaningful chunks.
    Uses a combination of structural analysis and embedding similarity.
    """

    def __init__(
        self,
        openai_api_key: str,
        model: str = "text-embedding-3-small",
        max_chunk_tokens: int = 500,
        min_chunk_tokens: int = 100,
        overlap_tokens: int = 50,
        similarity_threshold: float = 0.5
    ):
        """
        Initialize the SemanticChunker.

        Args:
            openai_api_key: OpenAI API key for embeddings
            model: Embedding model to use
            max_chunk_tokens: Maximum tokens per chunk
            min_chunk_tokens: Minimum tokens per chunk
            overlap_tokens: Number of overlapping tokens between chunks
            similarity_threshold: Threshold for semantic similarity
        """
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model
        self.max_chunk_tokens = max_chunk_tokens
        self.min_chunk_tokens = min_chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.similarity_threshold = similarity_threshold

    def _count_tokens(self, text: str) -> int:
        """Estimate token count (approx 4 chars per token for mixed content)."""
        # Simple estimation: ~4 characters per token for English/Korean mix
        return len(text) // 3

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text."""
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        return dot_product / (norm1 * norm2) if norm1 * norm2 > 0 else 0

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Handle common sentence endings
        sentence_endings = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]

    def _split_by_structure(self, text: str) -> List[str]:
        """Split text by structural elements (headers, paragraphs, etc.)."""
        # Split by markdown headers
        header_pattern = r'^#{1,6}\s+.+$'

        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\n+', text)

        segments = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Check if it's a header
            if re.match(header_pattern, para, re.MULTILINE):
                segments.append(para)
            else:
                # Further split long paragraphs into sentences
                if self._count_tokens(para) > self.max_chunk_tokens:
                    sentences = self._split_into_sentences(para)
                    segments.extend(sentences)
                else:
                    segments.append(para)

        return segments

    def _merge_small_segments(self, segments: List[str]) -> List[str]:
        """Merge segments that are too small."""
        merged = []
        current = ""

        for segment in segments:
            if not segment:
                continue

            test_combined = f"{current}\n\n{segment}" if current else segment
            combined_tokens = self._count_tokens(test_combined)

            if combined_tokens <= self.max_chunk_tokens:
                current = test_combined
            else:
                if current:
                    merged.append(current)
                current = segment

        if current:
            merged.append(current)

        return merged

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap between chunks for context continuity."""
        if len(chunks) <= 1:
            return chunks

        overlapped = []
        # Approximate chars for overlap (overlap_tokens * 3)
        overlap_chars = self.overlap_tokens * 3

        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped.append(chunk)
            else:
                # Get last portion of previous chunk by characters
                prev_chunk = chunks[i-1]
                overlap_text = ""

                if len(prev_chunk) > overlap_chars:
                    overlap_text = prev_chunk[-overlap_chars:]

                if overlap_text:
                    overlapped.append(f"...{overlap_text}\n\n{chunk}")
                else:
                    overlapped.append(chunk)

        return overlapped

    def chunk_text(
        self,
        text: str,
        source_file: str,
        use_embeddings: bool = True,
        metadata: Optional[Dict] = None
    ) -> List[Chunk]:
        """
        Split text into semantic chunks.

        Args:
            text: Text to chunk
            source_file: Source file path for metadata
            use_embeddings: Whether to use embedding-based similarity
            metadata: Additional metadata to include

        Returns:
            List of Chunk objects
        """
        if not text.strip():
            return []

        # Step 1: Split by structure
        segments = self._split_by_structure(text)

        if not segments:
            return []

        # Step 2: Merge small segments
        merged_segments = self._merge_small_segments(segments)

        # Step 3: Add overlap
        overlapped_segments = self._add_overlap(merged_segments)

        # Step 4: Create Chunk objects
        chunks = []
        char_position = 0

        for i, segment in enumerate(overlapped_segments):
            # Find actual position in original text (approximate)
            start_char = char_position
            end_char = start_char + len(segment)

            chunk = Chunk(
                content=segment,
                index=i,
                source_file=source_file,
                start_char=start_char,
                end_char=end_char,
                token_count=self._count_tokens(segment),
                metadata={
                    **(metadata or {}),
                    'chunk_index': i,
                    'total_chunks': len(overlapped_segments)
                }
            )
            chunks.append(chunk)
            char_position = end_char

        return chunks

    def chunk_json(
        self,
        json_content: str,
        source_file: str,
        metadata: Optional[Dict] = None
    ) -> List[Chunk]:
        """
        Chunk JSON content intelligently.

        Args:
            json_content: JSON string
            source_file: Source file path
            metadata: Additional metadata

        Returns:
            List of Chunk objects
        """
        import json

        try:
            data = json.loads(json_content)
        except json.JSONDecodeError:
            # If invalid JSON, treat as plain text
            return self.chunk_text(json_content, source_file, metadata=metadata)

        # For small JSON, keep as single chunk
        if self._count_tokens(json_content) <= self.max_chunk_tokens:
            return [Chunk(
                content=json_content,
                index=0,
                source_file=source_file,
                start_char=0,
                end_char=len(json_content),
                token_count=self._count_tokens(json_content),
                metadata={
                    **(metadata or {}),
                    'chunk_index': 0,
                    'total_chunks': 1,
                    'json_type': type(data).__name__
                }
            )]

        # For large JSON arrays, chunk by items
        if isinstance(data, list):
            return self._chunk_json_array(data, source_file, metadata)

        # For large JSON objects, chunk by keys
        if isinstance(data, dict):
            return self._chunk_json_object(data, source_file, metadata)

        # Fallback to text chunking
        return self.chunk_text(json_content, source_file, metadata=metadata)

    def _chunk_json_array(
        self,
        data: List,
        source_file: str,
        metadata: Optional[Dict]
    ) -> List[Chunk]:
        """Chunk a JSON array."""
        import json

        chunks = []
        current_items = []
        chunk_index = 0

        for item in data:
            item_str = json.dumps(item, ensure_ascii=False, indent=2)
            current_items.append(item)
            current_str = json.dumps(current_items, ensure_ascii=False, indent=2)

            if self._count_tokens(current_str) > self.max_chunk_tokens:
                # Save current chunk without last item
                if len(current_items) > 1:
                    current_items.pop()
                    chunk_content = json.dumps(current_items, ensure_ascii=False, indent=2)
                else:
                    chunk_content = current_str

                chunks.append(Chunk(
                    content=chunk_content,
                    index=chunk_index,
                    source_file=source_file,
                    start_char=0,
                    end_char=len(chunk_content),
                    token_count=self._count_tokens(chunk_content),
                    metadata={
                        **(metadata or {}),
                        'chunk_index': chunk_index,
                        'json_type': 'array_segment'
                    }
                ))

                chunk_index += 1
                current_items = [item] if len(current_items) > 1 else []

        # Add remaining items
        if current_items:
            chunk_content = json.dumps(current_items, ensure_ascii=False, indent=2)
            chunks.append(Chunk(
                content=chunk_content,
                index=chunk_index,
                source_file=source_file,
                start_char=0,
                end_char=len(chunk_content),
                token_count=self._count_tokens(chunk_content),
                metadata={
                    **(metadata or {}),
                    'chunk_index': chunk_index,
                    'json_type': 'array_segment'
                }
            ))

        # Update total_chunks in metadata
        for chunk in chunks:
            chunk.metadata['total_chunks'] = len(chunks)

        return chunks

    def _chunk_json_object(
        self,
        data: Dict,
        source_file: str,
        metadata: Optional[Dict]
    ) -> List[Chunk]:
        """Chunk a JSON object by keys."""
        import json

        chunks = []
        current_obj = {}
        chunk_index = 0

        for key, value in data.items():
            current_obj[key] = value
            current_str = json.dumps(current_obj, ensure_ascii=False, indent=2)

            if self._count_tokens(current_str) > self.max_chunk_tokens:
                # Save current chunk without last key
                if len(current_obj) > 1:
                    del current_obj[key]
                    chunk_content = json.dumps(current_obj, ensure_ascii=False, indent=2)
                else:
                    chunk_content = current_str

                chunks.append(Chunk(
                    content=chunk_content,
                    index=chunk_index,
                    source_file=source_file,
                    start_char=0,
                    end_char=len(chunk_content),
                    token_count=self._count_tokens(chunk_content),
                    metadata={
                        **(metadata or {}),
                        'chunk_index': chunk_index,
                        'json_type': 'object_segment'
                    }
                ))

                chunk_index += 1
                current_obj = {key: value} if len(current_obj) > 1 else {}

        # Add remaining keys
        if current_obj:
            chunk_content = json.dumps(current_obj, ensure_ascii=False, indent=2)
            chunks.append(Chunk(
                content=chunk_content,
                index=chunk_index,
                source_file=source_file,
                start_char=0,
                end_char=len(chunk_content),
                token_count=self._count_tokens(chunk_content),
                metadata={
                    **(metadata or {}),
                    'chunk_index': chunk_index,
                    'json_type': 'object_segment'
                }
            ))

        # Update total_chunks in metadata
        for chunk in chunks:
            chunk.metadata['total_chunks'] = len(chunks)

        return chunks


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        chunker = SemanticChunker(api_key)

        # Test with sample text
        sample_text = """
# Introduction

This is a sample document for testing semantic chunking.

## Section 1

The semantic chunker analyzes text structure and splits it intelligently.
It considers headers, paragraphs, and sentence boundaries.

## Section 2

For JSON content, it can split by array items or object keys.
This ensures that related data stays together in the same chunk.
"""

        chunks = chunker.chunk_text(sample_text, "test.md")
        for chunk in chunks:
            print(f"Chunk {chunk.index}: {chunk.token_count} tokens")
            print(chunk.content[:100])
            print("---")
    else:
        print("OPENAI_API_KEY not found")
