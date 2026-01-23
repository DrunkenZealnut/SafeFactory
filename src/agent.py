"""
Main Pinecone Agent
Orchestrates the entire process of loading files, generating embeddings,
and uploading to Pinecone.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from tqdm import tqdm

from .file_loader import FileLoader, FileType, LoadedFile
from .image_describer import ImageDescriber
from .semantic_chunker import SemanticChunker, Chunk
from .embedding_generator import EmbeddingGenerator
from .pinecone_uploader import PineconeUploader, VectorData


@dataclass
class ProcessingResult:
    """Result of processing a folder."""
    total_files: int
    processed_files: int
    total_chunks: int
    uploaded_vectors: int
    failed_uploads: int
    errors: List[str]


class PineconeAgent:
    """
    Main agent that processes files and uploads to Pinecone.

    Supports:
    - Image files (with AI-generated descriptions)
    - Markdown files (with semantic chunking)
    - JSON files (with intelligent splitting)
    """

    def __init__(
        self,
        openai_api_key: str,
        pinecone_api_key: str,
        pinecone_index_name: str,
        embedding_model: str = "text-embedding-3-small",
        vision_model: str = "gpt-4o-mini",
        max_chunk_tokens: int = 500,
        create_index_if_not_exists: bool = True
    ):
        """
        Initialize the PineconeAgent.

        Args:
            openai_api_key: OpenAI API key
            pinecone_api_key: Pinecone API key
            pinecone_index_name: Name of the Pinecone index
            embedding_model: OpenAI embedding model
            vision_model: OpenAI vision model for image descriptions
            max_chunk_tokens: Maximum tokens per chunk
            create_index_if_not_exists: Create index if it doesn't exist
        """
        # Initialize components
        self.image_describer = ImageDescriber(
            api_key=openai_api_key,
            model=vision_model
        )

        self.chunker = SemanticChunker(
            openai_api_key=openai_api_key,
            model=embedding_model,
            max_chunk_tokens=max_chunk_tokens
        )

        self.embedding_generator = EmbeddingGenerator(
            api_key=openai_api_key,
            model=embedding_model
        )

        # Get dimension from embedding model
        dimension = self.embedding_generator.MODELS.get(embedding_model, 1536)

        self.pinecone_uploader = PineconeUploader(
            api_key=pinecone_api_key,
            index_name=pinecone_index_name,
            dimension=dimension,
            create_if_not_exists=create_index_if_not_exists
        )

        self.openai_api_key = openai_api_key

    def _process_image(self, loaded_file: LoadedFile) -> List[Chunk]:
        """Process an image file by generating a description."""
        try:
            description = self.image_describer.describe_from_base64(
                base64_image=loaded_file.content,
                extension=loaded_file.metadata.get('extension', '.png')
            )

            # Create a single chunk with the image description
            return [Chunk(
                content=f"[이미지: {loaded_file.filename}]\n\n{description}",
                index=0,
                source_file=loaded_file.path,
                start_char=0,
                end_char=len(description),
                token_count=self.chunker._count_tokens(description),
                metadata={
                    'file_type': 'image',
                    'filename': loaded_file.filename,
                    'original_extension': loaded_file.metadata.get('extension'),
                    'relative_path': loaded_file.metadata.get('relative_path')
                }
            )]
        except Exception as e:
            print(f"Error processing image {loaded_file.filename}: {e}")
            return []

    def _process_markdown(self, loaded_file: LoadedFile) -> List[Chunk]:
        """Process a markdown file with semantic chunking."""
        try:
            chunks = self.chunker.chunk_text(
                text=loaded_file.content,
                source_file=loaded_file.path,
                metadata={
                    'file_type': 'markdown',
                    'filename': loaded_file.filename,
                    'relative_path': loaded_file.metadata.get('relative_path')
                }
            )
            return chunks
        except Exception as e:
            print(f"Error processing markdown {loaded_file.filename}: {e}")
            return []

    def _process_json(self, loaded_file: LoadedFile) -> List[Chunk]:
        """Process a JSON file with intelligent chunking."""
        try:
            chunks = self.chunker.chunk_json(
                json_content=loaded_file.content,
                source_file=loaded_file.path,
                metadata={
                    'file_type': 'json',
                    'filename': loaded_file.filename,
                    'relative_path': loaded_file.metadata.get('relative_path'),
                    'is_valid_json': loaded_file.metadata.get('is_valid_json')
                }
            )
            return chunks
        except Exception as e:
            print(f"Error processing JSON {loaded_file.filename}: {e}")
            return []

    def process_file(self, loaded_file: LoadedFile) -> List[Chunk]:
        """Process a single file based on its type."""
        if loaded_file.file_type == FileType.IMAGE:
            return self._process_image(loaded_file)
        elif loaded_file.file_type == FileType.MARKDOWN:
            return self._process_markdown(loaded_file)
        elif loaded_file.file_type == FileType.JSON:
            return self._process_json(loaded_file)
        return []

    def process_folder(
        self,
        folder_path: str,
        namespace: str = "",
        recursive: bool = True,
        batch_size: int = 50,
        verbose: bool = True
    ) -> ProcessingResult:
        """
        Process all files in a folder and upload to Pinecone.

        Args:
            folder_path: Path to the folder to process
            namespace: Pinecone namespace
            recursive: Whether to scan subdirectories
            batch_size: Batch size for embedding and upload
            verbose: Whether to show progress

        Returns:
            ProcessingResult with statistics
        """
        # Initialize file loader
        loader = FileLoader(folder_path, recursive=recursive)
        summary = loader.get_file_summary()

        if verbose:
            print(f"\n📁 폴더 스캔 완료: {folder_path}")
            print(f"   - 이미지: {summary['images']}개")
            print(f"   - 마크다운: {summary['markdown']}개")
            print(f"   - JSON: {summary['json']}개")
            print(f"   - 총 파일: {summary['total']}개\n")

        # Process all files
        all_chunks: List[Chunk] = []
        processed_files = 0
        errors: List[str] = []

        files_iterator = loader.load_all()
        if verbose:
            files_iterator = tqdm(list(files_iterator), desc="파일 처리 중")

        for loaded_file in files_iterator:
            try:
                chunks = self.process_file(loaded_file)
                all_chunks.extend(chunks)
                processed_files += 1

                if verbose and not isinstance(files_iterator, tqdm):
                    print(f"✓ {loaded_file.filename}: {len(chunks)}개 청크 생성")

            except Exception as e:
                error_msg = f"Error processing {loaded_file.filename}: {e}"
                errors.append(error_msg)
                if verbose:
                    print(f"✗ {error_msg}")

        if verbose:
            print(f"\n📊 총 {len(all_chunks)}개 청크 생성됨")
            print("🔢 임베딩 생성 중...")

        # Generate embeddings in batches
        vectors: List[VectorData] = []
        chunk_texts = [chunk.content for chunk in all_chunks]

        for i in range(0, len(chunk_texts), batch_size):
            batch_chunks = all_chunks[i:i + batch_size]
            batch_texts = chunk_texts[i:i + batch_size]

            try:
                embeddings = self.embedding_generator.generate_batch(batch_texts)

                for chunk, emb_result in zip(batch_chunks, embeddings):
                    vector = self.pinecone_uploader.prepare_vector(
                        embedding=emb_result.embedding,
                        content=chunk.content,
                        source_file=chunk.source_file,
                        chunk_index=chunk.index,
                        metadata=chunk.metadata
                    )
                    vectors.append(vector)

            except Exception as e:
                error_msg = f"Error generating embeddings for batch {i//batch_size}: {e}"
                errors.append(error_msg)
                if verbose:
                    print(f"✗ {error_msg}")

        if verbose:
            print(f"✓ {len(vectors)}개 임베딩 생성 완료")
            print("📤 Pinecone 업로드 중...")

        # Upload to Pinecone
        upload_result = self.pinecone_uploader.upload_batch(
            vectors=vectors,
            namespace=namespace,
            batch_size=batch_size
        )

        if verbose:
            print(f"✓ 업로드 완료: {upload_result['success']}개 성공, {upload_result['failed']}개 실패")

        return ProcessingResult(
            total_files=summary['total'],
            processed_files=processed_files,
            total_chunks=len(all_chunks),
            uploaded_vectors=upload_result['success'],
            failed_uploads=upload_result['failed'],
            errors=errors
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        namespace: str = "",
        filter: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar content in Pinecone.

        Args:
            query: Search query
            top_k: Number of results
            namespace: Pinecone namespace
            filter: Metadata filter

        Returns:
            List of search results
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.generate(query)

        # Search in Pinecone
        results = self.pinecone_uploader.query(
            vector=query_embedding.embedding,
            top_k=top_k,
            namespace=namespace,
            filter=filter,
            include_metadata=True
        )

        return results

    def get_stats(self) -> Dict:
        """Get Pinecone index statistics."""
        return self.pinecone_uploader.get_stats()


def create_agent_from_env() -> PineconeAgent:
    """Create a PineconeAgent using environment variables."""
    from dotenv import load_dotenv
    load_dotenv()

    openai_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "document-index")

    if not openai_key:
        raise ValueError("OPENAI_API_KEY not found in environment")
    if not pinecone_key:
        raise ValueError("PINECONE_API_KEY not found in environment")

    return PineconeAgent(
        openai_api_key=openai_key,
        pinecone_api_key=pinecone_key,
        pinecone_index_name=index_name
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.agent <folder_path> [namespace]")
        sys.exit(1)

    folder_path = sys.argv[1]
    namespace = sys.argv[2] if len(sys.argv) > 2 else ""

    agent = create_agent_from_env()
    result = agent.process_folder(folder_path, namespace=namespace)

    print("\n" + "="*50)
    print("📋 처리 결과 요약")
    print("="*50)
    print(f"총 파일 수: {result.total_files}")
    print(f"처리된 파일: {result.processed_files}")
    print(f"생성된 청크: {result.total_chunks}")
    print(f"업로드된 벡터: {result.uploaded_vectors}")
    print(f"실패한 업로드: {result.failed_uploads}")

    if result.errors:
        print(f"\n⚠️ 에러 ({len(result.errors)}개):")
        for error in result.errors[:5]:
            print(f"  - {error}")
        if len(result.errors) > 5:
            print(f"  ... 외 {len(result.errors) - 5}개")
