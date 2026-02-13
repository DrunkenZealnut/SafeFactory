"""
Pinecone Uploader Module
Handles uploading vectors to Pinecone index.
"""

import hashlib
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from pinecone import Pinecone, ServerlessSpec


@dataclass
class VectorData:
    """Represents a vector to be uploaded to Pinecone."""
    id: str
    values: List[float]
    metadata: Dict[str, Any]


class PineconeUploader:
    """Uploads vectors to Pinecone index."""

    def __init__(
        self,
        api_key: str,
        index_name: str,
        dimension: int = 1536,
        metric: str = "cosine",
        create_if_not_exists: bool = True,
        cloud: str = "aws",
        region: str = "us-east-1"
    ):
        """
        Initialize the PineconeUploader.

        Args:
            api_key: Pinecone API key
            index_name: Name of the Pinecone index
            dimension: Vector dimension (must match embedding model)
            metric: Distance metric (cosine, euclidean, dotproduct)
            create_if_not_exists: Create index if it doesn't exist
            cloud: Cloud provider (aws, gcp, azure)
            region: Cloud region
        """
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric

        # Check if index exists
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]

        if index_name not in existing_indexes:
            if create_if_not_exists:
                print(f"Creating index: {index_name}")
                self.pc.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric=metric,
                    spec=ServerlessSpec(
                        cloud=cloud,
                        region=region
                    )
                )
                # Wait for index to be ready
                self._wait_for_index_ready()
            else:
                raise ValueError(f"Index {index_name} does not exist")

        self.index = self.pc.Index(index_name)

    def _wait_for_index_ready(self, timeout: int = 120):
        """Wait for index to be ready."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                index_info = self.pc.describe_index(self.index_name)
                if index_info.status.ready:
                    print(f"Index {self.index_name} is ready")
                    return
            except Exception:
                pass
            time.sleep(2)
        raise TimeoutError(f"Index {self.index_name} not ready after {timeout}s")

    def generate_id(self, content: str, source_file: str, chunk_index: int) -> str:
        """
        Generate a unique ID for a vector.

        Args:
            content: Chunk content
            source_file: Source file path
            chunk_index: Chunk index

        Returns:
            Unique ID string
        """
        combined = f"{source_file}:{chunk_index}:{content[:100]}"
        return hashlib.md5(combined.encode()).hexdigest()

    def prepare_vector(
        self,
        embedding: List[float],
        content: str,
        source_file: str,
        chunk_index: int,
        metadata: Optional[Dict] = None
    ) -> VectorData:
        """
        Prepare a vector for upload.

        Args:
            embedding: Vector embedding
            content: Original text content
            source_file: Source file path
            chunk_index: Chunk index
            metadata: Additional metadata

        Returns:
            VectorData object
        """
        vector_id = self.generate_id(content, source_file, chunk_index)

        # Combine metadata
        full_metadata = {
            "source_file": source_file,
            "chunk_index": chunk_index,
            "content": content[:40000],          # 전문 저장 (Pinecone 40KB 한도)
            "content_preview": content[:1000],    # UI 미리보기용
            "content_length": len(content),
            **(metadata or {})
        }

        # Ensure metadata values are valid types
        sanitized_metadata = self._sanitize_metadata(full_metadata)

        return VectorData(
            id=vector_id,
            values=embedding,
            metadata=sanitized_metadata
        )

    def _sanitize_metadata(self, metadata: Dict) -> Dict:
        """Sanitize metadata to ensure valid types for Pinecone."""
        sanitized = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, list):
                # Only keep lists of strings
                if all(isinstance(v, str) for v in value):
                    sanitized[key] = value
                else:
                    sanitized[key] = str(value)
            elif value is None:
                sanitized[key] = ""
            else:
                sanitized[key] = str(value)
        return sanitized

    def upload_single(self, vector: VectorData, namespace: str = "") -> bool:
        """
        Upload a single vector.

        Args:
            vector: VectorData to upload
            namespace: Pinecone namespace

        Returns:
            True if successful
        """
        try:
            self.index.upsert(
                vectors=[{
                    "id": vector.id,
                    "values": vector.values,
                    "metadata": vector.metadata
                }],
                namespace=namespace
            )
            return True
        except Exception as e:
            print(f"Error uploading vector {vector.id}: {e}")
            return False

    def upload_batch(
        self,
        vectors: List[VectorData],
        namespace: str = "",
        batch_size: int = 100
    ) -> Dict[str, int]:
        """
        Upload multiple vectors in batches.

        Args:
            vectors: List of VectorData to upload
            namespace: Pinecone namespace
            batch_size: Number of vectors per batch

        Returns:
            Dictionary with success/failure counts
        """
        results = {"success": 0, "failed": 0}

        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]

            try:
                self.index.upsert(
                    vectors=[{
                        "id": v.id,
                        "values": v.values,
                        "metadata": v.metadata
                    } for v in batch],
                    namespace=namespace
                )
                results["success"] += len(batch)
            except Exception as e:
                print(f"Error uploading batch {i//batch_size}: {e}")
                results["failed"] += len(batch)

        return results

    def query(
        self,
        vector: List[float],
        top_k: int = 5,
        namespace: str = "",
        filter: Optional[Dict] = None,
        include_metadata: bool = True
    ) -> List[Dict]:
        """
        Query the index for similar vectors.

        Args:
            vector: Query vector
            top_k: Number of results to return
            namespace: Pinecone namespace
            filter: Metadata filter
            include_metadata: Whether to include metadata

        Returns:
            List of matching results
        """
        results = self.index.query(
            vector=vector,
            top_k=top_k,
            namespace=namespace,
            filter=filter,
            include_metadata=include_metadata
        )

        return [
            {
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata if include_metadata else None
            }
            for match in results.matches
        ]

    def query_namespaces(
        self,
        vector: List[float],
        namespaces: List[str],
        top_k: int = 5,
        filter: Optional[Dict] = None,
        include_metadata: bool = True
    ) -> List[Dict]:
        """
        Query multiple namespaces simultaneously using Pinecone server-side parallelism.

        Args:
            vector: Query vector
            namespaces: List of namespace names to query
            top_k: Number of results per namespace
            filter: Metadata filter
            include_metadata: Whether to include metadata

        Returns:
            Combined list of results from all namespaces, sorted by score
        """
        try:
            results = self.index.query_namespaces(
                vector=vector,
                namespaces=namespaces,
                top_k=top_k,
                filter=filter or {},
                include_metadata=include_metadata,
                metric="cosine"
            )

            combined = []
            for match in results.matches:
                combined.append({
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata if include_metadata else None,
                    "namespace": match.namespace
                })

            # Already sorted by score from Pinecone
            return combined[:top_k]

        except Exception as e:
            import logging
            logging.error(f"Multi-namespace query failed: {e}")
            # Fallback: sequential queries
            all_results = []
            for ns in namespaces:
                try:
                    ns_results = self.query(
                        vector=vector, top_k=top_k,
                        namespace=ns, filter=filter,
                        include_metadata=include_metadata
                    )
                    for r in ns_results:
                        r['namespace'] = ns
                    all_results.extend(ns_results)
                except Exception:
                    pass
            all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            return all_results[:top_k]

    def delete_by_ids(self, ids: List[str], namespace: str = "") -> bool:
        """Delete vectors by IDs."""
        try:
            self.index.delete(ids=ids, namespace=namespace)
            return True
        except Exception as e:
            print(f"Error deleting vectors: {e}")
            return False

    def delete_by_filter(self, filter: Dict, namespace: str = "") -> bool:
        """Delete vectors by metadata filter."""
        try:
            self.index.delete(filter=filter, namespace=namespace)
            return True
        except Exception as e:
            print(f"Error deleting vectors: {e}")
            return False

    def get_stats(self) -> Dict:
        """Get index statistics."""
        stats = self.index.describe_index_stats()
        return {
            "dimension": stats.dimension,
            "total_vector_count": stats.total_vector_count,
            "namespaces": stats.namespaces
        }


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "test-index")

    if api_key:
        uploader = PineconeUploader(
            api_key=api_key,
            index_name=index_name,
            dimension=1536
        )
        print(f"Index stats: {uploader.get_stats()}")
    else:
        print("PINECONE_API_KEY not found")
