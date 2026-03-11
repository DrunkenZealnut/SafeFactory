"""
DDM Pinecone Index Client
ddm 인덱스에 직접 연결하는 클라이언트
Host: https://ddm-lhicikc.svc.aped-4627-b74a.pinecone.io
"""

import os
import logging
from typing import List, Dict, Optional, Any

from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

DDM_INDEX_HOST = "https://ddm-lhicikc.svc.aped-4627-b74a.pinecone.io"
DDM_INDEX_NAME = "ddm"


class DDMClient:
    """DDM Pinecone 인덱스 클라이언트."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: Pinecone API 키. 없으면 환경변수 PINECONE_API_KEY 사용.
        """
        key = api_key or os.getenv("PINECONE_API_KEY")
        if not key:
            raise ValueError("PINECONE_API_KEY가 설정되지 않았습니다.")

        self.pc = Pinecone(api_key=key)
        # host를 직접 지정해 네임 조회 없이 바로 연결
        self.index = self.pc.Index(host=DDM_INDEX_HOST)
        logger.info("DDM 인덱스 연결 완료: %s", DDM_INDEX_HOST)

    # ── 조회 ──────────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """인덱스 통계를 반환합니다."""
        raw = self.index.describe_index_stats()
        return {
            "dimension": raw.dimension,
            "total_vector_count": raw.total_vector_count,
            "namespaces": {
                ns: {"vector_count": info.vector_count}
                for ns, info in (raw.namespaces or {}).items()
            },
        }

    def query(
        self,
        vector: List[float],
        top_k: int = 5,
        namespace: str = "",
        filter: Optional[Dict] = None,
        include_metadata: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        벡터로 유사 항목을 검색합니다.

        Args:
            vector: 쿼리 벡터
            top_k: 반환할 결과 수
            namespace: Pinecone 네임스페이스
            filter: 메타데이터 필터
            include_metadata: 메타데이터 포함 여부

        Returns:
            [{"id": ..., "score": ..., "metadata": ...}, ...]
        """
        res = self.index.query(
            vector=vector,
            top_k=top_k,
            namespace=namespace,
            filter=filter,
            include_metadata=include_metadata,
        )
        return [
            {
                "id": m.id,
                "score": m.score,
                "metadata": m.metadata if include_metadata else None,
            }
            for m in res.matches
        ]

    def fetch(self, ids: List[str], namespace: str = "") -> Dict[str, Any]:
        """ID 목록으로 벡터를 가져옵니다."""
        return self.index.fetch(ids=ids, namespace=namespace)

    # ── 업서트 / 삭제 ─────────────────────────────────────────────────────────

    def upsert(
        self,
        vectors: List[Dict[str, Any]],
        namespace: str = "",
        batch_size: int = 100,
    ) -> Dict[str, int]:
        """
        벡터를 업서트합니다.

        Args:
            vectors: [{"id": str, "values": List[float], "metadata": dict}, ...]
            namespace: Pinecone 네임스페이스
            batch_size: 배치 크기

        Returns:
            {"success": int, "failed": int}
        """
        results = {"success": 0, "failed": 0}
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            try:
                self.index.upsert(vectors=batch, namespace=namespace)
                results["success"] += len(batch)
            except Exception as e:
                logger.error("배치 업서트 실패 (batch %d): %s", i // batch_size, e)
                results["failed"] += len(batch)
        return results

    def delete_by_ids(self, ids: List[str], namespace: str = "") -> bool:
        """ID 목록으로 벡터를 삭제합니다."""
        try:
            self.index.delete(ids=ids, namespace=namespace)
            return True
        except Exception as e:
            logger.error("벡터 삭제 실패: %s", e)
            return False

    def delete_by_filter(self, filter: Dict, namespace: str = "") -> bool:
        """메타데이터 필터로 벡터를 삭제합니다."""
        try:
            self.index.delete(filter=filter, namespace=namespace)
            return True
        except Exception as e:
            logger.error("필터 삭제 실패: %s", e)
            return False


# ── 단독 실행 테스트 ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    client = DDMClient()

    print("=== DDM 인덱스 통계 ===")
    import json
    print(json.dumps(client.stats(), indent=2, ensure_ascii=False))
