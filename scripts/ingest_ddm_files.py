#!/usr/bin/env python3
"""
동대문구의회 회의록 개별 .md 파일 → DDM Pinecone 인덱스 업로드

사용법:
    python scripts/ingest_ddm_files.py /path/to/회차별/
    python scripts/ingest_ddm_files.py /path/to/회차별/ --namespace ddm
    python scripts/ingest_ddm_files.py /path/to/회차별/ --dry-run
    python scripts/ingest_ddm_files.py /path/to/회차별/ --delete-first
"""

import argparse
import hashlib
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DDM_INDEX_HOST = "https://ddm-lhicikc.svc.aped-4627-b74a.pinecone.io"
EMBEDDING_MODEL = "text-embedding-3-small"
DIMENSION = 512
KOREAN_TOKEN_RATIO = 1.8
MAX_ITEM_TOKENS = 7000


# ── 1. 파싱 ──────────────────────────────────────────────────────────────────

def parse_file(path: Path) -> Tuple[Dict, str]:
    """YAML 프론트매터와 본문을 분리해 반환."""
    text = path.read_text(encoding="utf-8")

    # 첫 번째 --- 와 두 번째 --- 사이가 YAML 프론트매터
    fm_match = re.match(r"^---\n(.*?)\n---\n?", text, re.DOTALL)
    meta: Dict = {"source_file": path.name, "filename": path.stem}
    if fm_match:
        fm = fm_match.group(1)
        body = text[fm_match.end():]
        meta.update(_parse_yaml_fields(fm))
    else:
        body = text

    return meta, body.strip()


def _parse_yaml_fields(fm: str) -> Dict:
    """간단한 YAML 스칼라 필드 파싱 (speakers/attendance 등 복잡한 필드는 제외)."""
    result: Dict = {}

    def _get(key: str) -> Optional[str]:
        m = re.search(rf"^{key}:\s*(.+)$", fm, re.MULTILINE)
        return m.group(1).strip().strip('"') if m else None

    for field in ("title", "date", "session", "assembly", "order", "meeting_type",
                  "url", "time_open", "time_close"):
        v = _get(field)
        if v:
            result[field] = v

    members = _get("members_present")
    if members:
        try:
            result["members_present"] = int(members)
        except ValueError:
            pass

    # 발언자 목록: 쉼표 구분 문자열로 저장
    speakers = re.findall(r"^\s+- name:\s*(.+)$", fm, re.MULTILINE)
    if speakers:
        result["speakers"] = ", ".join(speakers[:20])  # 상위 20명까지

    return result


# ── 2. 청킹 ──────────────────────────────────────────────────────────────────

def _count_tokens(text: str) -> int:
    return int(len(text) * KOREAN_TOKEN_RATIO)


def chunk_body(body: str, max_tokens: int = 800, overlap_lines: int = 2) -> List[str]:
    """단락 단위 청킹. max_tokens 초과 시 새 청크 시작."""
    paragraphs = re.split(r"\n{2,}", body)
    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        t = _count_tokens(para)
        if current_tokens + t > max_tokens and current:
            chunks.append("\n\n".join(current))
            current = current[-overlap_lines:] if overlap_lines else []
            current_tokens = sum(_count_tokens(p) for p in current)
        current.append(para)
        current_tokens += t

    if current:
        chunks.append("\n\n".join(current))

    return [c for c in chunks if c.strip()]


def build_chunks(meta: Dict, body: str, max_tokens: int = 800) -> List[Dict]:
    """파일 1개를 청크 리스트로 변환."""
    parts = chunk_body(body, max_tokens=max_tokens)
    result = []
    for idx, part in enumerate(parts):
        chunk_meta = {
            **meta,
            "chunk_index": idx,
            "chunk_count": len(parts),
            "content_preview": part[:500],
            "content": part[:10000],
        }
        raw = f"{meta['filename']}:{idx}:{part[:80]}"
        vector_id = hashlib.md5(raw.encode()).hexdigest()
        result.append({"id": vector_id, "text": part, "metadata": chunk_meta})
    return result


# ── 3. 임베딩 ─────────────────────────────────────────────────────────────────

def _truncate(text: str) -> str:
    return text[:int(MAX_ITEM_TOKENS / KOREAN_TOKEN_RATIO)]


def embed_batch(client: OpenAI, texts: List[str]) -> List[List[float]]:
    safe = [_truncate(t) for t in texts]
    resp = client.embeddings.create(input=safe, model=EMBEDDING_MODEL, dimensions=DIMENSION)
    return [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]


# ── 4. 업로드 ─────────────────────────────────────────────────────────────────

def _sanitize(meta: Dict) -> Dict:
    out = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, list) and all(isinstance(x, str) for x in v):
            out[k] = v
        elif v is None:
            out[k] = ""
        else:
            out[k] = str(v)
    return out


def upsert_chunks(index, chunks: List[Dict], client: OpenAI,
                  batch_size: int, namespace: str) -> Tuple[int, int]:
    success = failed = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        try:
            embeddings = embed_batch(client, [c["text"] for c in batch])
        except Exception as e:
            logger.error("임베딩 실패 (batch %d): %s", i // batch_size, e)
            failed += len(batch)
            time.sleep(2)
            continue
        vectors = [
            {"id": c["id"], "values": emb, "metadata": _sanitize(c["metadata"])}
            for c, emb in zip(batch, embeddings)
        ]
        try:
            index.upsert(vectors=vectors, namespace=namespace)
            success += len(batch)
        except Exception as e:
            logger.error("Pinecone 업서트 실패: %s", e)
            failed += len(batch)
    return success, failed


# ── 5. 메인 ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DDM 회차별 개별 파일 Pinecone 업로드")
    parser.add_argument("directory", help="회차별 .md 파일이 있는 디렉터리")
    parser.add_argument("--namespace", default="", help="Pinecone 네임스페이스 (기본: 빈 문자열)")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--delete-first", action="store_true",
                        help="업로드 전 네임스페이스 전체 삭제")
    parser.add_argument("--dry-run", action="store_true",
                        help="임베딩/업로드 없이 청킹 결과만 확인")
    args = parser.parse_args()

    openai_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    if not openai_key:
        sys.exit("OPENAI_API_KEY가 .env에 없습니다.")
    if not pinecone_key:
        sys.exit("PINECONE_API_KEY가 .env에 없습니다.")

    md_files = sorted(Path(args.directory).glob("*.md"))
    logger.info("처리 대상: %d개 .md 파일", len(md_files))

    if args.dry_run:
        total_chunks = 0
        for fp in md_files[:5]:
            meta, body = parse_file(fp)
            chunks = build_chunks(meta, body, args.max_tokens)
            total_chunks += len(chunks)
            print(f"\n{fp.name}: {len(chunks)}청크, 첫 청크 미리보기:")
            print(chunks[0]["text"][:200] if chunks else "(비어있음)")
        print(f"\n[dry-run] 샘플 5개 파일 → {total_chunks}청크 (전체 {len(md_files)}파일 추정 {total_chunks // 5 * len(md_files)}청크)")
        return

    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index(host=DDM_INDEX_HOST)
    oai = OpenAI(api_key=openai_key)

    if args.delete_first:
        logger.info("네임스페이스 '%s' 전체 삭제 중...", args.namespace)
        index.delete(delete_all=True, namespace=args.namespace)
        logger.info("삭제 완료")

    total_success = total_failed = 0
    for i, fp in enumerate(md_files, 1):
        meta, body = parse_file(fp)
        if not body:
            logger.warning("[%d/%d] %s: 본문 없음, 건너뜀", i, len(md_files), fp.name)
            continue
        chunks = build_chunks(meta, body, args.max_tokens)
        s, f = upsert_chunks(index, chunks, oai, args.batch_size, args.namespace)
        total_success += s
        total_failed += f
        logger.info("[%d/%d] %s → %d청크 업로드 (성공 %d, 실패 %d)",
                    i, len(md_files), fp.name, len(chunks), s, f)

    stats = index.describe_index_stats()
    print(f"\n완료: 성공 {total_success}개, 실패 {total_failed}개 청크")
    print(f"인덱스 총 벡터 수: {stats.total_vector_count}")


if __name__ == "__main__":
    main()
