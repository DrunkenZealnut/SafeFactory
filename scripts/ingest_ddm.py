"""
동대문구의회 9대 본회의 회의록 → DDM Pinecone 인덱스 업로드 스크립트

사용법:
    python scripts/ingest_ddm.py /Users/zealnutkim/Downloads/동대문구의회_9대_본회의_통합.md
    python scripts/ingest_ddm.py /path/to/file.md --batch-size 50 --max-tokens 600
"""

import os
import re
import sys
import hashlib
import logging
import argparse
from typing import List, Dict, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DDM_INDEX_HOST = "https://ddm-lhicikc.svc.aped-4627-b74a.pinecone.io"
EMBEDDING_MODEL = "text-embedding-3-small"
DIMENSION = 512   # ddm 인덱스는 512차원으로 생성됨
NAMESPACE = ""    # 필요하면 변경

# OpenAI text-embedding-3-small은 8191 토큰/item 제한
# 한국어는 1글자 ≈ 1.5~2 OpenAI 토큰 → 보수적으로 1.8 사용
# 배치 전송 시 총 토큰 합산 제한도 있으므로 배치 크기는 작게 유지
KOREAN_TOKEN_RATIO = 1.8   # 한국어 글자 → OpenAI 토큰 변환 비율
MAX_ITEM_TOKENS = 7000     # item당 OpenAI 토큰 상한 (8191 여유분)


# ── 1. 파일 분할 ──────────────────────────────────────────────────────────────

def split_into_meetings(text: str) -> List[Dict]:
    """
    ## [N/77] filename 패턴으로 회의별로 분할하고 메타데이터를 추출한다.
    """
    # 각 회의 섹션의 시작 위치를 찾는다
    pattern = re.compile(r"^## \[(\d+)/\d+\] (.+)$", re.MULTILINE)
    matches = list(pattern.finditer(text))

    meetings = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()

        seq = int(m.group(1))      # 1~77
        filename = m.group(2).strip()

        # YAML 프론트매터 파싱 (--- ... --- 블록)
        meta = parse_frontmatter(section_text, filename)
        meta["seq"] = seq
        meta["filename"] = filename

        # 프론트매터를 제거한 본문만 추출
        body = strip_frontmatter(section_text)

        meetings.append({"meta": meta, "body": body})

    logger.info("총 %d개 회의 분리 완료", len(meetings))
    return meetings


def parse_frontmatter(section: str, filename: str) -> Dict:
    """YAML 프론트매터에서 핵심 필드를 추출한다."""
    meta: Dict = {"source_file": filename}

    # --- ... --- 블록
    fm_match = re.search(r"---\n(.*?)\n---", section, re.DOTALL)
    if not fm_match:
        return meta

    fm = fm_match.group(1)

    def _get(key: str) -> Optional[str]:
        m = re.search(rf"^{key}:\s*(.+)$", fm, re.MULTILINE)
        return m.group(1).strip().strip('"') if m else None

    meta["title"]        = _get("title") or filename
    meta["date"]         = _get("date") or ""
    meta["session"]      = _get("session") or ""
    meta["assembly"]     = _get("assembly") or ""
    meta["order"]        = _get("order") or ""
    meta["meeting_type"] = _get("meeting_type") or "본회의"
    meta["url"]          = _get("url") or ""
    meta["time_open"]    = _get("time_open") or ""
    meta["time_close"]   = _get("time_close") or ""

    members = _get("members_present")
    if members:
        try:
            meta["members_present"] = int(members)
        except ValueError:
            pass

    return meta


def strip_frontmatter(section: str) -> str:
    """## 헤더와 YAML 프론트매터를 제거하고 본문만 반환한다."""
    # 첫 번째 ## 헤더 제거
    text = re.sub(r"^## \[\d+/\d+\] .+\n", "", section, count=1)
    # --- ... --- 블록 제거
    text = re.sub(r"---\n.*?\n---\n?", "", text, flags=re.DOTALL)
    return text.strip()


# ── 2. 청킹 ──────────────────────────────────────────────────────────────────

def count_tokens(text: str) -> int:
    """한/영 혼합 근사: 글자수 × 1.8 (한국어 보수적 추정)"""
    return int(len(text) * KOREAN_TOKEN_RATIO)


def chunk_text(text: str, max_tokens: int = 500, overlap_lines: int = 2) -> List[str]:
    """
    단락 단위로 청킹한다. max_tokens 초과 시 새 청크를 시작한다.
    """
    paragraphs = re.split(r"\n{2,}", text)
    chunks: List[str] = []
    current_lines: List[str] = []
    current_tokens = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        t = count_tokens(para)
        if current_tokens + t > max_tokens and current_lines:
            chunks.append("\n\n".join(current_lines))
            # 오버랩: 마지막 N개 단락을 다음 청크에 포함
            current_lines = current_lines[-overlap_lines:] if overlap_lines else []
            current_tokens = sum(count_tokens(p) for p in current_lines)
        current_lines.append(para)
        current_tokens += t

    if current_lines:
        chunks.append("\n\n".join(current_lines))

    return [c for c in chunks if c.strip()]


def build_chunks(meetings: List[Dict], max_tokens: int = 800) -> List[Dict]:
    """각 회의를 청킹하고 최종 업로드 단위를 만든다."""
    all_chunks = []
    for meeting in meetings:
        meta = meeting["meta"]
        body = meeting["body"]
        parts = chunk_text(body, max_tokens=max_tokens)

        for idx, part in enumerate(parts):
            chunk_meta = {
                **meta,
                "chunk_index": idx,
                "chunk_count": len(parts),
                # Pinecone 메타데이터 검색용 미리보기
                "content_preview": part[:500],
                "content": part[:10000],   # 한국어 3bytes/자 × 10000 ≈ 30KB < 40KB 한도
            }
            # 고유 ID: 파일명 + 회의 순번 + 청크 인덱스 + 내용 앞 80자
            raw = f"{meta['filename']}:{meta['seq']}:{idx}:{part[:80]}"
            vector_id = hashlib.md5(raw.encode()).hexdigest()

            all_chunks.append({"id": vector_id, "text": part, "metadata": chunk_meta})

    logger.info("총 %d개 청크 생성", len(all_chunks))
    return all_chunks


# ── 3. 임베딩 ─────────────────────────────────────────────────────────────────

def truncate_to_tokens(text: str, max_tokens: int = MAX_ITEM_TOKENS) -> str:
    """토큰 추정치 기준으로 텍스트를 잘라낸다."""
    max_chars = int(max_tokens / KOREAN_TOKEN_RATIO)
    return text[:max_chars]


def embed_batch(client: OpenAI, texts: List[str], model: str = EMBEDDING_MODEL) -> List[List[float]]:
    """OpenAI 임베딩 API 호출. 512차원으로 반환."""
    safe_texts = [truncate_to_tokens(t) for t in texts]
    response = client.embeddings.create(
        input=safe_texts,
        model=model,
        dimensions=DIMENSION,   # text-embedding-3-small은 dimensions 파라미터 지원
    )
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


# ── 4. 업로드 ─────────────────────────────────────────────────────────────────

def upsert_to_pinecone(
    index,
    chunks: List[Dict],
    openai_client: OpenAI,
    batch_size: int = 50,
    namespace: str = NAMESPACE,
) -> Tuple[int, int]:
    """청크를 임베딩 후 Pinecone에 업서트한다. (success, failed) 반환."""
    success = failed = 0
    total = len(chunks)

    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]

        try:
            embeddings = embed_batch(openai_client, texts)
        except Exception as e:
            logger.error("임베딩 실패 (batch %d): %s", i // batch_size, e)
            failed += len(batch)
            continue

        vectors = [
            {
                "id": chunk["id"],
                "values": emb,
                "metadata": _sanitize(chunk["metadata"]),
            }
            for chunk, emb in zip(batch, embeddings)
        ]

        try:
            index.upsert(vectors=vectors, namespace=namespace)
            success += len(batch)
            logger.info(
                "업로드 %d/%d (batch %d → %d개)",
                min(i + batch_size, total),
                total,
                i // batch_size + 1,
                len(batch),
            )
        except Exception as e:
            logger.error("Pinecone 업서트 실패 (batch %d): %s", i // batch_size, e)
            failed += len(batch)

    return success, failed


def _sanitize(meta: Dict) -> Dict:
    """Pinecone 허용 타입(str/int/float/bool/List[str])으로 정제."""
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


# ── 5. 메인 ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DDM 회의록 Pinecone 업로드")
    parser.add_argument("file", help="업로드할 .md 파일 경로")
    parser.add_argument("--batch-size", type=int, default=20, help="업로드 배치 크기 (기본 20)")
    parser.add_argument("--max-tokens", type=int, default=800, help="청크당 최대 추정 토큰 수 (기본 800 = ~444자)")
    parser.add_argument("--namespace", default=NAMESPACE, help="Pinecone 네임스페이스")
    parser.add_argument("--dry-run", action="store_true", help="임베딩/업로드 없이 청킹 결과만 확인")
    args = parser.parse_args()

    # API 키 확인
    openai_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    if not openai_key:
        sys.exit("OPENAI_API_KEY가 .env에 없습니다.")
    if not pinecone_key:
        sys.exit("PINECONE_API_KEY가 .env에 없습니다.")

    # 파일 읽기
    logger.info("파일 읽기: %s", args.file)
    with open(args.file, encoding="utf-8") as f:
        text = f.read()

    # 파이프라인
    meetings = split_into_meetings(text)
    chunks = build_chunks(meetings, max_tokens=args.max_tokens)

    if args.dry_run:
        print(f"\n[dry-run] 회의 수: {len(meetings)}, 청크 수: {len(chunks)}")
        for c in chunks[:3]:
            print(f"\n--- {c['metadata'].get('title','')} chunk {c['metadata']['chunk_index']} ---")
            print(c["text"][:200])
        return

    # Pinecone 연결
    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index(host=DDM_INDEX_HOST)
    logger.info("Pinecone 연결: %s", DDM_INDEX_HOST)

    # OpenAI 클라이언트
    oai = OpenAI(api_key=openai_key)

    # 업로드
    success, failed = upsert_to_pinecone(
        index, chunks, oai,
        batch_size=args.batch_size,
        namespace=args.namespace,
    )

    print(f"\n완료: 성공 {success}개, 실패 {failed}개 (전체 {len(chunks)}개 청크)")

    # 최종 통계
    stats = index.describe_index_stats()
    print(f"인덱스 총 벡터 수: {stats.total_vector_count}")


if __name__ == "__main__":
    main()
