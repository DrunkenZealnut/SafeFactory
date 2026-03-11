"""DDM 인덱스 검색 테스트"""
import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

DDM_INDEX_HOST = "https://ddm-lhicikc.svc.aped-4627-b74a.pinecone.io"
EMBEDDING_MODEL = "text-embedding-3-small"
DIMENSION = 512

oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc  = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
idx = pc.Index(host=DDM_INDEX_HOST)

queries = [
    "의장 선거",
    "예산안 심의",
    "주민 청원",
]

for q in queries:
    emb = oai.embeddings.create(input=[q], model=EMBEDDING_MODEL, dimensions=DIMENSION).data[0].embedding
    res = idx.query(vector=emb, top_k=3, include_metadata=True)

    print(f"\n{'='*60}")
    print(f"쿼리: {q}")
    print(f"{'='*60}")
    for i, m in enumerate(res.matches, 1):
        meta = m.metadata or {}
        print(f"\n[{i}] score={m.score:.4f}")
        print(f"    날짜: {meta.get('date','')}  회의: {meta.get('assembly','')} {meta.get('order','')}")
        print(f"    제목: {meta.get('title','')}")
        preview = meta.get('content_preview', '')[:150].replace('\n', ' ')
        print(f"    내용: {preview}...")
