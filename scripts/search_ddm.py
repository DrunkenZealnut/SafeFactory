"""DDM 인덱스 검색 스크립트
사용법: python scripts/search_ddm.py "검색어" [--top-k 20]
"""
import os
import argparse
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

DDM_INDEX_HOST = "https://ddm-lhicikc.svc.aped-4627-b74a.pinecone.io"

def search(query: str, top_k: int = 20):
    oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    pc  = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    idx = pc.Index(host=DDM_INDEX_HOST)

    emb = oai.embeddings.create(input=[query], model="text-embedding-3-small", dimensions=512).data[0].embedding
    res = idx.query(vector=emb, top_k=top_k, include_metadata=True)

    for i, m in enumerate(res.matches, 1):
        meta = m.metadata or {}
        print(f"\n[{i}] score={m.score:.4f} | {meta.get('date','')} | {meta.get('title','')} | chunk {meta.get('chunk_index','')}")
        print(meta.get('content_preview','')[:400])
        print("---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="검색어")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    search(args.query, args.top_k)
