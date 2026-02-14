"""Test metadata filtering for search quality comparison."""
import os, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')
from src.embedding_generator import EmbeddingGenerator
from pinecone import Pinecone

pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index = pc.Index(os.getenv('PINECONE_INDEX_NAME', 'document-index'))
embedder = EmbeddingGenerator(api_key=os.getenv('OPENAI_API_KEY'))

query = '반도체 장비 유지보수 절차'
emb = embedder.generate(query)

# 1) No filter
print('=== 필터 없이 검색 ===')
results = index.query(vector=emb.embedding, top_k=5, namespace='semiconductor', include_metadata=True)
for i, m in enumerate(results['matches']):
    meta = m.get('metadata', {})
    cat = meta.get('ncs_category', 'N/A')
    title = meta.get('ncs_document_title', meta.get('filename', ''))
    print(f'  [{i+1}] {m["score"]:.3f} | {cat} | {title}')

# 2) ncs_category filter
print('\n=== ncs_category=반도체장비 필터 ===')
results2 = index.query(
    vector=emb.embedding, top_k=5, namespace='semiconductor',
    include_metadata=True,
    filter={"ncs_category": {"$eq": "반도체장비"}}
)
for i, m in enumerate(results2['matches']):
    meta = m.get('metadata', {})
    cat = meta.get('ncs_category', 'N/A')
    title = meta.get('ncs_document_title', meta.get('filename', ''))
    print(f'  [{i+1}] {m["score"]:.3f} | {cat} | {title}')

# 3) Image filter
print('\n=== file_type=image 필터 ===')
results3 = index.query(
    vector=emb.embedding, top_k=5, namespace='semiconductor',
    include_metadata=True,
    filter={"file_type": {"$eq": "image"}}
)
for i, m in enumerate(results3['matches']):
    meta = m.get('metadata', {})
    cat = meta.get('ncs_category', 'N/A')
    fname = meta.get('filename', '?')
    preview = meta.get('content_preview', '')[:100]
    print(f'  [{i+1}] {m["score"]:.3f} | {cat} | {fname}')
    print(f'      {preview}')
