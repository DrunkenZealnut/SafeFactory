"""Test section type filtering after patch."""
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

def search(query, filt=None, label=""):
    emb = embedder.generate(query)
    kwargs = dict(vector=emb.embedding, top_k=5, namespace='semiconductor', include_metadata=True)
    if filt:
        kwargs['filter'] = filt
    results = index.query(**kwargs)
    print(f'\n=== {label or query} ===')
    for i, m in enumerate(results['matches']):
        meta = m.get('metadata', {})
        stype = meta.get('ncs_section_type', '?')
        cat = meta.get('ncs_category', '?')
        title = meta.get('section_title', '')[:70]
        print(f'  [{i+1}] {m["score"]:.3f} | {stype:25s} | {cat} | {title}')

# Test 1: performance_procedure filter
search('반도체 패키지 설계 수행 순서',
       {"ncs_section_type": {"$eq": "performance_procedure"}},
       "수행 순서 필터")

# Test 2: required_knowledge filter
search('반도체 식각 공정 이론',
       {"ncs_section_type": {"$eq": "required_knowledge"}},
       "필요 지식 필터")

# Test 3: safety_notes filter
search('반도체 안전 유의사항',
       {"ncs_section_type": {"$eq": "safety_notes"}},
       "안전 유의사항 필터")

# Test 4: evaluation_method filter
search('반도체 평가 방법',
       {"ncs_section_type": {"$eq": "evaluation_method"}},
       "평가 방법 필터")

# Test 5: equipment filter
search('반도체 장비 기기 공구',
       {"ncs_section_type": {"$eq": "equipment"}},
       "장비/기기 필터")

# Test 6: no filter (baseline)
search('반도체 패키지 설계 수행 순서', label="필터 없음 (baseline)")
