"""Verify ncs_code and document_summary patches were applied correctly."""
import sys, os, random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')
from pinecone import Pinecone
from src.embedding_generator import EmbeddingGenerator

pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index = pc.Index(os.getenv('PINECONE_INDEX_NAME', 'document-index'))

# Collect all vector IDs
print("Listing vector IDs...", flush=True)
all_ids = []
for ids_batch in index.list(namespace='semiconductor'):
    all_ids.extend(ids_batch)
print(f"Total vectors: {len(all_ids)}")

# Sample and check
sample_ids = random.sample(all_ids, min(500, len(all_ids)))
fetch_batch = 100

stats = {
    'total_sampled': 0,
    'has_ncs_code': 0,
    'missing_ncs_code': 0,
    'has_source_with_LM': 0,
    'summary_checked': 0,
    'summary_good': 0,
    'summary_bad': 0,
}

missing_ncs_examples = []

for i in range(0, len(sample_ids), fetch_batch):
    batch = sample_ids[i:i + fetch_batch]
    result = index.fetch(ids=batch, namespace='semiconductor')

    for vid, vdata in result.get('vectors', {}).items():
        meta = vdata.get('metadata', {})
        stats['total_sampled'] += 1
        source = meta.get('source_file', '')

        # ncs_code check
        ncs_code = meta.get('ncs_code', '')
        has_lm = 'LM1903' in source
        if has_lm:
            stats['has_source_with_LM'] += 1
        if ncs_code:
            stats['has_ncs_code'] += 1
        elif has_lm:
            stats['missing_ncs_code'] += 1
            if len(missing_ncs_examples) < 5:
                missing_ncs_examples.append(source[:100])

        # document_summary check (chunk_index=0 only)
        chunk_idx = meta.get('chunk_index', -1)
        if chunk_idx == 0 and meta.get('file_type') == 'markdown':
            stats['summary_checked'] += 1
            summary = meta.get('document_summary', '')
            if summary and not summary.startswith('![') and not summary.startswith('|') and len(summary.strip()) > 20:
                stats['summary_good'] += 1
            else:
                stats['summary_bad'] += 1

print(f"\n=== Verification Results (sample={stats['total_sampled']}) ===")
print(f"ncs_code present: {stats['has_ncs_code']}/{stats['has_source_with_LM']} LM-sourced vectors")
print(f"ncs_code missing: {stats['missing_ncs_code']}")
if missing_ncs_examples:
    print(f"  Missing examples:")
    for ex in missing_ncs_examples:
        print(f"    {ex}")
print(f"document_summary checked: {stats['summary_checked']}")
print(f"  Good: {stats['summary_good']}, Bad: {stats['summary_bad']}")

# Also check specific unversioned document
print("\n=== Unversioned document spot check ===")
embedder = EmbeddingGenerator(api_key=os.getenv('OPENAI_API_KEY'))
emb = embedder.generate("반도체 신뢰성 평가")
results = index.query(
    vector=emb.embedding, top_k=5, namespace='semiconductor',
    include_metadata=True,
    filter={"ncs_code": {"$eq": "LM1903060110"}}
)
print(f"Query for ncs_code=LM1903060110 (unversioned): {len(results['matches'])} results")
for m in results['matches'][:3]:
    meta = m.get('metadata', {})
    print(f"  score={m['score']:.3f} | code={meta.get('ncs_code','')} | title={meta.get('ncs_document_title','')[:50]}")
