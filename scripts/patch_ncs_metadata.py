"""Patch ncs_category and ncs_section_type for existing vectors in Pinecone."""
import sys
import unicodedata
import re
import time
import concurrent.futures
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')
from pinecone import Pinecone
from src.ncs_utils import NCS_CATEGORIES, classify_section

pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index = pc.Index(os.getenv('PINECONE_INDEX_NAME', 'document-index'))

# Collect all vector IDs
print("Listing vector IDs...", flush=True)
all_ids = []
for ids_batch in index.list(namespace='semiconductor'):
    all_ids.extend(ids_batch)
print(f"Total vectors: {len(all_ids)}", flush=True)


# Phase 1: Fetch all metadata and compute updates
print("Fetching metadata and computing updates...", flush=True)
updates_to_apply = []
fetch_batch = 100

for i in range(0, len(all_ids), fetch_batch):
    batch_ids = all_ids[i:i + fetch_batch]
    fetch_result = index.fetch(ids=batch_ids, namespace='semiconductor')

    for vid, vdata in fetch_result.get('vectors', {}).items():
        meta = vdata.get('metadata', {})
        source_file = meta.get('source_file', '')
        normalized = unicodedata.normalize('NFC', source_file)

        updates = {}

        # Fix ncs_category
        current_cat = meta.get('ncs_category')
        if not current_cat:
            for cat in NCS_CATEGORIES:
                if cat in normalized:
                    updates['ncs_category'] = cat
                    break

        # Fix ncs_section_type
        section_title = meta.get('section_title', '')
        if section_title:
            normalized_title = unicodedata.normalize('NFC', str(section_title))
            stype, lu = classify_section(normalized_title)
            if stype != 'general' and meta.get('ncs_section_type') == 'general':
                updates['ncs_section_type'] = stype
                if lu is not None:
                    updates['learning_unit'] = lu

        if updates:
            updates_to_apply.append((vid, updates))

    done = min(i + fetch_batch, len(all_ids))
    if done % 2000 < fetch_batch:
        print(f"  Fetched {done}/{len(all_ids)} vectors...", flush=True)

print(f"\nUpdates needed: {len(updates_to_apply)}", flush=True)

if not updates_to_apply:
    print("No updates needed. Exiting.")
    sys.exit(0)


# Phase 2: Apply updates with ThreadPool
MAX_RETRIES = 5

def apply_update(args):
    vid, upd = args
    for attempt in range(MAX_RETRIES):
        try:
            index.update(id=vid, set_metadata=upd, namespace='semiconductor')
            return True
        except Exception as e:
            if '429' in str(e) or 'rate' in str(e).lower():
                wait = (2 ** attempt) * 0.5
                time.sleep(wait)
            elif attempt < MAX_RETRIES - 1:
                time.sleep(0.5)
            else:
                return False
    return False


print(f"Applying {len(updates_to_apply)} updates with 10 parallel workers...", flush=True)
start = time.time()

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(apply_update, updates_to_apply))

success = sum(1 for r in results if r)
fail = sum(1 for r in results if not r)
elapsed = time.time() - start

# Count categories
cat_counts = {}
for _, upd in updates_to_apply:
    cat = upd.get('ncs_category', '')
    if cat:
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

section_counts = {}
for _, upd in updates_to_apply:
    st = upd.get('ncs_section_type', '')
    if st:
        section_counts[st] = section_counts.get(st, 0) + 1

print(f"\n=== Update Complete ({elapsed:.1f}s) ===", flush=True)
print(f"Success: {success}, Failed: {fail}", flush=True)
print(f"Category distribution: {cat_counts}", flush=True)
print(f"Section type updates: {section_counts}", flush=True)
