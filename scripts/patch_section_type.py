"""Patch ncs_section_type for existing vectors using improved patterns.

Strips markdown bold markers and normalizes special characters before matching.
"""
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
from src.ncs_utils import classify_section

pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index = pc.Index(os.getenv('PINECONE_INDEX_NAME', 'document-index'))

# Collect all vector IDs
print("Listing vector IDs...", flush=True)
all_ids = []
for ids_batch in index.list(namespace='semiconductor'):
    all_ids.extend(ids_batch)
print(f"Total vectors: {len(all_ids)}", flush=True)


# Phase 1: Fetch all metadata and compute updates
print("Fetching metadata and computing section type updates...", flush=True)
updates_to_apply = []
fetch_batch = 100
current_types = {}

for i in range(0, len(all_ids), fetch_batch):
    batch_ids = all_ids[i:i + fetch_batch]
    fetch_result = index.fetch(ids=batch_ids, namespace='semiconductor')

    for vid, vdata in fetch_result.get('vectors', {}).items():
        meta = vdata.get('metadata', {})
        file_type = meta.get('file_type', '')

        # Only update markdown vectors (images don't have meaningful section_titles)
        if file_type != 'markdown':
            continue

        section_title = meta.get('section_title', '')
        if not section_title:
            continue

        current_stype = meta.get('ncs_section_type', 'general')
        new_stype, lu = classify_section(section_title)

        # Track current types for reporting
        current_types[current_stype] = current_types.get(current_stype, 0) + 1

        if new_stype != 'general' and new_stype != current_stype:
            updates = {'ncs_section_type': new_stype}
            if lu is not None:
                updates['learning_unit'] = lu
            updates_to_apply.append((vid, updates))

    done = min(i + fetch_batch, len(all_ids))
    if done % 3000 < fetch_batch:
        print(f"  Scanned {done}/{len(all_ids)} vectors...", flush=True)

print(f"\nCurrent section type distribution: {current_types}", flush=True)
print(f"Updates needed: {len(updates_to_apply)}", flush=True)

# Preview update distribution
update_counts = {}
for _, upd in updates_to_apply:
    st = upd.get('ncs_section_type', '')
    update_counts[st] = update_counts.get(st, 0) + 1
print(f"Update distribution: {update_counts}", flush=True)

if not updates_to_apply:
    print("No updates needed. Exiting.")
    sys.exit(0)


# Phase 2: Apply updates
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


print(f"\nApplying {len(updates_to_apply)} updates with 10 parallel workers...", flush=True)
start = time.time()

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(apply_update, updates_to_apply))

success = sum(1 for r in results if r)
fail = sum(1 for r in results if not r)
elapsed = time.time() - start

print(f"\n=== Update Complete ({elapsed:.1f}s) ===", flush=True)
print(f"Success: {success}, Failed: {fail}", flush=True)
print(f"Section type updates: {update_counts}", flush=True)
