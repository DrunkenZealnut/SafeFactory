"""Patch ncs_code for unversioned NCS documents (e.g. LM1903060107 without _23v6).

Also patches document_summary for chunk_index=0 vectors with poor summaries.
"""
import sys
import re
import time
import concurrent.futures
import unicodedata
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')
from pinecone import Pinecone

pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index = pc.Index(os.getenv('PINECONE_INDEX_NAME', 'document-index'))

# Collect all vector IDs
print("Listing vector IDs...", flush=True)
all_ids = []
for ids_batch in index.list(namespace='semiconductor'):
    all_ids.extend(ids_batch)
print(f"Total vectors: {len(all_ids)}", flush=True)

# === Phase 1: Find vectors needing ncs_code fix ===
print("\n=== Phase 1: Scanning for ncs_code fixes ===", flush=True)

# New regex that handles unversioned filenames
NCS_CODE_PATTERN = re.compile(r'(LM\d{10}(?:_\d+v\d+)?)')
NCS_TITLE_PATTERN = re.compile(r'LM\d{10}(?:_\d+v\d+)?_(.+?)(?:/|$)')

def _generate_better_summary(text, max_length=300):
    """Generate a meaningful summary from text content."""
    paragraphs = text.split('\n\n') if '\n\n' in text else text.split('\n')
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if para.startswith('#'):
            continue
        if para.startswith('![') or para.startswith('![]'):
            continue
        if para.startswith('|') or para.count('|') > 3:
            continue
        if para.startswith('<!--'):
            continue
        if len(para) < 30:
            continue
        if re.match(r'^[-=*_\s]+$', para):
            continue
        if len(para) > max_length:
            return para[:max_length] + "..."
        return para

    # Fallback: collect header texts
    headers = []
    for para in (text.split('\n\n') if '\n\n' in text else text.split('\n')):
        para = para.strip()
        if para.startswith('#'):
            header_text = re.sub(r'^#+\s*', '', para).strip()
            if header_text and len(header_text) > 5:
                headers.append(header_text)
        if len(' | '.join(headers)) > max_length:
            break
    if headers:
        return ' | '.join(headers)[:max_length]

    return text[:max_length].strip() + "..." if len(text) > max_length else text.strip()


ncs_code_updates = []
summary_updates = []
fetch_batch = 100

for i in range(0, len(all_ids), fetch_batch):
    batch_ids = all_ids[i:i + fetch_batch]
    fetch_result = index.fetch(ids=batch_ids, namespace='semiconductor')

    for vid, vdata in fetch_result.get('vectors', {}).items():
        meta = vdata.get('metadata', {})
        source_file = meta.get('source_file', '')
        file_type = meta.get('file_type', '')

        # --- ncs_code fix ---
        current_ncs_code = meta.get('ncs_code', '')
        if not current_ncs_code and source_file:
            # Normalize path for matching
            normalized = unicodedata.normalize('NFC', source_file)
            code_match = NCS_CODE_PATTERN.search(normalized)
            if code_match:
                new_code = code_match.group(1)
                updates = {'ncs_code': new_code}

                # Also extract document title if missing
                current_title = meta.get('ncs_document_title', '')
                if not current_title:
                    title_match = NCS_TITLE_PATTERN.search(normalized)
                    if title_match:
                        raw_title = title_match.group(1).replace('_', ' ').strip()
                        updates['ncs_document_title'] = raw_title

                # Extract category if missing
                current_cat = meta.get('ncs_category', '')
                if not current_cat:
                    for cat in ['반도체개발', '반도체장비', '반도체재료', '반도체제조']:
                        if cat in normalized:
                            updates['ncs_category'] = cat
                            break

                ncs_code_updates.append((vid, updates))

        # --- document_summary fix for chunk_index=0 ---
        chunk_index = meta.get('chunk_index', -1)
        if chunk_index == 0 and file_type == 'markdown':
            current_summary = meta.get('document_summary', '')
            # Check if summary is poor quality (image ref, table, too short, empty)
            needs_fix = False
            if not current_summary:
                needs_fix = True
            elif current_summary.startswith('!['):
                needs_fix = True
            elif current_summary.startswith('|') or current_summary.count('|') > 3:
                needs_fix = True
            elif current_summary.startswith('<!--'):
                needs_fix = True
            elif len(current_summary.strip()) < 20:
                needs_fix = True

            if needs_fix:
                # Try to generate better summary from content_preview
                content = meta.get('content_preview', '') or meta.get('text', '')
                if content:
                    new_summary = _generate_better_summary(content)
                    if new_summary and new_summary != current_summary:
                        summary_updates.append((vid, {'document_summary': new_summary}))

    done = min(i + fetch_batch, len(all_ids))
    if done % 3000 < fetch_batch:
        print(f"  Scanned {done}/{len(all_ids)} vectors...", flush=True)


# Report findings
print(f"\n=== Scan Complete ===", flush=True)
print(f"ncs_code updates needed: {len(ncs_code_updates)}", flush=True)
print(f"document_summary updates needed: {len(summary_updates)}", flush=True)

if ncs_code_updates:
    print("\nncs_code updates preview:", flush=True)
    for vid, upd in ncs_code_updates[:10]:
        print(f"  {vid[:20]}... → {upd}", flush=True)

if summary_updates:
    print("\ndocument_summary updates preview:", flush=True)
    for vid, upd in summary_updates[:5]:
        preview = upd.get('document_summary', '')[:80]
        print(f"  {vid[:20]}... → {preview}", flush=True)

# Combine all updates
all_updates = ncs_code_updates + summary_updates
if not all_updates:
    print("\nNo updates needed. Exiting.")
    sys.exit(0)


# === Phase 2: Apply updates ===
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
                print(f"  FAIL {vid[:20]}...: {e}", flush=True)
                return False
    return False


print(f"\nApplying {len(all_updates)} updates with 10 parallel workers...", flush=True)
start = time.time()

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(apply_update, all_updates))

success = sum(1 for r in results if r)
fail = sum(1 for r in results if not r)
elapsed = time.time() - start

print(f"\n=== Update Complete ({elapsed:.1f}s) ===", flush=True)
print(f"Success: {success}, Failed: {fail}", flush=True)
print(f"  ncs_code fixes: {len(ncs_code_updates)}", flush=True)
print(f"  document_summary fixes: {len(summary_updates)}", flush=True)
