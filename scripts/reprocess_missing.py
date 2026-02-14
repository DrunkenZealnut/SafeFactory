"""Re-process missing image files that failed during initial upload.

Reads missing_files.txt, processes each image through the full pipeline
(Vision API → embedding → Pinecone upload) with rate-limit-aware delays.
"""
import sys
import os
import re
import time
import base64
import unicodedata
import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')

from src.image_describer import ImageDescriber
from src.embedding_generator import EmbeddingGenerator
from src.pinecone_uploader import PineconeUploader

# --- Config ---
NAMESPACE = 'semiconductor'
BATCH_SIZE = 20           # images per embedding batch
VISION_DELAY = 1.0        # seconds between Vision API calls (rate limit)
EMBED_DELAY = 2.0         # seconds between embedding batches
MISSING_FILE = PROJECT_ROOT / 'scripts' / 'missing_files.txt'

openai_key = os.getenv('OPENAI_API_KEY')
pinecone_key = os.getenv('PINECONE_API_KEY')
pinecone_index = os.getenv('PINECONE_INDEX_NAME', 'document-index')

if not openai_key:
    print("❌ OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
    sys.exit(1)
if not pinecone_key:
    print("❌ PINECONE_API_KEY 환경 변수가 설정되지 않았습니다.")
    sys.exit(1)

# --- Initialize components ---
print("Initializing components...", flush=True)
describer = ImageDescriber(api_key=openai_key, model='gpt-4o-mini', max_tokens=1000)
embedder = EmbeddingGenerator(api_key=openai_key, model='text-embedding-3-small')
uploader = PineconeUploader(
    api_key=pinecone_key,
    index_name=pinecone_index,
    dimension=1536,
    create_if_not_exists=False
)

# --- Read missing files ---
with open(MISSING_FILE, 'r', encoding='utf-8') as f:
    missing_files = [line.strip() for line in f if line.strip()]

print(f"Total missing files: {len(missing_files)}", flush=True)

# --- NCS metadata extraction ---
from src.ncs_utils import extract_ncs_metadata

def generate_vector_id(content, source_file, chunk_index):
    combined = f"{source_file}:{chunk_index}:{content[:100]}"
    return hashlib.md5(combined.encode()).hexdigest()

# --- Process images ---
stats = {'described': 0, 'desc_failed': 0, 'embedded': 0, 'embed_failed': 0,
         'uploaded': 0, 'upload_failed': 0}

# Phase 1: Describe all images with Vision API (rate-limited)
print(f"\n=== Phase 1: Image Description ({len(missing_files)} images) ===", flush=True)
described_items = []  # (filepath, description, metadata)

for i, filepath in enumerate(missing_files):
    filename = os.path.basename(filepath)
    ext = os.path.splitext(filename)[1].lower()

    try:
        with open(filepath, 'rb') as img_f:
            b64 = base64.b64encode(img_f.read()).decode('utf-8')

        description = describer.describe_from_base64(base64_image=b64, extension=ext)

        # Build metadata
        page_match = re.search(r'_page_(\d+)_', filename)
        source_page = int(page_match.group(1)) if page_match else None

        img_meta = {
            'file_type': 'image',
            'filename': filename,
            'original_extension': ext,
        }
        if source_page is not None:
            img_meta['source_page'] = source_page

        # NCS metadata
        ncs_meta = extract_ncs_metadata(filepath)
        img_meta.update(ncs_meta)
        if ncs_meta:
            img_meta['ncs_section_type'] = 'general'

        content = f"[이미지: {filename}]\n\n{description}"
        described_items.append((filepath, content, img_meta))
        stats['described'] += 1

    except Exception as e:
        stats['desc_failed'] += 1
        err_msg = str(e)[:100]
        print(f"  FAIL [{i+1}] {filename}: {err_msg}", flush=True)

    if (i + 1) % 50 == 0:
        print(f"  Progress: {i+1}/{len(missing_files)} "
              f"(OK: {stats['described']}, FAIL: {stats['desc_failed']})", flush=True)

    # Rate limit delay
    time.sleep(VISION_DELAY)

print(f"\nPhase 1 done: {stats['described']} described, {stats['desc_failed']} failed", flush=True)

if not described_items:
    print("No images to process. Exiting.")
    sys.exit(0)

# Phase 2: Generate embeddings in batches
print(f"\n=== Phase 2: Embedding ({len(described_items)} items) ===", flush=True)
vectors = []

for batch_start in range(0, len(described_items), BATCH_SIZE):
    batch = described_items[batch_start:batch_start + BATCH_SIZE]
    texts = [item[1] for item in batch]

    try:
        results = embedder.generate_batch(texts, batch_size=BATCH_SIZE)

        for j, emb_result in enumerate(results):
            filepath, content, meta = batch[j]
            vector_id = generate_vector_id(content, filepath, 0)

            full_meta = {
                'source_file': filepath,
                'chunk_index': 0,
                'content': content[:40000],
                'content_preview': content[:1000],
                'content_length': len(content),
                **meta
            }
            # Sanitize metadata
            sanitized = {}
            for k, v in full_meta.items():
                if isinstance(v, (str, int, float, bool)):
                    sanitized[k] = v
                elif isinstance(v, list):
                    sanitized[k] = [str(x) for x in v]
                elif v is not None:
                    sanitized[k] = str(v)
            vectors.append({
                'id': vector_id,
                'values': emb_result.embedding,
                'metadata': sanitized
            })
            stats['embedded'] += 1

    except Exception as e:
        stats['embed_failed'] += len(batch)
        err_msg = str(e)[:100]
        print(f"  Embedding batch FAIL at {batch_start}: {err_msg}", flush=True)

    done = min(batch_start + BATCH_SIZE, len(described_items))
    if done % 100 < BATCH_SIZE or done == len(described_items):
        print(f"  Embedded: {done}/{len(described_items)} "
              f"(OK: {stats['embedded']}, FAIL: {stats['embed_failed']})", flush=True)

    time.sleep(EMBED_DELAY)

print(f"\nPhase 2 done: {stats['embedded']} embedded, {stats['embed_failed']} failed", flush=True)

# Phase 3: Upload to Pinecone
print(f"\n=== Phase 3: Upload ({len(vectors)} vectors) ===", flush=True)
UPLOAD_BATCH = 100

for batch_start in range(0, len(vectors), UPLOAD_BATCH):
    batch = vectors[batch_start:batch_start + UPLOAD_BATCH]
    try:
        uploader.index.upsert(
            vectors=batch,
            namespace=NAMESPACE
        )
        stats['uploaded'] += len(batch)
    except Exception as e:
        stats['upload_failed'] += len(batch)
        err_msg = str(e)[:100]
        print(f"  Upload batch FAIL at {batch_start}: {err_msg}", flush=True)

    done = min(batch_start + UPLOAD_BATCH, len(vectors))
    print(f"  Uploaded: {done}/{len(vectors)}", flush=True)

# Final report
print(f"\n{'='*50}", flush=True)
print(f"=== Re-processing Complete ===", flush=True)
print(f"  Vision API:  {stats['described']} OK / {stats['desc_failed']} failed", flush=True)
print(f"  Embeddings:  {stats['embedded']} OK / {stats['embed_failed']} failed", flush=True)
print(f"  Upload:      {stats['uploaded']} OK / {stats['upload_failed']} failed", flush=True)
print(f"{'='*50}", flush=True)
