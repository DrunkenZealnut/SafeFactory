# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Korean-language RAG system with dual interfaces: CLI (`main.py`) and Flask web app (`web_app.py`). Ingests files (images, markdown, JSON), generates semantic embeddings via OpenAI, stores in Pinecone, and serves a streaming chat interface powered by GPT-4o-mini. Primary domains: semiconductor NCS technical docs, Korean labor law (with wage/insurance calculators), and MSDS chemical lookup.

## Commands

### Setup
```bash
pip install -r requirements.txt
cp .env.example .env    # Edit with API keys
```

### Run
```bash
python web_app.py                          # Flask dev server on port 5001
lsof -ti:5001 | xargs kill -15            # Kill server
```

### CLI
```bash
python main.py process ./folder [--namespace NAME] [--no-recursive] [--batch-size 50] [--max-chunk-tokens 500]
python main.py search "query" [--top-k 5] [--namespace NAME] [--filter-file-type image|markdown|json]
python main.py stats
python main.py delete --source-file "/path/to/file"
```

### Module Testing
Each `src/` module has standalone `if __name__ == "__main__"` blocks:
```bash
python -m src.file_loader
python -m src.semantic_chunker
python -m src.embedding_generator
python -m src.pinecone_uploader
python -m src.agent ./folder [namespace]
```

No formal test suite or linter configured. Syntax verification: `python -c "import py_compile; py_compile.compile('file.py', doraise=True)"`

## Architecture

### Entry Points
- **`main.py`** — CLI (argparse): process, search, stats, delete
- **`web_app.py`** — Flask server: streaming chat, domain prompts, calculator APIs, MSDS search

### Ingestion Pipeline (`src/`)
```
FileLoader → ImageDescriber / SemanticChunker → EmbeddingGenerator → PineconeUploader → MetadataManager
```
- **`PineconeAgent`** (`agent.py`): Orchestrator. Returns `ProcessingResult` dataclass. Also exposes `search()` and `search_all_namespaces()` for retrieval.
- **`FileLoader`**: Discovers files → `LoadedFile` dataclass. `SUPPORTED_EXTENSIONS` defines types.
- **`ImageDescriber`**: GPT-4o-mini Vision → Korean text descriptions
- **`SemanticChunker`**: Token-aware chunking with overlap → `Chunk` dataclass. `build_page_line_map()` maps lines to PDF pages using image anchors (`_page_N_*.jpeg`).
- **`EmbeddingGenerator`**: Batches up to 100 texts → `EmbeddingResult`
- **`PineconeUploader`**: Batch upload/query/delete/query_namespaces → `VectorData` dataclass
- **`MetadataManager`**: MySQL tracking with SHA256 hashing. Skips unchanged files. Table: `pinecone_agent` in `kcsvictory` DB.

### RAG Search Pipeline (7 phases in `_run_rag_pipeline()`)
```
Query → QueryEnhancer → Multi-Query Search → @Mention Filter → BM25+RRF → Reranker → ContextOptimizer → LLM
```

| Phase | Component | What it does |
|-------|-----------|-------------|
| 1 | `QueryEnhancer` | Multi-query expansion + HyDE (for queries ≥30 chars) + keyword extraction |
| 2 | `agent.search()` / `agent.search_all_namespaces()` | Vector search with NCS metadata filtering. `namespace='all'` triggers multi-namespace |
| 3 | `parse_mentions()` | Client-side @mention filtering (file/folder/keyword) with NFC unicode normalization |
| 4 | `HybridSearcher` | BM25 + keyword boosting via RRF. Skippable with `SKIP_BM25_HYBRID=true` |
| 5 | `Reranker` | Cross-encoder reranking. Priority: PineconeReranker → local CrossEncoder → LightweightReranker |
| 6 | `ContextOptimizer` | Jaccard deduplication + Lost-in-the-Middle reordering for LLM attention |
| 7 | Build context | Numbered doc blocks with source attribution + related image discovery |

### Reranker System (`src/reranker.py`)

Three implementations with automatic fallback via `get_reranker()` factory:
1. **`PineconeReranker`** — Pinecone Inference API (`bge-reranker-v2-m3`, multilingual, 1024 tokens). No local GPU needed.
2. **`Reranker`** — Local cross-encoder (`mmarco-mMiniLMv2-L12-H384-v1`). Requires `sentence-transformers`.
3. **`LightweightReranker`** — Keyword overlap scoring. Zero dependencies.

Factory priority: Pinecone API → local CrossEncoder → Lightweight. Override with `USE_LOCAL_RERANKER=true`.

### Calculator Modules (`calculator/`)
- **`WageCalculator`**: Net salary with 4대보험 deductions, income tax (2025 rates), dependent/child deductions
- **`InsuranceCalculator`**: Insurance by `CompanySize` enum (4 tiers) and `IndustryType` enum (29 categories, 산재보험 0.5%–18.5%)

### Web App (`web_app.py`)

**Two answer endpoints**:
- **`/api/ask`** (POST) — Returns complete JSON response
- **`/api/ask/stream`** (POST) — SSE streaming. Event types: `metadata` (sources/images first), `calculations` (tool results), `token` (individual tokens), `done` (completion), `error`

Both share `_run_rag_pipeline()` for phases 1-7, then diverge for LLM generation.

**Domain system**: `DOMAIN_PROMPTS` dict maps domain names to system prompts. The `laborlaw` prompt includes GPT function-calling tool definitions (`calculate_wage`, `calculate_insurance`). Tool calls: first GPT call detects tool use (non-streaming) → execute functions → second call streams with results.

**Lazy initialization**: Global singletons (`_agent`, `_openai_client`, `_query_enhancer`, `_context_optimizer`, `_reranker`, `_hybrid_searcher`) via `get_*()` functions. `get_pinecone_client()` tries agent's client first, falls back to standalone `Pinecone()`.

**Key routes**: `/` (chat), `/domain/<domain>` (domain UI), `/api/ask/stream` (SSE), `/api/search` (vector search), `/msds` (chemical search), `/calculate/wage`, `/calculate/insurance`, `/api/stats`, `/api/namespaces`, `/api/sources`, `/documents/<path>` (serve images)

**NCS metadata filtering**: `_build_ncs_filter()` detects NCS-related queries and builds Pinecone metadata filters using `ncs_category`, `ncs_section_type`, `learning_unit` fields.

**@mention system**: `parse_mentions()` extracts `@파일명.md` (file), `@폴더명/` (folder), `@키워드` (keyword) from query. Filtering happens post-query in Phase 3 since Pinecone doesn't support substring matching.

**Templates**: `index.html` (main chat), `domain.html` (domain chat with default namespace), `home.html` (intro), `msds.html` (chemical search). Both chat templates handle SSE with `ReadableStream` + real-time markdown rendering.

## Environment Variables

Required in `.env`:
```
OPENAI_API_KEY=xxx           # Embeddings + Vision + Chat
PINECONE_API_KEY=xxx         # Vector database
PINECONE_INDEX_NAME=xxx      # Auto-created if missing
```

Optional:
```
MSDS_API_KEY=xxx                # KOSHA MSDS API (data.go.kr)
USE_LOCAL_RERANKER=true         # Force local cross-encoder instead of Pinecone Inference API
SKIP_BM25_HYBRID=true           # Skip BM25+RRF phase (rely on Pinecone reranker alone)
DB_HOST/DB_PORT/DB_USER/DB_PASSWORD/DB_NAME  # MySQL metadata tracking
```

## Key Implementation Details

- **Token counting**: `len(text) // 3` approximation for mixed Korean/English
- **Vector IDs**: MD5 hash of `source_file + chunk_index + content_preview[:100]`
- **Pinecone**: ServerlessSpec on AWS us-east-1, cosine similarity, 1536d (text-embedding-3-small)
- **Namespaces**: `semiconductor` (~14K vectors), `laborlaw` (~3K), `field-training` (~400). Multi-namespace query via `index.query_namespaces()` with sequential fallback.
- **SSL**: Explicit `certifi` config at startup (`SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`) before importing httpx/urllib3
- **Unicode**: macOS NFD → NFC normalization for Korean filenames in image URLs and @mention filtering
- **Streaming**: Flask SSE with `stream_with_context()`. Metadata event sent before tokens so frontend gets sources immediately.
- **Korean optimization**: BM25 tokenizer strips Korean particles (조사); QueryEnhancer uses Korean-specific prompt templates

## Extension Points

- **Add file type**: Extend `FileLoader.SUPPORTED_EXTENSIONS` + add `_process_*` method in `PineconeAgent`
- **Add domain**: Add entry to `DOMAIN_PROMPTS` dict in `web_app.py` + create template
- **Add calculator**: Create module in `calculator/`, add to `CALCULATOR_FUNCTIONS` list, wire into domain prompt tool definitions
- **Add metadata fields**: Update `Chunk`/`VectorData` dataclasses → modify `PineconeAgent._create_vector_data()`
- **Modify chunking**: `SemanticChunker._split_by_structure()` for text, `_chunk_json_*()` for JSON
