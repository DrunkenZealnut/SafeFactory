# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SafeFactory is a multi-domain RAG (Retrieval-Augmented Generation) system with two interfaces:

1. **CLI** (`main.py`): Processes documents (images, markdown, JSON) into vector embeddings and uploads to Pinecone
2. **Web App** (`web_app.py`): Flask application providing semantic search, AI-powered Q&A, community forum, MSDS lookup, and admin panel

The system serves Korean-language knowledge bases across 4 active domains: semiconductor manufacturing (NCS), workplace safety training, OSHA-equivalent safety guidelines (KOSHA), and chemical safety (MSDS). The laborlaw domain is currently disabled (`[LABORLAW_DISABLED]` markers across 14 files — search `grep -rn "LABORLAW_DISABLED"` to find all disable points).

## Commands

```bash
# Setup
pip install -r requirements.txt
cp .env.example .env  # Then fill in API keys

# Run web app (dev)
source venv/bin/activate
python web_app.py  # http://localhost:5001

# Run web app (production)
gunicorn web_app:app --bind 127.0.0.1:5001 --workers 2 --timeout 180

# Kill existing process on port 5001
lsof -ti:5001 | xargs kill -9

# CLI: Process documents into Pinecone
python main.py process ./documents --namespace semiconductor [--batch-size 50] [--max-chunk-tokens 500]
python main.py search "query" [--top-k 5] [--namespace NAME]
python main.py stats
python main.py delete --source-file "/path/to/file"

# GraphRAG: Community detection and global search
python main.py build-community --namespace semiconductor-v2 [--resolution 1.0] [--reset] [--skip-summary]
python main.py community-stats [--namespace NAME]
python main.py graph-stats [--namespace NAME]

# Module standalone tests (each src/ module has if __name__ == "__main__" blocks)
python -m src.file_loader
python -m src.semantic_chunker
python -m src.embedding_generator
python -m src.pinecone_uploader

# Production deploy
cd ~/SafeFactory && git pull origin main && source venv/bin/activate
pip install -r requirements.txt
pkill -f gunicorn; sleep 2
nohup venv/bin/gunicorn web_app:app --bind 127.0.0.1:5001 --workers 2 --timeout 180 > app.log 2>&1 &
```

## Environment Variables

Required in `.env` (see `.env.example` for full list):
- `OPENAI_API_KEY` — embeddings (text-embedding-3-small) and Vision API (gpt-4o-mini)
- `GEMINI_API_KEY` — RAG answer generation via Google Gemini + Gemini Embedding 2 (optional)
- `PINECONE_API_KEY` / `PINECONE_INDEX_NAME` — vector database
- `SECRET_KEY` — Flask sessions and Fernet token encryption
- `GOOGLE_CLIENT_ID/SECRET`, `KAKAO_CLIENT_ID/SECRET` — OAuth social login
- Optional: `MSDS_API_KEY`, `RATE_LIMIT_ENABLED`, `CORS_ORIGINS`, `USE_LOCAL_RERANKER`, `SKIP_BM25_HYBRID`

## Architecture

### Two-System Design

**Document Processing Pipeline** (`src/` → CLI):
```
FileLoader → ImageDescriber(Vision API) / SemanticChunker → EmbeddingGenerator → PineconeUploader
                                                                                    ↓
                                                                              MetadataManager (SQLite tracking)
```

**RAG Pipeline** (`services/rag_pipeline.py` → Web, 11-phase):
```
Query → P0:DomainClassifier → QueryTypeClassifier(5 types) → P1:QueryEnhancer(multi-query+HyDE)
      → P2:VectorSearch(Pinecone) → P3:GraphRAG(entity enrichment) → P3.5:CommunitySearch(global)
      → P4:HybridSearch(BM25+Vector RRF) → P5:Reranker(cross-encoder 70%+original 30%)
      → P5.5:MMR(dedup) → P6:ContextOptimizer(Lost-in-Middle) → P7:BuildContext+CrossSearch
      → P8:LLM(Gemini/OpenAI/Anthropic) → Response with citations
```

### Service Layer Pattern

`services/singletons.py` provides thread-safe, lazy-initialized global instances (OpenAI client, Gemini client, Anthropic client, PineconeAgent, QueryEnhancer, ContextOptimizer, Reranker, HybridSearcher, GraphSearcher, CommunitySearcher). Uses double-checked locking with `threading.RLock()`. Settings changes trigger cache invalidation by nullifying the relevant singleton.

### Multi-Domain System

`services/domain_config.py` defines 5 domains, each with:
- Pinecone namespace mapping (`DIRECTORY_NAMESPACE_MAP`)
- Custom LLM system prompt (`DOMAIN_PROMPTS`)
- Domain-specific metadata filters (`services/filters.py`)

Active domains: `semiconductor-v2` (default), `field-training`, `kosha`, `msds`. Disabled: `laborlaw` (`[LABORLAW_DISABLED]`)

### Automatic Domain Routing

`services/query_router.py` contains `classify_domain()` which auto-routes queries to the best-matching namespace using keyword scoring. Each namespace has `high` (weight 1.0) and `low` (weight 0.3) keywords. The current page's namespace gets a bonus (+0.5). Below the confidence threshold (0.3), queries stay in the default namespace. This replaced the previous @mention-based manual filtering system.

### API Structure

All API routes registered as Flask blueprints under `/api/v1/` in `api/v1/__init__.py`:
- `search.py` — `/search`, `/ask`, `/ask/stream`, PDF resolution
- `admin.py` — settings CRUD, index stats, namespace sync/delete
- `community.py` — posts, comments, likes (CRUD)
- `questions.py` — shared questions, likes, word cloud keywords
- `bookmarks.py` — user document bookmarks
- `msds.py` — KOSHA chemical safety search
- `calculator.py` — wage and insurance calculations
- `index_ops.py` — Pinecone index operations (stats, namespaces, delete)
- `health.py` — system health check
- `news.py` — news aggregation
- `auth.py` — OAuth callbacks

API responses use `api/response.py` helpers (`success_response`, `error_response`) for consistent JSON format. Rate limiting via `api.rate_limit()` decorator (no-op when flask-limiter unavailable).

### Database

SQLite (`instance/app.db`) via Flask-SQLAlchemy. `models.py` defines:
- Auth: `User`, `SocialAccount` (OAuth with encrypted tokens)
- Community: `Post`, `Comment`, `PostLike`, `CommentLike`, `Category`, `PostAttachment`
- Social Q&A: `SharedQuestion`, `QuestionLike` (shared AI questions with likes)
- User Data: `SearchHistory`, `UserBookmark`, `Document`
- Config: `SystemSetting`, `AdminLog`
- Content: `NewsArticle`
- Knowledge Graph: `KGEntity`, `KGRelation`, `KGEntityChunk` (GraphRAG entity-relation store)
- Community: `KGCommunity`, `KGCommunityMember` (Leiden algorithm community clusters with LLM summaries)
- Metadata: `safe_factory` table for document processing tracking (file hash, chunk count, vector IDs)

### Authentication

OAuth-only (Google + Kakao) via authlib. Access/refresh tokens encrypted with Fernet derived from `SECRET_KEY`. CSRF protection via Flask-WTF on form routes; API routes (`v1_bp`) are CSRF-exempt.

### Services Layer

Key service modules beyond `singletons.py` and `rag_pipeline.py`:
- `keyword_extractor.py` — regex-based Korean/English keyword extraction for word clouds
- `law_api.py` / `law_drf_client.py` — external legal information API integration
- `legal_source_router.py` — routes legal queries to appropriate data sources
- `labor_calculator.py` / `labor_classifier.py` — wage/insurance calculation logic
- `query_router.py` — query type classification (factual/procedural/comparison/calculation/overview) and automatic domain routing (`classify_domain`, `classify_query_type`)
- `graph_config.py` — per-namespace GraphRAG and community detection configuration
- `community_searcher.py` — Leiden community-based global search for overview queries
- `settings.py` — admin settings management with singleton cache invalidation

### Frontend

Jinja2 templates in `templates/` with inline `<script>` blocks. All pages extend `base.html` (common nav, footer with 청년노동자인권센터 info). `templates/domain.html` is the main search/Q&A interface shared across all 5 domains. `templates/admin.html` uses `admin-` prefixed CSS classes to avoid collision with base.html. CDN libraries: marked.js (markdown), DOMPurify (XSS), Chart.js, wordcloud2.js. Design system: `static/css/theme.css` defines CSS variables (`--sf-*`) used across all templates.

## Key Implementation Details

- **Token counting**: `len(text) // 3` approximation for mixed Korean/English
- **Vector IDs**: MD5 hash of `source_file + chunk_index + content_preview` (deterministic/idempotent)
- **Metadata limit**: Content previews truncated to 1000 chars for Pinecone
- **Serverless Pinecone**: `ServerlessSpec` with AWS us-east-1
- **Embedding models**: OpenAI (`text-embedding-3-small` 1536D, `text-embedding-3-large` 3072D) and Gemini (`gemini-embedding-2-preview` 3072D native, MRL 1536D). `EmbeddingGenerator` auto-detects provider from model name. Gemini models use `task_type` parameter (`RETRIEVAL_QUERY` for search, `RETRIEVAL_DOCUMENT` for ingest) and `output_dimensionality` for MRL dimension control.
- **Reranking**: Optional — either Pinecone Inference API or local cross-encoder (`sentence-transformers`)
- **Streaming**: `/ask/stream` uses Server-Sent Events (SSE) for real-time LLM responses
- **Emergency responder**: `services/emergency_responder.py` detects emergency queries (화학물질 노출, 감전, 화상 등) and returns immediate first-aid guidance before RAG pipeline runs. Frontend renders emergency banner with `renderAskResults()` in `domain.html`
- **Safety cross-search**: Semiconductor domain answers automatically search `kosha` namespace for safety/health context and append it as supplementary references
- **MSDS cross-search**: Automatically detects chemical names in queries/context and fetches MSDS data
- **GraphRAG**: Entity-relation graph enriches vector search with knowledge graph traversal (hop_depth=2)
- **Community search**: Leiden algorithm detects entity communities; LLM generates summaries for overview queries
- **LLM namespace override**: `_NAMESPACE_MODEL_OVERRIDE` in `api/v1/search.py` maps namespaces to specific LLM models (currently empty — add entries to override per-domain)
- **CSS convention**: All templates use `--sf-*` CSS variables from `theme.css`. Max-width 960px. Admin page uses `admin-` class prefix

## Extension Points

- **Add domain**: Add entry to `DIRECTORY_NAMESPACE_MAP` and `DOMAIN_PROMPTS` in `services/domain_config.py`, add filter builder in `services/filters.py`
- **Add API endpoint**: Create module in `api/v1/`, import it in `api/v1/__init__.py`
- **Add file type**: Extend `FileLoader.SUPPORTED_EXTENSIONS`, add `_process_*` method in `src/agent.py`
- **Modify RAG phases**: Edit `run_rag_pipeline()` in `services/rag_pipeline.py`
- **Change chunking**: Edit `SemanticChunker._split_by_structure()` in `src/semantic_chunker.py`
- **Add GraphRAG config**: Add `community` block to namespace in `services/graph_config.py`
- **Add LLM provider**: Add provider branch in `api/v1/search.py` `api_ask()` Phase 8 section
- **Add embedding model**: Add to `EmbeddingGenerator.MODELS` dict and `GEMINI_MODELS` set (if Gemini) in `src/embedding_generator.py`, add to `_VALID_SETTING_VALUES['embedding_model']` and `embedding_models` list in `api/v1/admin.py`
- **Benchmark embeddings**: `python scripts/benchmark_embeddings.py --domain semiconductor-v2` compares OpenAI vs Gemini search quality. Requires `*-gemini` namespace to be ingested via `python scripts/ingest_gemini_test.py`

## Metadata Key Differences by Namespace

Different namespaces store text in different metadata keys. When accessing document text from Pinecone vectors:
- `semiconductor-v2`, `kosha`, `safeguide`, `field-training`: `content_preview` (also `content`)
- `laborlaw-v2`, `counsel`, `precedent` (disabled): `chunk_text` (also `text` for counsel)

Always check for `content_preview` → `chunk_text` → `text` → `content` fallback order.

# currentDate
Today's date is 2026-03-19.
