# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SafeFactory is a multi-domain RAG (Retrieval-Augmented Generation) system with two interfaces:

1. **CLI** (`main.py`): Processes documents (images, markdown, JSON) into vector embeddings and uploads to Pinecone
2. **Web App** (`web_app.py`): Flask application providing semantic search, AI-powered Q&A, community forum, MSDS lookup, and admin panel

The system serves Korean-language knowledge bases across 5 domains: semiconductor manufacturing (NCS), Korean labor law, workplace safety training, OSHA-equivalent safety guidelines, and chemical safety (MSDS).

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
- `GEMINI_API_KEY` — RAG answer generation via Google Gemini
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

**7-Phase RAG Pipeline** (`services/rag_pipeline.py` → Web):
```
Query → QueryEnhancer(multi-query) → @mention filter → Vector+BM25 search
      → RRF fusion → Reranker(cross-encoder) → ContextOptimizer → Gemini LLM → Response with citations
```

### Service Layer Pattern

`services/singletons.py` provides thread-safe, lazy-initialized global instances (OpenAI client, Gemini client, PineconeAgent, QueryEnhancer, ContextOptimizer, Reranker, HybridSearcher). Uses double-checked locking with `threading.RLock()`. Settings changes trigger cache invalidation by nullifying the relevant singleton.

### Multi-Domain System

`services/domain_config.py` defines 5 domains, each with:
- Pinecone namespace mapping (`DIRECTORY_NAMESPACE_MAP`)
- Custom LLM system prompt (`DOMAIN_PROMPTS`)
- Domain-specific metadata filters (`services/filters.py`)

Domains: `''` (semiconductor/default), `laborlaw`, `field-training`, `safeguide`, `msds`

### API Structure

All API routes registered as Flask blueprints under `/api/v1/` in `api/v1/__init__.py`:
- `search.py` — `/search`, `/ask`, `/ask/stream`, PDF resolution
- `admin.py` — settings CRUD, index stats, namespace sync/delete
- `community.py` — posts, comments, likes (CRUD)
- `msds.py` — KOSHA chemical safety search
- `calculator.py` — wage and insurance calculations
- `health.py` — system health check
- `news.py` — news aggregation
- `auth.py` — OAuth callbacks

API responses use `api/response.py` helpers for consistent JSON format.

### Database

SQLite (`instance/app.db`) via Flask-SQLAlchemy. `models.py` defines:
- Auth: `User`, `SocialAccount` (OAuth with encrypted tokens)
- Community: `Post`, `Comment`, `PostLike`, `CommentLike`, `Category`, `PostAttachment`
- Config: `SystemSettings`, `ProviderSettings`
- Metadata: `safe_factory` table for document processing tracking (file hash, chunk count, vector IDs)

### Authentication

OAuth-only (Google + Kakao) via authlib. Access/refresh tokens encrypted with Fernet derived from `SECRET_KEY`. CSRF protection via Flask-WTF on form routes; API routes (`v1_bp`) are CSRF-exempt.

## Key Implementation Details

- **Token counting**: `len(text) // 3` approximation for mixed Korean/English
- **Vector IDs**: MD5 hash of `source_file + chunk_index + content_preview` (deterministic/idempotent)
- **Metadata limit**: Content previews truncated to 1000 chars for Pinecone
- **Serverless Pinecone**: `ServerlessSpec` with AWS us-east-1
- **Embedding dimensions**: 1536 (text-embedding-3-small) or 3072 (text-embedding-3-large)
- **Reranking**: Optional — either Pinecone Inference API or local cross-encoder (`sentence-transformers`)

## Extension Points

- **Add domain**: Add entry to `DIRECTORY_NAMESPACE_MAP` and `DOMAIN_PROMPTS` in `services/domain_config.py`, add filter builder in `services/filters.py`
- **Add API endpoint**: Create module in `api/v1/`, import it in `api/v1/__init__.py`
- **Add file type**: Extend `FileLoader.SUPPORTED_EXTENSIONS`, add `_process_*` method in `src/agent.py`
- **Modify RAG phases**: Edit `run_rag_pipeline()` in `services/rag_pipeline.py`
- **Change chunking**: Edit `SemanticChunker._split_by_structure()` in `src/semantic_chunker.py`
