# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pinecone Agent is a Python CLI tool that processes files (images, markdown, JSON) and uploads their semantic embeddings to Pinecone vector database for semantic search.

## Commands

### Installation
```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with API keys
```

### CLI Usage
```bash
# Process folder and upload to Pinecone
python main.py process ./folder [--namespace NAME] [--no-recursive] [--batch-size 50] [--max-chunk-tokens 500]

# Search vectors
python main.py search "query" [--top-k 5] [--namespace NAME] [--filter-file-type image|markdown|json]

# View index statistics
python main.py stats

# Delete vectors
python main.py delete --source-file "/path/to/file"
python main.py delete --all  # Requires confirmation
```

### Module Testing
Each module in `src/` has standalone test code via `if __name__ == "__main__"` blocks:
```bash
python -m src.file_loader
python -m src.semantic_chunker
python -m src.embedding_generator
python -m src.pinecone_uploader
```

## Architecture

### Processing Pipeline
```
FileLoader → ImageDescriber/SemanticChunker → EmbeddingGenerator → PineconeUploader
```

### Key Classes (src/)

- **`PineconeAgent`** (`agent.py`): Main orchestrator that coordinates the entire pipeline
- **`FileLoader`** (`file_loader.py`): Discovers and loads files with metadata, returns `LoadedFile` dataclass
- **`ImageDescriber`** (`image_describer.py`): Converts images to text descriptions via OpenAI Vision API (gpt-4o-mini)
- **`SemanticChunker`** (`semantic_chunker.py`): Splits text into meaningful chunks with token awareness and overlap support
- **`EmbeddingGenerator`** (`embedding_generator.py`): Generates OpenAI embeddings in configurable batches
- **`PineconeUploader`** (`pinecone_uploader.py`): Manages Pinecone index operations and batch uploads

### Data Flow
1. `FileLoader` scans folder → `LoadedFile` objects (with base64 for images)
2. Images → `ImageDescriber` → text descriptions
3. Text content → `SemanticChunker` → `Chunk` objects
4. Chunks → `EmbeddingGenerator` → `EmbeddingResult` objects
5. Embeddings → `PineconeUploader` → `VectorData` uploaded to Pinecone

### File Type Processing
| Type | Extensions | Processing |
|------|------------|------------|
| Image | .png, .jpg, .jpeg, .gif, .bmp, .webp | Vision API description |
| Markdown | .md, .markdown | Structure-aware semantic chunking |
| JSON | .json | Array/object intelligent splitting |

## Environment Variables

Required in `.env`:
```
OPENAI_API_KEY=xxx       # For embeddings and vision
PINECONE_API_KEY=xxx     # Pinecone access
PINECONE_INDEX_NAME=xxx  # Target index (auto-created if missing)
```

## Extension Points

- **Add file type support**: Extend `FileLoader.SUPPORTED_EXTENSIONS` and add `_process_*` method in `PineconeAgent`
- **Modify chunking**: Edit `SemanticChunker._split_by_structure()` for text or `_chunk_json_*()` for JSON
- **Add metadata fields**: Update `Chunk` and `VectorData` dataclasses, then modify metadata pipeline in `PineconeAgent._create_vector_data()`

## Key Implementation Details

- **Token counting**: Approximates as `len(text) // 3` for mixed Korean/English
- **Vector IDs**: MD5 hash of `source_file + chunk_index + content_preview`
- **Metadata limit**: Content previews truncated to 1000 chars for Pinecone
- **Serverless Pinecone**: Uses `ServerlessSpec` with AWS us-east-1 by default
- **Embedding dimensions**: 1536 (text-embedding-3-small) or 3072 (text-embedding-3-large)
