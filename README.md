# rag-search-engine

A Python RAG-style movie search project combining:
- **Keyword retrieval** (inverted index + BM25)
- **Semantic retrieval** (SentenceTransformers embeddings)
- **Hybrid ranking** (weighted fusion or Reciprocal Rank Fusion)
- **Optional LLM query enhancement and reranking** via OpenRouter
- **Optional multimodal/image-based retrieval**

## Architecture

The system is organized around CLI entry points in `/home/runner/work/rag-search-engine/rag-search-engine/cli` and library modules in `/home/runner/work/rag-search-engine/rag-search-engine/cli/lib`.

### Core components

1. **Keyword index + BM25** (`cli/index.py`)
   - Builds and loads an inverted index
   - Computes BM25 term/document scores
   - Persists cache files under `cache/` (`index.pkl`, `docmap.pkl`, `term_frequencies.pkl`, `doc_lengths.pkl`)

2. **Semantic search** (`cli/lib/semantic_search.py`)
   - Uses `SentenceTransformer("all-MiniLM-L6-v2")`
   - Embeds movie text (`title + description`)
   - Stores embedding cache in `cache/movie_embeddings.npy`

3. **Chunked semantic search** (`ChunkedSemanticSearch`)
   - Splits descriptions into semantic chunks
   - Embeds chunks and stores:
     - `cache/chunk_embeddings.npy`
     - `cache/chunk_metadata.json`

4. **Hybrid search** (`cli/lib/hybrid_search.py`)
   - Runs BM25 and chunked semantic retrieval in parallel conceptually
   - Supports:
     - **Weighted fusion**: `alpha * bm25 + (1 - alpha) * semantic`
     - **RRF**: `1 / (k + rank)` across keyword + semantic ranks

5. **Query enhancement / reranking / generation** (`cli/lib/query_enhancement.py`)
   - LLM-based spell correction, rewrite, expansion
   - LLM reranking (single-doc or batch)
   - RAG answer/summarization/citation style helpers

6. **Multimodal retrieval** (`cli/lib/multimodal_search.py`)
   - Uses CLIP model (`clip-ViT-B-32`)
   - Encodes image query and compares against text embeddings

### High-level retrieval flow (RRF path)

`User query -> BM25 ranking + semantic chunk ranking -> fusion (RRF) -> optional LLM rerank -> top-k results`

## Installation

### Prerequisites

- Python **3.13+**
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Setup with uv (recommended)

```bash
git clone https://github.com/Suyash-24/rag-search-engine.git
cd rag-search-engine
uv sync
```

### Setup with pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Data files expected

This repo ignores `/data` in git, so add your local dataset files:

- `/home/runner/work/rag-search-engine/rag-search-engine/data/movies.json`
- `/home/runner/work/rag-search-engine/rag-search-engine/data/stopwords.txt`
- `/home/runner/work/rag-search-engine/rag-search-engine/data/golden_dataset.json` (for evaluation)

`movies.json` is expected to contain a top-level `movies` list with items that include at least: `id`, `title`, `description`.

## Environment variables

Create `/home/runner/work/rag-search-engine/rag-search-engine/.env` (or export in shell):

```bash
OPENROUTER_API_KEY=...
# Optional, only for helper scripts:
GROQ_API_KEY=...
GEMINI_API_KEY=...
```

### Variable usage

- `OPENROUTER_API_KEY`: required for query enhancement/rerank/generation (`query_enhancement.py`, `describe_image_cli.py`)
- `GROQ_API_KEY`: used by `cli/check_model.py`
- `GEMINI_API_KEY`: used by `cli/test_llm.py`

## Sample commands

Run commands from repository root:

### 1) Build keyword index

```bash
uv run python cli/keyword_search_cli.py build
```

### 2) Keyword BM25 search

```bash
uv run python cli/keyword_search_cli.py bm25search "family movie about bears in the woods" --limit 5
```

### 3) Semantic search

```bash
uv run python cli/semantic_search_cli.py search "family movie about bears in the woods" --limit 5
```

### 4) Hybrid weighted search

```bash
uv run python cli/hybrid_search_cli.py weighted-search "family movie about bears in the woods" --alpha 0.5 --limit 5
```

### 5) Hybrid RRF search

```bash
uv run python cli/hybrid_search_cli.py rrf-search "family movie about bears in the woods" -k 60 --limit 5
```

### 6) RRF + LLM rerank

```bash
uv run python cli/hybrid_search_cli.py rrf-search "family movie about bears in the woods" --rerank-method individual --limit 3
```

### 7) Evaluate retrieval quality

```bash
uv run python cli/evaluation_cli.py --limit 5
```

### 8) Image-based search

```bash
uv run python cli/multimodal_search_cli.py image_search /absolute/path/to/image.jpg
```

## Real example: query -> ranked results

The repository includes a concrete regression-style query in `cli/check_candidates.py` and `cli/check_rerank.py`:

- **Query**: `family movie about bears in the woods`
- **Pipeline**: RRF candidate generation (`k=60`, top 15) -> optional LLM rerank -> top 3
- **Expected behavior**: bear-related titles (for example, *The Berenstain Bears Christmas Tree*, *Paddington*) should rank above unrelated titles.

Reproduce this flow:

```bash
uv run python cli/check_candidates.py
uv run python cli/check_rerank.py
```

## Notes

- First semantic/hybrid/multimodal runs can be slower because embedding caches are created.
- Cache artifacts are written under `cache/`.
- If keys are missing, LLM-dependent commands raise explicit runtime errors.
