# Insn — semantic paper explorer

[Demo video](https://www.youtube.com/watch?v=0hZhV7mirQE)

A Flask research-paper prototype that fetches arXiv metadata, embeds titles and abstracts, searches a FAISS index, and presents a three-dimensional paper landscape.

## Overview

This implementation computes its own semantic paper index and PCA coordinates. It is separate from `Insn_`, which renders remotely supplied Datamapplot example data.

## Features

- arXiv `cs.AI` paper ingestion and local metadata caching.
- SentenceTransformer embeddings and FAISS nearest-neighbor search.
- Related-paper links and three-dimensional PCA coordinates for visualization.
- Paper details and citation/reference fields.
- Gemini-assisted questions about a selected paper's title, authors, and abstract.

## Architecture

`app.py` loads the embedding model and paper dataset at module import. It reuses local cache files when available, otherwise fetches records, embeds them, builds a FAISS L2 index, computes neighbor relationships and PCA coordinates, and saves the artifacts. Flask then exposes data and query routes to `templates/index.html`.

The target dataset in this snapshot is 1,000 papers. A first startup may perform many external requests and model downloads before the HTTP server is available.

## Tech stack

Python, Flask, NumPy, Sentence Transformers (`all-MiniLM-L6-v2`), FAISS, scikit-learn PCA, arXiv, Requests, Beautiful Soup, tqdm, and the `google-generativeai` client.

## Project structure

- `app.py` — ingestion, indexing, plotting coordinates, model integration, and API.
- `templates/index.html` — browser visualization/interface.
- `requirements.txt` — partial dependency list.
- Runtime `papers.json`, `embeddings.npy`, and `faiss_index.bin` — metadata, vectors, and search index.

## Run locally

Create and activate a Python virtual environment before installing dependencies.

```bash
git clone https://github.com/anishkganesh/Insn.git
cd Insn
python -m venv .venv
pip install -r requirements.txt
pip install beautifulsoup4 tqdm google-generativeai
```

The additional install covers imports absent from the committed requirements file. Configure your own Gemini credential locally before starting: the code has an inline `genai.configure(api_key=...)` assignment, so setting an environment variable alone does not replace it. Do not use or redistribute a committed credential.

```bash
python app.py
```

Open `http://localhost:5000` after initialization finishes. The configured model is `gemini-1.5-flash`; the repository's older provider/client integration requires an availability and compatibility check in your own environment.

## Configuration and data

Cache files are relative to the working directory and must remain consistent with one another. Embeddings are derived from title-plus-abstract text, not full PDFs. PCA coordinates are a visualization projection, not a measured scientific relationship.

Semantic Scholar is queried for citation information, but **this code substitutes random citation/reference counts when metadata is missing or a lookup fails, and can substitute a random citation count for zero**. Displayed counts must not be represented as verified bibliometrics.

## Usage

Browse the paper landscape, search with text or a selected paper, and inspect neighboring results. Ask questions using the paper's abstract context, then verify answers against the paper itself.

Routes are `/`, `/api/papers`, `/api/citations`, `/api/query`, and `/api/llm`.

## Validation

No automated test suite is included. Check initialization, cache reuse, index/metadata alignment, search results, visualization, and the optional Gemini path. This documentation review did not fetch the corpus or execute model calls.

## Deployment

Use a persistent Python service with sufficient startup time, memory, writable storage, and outbound network access. Flask's debug runner is a local development entry point. No complete production deployment configuration is provided.

## Limitations

- Provider/model compatibility and network quotas can prevent initialization or answering.
- Citation fallback values are synthetic.
- Search quality is not benchmarked; PCA reduces a high-dimensional space to three coordinates.
- Growing or mismatched caches can cause index/record inconsistencies.
- The app processes abstracts, not full-text scholarly evidence.

## Attribution and license

arXiv, Semantic Scholar, embedding models, and dependencies retain their respective terms. No standalone project license file is included; this README does not grant a new license.
