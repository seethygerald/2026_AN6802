## Move from local Chroma to hosted Qdrant (Render 512MB-friendly)

### Current branch + commit
- Branch: `work`
- Commit at start of this update: `3c5f2b7ec784a71bd37cab6b73cef98265615d0f`

### Recommended production architecture (Render 512MB)
Use hosted Qdrant + hosted Hugging Face inference for query embeddings:

- Ingestion (one-time/offline):
  - `scripts_build_vector_store.py` creates vectors with `all-MiniLM-L6-v2` and loads `demo_collection` in Qdrant.
- Runtime (Render web service):
  - `rag_service.py` defaults to `EMBEDDING_PROVIDER=hf_api`.
  - Query embeddings are generated through `https://api-inference.huggingface.co/...`.
  - App sends vector to Qdrant and uses Gemini for generation.

Why this fits 512MB:
- Avoids loading `sentence-transformers` model into Render runtime memory.
- Keeps heavy embedding dependencies out of runtime `requirements.txt`.

### Temporary fallback for Codespaces/local testing
If DNS/network cannot reach Hugging Face inference endpoint, set:

- `EMBEDDING_PROVIDER=local`
- optional `LOCAL_EMBED_MODEL=all-MiniLM-L6-v2`

This local mode lazily imports `langchain_huggingface` only when selected.

> Important: to preserve embedding-space consistency, local fallback must use the same model family used for ingestion (`all-MiniLM-L6-v2`).

### Environment variables
Required for runtime:
- `GEMINI_API_KEY`
- `QDRANT_URL`
- `QDRANT_API_KEY` (if your cluster requires it)
- `QDRANT_COLLECTION=demo_collection`
- `EMBEDDING_PROVIDER=hf_api` (default)
- `HF_API_KEY` (required for `hf_api`)
- `HF_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2` (default)

Optional local fallback vars:
- `EMBEDDING_PROVIDER=local`
- `LOCAL_EMBED_MODEL=all-MiniLM-L6-v2`

### Dependency split
- Runtime (`requirements.txt`) intentionally stays lightweight.
- Ingestion/local-embed extras in `requirements_for_ingestion.txt`:
  - `langchain-huggingface`
  - `sentence-transformers`
  - `unstructured[pdf]`
  - `pypdf`

### Deployment checklist
1. In Render environment, set all runtime env vars above.
2. Keep `EMBEDDING_PROVIDER=hf_api` for low memory.
3. Deploy web service.
4. Smoke test `/equity/query` with a known prompt.
5. If DNS failure appears in non-production dev envs, switch only that env to `EMBEDDING_PROVIDER=local`.

### Validation commands and expected results
- `python3 -m compileall app.py rag_service.py`
  - Expected: both files compile successfully.
- `python3 - <<'PY' ... PY` (instantiate service with `EMBEDDING_PROVIDER=local` but without local deps)
  - Expected: explicit runtime error guiding to install ingestion requirements or switch to hf_api.
- `python3 - <<'PY' ... PY` (instantiate service with `EMBEDDING_PROVIDER=hf_api` without HF key)
  - Expected: clear missing-HF key error.
