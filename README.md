## Move from local Chroma to hosted Qdrant (Render 512MB-friendly)

This app now uses **Qdrant** instead of local `./chroma_db`, so your Render instance no longer needs to store/process the vector index on ephemeral disk.

### What changed in code

- `rag_service.py`
  - Replaced `langchain_chroma.Chroma` with `langchain_qdrant.QdrantVectorStore`.
  - Reads `QDRANT_URL`, `QDRANT_API_KEY` (optional for local Qdrant), and `QDRANT_COLLECTION`.
  - Connects through `QdrantClient` and creates retriever from that collection.

- `scripts_build_vector_store.py`
  - Replaced local Chroma persistence logic with `QdrantVectorStore.from_documents(...)`.
  - No more `persist_directory`; chunks are pushed directly to Qdrant.

- `requirements.txt`
  - Removed `langchain-chroma`.
  - Added `langchain-qdrant` and `qdrant-client`.

---

## Step-by-step setup

### 1) Create a Qdrant instance

Pick one:
- **Qdrant Cloud** (recommended for Render free tier)
- Self-hosted Qdrant (Docker/VM)

After creation, copy:
- **Cluster URL** → use as `QDRANT_URL`
- **API Key** → use as `QDRANT_API_KEY` (if your cluster requires auth)

### 2) Set environment variables locally

```bash
export GEMINI_API_KEY="your_gemini_key"
export QDRANT_URL="https://<cluster-id>.<region>.aws.cloud.qdrant.io:6333"
export QDRANT_API_KEY="your_qdrant_api_key"
export QDRANT_COLLECTION="demo_collection"
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Ingest documents into Qdrant

Ensure your docs are in `./data`, then run:

```bash
python scripts_build_vector_store.py
```

This creates/updates the collection in Qdrant with embedded chunks.

### 5) Run app locally and verify retrieval

```bash
python app.py
```

Ask one question you know is answered in your PDFs and verify RAG answer quality.

### 6) Configure Render environment variables

In Render dashboard for your service, add:
- `GEMINI_API_KEY`
- `QDRANT_URL`
- `QDRANT_API_KEY`
- `QDRANT_COLLECTION`

Then redeploy.

### 7) Keep ingestion off your web dyno (recommended)

For free/low-memory deployments, avoid building embeddings during web startup.

Preferred approach:
- run `scripts_build_vector_store.py` from your laptop/CI whenever docs change,
- keep the web service read-only against existing Qdrant collection.

---

## Operational notes

- If you change embedding model later, create a **new collection** and re-ingest (dimension mismatch otherwise).
- You can tune retriever depth in `rag_service.py` via `search_kwargs={"k": 10}`.
- `QDRANT_API_KEY` may be omitted only for unsecured local Qdrant.
