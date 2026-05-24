
For codespaces deployment:

# Required
export GEMINI_API_KEY="YOUR_GEMINI_KEY"
export QDRANT_URL="YOUR_QDRANT_URL"
export QDRANT_API_KEY="YOUR_QDRANT_API_KEY"   # omit only if your cluster is public/no key
export QDRANT_COLLECTION="demo_collection"

# Embedding mode for Codespaces DNS-blocked environments
export EMBEDDING_PROVIDER="local"
export LOCAL_EMBED_MODEL="all-MiniLM-L6-v2"

# Keep fallback off; you are already explicitly using local mode
export HF_FALLBACK_TO_LOCAL="false"