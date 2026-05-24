import os
from functools import lru_cache
from requests.exceptions import RequestException

import requests
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from qdrant_client import QdrantClient


class EquityRAGService:
    def __init__(self, collection_name: str | None = None):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY environment variable.")

        qdrant_url = os.getenv("QDRANT_URL")
        if not qdrant_url:
            raise RuntimeError("Missing QDRANT_URL environment variable.")

        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION", "demo_collection")

        self.embedding_provider = os.getenv("EMBEDDING_PROVIDER", "hf_api").strip().lower()
        self.hf_api_key = os.getenv("HF_API_KEY")
        self.hf_embed_model = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.local_embed_model = os.getenv("LOCAL_EMBED_MODEL", "all-MiniLM-L6-v2")
        self.hf_fallback_to_local = os.getenv("HF_FALLBACK_TO_LOCAL", "false").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        if self.embedding_provider == "hf_api" and not self.hf_api_key:
            raise RuntimeError("Missing HF_API_KEY environment variable for EMBEDDING_PROVIDER=hf_api.")

        self._local_embeddings = None

        self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

        self.prompt = ChatPromptTemplate.from_template(
            """Answer the question based only on the following context:
{context}

Question: {question}
"""
        )
        self.output_parser = StrOutputParser()

    def _embed_query_hf_api(self, query: str) -> list[float]:
        url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.hf_embed_model}"
        headers = {
            "Authorization": f"Bearer {self.hf_api_key}",
            "Content-Type": "application/json",
        }
        payload = {"inputs": query, "options": {"wait_for_model": True}}

        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, list) and data and isinstance(data[0], list):
            return data[0]
        return data

    def _embed_query_local(self, query: str) -> list[float]:
        if self._local_embeddings is None:
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
            except ImportError as exc:
                raise RuntimeError(
                    "Local embeddings are enabled but optional local-embedding dependencies are missing. "
                    "Install requirements_for_ingestion.txt for dev/testing, or use EMBEDDING_PROVIDER=hf_api "
                    "for low-memory production environments like Render."
                ) from exc

            self._local_embeddings = HuggingFaceEmbeddings(model_name=self.local_embed_model)

        return self._local_embeddings.embed_query(query)

    def _embed_query(self, query: str) -> list[float]:
        if self.embedding_provider == "hf_api":
            try:
                return self._embed_query_hf_api(query)
            except RequestException as exc:
                if self.hf_fallback_to_local:
                    return self._embed_query_local(query)

                raise RuntimeError(
                    "Hugging Face API embedding request failed. "
                    "If DNS/network is blocked in this environment, set EMBEDDING_PROVIDER=local "
                    "for Codespaces testing, or enable HF_FALLBACK_TO_LOCAL=true if local embedding "
                    "dependencies are installed."
                ) from exc
        if self.embedding_provider == "local":
            return self._embed_query_local(query)
        raise RuntimeError("Unsupported EMBEDDING_PROVIDER. Use 'hf_api' or 'local'.")

    def _retrieve_context(self, query: str, k: int = 10) -> str:
        query_vector = self._embed_query(query)

        if hasattr(self.qdrant_client, "search"):
            hits = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=k,
                with_payload=True,
            )
        else:
            response = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=k,
                with_payload=True,
            )
            hits = response.points

        parts = []
        for hit in hits:
            payload = hit.payload or {}
            text = payload.get("page_content") or payload.get("text") or ""
            if text:
                parts.append(text)

        return "\n\n".join(parts)

    def ask(self, query: str) -> dict:
        context = self._retrieve_context(query, k=10)
        prompt_value = self.prompt.invoke({"context": context, "question": query})
        rag_answer = self.output_parser.invoke(self.llm.invoke(prompt_value))
        llm_answer = self.llm.invoke(query).content
        return {"rag_answer": rag_answer, "llm_answer": llm_answer}


@lru_cache(maxsize=1)
def get_equity_rag_service() -> EquityRAGService:
    return EquityRAGService()
