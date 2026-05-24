import os
from functools import lru_cache

from dotenv import load_dotenv
from requests.exceptions import RequestException

import requests
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from qdrant_client import QdrantClient

load_dotenv()


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
        self.local_embed_model = os.getenv("LOCAL_EMBED_MODEL", "all-MiniLM-L6-v2")
        self.external_embed_url = os.getenv("EMBEDDING_ENDPOINT_URL", "").strip()
        self.external_embed_token = os.getenv("EMBEDDING_ENDPOINT_TOKEN", "").strip()

        if self.embedding_provider == "external" and not self.external_embed_url:
            raise RuntimeError("Missing EMBEDDING_ENDPOINT_URL for EMBEDDING_PROVIDER=external.")

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

    def _embed_query_external(self, query: str) -> list[float]:
        headers = {"Content-Type": "application/json"}
        if self.external_embed_token:
            headers["Authorization"] = f"Bearer {self.external_embed_token}"

        resp = requests.post(
            self.external_embed_url,
            headers=headers,
            json={"text": query},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()

        embedding = data.get("embedding") if isinstance(data, dict) else None
        if not isinstance(embedding, list) or not embedding:
            raise RuntimeError("External embedding endpoint returned an invalid response payload.")

        return embedding

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
        if self.embedding_provider == "local":
            return self._embed_query_local(query)
        if self.embedding_provider == "external":
            try:
                return self._embed_query_external(query)
            except RequestException as exc:
                raise RuntimeError(
                    "External embedding endpoint request failed. "
                    "For offline Codespaces testing, set EMBEDDING_PROVIDER=local and install "
                    "requirements_for_ingestion.txt."
                ) from exc
        raise RuntimeError("Unsupported EMBEDDING_PROVIDER. Use 'external' or 'local'.")

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
        try:
            context = self._retrieve_context(query, k=10)
            prompt_value = self.prompt.invoke({"context": context, "question": query})
            rag_answer = self.output_parser.invoke(self.llm.invoke(prompt_value))
            retrieval_error = None
        except Exception as exc:
            context = ""
            retrieval_error = str(exc)
            rag_answer = (
                "RAG context retrieval is temporarily unavailable, so this answer is generated "
                "without vector search context."
            )

        try:
            llm_answer = self.llm.invoke(query).content
            llm_error = None
        except Exception as exc:
            llm_answer = "LLM response is temporarily unavailable. Please try again shortly."
            llm_error = str(exc)

        response = {"rag_answer": rag_answer, "llm_answer": llm_answer}
        if retrieval_error:
            response["warning"] = retrieval_error
        if llm_error:
            response["llm_warning"] = llm_error
        return response


@lru_cache(maxsize=1)
def get_equity_rag_service() -> EquityRAGService:
    return EquityRAGService()
