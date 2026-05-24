import os
from functools import lru_cache

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

        self.hf_api_key = os.getenv("HF_API_KEY")
        if not self.hf_api_key:
            raise RuntimeError("Missing HF_API_KEY environment variable.")

        self.hf_embed_model = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

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

        # HF may return [dim] or [[dim]]
        if isinstance(data, list) and data and isinstance(data[0], list):
            return data[0]
        return data

    def _retrieve_context(self, query: str, k: int = 10) -> str:
        query_vector = self._embed_query_hf_api(query)

        hits = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=k,
            with_payload=True,
        )

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
