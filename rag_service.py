import os
from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
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
        collection_name = collection_name or os.getenv("QDRANT_COLLECTION", "demo_collection")

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
        )

        self.retriever = vector_store.as_retriever(search_kwargs={"k": 10})
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

        template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
        prompt = ChatPromptTemplate.from_template(template)

        self.qa_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self, query: str) -> dict:
        rag_answer = self.qa_chain.invoke(query)
        llm_answer = self.llm.invoke(query).content
        return {"rag_answer": rag_answer, "llm_answer": llm_answer}


@lru_cache(maxsize=1)
def get_equity_rag_service() -> EquityRAGService:
    return EquityRAGService()
