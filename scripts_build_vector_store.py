"""Build a Qdrant collection from files in ./data."""

import os
import uuid
from pathlib import Path

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

DATA_DIR = Path("./data")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "demo_collection")


def _load_pdf_documents(pdf_path: Path) -> list[Document]:
    reader = PdfReader(str(pdf_path))
    docs: list[Document] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={"source": str(pdf_path), "page": page_number},
            )
        )

    return docs


def _load_text_document(text_path: Path) -> list[Document]:
    text = text_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return [Document(page_content=text, metadata={"source": str(text_path)})]


def load_documents() -> list[Document]:
    if not DATA_DIR.exists():
        raise RuntimeError("Missing ./data directory. Add your source files there first.")

    documents: list[Document] = []

    for pdf_path in sorted(DATA_DIR.rglob("*.pdf")):
        documents.extend(_load_pdf_documents(pdf_path))

    for text_path in sorted(DATA_DIR.rglob("*.txt")):
        documents.extend(_load_text_document(text_path))

    if not documents:
        raise RuntimeError(
            "No readable .pdf or .txt documents found in ./data. "
            "Add files to ./data and try again."
        )

    return documents

def _chunk_id(doc: Document) -> str:
    source = doc.metadata.get("source", "")
    page = doc.metadata.get("page", "")
    payload = f"{source}|{page}|{doc.page_content}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, payload))

def main() -> None:
    qdrant_url = os.getenv("QDRANT_URL")
    if not qdrant_url:
        raise RuntimeError("Missing QDRANT_URL environment variable.")

    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    documents = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    if not docs:
        raise RuntimeError("Document split produced zero chunks; cannot build vector store.")

    ids = [_chunk_id(d) for d in docs]

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name=COLLECTION_NAME,
        ids=ids,
    )

    print(f"Loaded {len(documents)} documents")
    print(f"Split into {len(docs)} chunks")
    print(f"Stored collection '{COLLECTION_NAME}' in Qdrant at {qdrant_url}")
    print(f"Retriever ready: {vector_store is not None}")


if __name__ == "__main__":
    main()
