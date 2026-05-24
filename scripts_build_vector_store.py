"""Build and upload vector embeddings from ./data into Qdrant."""

import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from qdrant_client import QdrantClient

DATA_DIR = Path("./data")
COLLECTION_NAME = "demo_collection"


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


def main() -> None:
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    if not qdrant_url:
        raise RuntimeError("Missing QDRANT_URL environment variable.")
    if not qdrant_api_key:
        raise RuntimeError("Missing QDRANT_API_KEY environment variable.")

    documents = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    if not docs:
        raise RuntimeError("Document split produced zero chunks; cannot build vector store.")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name=COLLECTION_NAME,
        force_recreate=True,
    )

    count = client.count(collection_name=COLLECTION_NAME, exact=True).count
    print(f"Loaded {len(documents)} documents")
    print(f"Split into {len(docs)} chunks")
    print(f"Uploaded {count} chunks to Qdrant collection '{COLLECTION_NAME}'")


if __name__ == "__main__":
    main()
