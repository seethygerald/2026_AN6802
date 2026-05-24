"""Build a persistent Chroma DB from files in ./data."""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

DATA_DIR = Path("./data")
PERSIST_DIR = "./chroma_db"
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
    documents = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    if not docs:
        raise RuntimeError("Document split produced zero chunks; cannot build vector store.")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(
        documents=docs,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )

    print(f"Loaded {len(documents)} documents")
    print(f"Split into {len(docs)} chunks")
    print(f"Stored {vector_store._collection.count()} chunks in {PERSIST_DIR}")


if __name__ == "__main__":
    main()
