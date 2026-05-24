"""Build a persistent Chroma DB from files in ./data."""

from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DATA_DIR = Path("./data")
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "demo_collection"


def load_documents():
    if not DATA_DIR.exists():
        raise RuntimeError("Missing ./data directory. Add your source files there first.")

    pdf_loader = DirectoryLoader(
        str(DATA_DIR),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        silent_errors=True,
    )
    txt_loader = DirectoryLoader(
        str(DATA_DIR),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        silent_errors=True,
    )

    documents = pdf_loader.load() + txt_loader.load()
    if not documents:
        raise RuntimeError(
            "No readable .pdf or .txt documents found in ./data. "
            "Add files to ./data and try again."
        )

    return documents


def main():
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
