"""Run once to build a persistent Chroma DB from files in ./data."""

from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def main():
    loader = DirectoryLoader(
        "./data",
        glob="**/*",
        loader_cls=UnstructuredFileLoader,
        silent_errors=True,
    )
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(
        documents=docs,
        collection_name="demo_collection",
        embedding=embeddings,
        persist_directory="./chroma_db",
    )
    print(f"Stored {vector_store._collection.count()} chunks in ./chroma_db")


if __name__ == "__main__":
    main()
