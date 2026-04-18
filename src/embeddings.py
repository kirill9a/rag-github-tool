import os
import shutil
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

CHROMA_PATH = "./chroma_db"


def load_documents(repo_path: str, file_paths: list[str]) -> list[Document]:
    """
    Reads files from disk and wraps them in LangChain Document objects.
    """
    docs = []

    for relative_path in file_paths:
        full_path = os.path.join(repo_path, relative_path)

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()

            if not content.strip():
                continue

            # store the relative path in metadata so the LLM can say
            # "I found this in src/main.go"
            doc = Document(
                page_content=content,
                metadata={"source": relative_path}
            )
            docs.append(doc)

        except UnicodeDecodeError:
            # binary files that slipped through the filter
            print(f"Skipping non-text file: {relative_path}")

    return docs


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Splits documents into smaller chunks for embedding.
    """
    # tries to split on \n\n first, then \n, then spaces
    # this avoids cutting functions or classes in half
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )

    return splitter.split_documents(documents)


def save_to_chroma(chunks: list[Document]) -> Chroma:
    """
    Embeds chunks and saves them to a local ChromaDB instance.
    """
    # always start fresh — one repo at a time for now
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    print(f"Embedding and saving {len(chunks)} chunks...")

    db = Chroma.from_documents(
        chunks,
        OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_base="https://api.vsellm.ru/v1"
        ),
        persist_directory=CHROMA_PATH
    )

    print(f"Saved to {CHROMA_PATH}")
    return db


# test

if __name__ == "__main__":
    from ingest import clone_repository, build_project_tree

    TEST_URL = "https://github.com/JuliusBrussee/caveman"

    print("1. Cloning repository...")
    repo_path = clone_repository(TEST_URL)

    print("\n2. Building file tree...")
    file_list = build_project_tree(repo_path)

    print("\n3. Loading files...")
    docs = load_documents(repo_path, file_list)
    print(f"Loaded {len(docs)} files")

    print("\n4. Chunking...")
    chunks = chunk_documents(docs)
    print(f"Got {len(chunks)} chunks")

    print("\n5. Saving to ChromaDB...")
    db = save_to_chroma(chunks)

    print("\n--- SEARCH TEST ---")
    query = "database connection or configuration"
    print(f"Query: '{query}'")

    results = db.similarity_search(query, k=3)

    for i, res in enumerate(results):
        print(f"\n--- Result {i+1} ({res.metadata['source']}) ---")
        print(res.page_content[:200] + "...")