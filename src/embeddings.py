import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


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
        chunk_overlap=200,  # overlap keeps context intact at chunk boundaries
        separators=["\n\n", "\n", " ", ""]
    )

    return splitter.split_documents(documents)