import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import FakeEmbeddings

# Directory where ChromaDB stores its data
CHROMA_PATH = "rag/chroma_db"
DOCS_PATH = "rag/osha_docs"

def get_embeddings():
    """
    Use fake embeddings for cloud deployment.
    Locally, sentence-transformers runs for real semantic search.
    """
    import os
    if os.getenv("RENDER"):
        # On Render cloud — use fake embeddings
        return FakeEmbeddings(size=384)
    else:
        # Locally — use real semantic embeddings
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )

def ingest_documents():
    """
    Read all OSHA documents and store them in ChromaDB.
    Only needs to run once — data persists in chroma_db folder.
    """
    print("Loading OSHA documents...")

    # Load all .txt files from osha_docs folder
    loader = DirectoryLoader(
        DOCS_PATH,
        glob="**/*.txt",
        loader_cls=TextLoader
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")

    # Split into smaller chunks
    # Why? LLMs have token limits — we split docs into pieces
    # and only retrieve the relevant pieces
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # each chunk = 500 characters
        chunk_overlap=50,    # 50 char overlap so context isn't lost
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    # Create embeddings and store in ChromaDB
    print("Creating vector embeddings (first time is slow)...")
    embeddings = get_embeddings()

    # Store in ChromaDB
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    print(f"Stored {len(chunks)} chunks in ChromaDB!")
    print(f"Database saved to: {CHROMA_PATH}")
    return vectorstore


def get_vectorstore():
    """Load existing ChromaDB — use this after first ingest"""
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )


if __name__ == "__main__":
    ingest_documents()