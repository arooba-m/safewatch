from rag.ingest import get_vectorstore, ingest_documents
import os

CHROMA_PATH = "rag/chroma_db"

def get_retriever():
    """Get the vector store, ingesting docs if needed"""
    if not os.path.exists(CHROMA_PATH):
        print("First run — ingesting documents...")
        ingest_documents()
    return get_vectorstore()


def retrieve_osha_context(query: str, k: int = 3) -> str:
    """
    Search ChromaDB for OSHA regulations relevant to the query.
    
    query: what to search for e.g. "worker without helmet"
    k: how many chunks to retrieve
    Returns: relevant OSHA text as a string
    """
    vectorstore = get_retriever()
    
    # Similarity search — finds most relevant chunks
    docs = vectorstore.similarity_search(query, k=k)
    
    if not docs:
        return "No specific OSHA regulations found for this scenario."
    
    # Combine retrieved chunks into one string
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])
    return context


def retrieve_with_scores(query: str, k: int = 3) -> list:
    """
    Same as above but also returns relevance scores.
    Score closer to 0 = more relevant.
    Useful for debugging and showing in dashboard.
    """
    vectorstore = get_retriever()
    results = vectorstore.similarity_search_with_score(query, k=k)
    
    return [
        {
            "content": doc.page_content,
            "relevance_score": round(float(score), 4),
            "source": doc.metadata.get("source", "unknown")
        }
        for doc, score in results
    ]


if __name__ == "__main__":
    # Test the retriever
    test_query = "worker without helmet on construction site"
    print(f"Query: {test_query}\n")
    
    results = retrieve_with_scores(test_query)
    for i, r in enumerate(results):
        print(f"Result {i+1} (score: {r['relevance_score']}):")
        print(r['content'][:200])
        print("---")