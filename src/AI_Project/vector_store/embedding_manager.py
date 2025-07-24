# embedding_manager.py

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def load_faiss_retriever(index_path="faiss_index"):
    """
    Loads the FAISS vector store and returns a retriever.

    Args:
        index_path (str): Path to FAISS index directory.

    Returns:
        retriever (VectorStoreRetriever): LangChain retriever object.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever()
    return retriever
