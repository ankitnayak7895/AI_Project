# src/AI_Project/llm/gpt_qa.py
import os
import sys

# Add root path to sys.path to resolve imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.AI_Project.llm.local_llm import load_llm  # ✅ Correct import now
from langchain.chains import RetrievalQA

def build_qa_pipeline(retriever):
    """
    Builds a RetrievalQA chain using a local LLM and the given retriever.

    Args:
        retriever: The retriever object (e.g. from FAISS vectorstore)

    Returns:
        RetrievalQA chain
    """
    llm = load_llm()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",  # You can change this to "map_reduce" if needed
        return_source_documents=True
    )
    return qa_chain
