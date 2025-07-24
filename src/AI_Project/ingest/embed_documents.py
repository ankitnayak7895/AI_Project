import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from src.AI_Project.ingest.document_loader import load_and_split_pdf

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings


def build_vector_store(pdf_path, db_path="vectorstore"):
    chunks = load_and_split_pdf(pdf_path)
    embedding = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(chunks, embedding)
    vectordb.save_local(db_path)
    print(f"Vector store saved at {db_path}")