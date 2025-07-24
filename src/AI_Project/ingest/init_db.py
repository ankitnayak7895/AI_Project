# init_db.py

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR = "notebooks/data"  # Folder with your PDFs
INDEX_DIR = "faiss_index"

def load_and_split_all_pdfs(folder_path):
    print("📁 Scanning PDF files in:", folder_path)
    documents = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            print(f"📄 Loading: {filename}")
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    print(f"✅ Total documents loaded: {len(documents)}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    print(f"✂️ Total chunks after splitting: {len(chunks)}")
    return chunks

def main():
    print("🚀 Starting ingestion...")
    chunks = load_and_split_all_pdfs(DATA_DIR)

    if not chunks:
        print("⚠️ No PDF chunks found. Make sure PDFs are in the 'data/' folder.")
        return  # Exit early to avoid crash

    print("🔤 Generating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("💾 Creating FAISS index...")
    db = FAISS.from_documents(chunks, embeddings)

    print(f"📂 Saving FAISS index to: {INDEX_DIR}")
    db.save_local(INDEX_DIR)

    print("✅ Ingestion complete!")

if __name__ == "__main__":
    main()
