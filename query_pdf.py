from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import LlamaCpp

# Step 1: Load FAISS index
db = FAISS.load_local(
    "faiss_index",
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    allow_dangerous_deserialization=True
)

# Step 2: Ask user a question
query = input("💬 Ask a question about the document:\n> ")

# Step 3: Init LLM (TinyLlama)
llm = LlamaCpp(
    model_path="./models/tinyllama.gguf",
    n_ctx=2048,
    n_gpu_layers=0,
    temperature=0.7,
    top_p=1,
    verbose=False
)

# Step 4: Build Retrieval QA chain
retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=False  # Optional
)

# Step 5: Run QA
result = qa.invoke({"query": query})

# Step 6: Print answer
print("\n🧠 Answer:")
print(result["result"])  # Show only the answer string

# Step 7: Manual cleanup to avoid exception
try:
    llm.client.__del__()  # or: llm.client.close() depending on version
except Exception:
    pass
