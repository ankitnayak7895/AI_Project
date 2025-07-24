from langchain_community.llms import LlamaCpp
from langchain_core.prompts import ChatPromptTemplate
import os

def load_llm():
    print("🔄 Loading TinyLlama model...")
    
    llm=LlamaCpp(
        model_path="/home/ankit/AI_Project/models/tinyllama.gguf",  # your downloaded .gguf model
        temperature=0.7,
        max_tokens=1024,
        n_batch=64,
        n_ctx=2048,
        verbose=True,
        chat_format="llama-2",
        n_gpu_layers=0,# set >0 if using GPU
        
        )

    print("✅ TinyLlama loaded.")
    return llm