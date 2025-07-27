import gradio as gr
import speech_recognition as sr
import tempfile
import pyttsx3
import os
import shutil
import time

from src.AI_Project.vector_store.embedding_manager import load_faiss_retriever
from src.AI_Project.llm.gpt_qa import build_qa_pipeline

# Ensure notebooks/data directory exists
os.makedirs("notebooks/data", exist_ok=True)

# Initialize pyttsx3 engine globally
try:
    tts_engine = pyttsx3.init()
except Exception as e:
    print(f"Failed to initialize pyttsx3: {e}")
    tts_engine = None

# Load retrieval and QA pipeline with optimized settings
try:
    retriever = load_faiss_retriever()
    # Optimize retriever: limit to top 2 documents
    retriever.search_kwargs = {"k": 2}  # NEW: Reduce retrieval overhead
    qa = build_qa_pipeline(retriever)
except Exception as e:
    print(f"Error loading retriever or QA pipeline: {e}")
    raise

# Text-to-speech conversion
def tts(text):
    if not tts_engine or not text:
        return None
    try:
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tts_engine.save_to_file(text, tmp_wav.name)
        tts_engine.runAndWait()
        tmp_wav.close()
        # Removed time.sleep to reduce delay
        if os.path.exists(tmp_wav.name):
            return tmp_wav.name
        else:
            print(f"TTS error: WAV file {tmp_wav.name} not created")
            return None
    except Exception as e:
        print(f"TTS error: {e}")
        return None

# Answer from microphone input
def answer_from_audio(audio):
    if audio is None:
        return "No audio input.", "", None
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio) as source:
            recognizer.adjust_for_ambient_noise(source)
            audio_data = recognizer.record(source)
        question = recognizer.recognize_google(audio_data)
        print(f"🗣️ You said: {question}")
        # Optimize QA: limit max tokens
        answer = qa.invoke({"query": question, "max_new_tokens": 100})["result"]  # NEW: Limit output tokens
        answer_audio = tts(answer)
        return question, answer, answer_audio
    except sr.UnknownValueError:
        return "❌ Speech not recognized.", "", None
    except sr.RequestError as e:
        return f"❌ Speech API error: {e}", "", None
    except Exception as e:
        return f"⚠️ Error: {e}", "", None

# Answer from text input
def answer_from_text(text):
    if not text:
        return "Please type a question.", None
    try:
        # Optimize QA: limit max tokens
        answer = qa.invoke({"query": text, "max_new_tokens": 100})["result"]  # NEW: Limit output tokens
        answer_audio = tts(answer)
        return answer, answer_audio
    except Exception as e:
        return f"⚠️ Error: {e}", None

# Handle PDF upload
def handle_pdf_upload(pdf):
    if pdf is None:
        return "No PDF uploaded."
    try:
        pdf_path = pdf.name if hasattr(pdf, 'name') else pdf
        if not pdf_path.lower().endswith('.pdf'):
            return "❌ Please upload a valid PDF file."
        os.makedirs("notebooks/data", exist_ok=True)
        save_path = os.path.join("notebooks/data", os.path.basename(pdf_path))
        shutil.copy(pdf_path, save_path)
        return f"✅ Uploaded `{os.path.basename(pdf_path)}`. Rebuild FAISS index to use this document."
    except Exception as e:
        return f"⚠️ Error uploading PDF: {e}"

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Local LLM QA Assistant (Gradio)")
    gr.Markdown("Ask questions using your **voice**, **text**, or upload **PDF files** for context.")

    with gr.Tab("🎙️ Audio Q&A"):
        gr.Interface(
            fn=answer_from_audio,
            inputs=gr.Audio(source="microphone", type="filepath", label="🎤 Speak your question"),
            outputs=[
                gr.Textbox(label="🗣️ You said"),
                gr.Textbox(label="💬 Answer"),
                gr.Audio(label="🔊 Spoken Answer", type="filepath")
            ],
            allow_flagging="never",
            live=False  # NEW: Disable live mode to reduce overhead
        )

    with gr.Tab("📝 Text Q&A"):
        gr.Interface(
            fn=answer_from_text,
            inputs=gr.Textbox(lines=2, placeholder="Type your question here..."),
            outputs=[
                gr.Textbox(label="💬 Answer"),
                gr.Audio(label="🔊 Spoken Answer", type="filepath")
            ],
            allow_flagging="never",
            live=False  # NEW: Disable live mode
        )

    with gr.Tab("📄 Upload PDF"):
        pdf_upload = gr.File(label="📄 Upload PDF", file_types=[".pdf"])
        upload_output = gr.Textbox(label="📂 Upload Status")
        pdf_upload.upload(handle_pdf_upload, inputs=pdf_upload, outputs=upload_output)

# Cleanup pyttsx3 engine on exit
try:
    demo.launch(debug=True)
finally:
    if tts_engine:
        tts_engine.stop()