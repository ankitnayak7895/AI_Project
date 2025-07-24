import gradio as gr
import speech_recognition as sr
import tempfile
import pyttsx3
import os
import shutil

from src.AI_Project.vector_store.embedding_manager import load_faiss_retriever
from src.AI_Project.llm.gpt_qa import build_qa_pipeline

# Load retriever and QA pipeline once
retriever = load_faiss_retriever()
qa = build_qa_pipeline(retriever)

# 🔊 Text-to-speech function
def tts(text):
    """Convert text to speech and return path to WAV file."""
    tts_engine = pyttsx3.init()
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tts_engine.save_to_file(text, tmp_wav.name)
    tts_engine.runAndWait()
    tmp_wav.close()
    return tmp_wav.name  # ✅ Return path to audio file

# 🎙️ Handle audio Q&A
def answer_from_audio(audio):
    if audio is None:
        return "No audio input.", "", None

    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio) as source:
            audio_data = recognizer.record(source)
        question = recognizer.recognize_google(audio_data)
        print(f"🗣️ You said: {question}")

        answer = qa.invoke({"query": question})["result"]
        answer_audio = tts(answer)
        return question, answer, answer_audio

    except sr.UnknownValueError:
        return "❌ Speech not recognized.", "", None
    except sr.RequestError as e:
        return f"❌ Speech API error: {e}", "", None
    except Exception as e:
        return f"⚠️ Error: {e}", "", None

# 📝 Handle text Q&A
def answer_from_text(text):
    if not text:
        return "Please type a question.", None
    answer = qa.invoke({"query": text})["result"]
    answer_audio = tts(answer)
    return answer, answer_audio

# 📁 Handle PDF upload
def handle_pdf_upload(pdf):
    if pdf is None:
        return "No PDF uploaded."
    save_path = os.path.join("notebooks/data", os.path.basename(pdf))
    shutil.copy(pdf, save_path)
    return f"✅ Uploaded `{os.path.basename(pdf)}`. Rebuild FAISS index to use this document."

# 🖥️ Gradio UI Layout
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Local LLM QA Assistant (Gradio)")
    gr.Markdown("Ask questions using your **voice**, **text**, or upload **PDF files** for context.")

    with gr.Tab("🎙️ Audio Q&A"):
        gr.Interface(
            fn=answer_from_audio,
            inputs=gr.Audio(type="filepath", label="🎤 Speak your question"),
            outputs=[
                gr.Textbox(label="🗣️ You said"),
                gr.Textbox(label="💬 Answer"),
                gr.Audio(label="🔊 Spoken Answer", type="filepath")  # ✅ MUST SET THIS
            ],
            live=False,
            allow_flagging="never"
        ).render()

    with gr.Tab("📝 Text Q&A"):
        gr.Interface(
            fn=answer_from_text,
            inputs=gr.Textbox(lines=2, placeholder="Type your question here..."),
            outputs=[
                gr.Textbox(label="💬 Answer"),
                gr.Audio(label="🔊 Spoken Answer", type="filepath")  # ✅ MUST SET THIS
            ],
            live=False,
            allow_flagging="never"
        ).render()

    with gr.Tab("📄 Upload PDF"):
        pdf_upload = gr.File(label="📄 Upload PDF", file_types=[".pdf"])
        upload_output = gr.Textbox(label="📂 Upload Status")
        pdf_upload.upload(handle_pdf_upload, inputs=pdf_upload, outputs=upload_output)

demo.launch()
