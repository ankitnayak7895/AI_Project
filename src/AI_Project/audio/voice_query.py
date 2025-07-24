# src/AI_Project/audio/audio_qa.py

import os
import sys
import speech_recognition as sr
import pyttsx3

# Set path to access src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.AI_Project.vector_store.embedding_manager import load_faiss_retriever
from src.AI_Project.llm.gpt_qa import build_qa_pipeline

def listen_to_audio():
    recognizer = sr.Recognizer()
    
    # Explicit configuration
    mic = sr.Microphone(
        device_index=0,  # Your ALC897 Analog device
        sample_rate=44100,  # Match your working arecord settings
        chunk_size=1024  # Standard chunk size
    )
    
    with mic as source:
        print("🔊 Adjusting for ambient noise (3 seconds)...")
        recognizer.adjust_for_ambient_noise(source, duration=3)
        recognizer.dynamic_energy_threshold = True
        recognizer.energy_threshold = 400  # Default is 300, increase if needed
        
        print("🎙️ Speak now (waiting for 5 seconds)...")
        try:
            audio = recognizer.listen(
                source, 
                timeout=5,
                phrase_time_limit=5
            )
            print(f"✅ Captured {len(audio.frame_data)} bytes")
            
            # Save for verification
            with open("debug_python.wav", "wb") as f:
                f.write(audio.get_wav_data())
            print("💾 Saved debug_python.wav - play this to verify")
            
            return audio
        except Exception as e:
            print(f"❌ Error: {type(e).__name__}: {str(e)}")
            return None
def speak_text(text):
    engine = pyttsx3.init()
    engine.setProperty("rate", 180)  # Speech rate (default ~200)
    engine.setProperty("volume", 1.0)  # Max volume
    engine.say(text)
    engine.runAndWait()
    engine.stop()


def main():
    print("🧠 Loading retriever and LLM...")
    retriever = load_faiss_retriever()
    qa = build_qa_pipeline(retriever)

    audio = listen_to_audio()
    if audio:
        
        recognizer = sr.Recognizer()
        try:
            question = recognizer.recognize_google(audio)
            print(f"📝 Transcribed: {question}")
            print("🤖 Thinking...")
            result = qa.invoke({"query": question})
            answer = result.get("result")
            print("💬 Answer:", result.get("result"))
            speak_text(answer)  # <--- Add this line to speak the answer
        except sr.UnknownValueError:
            print("❌ Could not understand audio.")
        except Exception as e:
            print(f"❗ Error during transcription or QA: {e}")



if __name__ == "__main__":
    main()
