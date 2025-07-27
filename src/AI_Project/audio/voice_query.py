import os
import sys
import speech_recognition as sr
import pyttsx3

# Setup project import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.AI_Project.vector_store.embedding_manager import load_faiss_retriever
from src.AI_Project.llm.gpt_qa import build_qa_pipeline

def listen_to_audio():
    recognizer = sr.Recognizer()

    print("🎤 Available microphones:")
    for i, name in enumerate(sr.Microphone.list_microphone_names()):
        print(f"  {i}: {name}")

    try:
        mic = sr.Microphone(device_index=0, sample_rate=44100, chunk_size=1024)
        with mic as source:
            print("🔊 Adjusting for ambient noise (2s)...")
            recognizer.adjust_for_ambient_noise(source, duration=2)
            print("🎙️ Listening... (timeout 5s)")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)

            # Save for debugging
            with open("debug_input.wav", "wb") as f:
                f.write(audio.get_wav_data())
            print("💾 Audio saved: debug_input.wav")

            return audio

    except sr.WaitTimeoutError:
        print("⏳ Timeout: No speech detected.")
    except Exception as e:
        print(f"🎤 Microphone error: {e}")

    return None

def speak_text(text):
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 180)
        engine.setProperty("volume", 1.0)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print(f"🔈 TTS error: {e}")

def fallback_text_query(qa):
    print("💬 Enter your question:")
    user_input = input("> ")
    result = qa.invoke({"query": user_input})
    if isinstance(result, dict):
        answer = result.get("result", "No answer found.")
    else:
        answer = str(result)
    print("🧠 Answer:", answer)
    speak_text(answer)

def main():
    print("🧠 Loading retriever and LLM...")
    retriever = load_faiss_retriever()
    qa = build_qa_pipeline(retriever)

    audio = listen_to_audio()
    recognizer = sr.Recognizer()

    if audio:
        try:
            question = recognizer.recognize_google(audio)
            print(f"📝 Transcribed: {question}")

            print("🤖 Querying...")
            result = qa.invoke({"query": question})

            if isinstance(result, dict):
                answer = result.get("result", "No answer found.")
            else:
                answer = str(result)

            print("💬 Answer:", answer)
            speak_text(answer)

        except sr.UnknownValueError:
            print("❌ Could not understand the audio.")
            fallback_text_query(qa)
        except sr.RequestError as e:
            print(f"🌐 Google API error: {e}")
            fallback_text_query(qa)
        except Exception as e:
            print(f"❗ Unexpected error: {e}")
            fallback_text_query(qa)
    else:
        print("⚠️ No audio received, switching to text input.")
        fallback_text_query(qa)

if __name__ == "__main__":
    main()
