import pyttsx3

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    

import pyttsx3
engine = pyttsx3.init()
for voice in engine.getProperty('voices'):
    print(voice.id)