import gradio as gr

def transcribe(audio):
    return "Audio received!"

demo = gr.Interface(fn=transcribe, inputs=gr.Audio(sources="microphone", type="filepath"), outputs="text")
demo.launch(share=True)  # use share=True to get HTTPS
