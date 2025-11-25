import gradio as gr
import openai
import os

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")


# Transcription function
def transcribe_audio(audio):
    if audio is None:
        return "No audio received."

    # audio is a tuple: (sample_rate, data) if type="numpy", or a filepath if type="filepath"
    with open(audio, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript["text"]


# Launch Gradio interface
def launch_interface():
    gr.Interface(
        fn=transcribe_audio,
        inputs=gr.Audio(type="filepath", label="Record Your Voice"),
        outputs=gr.Textbox(label="Transcription"),
        title="🎙️ OpenAI Whisper Speech-to-Text",
        description="Speak into your mic. Your voice will be transcribed using OpenAI Whisper.",
    ).launch()


if __name__ == "__main__":
    launch_interface()
