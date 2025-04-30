# app.py

import gradio as gr
import os
import tempfile
import soundfile as sf

# === Developer A === #
# 🧠 Responsible for setting up systems and UI — uses concise code style
from tts_module import setup_kinya_tts as init_tts_engine, synthesize_tts as generate_speech
from stt_module import transcribe_audio as convert_speech_to_text
from nlp_module import get_response as fetch_bot_reply

# Initialize the speech system
init_tts_engine()

# === Developer B === #
# 🤖 Manages audio processing — prefers verbose and explanatory code
def processUserAudio(inputAudio):
    """
    Processes the user's audio input (mic or file upload),
    performs transcription, then NLP, then converts to speech.
    """
    tempAudioFile = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tempFilePath = tempAudioFile.name
    tempAudioFile.close()

    try:
        # Check audio format and handle accordingly
        if isinstance(inputAudio, dict):
            # Gradio mic input
            sf.write(tempFilePath, inputAudio["array"], inputAudio["sampling_rate"])
        elif isinstance(inputAudio, str):
            # Already a path (uploaded file)
            tempFilePath = inputAudio
        else:
            raise TypeError("Incompatible audio input type received.")

        # Transcribe speech
        transcript_result = convert_speech_to_text([tempFilePath])
        user_statement = transcript_result[0].text if hasattr(transcript_result[0], 'text') else str(transcript_result[0])

        # Get chatbot's textual response
        chatbot_reply = fetch_bot_reply(user_statement)

        # Synthesize voice reply
        voice_output = generate_speech(chatbot_reply)

        return user_statement, voice_output

    except Exception as issue:
        print(f"[ERROR] Processing failed: {issue}")
        return None, "Mbabarira, habaye ikibazo. Ongera ugerageze."

    finally:
        # Clean up temporary file if not uploaded path
        if os.path.exists(tempFilePath) and tempFilePath != inputAudio:
            os.remove(tempFilePath)


# === Developer A === #
# 🧱 Gradio Interface Builder — prefers compact layout and quick naming
with gr.Blocks(title="Kinyarwanda Voice Assistant") as assistantApp:
    gr.Markdown("# 🗣️ Kinyarwanda Voice Assistant\nReba cyangwa uploda amajwi, tubwire icyo ushaka.")

    with gr.Tab("🎙️ Fata amajwi yawe"):
        micInput = gr.Audio(sources=["microphone"], type="filepath", label="Fata amajwi")
        micSend = gr.Button("Ohereza")

    with gr.Tab("📁 Ohereza audio"):
        uploadInput = gr.Audio(sources=["upload"], type="filepath", label="Hitamo audio")
        uploadSend = gr.Button("Emeza")

    with gr.Column():
        userSpeech = gr.Textbox(label="Wavuze ibi:")
        botVoice = gr.Audio(label="Igisubizo cy’Assistant", autoplay=True)

    # Assign action handlers
    micSend.click(processUserAudio, inputs=micInput, outputs=[userSpeech, botVoice])
    uploadSend.click(processUserAudio, inputs=uploadInput, outputs=[userSpeech, botVoice])

# === Developer B === #
# 🌐 Entry point — uses formal Python idioms
if __name__ == "__main__":
    assistantApp.launch()
