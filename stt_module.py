import os
import nemo.collections.asr as nemo_asr
from huggingface_hub import login 
from dotenv import load_dotenv

# === ENV SETUP ===
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")

# Authenticate with HuggingFace (handled via token in .env file)
login(token=HUGGINGFACE_TOKEN, add_to_git_credential=True)

# === LOAD STT MODEL ===
# Using a pretrained Kinyarwanda model from mbazaNLP
model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
    model_name="mbazaNLP/Kinyarwanda_nemo_stt_conformer_model"
)

def transcribe_audio(audio_input):
    """
    Transcribes audio files to text.
    Accepts either a directory path or a list of file paths.
    """

    supported_ext = ('.wav', '.mp3')  # Audio types we support for input

    # First dev’s logic: clean folder input
    if isinstance(audio_input, str):
        if not os.path.isdir(audio_input):
            raise ValueError(f"[ERROR] '{audio_input}' is not a valid directory.")
        
        audio_files = [
            os.path.join(audio_input, f)
            for f in sorted(os.listdir(audio_input))
            if f.lower().endswith(supported_ext)
        ]

    # Second dev’s style: direct list input with sanity check
    elif isinstance(audio_input, list):
        audio_files = [
            f for f in audio_input 
            if isinstance(f, str) and f.lower().endswith(supported_ext)
        ]

    else:
        raise ValueError("Input must be a path to a directory or a list of audio file paths.")

    if not audio_files:
        raise FileNotFoundError("⚠️ No supported audio files found for transcription.")

    # Go ahead and run inference
    return model.transcribe(audio_files)
