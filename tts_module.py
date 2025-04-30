# tts_module.py

import os
import uuid
import re
import torch
import torchaudio

from kinyatts.tts.commons import intersperse
from kinyatts.tts.utils import get_hparams_from_file, load_checkpoint
from kinyatts.tts.models import SynthesizerTrn
from kinyatts.tts.text import text_to_sequence
from kinyatts.tts.text.symbols import symbols

# Global inference engine tuple: (device, model, hparams, volume_boost)
inference_engine = (None, None, None, None)

def setup_kinya_tts():
    """
    Initializes the KinyaTTS text-to-speech engine.
    
    Loads the model configuration, model weights, sets up device (CPU/GPU),
    prepares audio transformation tools (volume boost), and initializes the inference model.

    Raises:
        ValueError: If the hyperparameter config file fails to load.
    """
    global inference_engine

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config_path = "./KinyaTTS/Inference/kinyatts/ms_ktjw_istft_vits2_base.json"
    model_path = "./TTS_MODEL_ms_ktjw_istft_vits2_base_1M.pt"

    # Load hyperparameters from config
    hps = get_hparams_from_file(config_path)
    if not hps:
        raise ValueError("Failed to load hyperparameters from the config file.")
    
    # Determine posterior encoder settings
    if getattr(hps.model, "use_mel_posterior_encoder", False):
        posterior_channels = 80
        hps.data.use_mel_posterior_encoder = True
    else:
        posterior_channels = hps.data.filter_length // 2 + 1
        hps.data.use_mel_posterior_encoder = False

    # Initialize Synthesizer model
    model = SynthesizerTrn(
        len(symbols),
        posterior_channels,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=getattr(hps.data, "n_speakers", 0),
        **hps.model
    ).to(device)

    _ = model.eval()  # Set to evaluation mode
    _ = load_checkpoint(model_path, model, None)  # Load pre-trained weights

    # Boost audio volume
    vol_boost = torchaudio.transforms.Vol(gain=6.0, gain_type="amplitude")

    # Store setup in the inference engine global
    inference_engine = (device, model, hps, vol_boost)
    print("✅ KinyaTTS initialized and ready!")


def text_to_input_tensor(text, hps):
    """
    Converts input text to a tensor sequence for the TTS model.

    Args:
        text (str): Input text to synthesize.
        hps (object): Loaded hyperparameters object.

    Returns:
        torch.LongTensor: Tensor representation of the input text.

    Raises:
        ValueError: If hyperparameters are not loaded.
    """
    if hps is None:
        raise ValueError("Hyperparameters (hps) are not loaded correctly. Ensure 'setup_kinya_tts' is called.")

    text_norm = text_to_sequence(text)

    # Optionally insert blank tokens for better alignment
    if getattr(hps.data, "add_blank", False):
        text_norm = intersperse(text_norm, 0)

    return torch.LongTensor(text_norm)


def speak(text, out_dir="outputs"):
    """
    Synthesizes speech audio from text input and saves it as a WAV file.

    Args:
        text (str): The input text to convert to speech.
        out_dir (str): Directory to save the resulting audio file.

    Returns:
        str: Path to the generated WAV file.

    Raises:
        ValueError: If the model hasn't been initialized.
    """
    global inference_engine
    device, model, hps, vol_boost = inference_engine

    if model is None:
        raise ValueError("Model has not been initialized. Call 'setup_kinya_tts()' first.")

    os.makedirs(out_dir, exist_ok=True)

    # Clean the input text
    text = re.sub(r"[(){}]", "", text)

    # Prepare input tensor
    input_tensor = text_to_input_tensor(text, hps)
    speed = 0.97  # Controls speech tempo (1.0 is default)

    with torch.no_grad():
        x = input_tensor.to(device).unsqueeze(0)
        x_len = torch.LongTensor([input_tensor.size(0)]).to(device)

        # Run inference to get waveform
        audio = model.infer(
            x, x_len,
            noise_scale=0.667,
            noise_scale_w=0.8,
            length_scale=1 / speed
        )[0][0, 0].cpu().float()

    # Boost the volume and reshape
    audio = vol_boost(audio.unsqueeze(0))

    # Generate output file name
    file_name = f"{out_dir}/response_{uuid.uuid4().hex[:8]}.wav"
    torchaudio.save(file_name, audio, hps.data.sampling_rate)

    return file_name


def synthesize_tts(text):
    """
    Public interface for generating TTS output from input text.

    Args:
        text (str): The text to synthesize.

    Returns:
        str: File path to the synthesized WAV audio.
    """
    return speak(text)
