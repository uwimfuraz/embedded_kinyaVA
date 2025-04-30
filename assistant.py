# assistant.py

# --- Module Imports ---
# Importing speech-to-text, text-to-speech, and natural language processing functionalities
from stt_module import transcribe_audio
from tts_module import synthesize_tts
from nlp_module import get_response


def run_assistant(audio_folder="./audio_samples"):
    """
    The main function that runs the virtual assistant pipeline.
    
    Steps:
    1. Transcribes all audio files from the specified folder.
    2. Processes each transcription to generate a text-based response using NLP.
    3. Synthesizes the generated response into a speech/audio format.
    
    Args:
        audio_folder (str): The path to the folder containing input audio samples.
                            Defaults to './audio_samples'.
    
    Returns:
        List[Tuple[str, bytes]]: A list of tuples, where each tuple contains:
            - The transcription of the audio (str)
            - The synthesized speech audio as WAV byte data (bytes)
    """
    # Transcribe all audio samples into a list of Hypothesis objects
    hypotheses = transcribe_audio(audio_folder)

    results = []

    for hypothesis in hypotheses:
        transcription = hypothesis.text  # Extract the transcribed text

        # Generate a natural language response based on the transcription
        response_text = get_response(transcription)

        # Synthesize the response into audio format
        response_wav = synthesize_tts(response_text)

        # Store the result as a (text, audio) tuple
        results.append((transcription, response_wav))

    return results
