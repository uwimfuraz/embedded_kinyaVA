import os
from assistant import run_assistant
from tts_module import setup_kinya_tts

# === SETUP SECTION ===
# Kicking off the TTS engine — do this once before running any synthesis
setup_kinya_tts()

# Define the folder path for audio input
audio_folder = os.path.join(os.path.dirname(__file__), "audio_samples")  # Using __file__ for flexibility

# === EXECUTION SECTION ===
# Run the assistant pipeline: STT ➡️ NLP ➡️ TTS
results = run_assistant(audio_folder)

# Output the results one by one
for idx, (transcription, response_wav) in enumerate(results):
    print()
    print(f"===== AUDIO SAMPLE #{idx + 1} =====")
    print("Transcript:", transcription)  # clean and straightforward
    print(f"Response Audio ➜ {response_wav}")  # arrow for visual flow

# Final log for batch processing complete
print("\n[✔] Finished processing all audio samples. Assistant is done!")
