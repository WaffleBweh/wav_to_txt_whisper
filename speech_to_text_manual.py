from datetime import timedelta
import os
import sys
import whisper

# Model DL : python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='bofenghuang/whisper-large-v3-french', filename='original_model.pt', local_dir='./models/whisper-large-v3-french')"
def transcribe_audio(path):
    model = whisper.load_model("./models/whisper-large-v3-french/original_model.pt")
    print("Whisper model loaded.")
    transcribe = model.transcribe(audio=path, language="fr",  temperature=0)
    segments = transcribe['segments']

    for segment in segments:
        text = segment['text']

        srtFilename = path + ".txt"
        with open(srtFilename, 'a', encoding='utf-8') as srtFile:
            srtFile.write(text)

    return srtFilename


# Get the audio file path from user input (command line)
if len(sys.argv) < 2:
    print("Usage: python script.py <path_to_audio_file>")
    sys.exit(1)

audio_file_path = sys.argv[1]  # Get the audio file path from the command line argument


print(f"Transcription saved to {transcribe_audio(audio_file_path)}")