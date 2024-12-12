from datetime import timedelta
import os
import sys
import whisper

# Model DL : python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='bofenghuang/whisper-large-v3-french', filename='original_model.pt', local_dir='./models/whisper-large-v3-french')"
def transcribe_audio(path):
    model = whisper.load_model("./models/whisper-large-v3-french/original_model.pt")
    print("Whisper model loaded.")
    transcribe = model.transcribe(audio=path, language="fr")
    segments = transcribe['segments']

    for segment in segments:
        startTime = str(0)+str(timedelta(seconds=int(segment['start'])))+',000'
        endTime = str(0)+str(timedelta(seconds=int(segment['end'])))+',000'
        text = segment['text']
        segmentId = segment['id']+1
        segment = f"{segmentId}\n{startTime} --> {endTime}\n{text[1:] if text[0] == ' ' else text}\n\n"

        srtFilename = path + ".srt"
        with open(srtFilename, 'a', encoding='utf-8') as srtFile:
            srtFile.write(segment)

    return srtFilename


# Get the audio file path from user input (command line)
if len(sys.argv) < 2:
    print("Usage: python script.py <path_to_audio_file>")
    sys.exit(1)

audio_file_path = sys.argv[1]  # Get the audio file path from the command line argument


print(f"Transcription saved to {transcribe_audio(audio_file_path)}")