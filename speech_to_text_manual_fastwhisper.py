from datetime import timedelta
import os
import sys
from tqdm import tqdm
from faster_whisper import WhisperModel

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Model : python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='bofenghuang/whisper-large-v3-french', local_dir='./models/whisper-large-v3-french', allow_patterns='ctranslate2/*')"
def transcribe_audio(path):
    model = WhisperModel("./models/whisper-large-v3-french/ctranslate2", device="cpu", cpu_threads=4, compute_type="int8") 
    print("Whisper model loaded.")

    segments, info = model.transcribe(audio=path, beam_size=5, language="fr")

    total_duration = round(info.duration, 2)  # Same precision as the Whisper timestamps.
    timestamps = 0.0  # to get the current segments

    with tqdm(total=total_duration, unit=" audio seconds") as pbar:
        for segment in segments:
            pbar.update(segment.end - timestamps)
            timestamps = segment.end
            if timestamps < info.duration: # silence at the end of the audio
                pbar.update(info.duration - timestamps)

            startTime = str(0)+str(timedelta(seconds=int(segment.start)))+',000'
            endTime = str(0)+str(timedelta(seconds=int(segment.end)))+',000'
            text = segment.text
            segmentId = segment.id
            segment = f"{segmentId}\n{startTime} --> {endTime}\n{text[1:] if text[0] == ' ' else text}\n\n"

            srtFilename = path+".srt"
            with open(srtFilename, 'a', encoding='utf-8') as srtFile:
                srtFile.write(segment)

    return srtFilename


# Get the audio file path from user input (command line)
if len(sys.argv) < 2:
    print("Usage: python script.py <path_to_audio_file>")
    sys.exit(1)

audio_file_path = sys.argv[1]  # Get the audio file path from the command line argument


print(f"Transcription saved to {transcribe_audio(audio_file_path)}")