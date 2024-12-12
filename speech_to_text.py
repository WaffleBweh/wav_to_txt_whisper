import torch
import torchaudio
import numpy as np
import sys
from datetime import timedelta
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load model
model_name_or_path = "bofenghuang/whisper-large-v3-french"
processor = AutoProcessor.from_pretrained(model_name_or_path)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_name_or_path,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
)
model.to(device)

# Save model and processor locally
local_model_path = "./whisper-large-v3-french"
model.save_pretrained(local_model_path)
processor.save_pretrained(local_model_path)

# Init pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    feature_extractor=processor.feature_extractor,
    tokenizer=processor.tokenizer,
    torch_dtype=torch_dtype,
    return_timestamps=True,
    device=device,
    generate_kwargs={"max_new_tokens": 128},
)

# Load local audio file (WAV format for example)
def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)  # Load audio
    # Resample to 16kHz (if not already)
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)
    
    # Ensure the audio is mono (single channel)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # Convert stereo to mono
    
    return waveform, 16000  # Returning waveform and the sampling rate

# Function to split audio into chunks of max `chunk_length_s` seconds
def split_audio(waveform, chunk_length_s, sample_rate):
    # Convert chunk length from seconds to number of frames
    chunk_size = chunk_length_s * sample_rate
    num_chunks = (waveform.size(1) + chunk_size - 1) // chunk_size  # Ceiling division
    
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, waveform.size(1))
        chunks.append(waveform[:, start:end])
    
    return chunks

# Get the audio file path from user input (command line)
if len(sys.argv) < 2:
    print("Usage: python script.py <path_to_audio_file>")
    sys.exit(1)

audio_file_path = sys.argv[1]  # Get the audio file path from the command line argument

# Load the audio
waveform, sample_rate = load_audio(audio_file_path)

# Split the audio into 30-second chunks
chunk_length_s = 15  # 30 seconds
audio_chunks = split_audio(waveform, chunk_length_s, sample_rate)

# Process each chunk and combine the results with a progress bar
final_transcription = []
idx = 0
for chunk in tqdm(audio_chunks, desc="Processing audio chunks", unit="chunk"):
    # Convert waveform to numpy array and prepare dictionary input
    audio_input = {
        "raw": chunk.squeeze().numpy(),  # Convert tensor to numpy array
        "sampling_rate": sample_rate
    }

    # Run pipeline for the chunk
    results = pipe(audio_input)
    final_transcription.append(results["text"])

# Combine all chunk transcriptions into one
combined_transcription = " ".join(final_transcription)

# Output transcription to a text file
output_file_path = audio_file_path+".txt"
with open(output_file_path, "w", encoding="utf-8") as f:
    f.write(combined_transcription)

print(f"Transcription saved to {output_file_path}")