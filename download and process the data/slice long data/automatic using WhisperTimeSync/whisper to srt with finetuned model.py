import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import librosa
import srt
from datetime import timedelta

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the fine-tuned model and processor
model = WhisperForConditionalGeneration.from_pretrained("ivrit-ai/whisper-v2-pd1-e1").to(device)
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")



# Load and preprocess audio
def load_audio(file_path, sample_rate=16000):
    audio, sr = librosa.load(file_path, sr=sample_rate)
    return audio

audio_path = "/app/audio_files_other/audio_001.wav"
audio = load_audio(audio_path)


# Generate timestamps
from transformers import pipeline

asr = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer, 
               feature_extractor=processor.feature_extractor, device=device)

result = asr(audio, return_timestamps="word")

print(result)
# Create SRT file
def create_srt(result, max_line_length=20):
    subtitles = []
    current_line = ""
    start_time = None
    
    for i, segment in enumerate(result["chunks"]):
        word = segment["text"].strip()
        if start_time is None:
            start_time = segment["timestamp"][0]
        
        if len(current_line + word) > max_line_length:
            end_time = segment["timestamp"][0]
            subtitles.append(srt.Subtitle(index=len(subtitles)+1,
                                          start=timedelta(seconds=start_time),
                                          end=timedelta(seconds=end_time),
                                          content=current_line.strip()))
            current_line = word + " "
            start_time = segment["timestamp"][0]
        else:
            current_line += word + " "
    
    # Add the last subtitle
    if current_line:
        subtitles.append(srt.Subtitle(index=len(subtitles)+1,
                                      start=timedelta(seconds=start_time),
                                      end=timedelta(seconds=result["chunks"][-1]["timestamp"][1]),
                                      content=current_line.strip()))
    
    return srt.compose(subtitles)

srt_content = create_srt(result)

# Write to file
with open("output.srt", "w", encoding="utf-8") as f:
    f.write(srt_content)