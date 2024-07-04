import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

def cut_audio(audio_path, length=25):
    """
    This function takes an audio file and cuts it into len second clips.
    """
    # load the audio file
    y, sr = librosa.load(audio_path, sr=None)
    # calculate the number of len second clips
    n = int(len(y) / (length * sr))
    # cut the audio into len second clips
    for i in range(n):
        y_clip = y[i * length * sr: (i + 1) * length * sr]
        # save the clip
        sf.write(f"{audio_path[:-4]}_clip{i}.wav", y_clip, sr)
    return
    
    
if __name__ == "__main__":
    # run the function on all the audio files (mp3 or wav) in the folder
    import os
    for file in tqdm(os.listdir()):
        if file.endswith(".mp3"):
            cut_audio(file, length=25)

