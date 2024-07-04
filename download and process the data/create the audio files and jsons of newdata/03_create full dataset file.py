import os
import json
from tqdm import tqdm
import librosa
import mutagen
import mutagen.mp3
import numpy as np

SR = 16000

def is_mp3(filename):
    try:
        mutagen.mp3.MP3(filename)
        return True
    except mutagen.MutagenError:
        return False


# # Load dataset.json
# with open('03_dataset.json', 'r', encoding='utf-8') as f:
#     dataset = json.load(f)

# audios = []
# texts = []
# missing_files = []
# for index, audio_file in enumerate(tqdm(dataset['maroko'])):
#     audio_path = os.path.join(audio_file)
#     if is_mp3(audio_path):
#         audio, sr = librosa.load(audio_path, sr=SR)
#         audios.append(audio)
#         texts.append(dataset['text'][index])
#     else:
#         missing_files.append((audio_path, dataset['text'][index], index))

# audios = np.asarray(audios, dtype=object)
# texts = np.asarray(texts, dtype=object)

# print(f"audios: {len(audios)}")
# print(f"texts: {len(texts)}")
# print(f"missing_files: {len(missing_files)}")
# print(f"first audio: {audios[0]}")
# print(f"first text: {texts[0]}")


# predataset = {
#     'text': texts,
#     'audio': audios
# }

# ## Save the dataset as a compressed NumPy file
# np.savez_compressed('predataset_maroko_wav.npz', **dataset)

# # Save the missing files
# with open('missing_files.json', 'w', encoding='utf-8') as f:
#     json.dump(missing_files, f, ensure_ascii=False, indent=4)



# load the predataset (the one with the audios and the texts)
predataset = np.load('predataset_maroko_wav.npz', allow_pickle=True)


# create a dataset with one mp3 audio file and a text files with the times and texts
texts = []
stats_times = [0]

for index, audio in enumerate(tqdm(predataset['audio'])):
    duration = audio.shape[0] / SR
    stats_times.append(stats_times[index] + duration)
    texts.append(predataset['text'][index])


# save the times as a text file with "," between each time
with open('times.txt', 'w', encoding='utf-8') as f:
    f.write(",".join([str(time) for time in stats_times]))

# save the texts as a text file with "\n" between each text
with open('texts.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join(texts))



# concatenate all the audios into one vector
audio = np.concatenate(predataset['audio'], axis=0)

dataset = {
    'text': texts,
    'audio': audio,
    'times': stats_times
}

# save the dataset as a compressed NumPy file
np.savez_compressed('dataset_maroko_wav.npz', **dataset)

# save the audio as a npz file
np.savez_compressed('audio.npz', audio=audio)


# load the dataset and the time it takes to load it
import time
start_time = time.time()
dataset = np.load('dataset_maroko_wav.npz', allow_pickle=True)
end_time = time.time()
execution_time = end_time - start_time
print(f"Time taken to load the dataset: {execution_time} seconds")

from pydub import AudioSegment
# save the audio as a wav file
AudioSegment(audio.tobytes(), frame_rate=SR, sample_width=audio.dtype.itemsize, channels=1).export("combined.wav", format="wav")

# save the audio as a mp3 file
AudioSegment(audio.tobytes(), frame_rate=SR, sample_width=audio.dtype.itemsize, channels=1).export("combined.mp3", format="mp3")

