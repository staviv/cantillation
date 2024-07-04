import os
import json
import requests
from tqdm import tqdm
import mutagen
import mutagen.mp3

def is_mp3(filename):
    try:
        mutagen.mp3.MP3(filename)
        return True
    except mutagen.MutagenError:
        return False


nusachim = ["maroko", "yerushalmi", "ashkenazi", "bavly"]


# Load the JSON file
with open('02_relevant_data.json', encoding='utf-8') as f:
    data = json.load(f)

# Create a directory to store the audio files
os.makedirs('audio_files', exist_ok=True)
os.makedirs('audio_files/audio_files_ben13', exist_ok=True)

import concurrent.futures
# Function to download and save a single audio file
def download_audio_file(url, filepath):
    response = requests.get(url)
    with open(filepath, 'wb') as f:
        f.write(response.content)

# Download and save the audio files in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
    futures = []
    total_files = len(data['audio url']) * len(nusachim)
    completed_files = 0
    with tqdm(total=total_files) as pbar:
        for i, relative_url in enumerate(data['audio url']):
            relative_url = relative_url.lower()
            for nusach in nusachim:
                url = "https://www.ben13.co.il/audio-files/" + nusach + "/" + relative_url
                filepath = os.path.join('audio_files/audio_files_ben13', nusach + "_" + relative_url.replace("/", "_"))
                futures.append(executor.submit(download_audio_file, url, filepath))
                completed_files += 1
        for future in concurrent.futures.as_completed(futures):
            pbar.update(1)

# Download and save the missing audio files (that failed to download in the previous step)
file_list = []
for i, relative_url in enumerate(data['audio url']):
    relative_url = relative_url.lower()
    for nusach in nusachim:
        filepath = os.path.join('audio_files', nusach + "_" + relative_url.replace("/", "_"))
        if not os.path.exists(filepath):
            file_link = "https://www.ben13.co.il/audio-files/" + nusach + "/" + relative_url
            file_list.append({
                'file_path': filepath,
                'file_link': file_link
            })

# Download and save the missing audio files in parallel with 10 workers
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = []
    with tqdm(total=len(file_list)) as pbar:
        for file_info in file_list:
            futures.append(executor.submit(download_audio_file, file_info['file_link'], file_info['file_path']))
        for future in concurrent.futures.as_completed(futures):
            pbar.update(1)

# check if all the audio files were downloaded
for i, relative_url in enumerate(data['audio url']):
    relative_url = relative_url.lower()
    for nusach in nusachim:
        filepath = os.path.join('audio_files/audio_files_ben13', nusach + "_" + relative_url.replace("/", "_"))
        if not os.path.exists(filepath):
            print("missing file:", filepath)

# Create a dataset linking the audio files to their corresponding texts
dataset = {
    'text': [],
    'maroko': [],
    'yerushalmi': [],
    'ashkenazi': [],
    'bavly': []
}
for i, relative_url in enumerate(data['audio url']):
    dataset['maroko'].append(os.path.join('audio_files/audio_files_ben13', "maroko_" + relative_url.lower().replace("/", "_")))
    dataset['yerushalmi'].append(os.path.join('audio_files/audio_files_ben13', "yerushalmi_" + relative_url.lower().replace("/", "_")))
    dataset['ashkenazi'].append(os.path.join('audio_files/audio_files_ben13', "ashkenazi_" + relative_url.lower().replace("/", "_")))
    dataset['bavly'].append(os.path.join('audio_files/audio_files_ben13', "bavly_" + relative_url.lower().replace("/", "_")))
    dataset['text'].append(data['text'][i])

# Save the dataset as a JSON file
with open('03_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)


print((len(data['text']))*len(nusachim))