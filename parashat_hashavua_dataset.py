import os
import librosa
import random
import numpy as np
import IPython.display as ipd
import pickle
import pandas as pd
from datasets import Dataset
from datasets import Audio
from transformers import WhisperProcessor
import mutagen.mp3
from tqdm import tqdm
import json
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, RoomSimulator
import srt
import re
from IPython.display import clear_output

#our libraries
from global_variables.training_vars import *
from global_variables.folders import *
from nikud_and_teamim import just_teamim,remove_nikud

path = "."
class parashat_hashavua_dataset:
        def __init__(self, new_data, processor, few_data=False, train = True ,validation=False, test=False, num_of_words_in_sample = 15, random = False, prob_for_num_of_parts=[], nusachim=["ashkenazi"], augment=False, load_srt_data = False):
                self.data = []
                self.few_data = few_data
                self.load_data(new_data, train, validation, test, nusachim=nusachim, load_srt_data=load_srt_data)
                if JUST_TEAMIM:
                        self.data['text'] = self.data['text'].apply(just_teamim)
                elif not NIKUD:
                        self.data['text'] = self.data['text'].apply(remove_nikud)
                self.data = self.data[self.data['text'] != ""] # remove empty texts (and their audio)
                self.num_of_words_in_sample = num_of_words_in_sample
                self.random = random
                self.start = 0
                self.is_eval_set = validation or test
                self.prob_for_num_of_parts = prob_for_num_of_parts if prob_for_num_of_parts else [1/self.num_of_words_in_sample for i in range(self.num_of_words_in_sample)]
                self.augment = augment
                self.processor = processor
                # prob_for_num_of_parts - the probability to take 1, 2, 3, etc. parts.
                # example of prob_for_num_of_parts: [0.1, 0.2, 0.3, 0.4] means that the probability to take 1 part is 0.1, 2 parts is 0.2, etc.

        def __getitem__(self, index):
                if self.is_eval_set:
                        audio, text_tokens, _ = self.get_sequence_(index*self.num_of_words_in_sample, num_of_words=self.num_of_words_in_sample)
                else:
                        if self.random:
                                # ensure that the sum of probabilities is 1
                                if np.sum(self.prob_for_num_of_parts) != 1:
                                        self.prob_for_num_of_parts = self.prob_for_num_of_parts / np.sum(self.prob_for_num_of_parts)
                                # get the number of parts
                                num_of_parts = np.random.choice(np.arange(1, len(self.prob_for_num_of_parts)+1), p=self.prob_for_num_of_parts)
                        # get the sequence
                                audio, text_tokens = self.get_random_words_sequence_audio_tokens(num_of_words=self.num_of_words_in_sample, num_of_parts=num_of_parts)
                        else:
                                audio, text_tokens, _ = self.get_sequence_(index, num_of_words=self.num_of_words_in_sample)
                if self.augment:
                        # augment the audio
                        audio = self.augment_audio(audio)

                # compute log-Mel input features from input audio array
                input_features = self.processor.feature_extractor(audio, sampling_rate=SR).input_features[0]
                # compute input length of audio sample in seconds
                input_length = len(audio) / SR
                # self.processor.tokenizer.decode(text_tokens)
                return {"input_features": input_features, "input_length": input_length, "labels": text_tokens}

        def __len__(self):
                if self.is_eval_set:
                        return int(len(self.data)/self.num_of_words_in_sample)
                else:
                        if self.random:
                                # high number because of the augmentation
                                return 100000
                        else:
                                # The length is the (number of word in the data)/(number of words in sequance)
                                return len(self.data)

        def get_sequence_audio_text(self, sequence):
                audio = np.concatenate(sequence['audio'].values)
                text = " ".join(sequence['text'])
                audio_len = len(audio) / 16000
                text_tokens = self.processor.tokenizer.encode(text)
                text_len = len(text_tokens)
                return sequence, audio, text, audio_len, text_tokens, text_len
        
        def load_data(self,new_data , train, validation, test, nusachim=["ashkenazi"], load_srt_data = False): 
                if load_srt_data:
                        self.load_data_srt_mp3(train, validation, test)
                elif new_data == "other":
                        nusachim = ["audio"]
                        if train:
                                self.load_data_new(nusachim, train=True, validation=False, test=False, other=True)
                        elif validation:
                                self.load_data_new(nusachim, train=False, validation=True, test=False, other=True)
                        elif test:
                                self.load_data_new(nusachim, train=False, validation=False, test=True, other=True)
                        else:
                                print("Invalid input. Please provide a valid input.")
                elif new_data:
                        if  (train==True and validation==False and test==False):
                                self.load_data_new(nusachim,train=True, validation=False, test=False)
                        elif (train==False and validation==True and test==False):
                                self.load_data_new(nusachim,train=False, validation=True, test=False)
                        elif (train==False and validation==False and test==True):
                                self.load_data_new(nusachim,train=False, validation=False, test=True)
                        else:
                                print(f"Invalid input. Please provide a valid input. train={train}, validation={validation}, test={test}")
                else:
                        self.load_data_old(validation)

        # methods for the new data
        def is_mp3_and_legal_length(self, filename, min_length=0.2, max_length=30):
                if filename.endswith(".wav"):
                        return True # we don't check the wav files for now
                
                try:
                        audio = mutagen.mp3.MP3(filename)
                        if audio.info.length < min_length or audio.info.length > max_length:
                                return False
                        else:
                                return True
                except mutagen.MutagenError:
                        return False

        
        def is_text_with_nikud(self, text):
                for char in text:
                        if char in "ְֱֲֳִֵֶַָֹֺֻּֽ֑֖֛֢֣֤֥֦֧֪֚֭֮֒֓֔֕֗֘֙֜֝֞֟֠֡֨֩֫֬֯־ֿ׀ׁׂ׃ׅׄ׆ׇ": # string of all the nikud characters ['֑', '֒', '֓', '֔', '֕', '֖', '֗', '֘', '֙', '֚', '֛', '֜', '֝', '֞', '֟', '֠', '֡', '֢', '֣', '֤', '֥', '֦', '֧', '֨', '֩', '֪', '֫', '֬', '֭', '֮', '֯', 'ְ', 'ֱ', 'ֲ', 'ֳ', 'ִ', 'ֵ', 'ֶ', 'ַ', 'ָ', 'ֹ', 'ֺ', 'ֻ', 'ּ', 'ֽ', '־', 'ֿ', '׀', 'ׁ', 'ׂ', '׃', 'ׄ', 'ׅ', '׆', 'ׇ']
                                return True
                return False

        def is_text_and_audio_pair_legal(self, text, filename):
                if not self.is_text_with_nikud(text):
                        print("the text doesn't have nikud: ", text)
                        return False
                if not self.is_mp3_and_legal_length(filename):
                        print("the audio is not mp3 or the length is not legal: ", filename)
                        return False
                return True

        def load_data_new(self, nusachim, train, validation, test, other=False):
                # Load dataset.json
                if other:
                        nusachim = ["audio"]
                        if train:
                                file_path = os.path.join(JSONS_FOLDER, 'train_data_other.json')
                        elif validation:
                                file_path = os.path.join(JSONS_FOLDER, 'validation_data_other.json')
                        elif test:
                                file_path = os.path.join(JSONS_FOLDER, 'test_data_other.json')  
                        else:
                                file_path = os.path.join(JSONS_FOLDER, '03_dataset.json') 
                else: 
                        if train:
                                file_path = os.path.join(JSONS_FOLDER, 'train_data.json')
                        elif validation:
                                file_path = os.path.join(JSONS_FOLDER, 'validation_data.json')
                        elif test:
                                file_path = os.path.join(JSONS_FOLDER, 'test_data.json')  
                        else:
                                file_path = os.path.join(JSONS_FOLDER, '03_dataset.json') 

                with open(file_path, 'r', encoding='utf-8') as f:
                        predataset = json.load(f)
                
                audios = []
                texts = []
                for nusach in nusachim:
                        file_path = "dataset_" + nusach + ".npy"
                        if os.path.exists(file_path) and False: # we don't want to use the saved data as 1 file right now
                                data = np.load(file_path, allow_pickle=True).item()
                                audios.extend(data['audio'])
                                texts.extend(data['text'])
                        else:
                                if self.few_data:
                                        predataset[nusach] = predataset[nusach][:500]
                                        predataset['text'] = predataset['text'][:500]
                                        
                                missing_files = []
                                print(predataset.keys())
                                for index, audio_file in enumerate(tqdm(predataset[nusach], desc=f"Loading {nusach} nusach ({nusachim.index(nusach)+1}/{len(nusachim)})")):
                                        audio_path = os.path.join(audio_file)
                                        if self.is_text_and_audio_pair_legal(predataset['text'][index], audio_path):
                                                audio, sr = librosa.load(audio_path, sr=SR)
                                                audios.append(audio)
                                                texts.append(predataset['text'][index])
                                        else:
                                                missing_files.append((audio_path, predataset['text'][index], index))
                                # Save the missing files
                                with open('missing_files' + nusach + '.json', 'w', encoding='utf-8') as f:
                                        json.dump(missing_files, f, ensure_ascii=False, indent=4)
                                print("Num of missing files in " + nusach + " nusach: ", len(missing_files))
                                # Save the data for the next time
                                data = {"audio": audios, "text": texts}
                # create the dataset
                self.data = {"audio": audios, "text": texts}
                self.data = pd.DataFrame(self.data)
                

        # methods for the old data
        def prepare_transcript_str_to_list_old_data(self, text:str) -> list:
                """
                this function get a string of words and return a list of the words
                """
                text = text.replace(" ׀ ", "׀").replace(" ׀ ", "׀").replace("׀", "׀ ").replace("־", "־ ").replace("[1]", "")
                text = re.sub(r'\s+|\n', ' ', text)  # replace multiple spaces or newline with a single space
                text_list = text.split(" ")
                if text_list[-1] == "":
                        text_list = text_list[:-1]
                return text_list

        def load_data_old(self, validation):
                success_count = 0
                fail_count = 0
                diff_list = []
                if validation:
                        transcript_folder = POKET_TORAH_FOLDER + '/text_val'
                else:
                        transcript_folder = POKET_TORAH_FOLDER + '/text'
                audio_folder = POKET_TORAH_FOLDER + '/audio'
                timing_folder = POKET_TORAH_FOLDER + '/time'
                audios = []
                text = []
                for filename in tqdm(os.listdir(transcript_folder)):
                        if filename.endswith(".txt"):
                                audio_path = os.path.join(audio_folder, filename.replace('.txt', '.mp3'))
                                transcript_path = os.path.join(transcript_folder, filename)
                                timing_path = os.path.join(timing_folder, filename)
                                audio, sr = librosa.load(audio_path, sr=16000)
                                with open(transcript_path, 'r', encoding='utf-8') as f:
                                        transcript = self.prepare_transcript_str_to_list_old_data(f.read())
                                with open(timing_path, 'r', encoding='utf-8') as f:
                                        timings = [float(time) for time in f.read().split(",")]
                                
                                if len(transcript) != len(timings):
                                        diff_list.append((len(transcript) - len(timings), filename))
                                        fail_count += 1
                                else:
                                        success_count += 1
                                        for i, (word, start_time) in enumerate(zip(transcript, timings)):
                                                if i == len(transcript) - 1:
                                                        end_time = len(audio) / sr
                                                else:
                                                        end_time = timings[i+1]
                                                word_audio = audio[int(start_time * sr):int(end_time * sr)]
                                                audios.append(word_audio)
                                                text.append(word)
                print("success_count: ", success_count)
                print("fail_count: ", fail_count)
                
                # diff_histogram:{1: 95, -1: 47, -6: 2, -2: 8, -5: 3, -3: 6, -7: 2, -4: 3, -8: 1, 3: 1, -9: 1, -26: 1}
                diff_histogram = {}
                for diff, filename in diff_list:
                        if diff in diff_histogram:
                                diff_histogram[diff] += 1
                        else:
                                diff_histogram[diff] = 1
                
                for key in sorted(diff_histogram.keys()):
                        print(f"{key}: {diff_histogram[key]}")
                
                print("diff_list: ", sorted(diff_list, key=lambda x: x[0]))
                self.failed_files = diff_list
                
                data_dict = {"audio": audios, "text": text}
                self.data = pd.DataFrame(data_dict)

        def check_failed_files_of_old_data(self):
                """
                check the failed files of the old data.
                listen to the audio and check the text.
                """
                for diff, filename in self.failed_files:
                        clear_output()
                        print("filename: ", filename)
                        audio_path = os.path.join(path + '/audio', filename.replace('.txt', '.mp3'))
                        transcript_path = os.path.join(path + '/text', filename)
                        timing_path = os.path.join(path + '/time', filename)
                        audio, sr = librosa.load(audio_path, sr=16000)
                        with open(transcript_path, 'r', encoding='utf-8') as f:
                                transcript = self.prepare_transcript_str_to_list_old_data(f.read())
                        with open(timing_path, 'r', encoding='utf-8') as f:
                                timings = [float(time) for time in f.read().split(",")]
                        print("num of words: ", len(transcript))
                        print("num of timings: ", len(timings))
                        
                        num = int(input("Enter the number of the word you want to listen to: "))
                        while num != -1:
                                if num == -2:
                                        # get the range that we want to search on
                                        start = int(input("Enter the start of the range: "))
                                        end = int(input("Enter the end of the range: "))
                                        num_steps = int(input("Enter the number of steps: "))
                                        for i in range(start, end, num_steps):
                                                self.check_failed_files_of_old_data_helper(timings, transcript, audio, sr, i)
                                else:
                                        self.check_failed_files_of_old_data_helper(timings, transcript, audio, sr, num)
                                num = int(input("Enter the number of the word you want to listen to: "))
                
                
        def check_failed_files_of_old_data_helper(self, timings, transcript, audio, sr, num):
                print("num: ", num)
                start_time = timings[num]
                end_time = timings[num+1] if num != len(timings) - 1 else len(audio) / sr
                print("start_time: ", start_time, " end_time: ", end_time)
                
                # print 3 words, 1 before the word, the word and 1 after the word (if exists)
                if num == 0:
                        print("word: ", transcript[num], " ", transcript[num+1])
                elif num == len(transcript) - 1:
                        print(transcript[num-1], " ", transcript[num])
                else:
                        print(transcript[num-1], " ", transcript[num], " ", transcript[num+1])
        
                # play the audio
                ipd.display(ipd.Audio(audio[int(start_time * sr):int(end_time * sr)], rate=SR))
        
        def load_data_srt_mp3(self, train, validation, test):
                import concurrent.futures
                import srt  # Make sure to import the srt module
                
                if train and not validation and not test:
                        folder = './train_data/'
                elif validation and not train and not test:
                        folder = './validation_data/'
                elif test and not train and not validation:
                        folder = './test_data/'
                else:
                        print("Invalid input. Please provide a valid input.")
                        return
                
                # Find all SRT files recursively
                srt_files = []
                
                def collect_srt_files(directory):
                        for item in os.listdir(directory):
                                item_path = os.path.join(directory, item)
                                if os.path.isdir(item_path):
                                        collect_srt_files(item_path)
                                elif item.endswith(".srt"):
                                        # Store tuple of (srt_path, directory, item)
                                        srt_files.append((item_path, directory, item))
                
                collect_srt_files(folder)
                print(f"Found {len(srt_files)} SRT files to process")
                
                # Worker function for SRT processing
                def process_srt_file(srt_info):
                        srt_path, directory, item = srt_info
                        base_name = item[:-4]  # Remove .srt extension
                        
                        results = {"audio": [], "text": []}
                        
                        # Look for corresponding audio file
                        audio_path = os.path.join(directory, base_name + '.mp3')
                        if not os.path.exists(audio_path):
                                audio_path = os.path.join(directory, base_name + '.wav')
                                if not os.path.exists(audio_path):
                                        # print(f"No audio file found for {item} in {directory}")
                                        return results
                        
                        # Load transcript
                        try:
                                with open(srt_path, 'r', encoding='utf-8') as f:
                                        transcript = list(srt.parse(f.read()))
                        except Exception as e:
                                print(f"Error parsing SRT file {srt_path}: {e}")
                                return results
                        
                        # Load audio
                        try:
                                audio, sr = librosa.load(audio_path, sr=16000)
                        except Exception as e:
                                print(f"Error loading audio file {audio_path}: {e}")
                                return results
                        
                        # Process each subtitle segment
                        for i, sub in enumerate(transcript):
                                start_time = sub.start.total_seconds()
                                end_time = sub.end.total_seconds()
                                
                                # Skip if times are invalid
                                if start_time >= end_time or start_time < 0 or end_time * sr >= len(audio):
                                        continue
                                
                                word_audio = audio[int(start_time * sr):int(end_time * sr)]
                                results["audio"].append(word_audio)
                                results["text"].append(sub.content)
                        
                        # print(f"Processed {item} - Added {len(results['audio'])} segments")
                        return results
                
                # Process files in parallel using ThreadPoolExecutor instead of ProcessPoolExecutor
                all_results = {"audio": [], "text": []}
                max_workers = min(16, len(srt_files)) 
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = [executor.submit(process_srt_file, srt_file) for srt_file in srt_files]
                        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), 
                                                        desc="Processing SRT files in parallel"):
                                result = future.result()
                                all_results["audio"].extend(result["audio"])
                                all_results["text"].extend(result["text"])
                
                self.data = pd.DataFrame(all_results)
                print(f"Total loaded: {len(self.data)} segments")

                        
        def get_data(self):
                return self.data

        def get_random_word(self):
                return random.choice(self.data)

        def get_sequence(self, start, end):
                return self.data[start:end]

        # the limit of whisper model
        # audio length of 30 seconds
        # text length of 448 tokens
        # I will take 20 words and check if the audio and text are in the limit
        def get_sequence_(self, start, num_of_words=20, random_cut_long=False):
                if start + num_of_words > len(self.data):
                        end = len(self.data)
                else:
                        end = start + num_of_words
                sequence = self.get_sequence(start, end)
                sequence, audio, text, audio_len, text_tokens, text_len = self.get_sequence_audio_text(sequence)
                if audio_len < 30 and text_len < 448:
                        return audio, text_tokens, end
                else: # cut the sequence
                        print("this sequence of ", num_of_words, " words is too long!")
                        print("sequence audio length: ", audio_len)
                        print("sequence text length(in tokens): ", text_len)
                        print("text: ", text)
                        # ipd.display(ipd.Audio(audio, rate=SR))

                        if random_cut_long:
                                # divide into 2 parts and randomaly take one of them
                                if random.randint(0, 1) == 0:
                                        start = start + int(num_of_words/2)

                        if num_of_words>=2:
                                return self.get_sequence_(start, num_of_words=int(num_of_words/2), random_cut_long=random_cut_long)
                        else:
                                return self.get_sequence_(end, num_of_words=num_of_words, random_cut_long=random_cut_long)


        def get_dataset_slice_to_sequences(self, num_of_words):
                audios = []
                labels = []
                start = 0
                while start < len(self.data):
                        audio, label_feature, start = self.get_sequence_(start,num_of_words)
                        audios.append(audio)
                        labels.append(label_feature)
                dataset = {"audios": audios, "labels": labels}
                dataset = pd.DataFrame(dataset)
                return dataset
        

        def get_random_sequence_(self, length=20):
                """
                get random sequence of "length" words
                """
                start = random.randint(0, len(self.data) - length)
                return self.get_sequence_(start)

        def get_random_sequence(self, length=20):
                """
                get random sequence of "length" words
                """
                start = random.randint(0, len(self.data) - length)
                return self.get_sequence(start, start+length)

        def get_random_words_sequence_audio_tokens(self, num_of_words, num_of_parts = None):
                """
                get sequence of random words (not logical sentences)
                createed from num_of_parts short sentences
                """
                if num_of_parts == None:
                        num_of_parts = num_of_words

                if num_of_parts > num_of_words:
                        print("num_of_parts can't be bigger than num_of_words")
                        print("so num_of_parts = num_of_words = ", num_of_words)
                        num_of_parts = num_of_words

                # num of words in each part
                num_of_words_in_parts = [num_of_words // num_of_parts + (1 if i < num_of_words % num_of_parts else 0) for i in range(num_of_parts)]

                sequence = {"audio": [], "text": []}
                for num_of_words_in_part in num_of_words_in_parts:
                        part = self.get_random_sequence(num_of_words_in_part)
                        sequence["audio"].extend(part["audio"])
                        sequence["text"].extend(part["text"])
                sequence = pd.DataFrame(sequence)
                sequence, audio, text, audio_len, text_tokens, text_len = self.get_sequence_audio_text(sequence)
                if audio_len < 30 and text_len < 448:
                        return audio, text_tokens
                else:
                        print("this sequence (of ", num_of_words, " words) is too long!")
                        print("sequence audio length: ", audio_len)
                        print("sequence text length(in tokens): ", text_len)
                        print("text: ", text)
                        # ipd.display(ipd.Audio(audio, rate=SR))
                        return self.get_random_words_sequence_audio_tokens(num_of_words, num_of_parts)


        def get_dataset_slice_to_sequences_random_words(self, num_of_words, num_of_sequences=None, times = 5):
                audios = []
                labels = []
                if num_of_sequences:
                        num_of_sequences = num_of_sequences
                else:
                        num_of_sequences = int(len(self.data)*times/num_of_words)
                for i in range(num_of_sequences):
                        audio, label_feature = self.get_random_words_sequence_audio_tokens(num_of_words)
                        audios.append(audio)
                        labels.append(label_feature)
                dataset = {"audios": audios, "labels": labels}
                dataset = pd.DataFrame(dataset)
                return dataset
        
        
        # methods for checking the data
        def get_longest_audio_index(self):
                """
                returns the index of longest audio in the dataset
                """
                index = np.argmax([len(audio) for audio in self.data['audio']])
                return index
        
        def get_longest_text_index(self):
                """
                returns the index of longest text in the dataset
                """
                index = np.argmax([len(text) for text in self.data['text']])
                return index
        
        def get_shortest_audio_index(self):
                """
                returns the index of shortest audio in the dataset
                """
                index = np.argmin([len(audio) for audio in self.data['audio']])
                return index
        
        def get_shortest_text_index(self):
                """
                returns the index of shortest text in the dataset
                """
                index = np.argmin([len(text) for text in self.data['text']])
                return index
        
        def check_the_data(self):
                """
                find the longest and shortest audio and text in the dataset
                and print and play them
                """
                index = self.get_longest_audio_index()
                print("longest audio index: ", index)
                print("longest audio text: ", self.data['text'][index])
                ipd.display(ipd.Audio(self.data['audio'][index], rate=SR))

                index = self.get_shortest_audio_index()
                print("shortest audio index: ", index)
                print("shortest audio text: ", self.data['text'][index])
                ipd.display(ipd.Audio(self.data['audio'][index], rate=SR))

                index = self.get_longest_text_index()
                print("longest text index: ", index)
                print("longest text: ", self.data['text'][index])
                ipd.display(ipd.Audio(self.data['audio'][index], rate=SR))

                index = self.get_shortest_text_index()
                print("shortest text index: ", index)
                print("shortest text: ", self.data['text'][index])
                ipd.display(ipd.Audio(self.data['audio'][index], rate=SR))


        def remove_word_by_index(self, index):
                """
                delete word from the dataset by index
                """
                if index < 0 or index >= len(self.data):
                        print("Invalid index. Please provide a valid index.")
                        return
                
                self.data.drop(index, inplace=True)
                self.data.reset_index(drop=True, inplace=True)

        def print_and_play_word_by_index(self,index):
                print(self.data['text'][index])
                ipd.display(ipd.Audio(self.data['audio'][index], rate=SR))

        def augment_audio(self, audio):
                """
                augment the audio using audiomentations.
                each augmentation is done with random values.
                """
                augment = Compose([
                    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.05, p=0.5),
                    TimeStretch(min_rate=0.8, max_rate=4, p=1),
                    PitchShift(min_semitones=-8, max_semitones=8, p=1),
                    Shift(min_shift=0, max_shift=2, shift_unit="seconds", rollover=False),
                    RoomSimulator(),
                ])
                
                return audio


class CombinedDataset:
    """
    This class combines multiple datasets and provides data from them sequentially.
    """
    def __init__(self, *datasets):
        self.datasets = datasets
        self.dataset_cycle = cycle(self.datasets)
        self.current_dataset = next(self.dataset_cycle)
        self.current_index = 0

    def __getitem__(self, index):
        # Check if we need to switch to the next dataset
        if self.current_index >= len(self.current_dataset):
            self.current_dataset = next(self.dataset_cycle)
            self.current_index = 0

        # Get data from the current dataset
        data = self.current_dataset[self.current_index]
        self.current_index += 1
        return data

    def __len__(self):
        # Return the total length of all datasets
        return sum(len(dataset) for dataset in self.datasets)