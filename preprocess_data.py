import os
import torch
import pickle
import argparse
from transformers import WhisperProcessor
from huggingface_hub import login

from global_variables.training_vars import *
from global_variables.folders import *
from parashat_hashavua_dataset import parashat_hashavua_dataset
from nikud_and_teamim import TEAMIM

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess and cache datasets for Whisper cantillation model')
    parser.add_argument('--include_srt', action='store_true', help='Include SRT data in the preprocessing')
    return parser.parse_args()

def preprocess_and_cache_datasets(include_srt=True):
    print("Loading and preprocessing datasets. This will be done only once.")
    
    # Load token
    with open("./cantillation/tokens/HF_token.txt", "r") as f:
        HF_TOKEN = f.read().strip()
    login(token=HF_TOKEN)
    
    # Initialize processor
    processor = WhisperProcessor.from_pretrained(BASE_MODEL_NAME, language="hebrew", task="transcribe")
    
    # Add tokens if needed
    tokens_added = (len(processor.tokenizer.encode('֟')) == 6)
    if ADDTOKENS and not tokens_added:
        if JUST_TEAMIM:
            new_tokens = [BASE_CHAR + c for c in TEAMIM]
        elif NIKUD:
            new_tokens = ['֑', '֒', '֓', '֔', '֕', '֖', '֗', '֘', '֙', '֚', '֛', '֜', '֝', '֞', '֟', '֠', '֡', '֢', '֣', '֤', '֥', '֦', '֧', '֨', '֩', '֪', '֫', '֬', '֭', '֮', '֯', 'ְ', 'ֱ', 'ֲ', 'ֳ', 'ִ', 'ֵ', 'ֶ', 'ַ', 'ָ', 'ֹ', 'ֺ', 'ֻ', 'ּ', 'ֽ', '־', 'ֿ', '׀', 'ׁ', 'ׂ', '׃', 'ׄ', 'ׅ', '׆', 'ׇ']
        else:
            new_tokens = TEAMIM
        processor.tokenizer.add_tokens(new_tokens)
    
    # Create cache directory
    os.makedirs("./cantillation/cached_datasets", exist_ok=True)
    
    # Process and cache validation data
    print("Processing validation data...")
    val_data = parashat_hashavua_dataset(new_data=True, processor=processor, load_srt_data=True, 
                                        num_of_words_in_sample=1, test=True, train=False)
    with open("./cantillation/cached_datasets/val_data.pkl", "wb") as f:
        pickle.dump(val_data, f)
    print(f"Validation data processed and cached. Size: {len(val_data)} samples")
    
    # Process and cache training data
    print("Processing training data...")
    train_data_ben13 = parashat_hashavua_dataset(new_data=True, few_data=FASTTEST, train=True, validation=False, 
                                                random=RANDOM, num_of_words_in_sample=4, 
                                                nusachim=NUSACHIM, augment=AUGMENT, processor=processor)
    with open("./cantillation/cached_datasets/train_data_ben13.pkl", "wb") as f:
        pickle.dump(train_data_ben13, f)
    print(f"Ben13 training data processed and cached. Size: {len(train_data_ben13)} samples")
    
    if include_srt:
        print("Processing SRT training data...")
        train_data_srt = parashat_hashavua_dataset(new_data=True, processor=processor, load_srt_data=True, 
                                                num_of_words_in_sample=1)
        with open("./cantillation/cached_datasets/train_data_srt.pkl", "wb") as f:
            pickle.dump(train_data_srt, f)
        print(f"SRT training data processed and cached. Size: {len(train_data_srt)} samples")
    
    # Save processor
    processor.save_pretrained("./cantillation/cached_datasets/processor")
    print("Dataset preprocessing complete!")
    
if __name__ == "__main__":
    args = parse_args()
    preprocess_and_cache_datasets(include_srt=args.include_srt)
