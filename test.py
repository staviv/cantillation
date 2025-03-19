# %%
import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available.")
else:
    print("CUDA is not available. Check if your GPU drivers are properly installed.")


# %%
import os
os.chdir('/app/')


import librosa
import random
import numpy as np
import IPython.display as ipd
import pickle
import pandas as pd
from datasets import Dataset
from datasets import Audio
from torch.utils.data import ConcatDataset
from transformers import WhisperProcessor
import mutagen.mp3
from tqdm import tqdm
import json
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, RoomSimulator
import srt
import re
from IPython.display import clear_output
from transformers import WhisperForConditionalGeneration
from huggingface_hub import login



#our libraries
from global_variables.training_vars import *
from global_variables.folders import *
from parashat_hashavua_dataset import *
from nikud_and_teamim import just_teamim,remove_nikud


# %%
# # load the token from txt file
# with open("./tokens/HF_token.txt", "r") as f:
#     HF_TOKEN = f.read().strip() # strip() removes the trailing "\n" if it exists
# login(token=HF_TOKEN)


# %%
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

# %%
import concurrent.futures
import evaluate
import time
import cantilLocations_evaluation


# # possible metrics : "wer", "cer", "bleu", "rouge", "sacrebleu", "sari":
# # 1. `wer`: Word Error Rate.
# # 2. `cer`: Character Error Rate.
# # 3. `bleu`: Bilingual Evaluation Understudy.
# # 4. `rouge`: Recall-Oriented Understudy for Gisting Evaluation.
# # 5. `sacrebleu`: A standardized BLEU score implementation for more consistent machine translation evaluation.
# # 6. `sari`: System Agnostic Refinement Index. 

WER_CALCULATOR = evaluate.load("wer")
def compute_metrics(pred):
    eval_list = cantilLocations_evaluation.calculate_precision_recall_f1_for_string_list_with_method_list
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    # method to calculate the metrics(method can be "Exact", "Letter_Shift", "Word_Level", "Word_Shift")
    methods = ["Exact", "Letter_Shift", "Word_Level", "Word_Shift"]

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    
    start_time = time.time()
    
    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    decode_time = time.time() - start_time
    
    # evaluate the metrics
    results = eval_list(pred_str, label_str, methods)
    
    
    
    # compute the average of each metric
    avg = {}
    for i in range(4):
        avg["avg_precision_" + methods[i]] = np.mean(results[i][0])
        avg["avg_recall_" + methods[i]] = np.mean(results[i][1])
        avg["avg_f1_" + methods[i]] = np.mean(results[i][2])
    
    precision_list_exact = results[methods.index("Exact")][0]
    recall_list_exact = results[methods.index("Exact")][1]
    f1_list_exact = results[methods.index("Exact")][2]
    
    # compute the median
    precision_median_exact = np.median(precision_list_exact)
    recall_median_exact = np.median(recall_list_exact)
    f1_median_exact = np.median(f1_list_exact)
    
    
    # max and min:
    precision_max_exact = np.max(precision_list_exact)
    recall_max_exact = np.max(recall_list_exact)
    f1_max_exact = np.max(f1_list_exact)
    best_index = np.argmax(f1_list_exact)
    
    f1_min = [0, 0, 0, 0]
    recall_min = [0, 0, 0, 0]
    precision_min = [0, 0, 0, 0]
    
    for i in range(4):
        precision_min[i] = np.min(results[i][0])
        recall_min[i] = np.min(results[i][1])
        f1_min[i] = np.min(results[i][2])
    
    worst_index = [np.argmin(results[i][2]) for i in range(4)] 
    
    
    
    start_time = time.time()
    # WER
    wer = 100 * WER_CALCULATOR.compute(predictions=pred_str, references=label_str)
    
    wer_time = time.time() - start_time
    
    best_pred = pred_str[best_index]
    best_label = label_str[best_index]
    worst_pred = [pred_str[worst_index[i]] for i in range(4)]
    worst_label = [label_str[worst_index[i]] for i in range(4)]
    
    # # print
    # ## best:
    # print(f"best f1 for {methods[0]}: {f1_max_exact}\nbest pred: {best_pred}\nbest label: {best_label}\n")
    
    # ## worst (the worst for each method):
    # for i in range(4):
    #     print(f"worst f1 for {methods[i]}: {f1_min[i]}\nworst pred: {worst_pred[i]}\nworst label: {worst_label[i]}\n")
    
    
    
    # print("Time taken for each part:")
    # print(f"Decode calculation: {decode_time} seconds")
    # print(f"WER calculation: {wer_time} seconds")
    
    # matric_dict = {"wer": wer, "precision": precision_avg, "recall": recall_avg, "f1": f1_avg, "precision_median": precision_median, "recall_median": recall_median, "f1_median": f1_median, "precision_max": precision_max, "recall_max": recall_max, "f1_max": f1_max, "precision_min": precision_min, "recall_min": recall_min, "f1_min": f1_min}
    
    # create the matric_dict with the metrics
    matric_dict = {"wer": wer}
    for i in range(4):
        matric_dict["avg_precision_" + methods[i]] = avg["avg_precision_" + methods[i]]
        matric_dict["avg_recall_" + methods[i]] = avg["avg_recall_" + methods[i]]
        matric_dict["avg_f1_" + methods[i]] = avg["avg_f1_" + methods[i]]
    matric_dict["precision_median_exact"] = precision_median_exact
    matric_dict["recall_median_exact"] = recall_median_exact
    matric_dict["f1_median_exact"] = f1_median_exact
    matric_dict["precision_max_exact"] = precision_max_exact
    matric_dict["recall_max_exact"] = recall_max_exact
    matric_dict["f1_max_exact"] = f1_max_exact
    for i in range(4):
        matric_dict["precision_min_" + methods[i]] = precision_min[i]
        matric_dict["recall_min_" + methods[i]] = recall_min[i]
        matric_dict["f1_min_" + methods[i]] = f1_min[i]
    # print(matric_dict)
    return matric_dict

# %% [markdown]
# # Test the model

# %%
JUST_ADD_NEW_RESULTS = True

# %%
if JUST_ADD_NEW_RESULTS:
    models = ["cantillation/Teamim-medium_Random_WeightDecay-0.005_Augmented_New-Data_date-11-03-2025"]
else:
    models = ["cantillation/Teamim-base_WeightDecay-0.05_Augmented_Combined-Data_date-11-07-2024_05-09", "cantillation/Teamim-large-v2-pd1-e1_WeightDecay-0.05_Augmented_Combined-Data_date-14-07-2024_18-24", "cantillation/Teamim-medium_WeightDecay-0.05_Augmented_Combined-Data_date-13-07-2024_18-40", "cantillation/Teamim-small_WeightDecay-0.05_Augmented_Combined-Data_date-11-07-2024_12-42", "cantillation/Teamim-small_WeightDecay-0.05_Augmented_New-Data_date-19-07-2024_15-41", "cantillation/Teamim-small_WeightDecay-0.05_Combined-Data_date-17-07-2024_10-08", "cantillation/Teamim-tiny_WeightDecay-0.05_Augmented_Combined-Data_date-10-07-2024_14-33", "cantillation/Teamim-tiny_WeightDecay-0.05_Combined-Data_date-17-07-2024_10-10", "cantillation/Teamim-small_Random_WeightDecay-0.05_Augmented_Old-Data_date-21-07-2024_14-33","cantillation/Teamim-small_WeightDecay-0.05_Augmented_Old-Data_date-21-07-2024_14-34_WithNikud","cantillation/Teamim-small_WeightDecay-0.05_Augmented_Old-Data_date-23-07-2024", "cantillation/Teamim-small_WeightDecay-0.05_Augmented_New-Data_nusach-yerushalmi_date-24-07-2024", "cantillation/Teamim-large-v2_WeightDecay-0.05_Augmented_Combined-Data_date-25-07-2024", "cantillation/Teamim-small_Random_WeightDecay-0.05_Augmented_New-Data_date-02-08-2024"]


# %%
# Our dataset class needs the processor to check if the length of the audio or the text is too long
# We use the processor that we updated with teamim
processor = WhisperProcessor.from_pretrained(models[0])
# Load the test data
test_data = parashat_hashavua_dataset(new_data="other", few_data=FASTTEST, train=False, validation=False, test=True, random=False, num_of_words_in_sample=1, augment=False, processor=processor, load_srt_data=False)
# test_data = parashat_hashavua_dataset(new_data="other", few_data=FASTTEST, train=False, validation=False, test=True, random=False, num_of_words_in_sample=8, augment=False, processor=processor, load_srt_data=True)


# %%
# lens = test_data.data["audio"].apply(lambda x: len(x))/16000
# # sum each 4 samples to get the total length of the audio
# lens = lens.rolling(5).sum().dropna()
# # plot the length of the audio histogram
# lens.hist(bins=100)


# %%
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer

def test_model(model_name):
    torch.cuda.empty_cache()
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    processor = WhisperProcessor.from_pretrained(model_name)
    training_args = Seq2SeqTrainingArguments(
        output_dir= "evalutions_on_other_data/test_" + model_name.split("/")[-1],
        predict_with_generate=True,
    )
    
    # create the data collator (using the processor)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        eval_dataset=test_data,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )
    
    
    return trainer.evaluate()
    

def test_models(models_names):
    results = []
    for model_name in models_names:
        with torch.no_grad():
            result = {"model": model_name}
            result.update(test_model(model_name))
            results.append(result)
            
    # return the results
    return results



# %%
# test the models
results = test_models(models)

# %%
df = pd.DataFrame(results)



# %%
# Each column label with "eval_" replaced with ""
df.columns = df.columns.str.replace("eval_", "")


# %%
def reorder_columns(df, priority_columns):
    """
    Reorders the DataFrame columns, placing the priority columns first.

    Parameters:
    - df (pd.DataFrame): The DataFrame to reorder.
    - priority_columns (list of str): The columns to place at the beginning.

    Returns:
    - pd.DataFrame: The DataFrame with reordered columns.
    """
    # Ensure all priority columns are in the DataFrame's columns
    priority_columns = [col for col in priority_columns if col in df.columns]
    # Reorder columns, placing priority columns first
    reordered_columns = priority_columns + [col for col in df.columns if col not in priority_columns]
    return df[reordered_columns]
df = reorder_columns(df, ["avg_f1_Exact", "avg_recall_Exact", "avg_precision_Exact","wer"])
df

# %%
import re
import pandas as pd

def extract_model_info(model_str):
  """
  Extracts information from a model string.

  Args:
    model_str: A model string from the 'model' column.

  Returns:
    A dictionary containing the extracted model information.
  """

  model_info = {}
  # model_info['prefix'] = model_str.split('/')[0]

  parts = model_str.split('/')[1].split('_')
  for part in parts:
    if 'Teamim-' in part:
      model_info['model'] = part.replace('Teamim-', '')
    elif 'nusach-' in part:
      model_info['nusach'] = part.replace('nusach-', '')
    elif 'WeightDecay-' in part:
      model_info['L2_reg'] = part.replace('WeightDecay-', '')
    elif 'Augmented' in part:
      model_info['augmented'] = True
    elif 'Combined-Data' in part:
      model_info['data_type'] = 'Combined'
    elif 'New-Data' in part:
      model_info['data_type'] = 'Ben13'
    elif 'Old-Data' in part:
      model_info['data_type'] = 'PocketTorah'
    elif 'WithNikud' in part:
      model_info['with_nikud'] = True
    elif 'Random' in part:
      model_info['random'] = True
    # elif re.match(r'date-\d{2}-\d{2}-\d{4}', part):
      # model_info['date'] = part.replace('date-', '')
    # elif re.match(r'\d{2}-\d{2}', part): # must be after the date because the date also matches this pattern
      # model_info['time'] = part
  if 'augmented' not in model_info:
    model_info['augmented'] = False
  if 'with_nikud' not in model_info:
    model_info['with_nikud'] = False
  if 'random' not in model_info:
    model_info['random'] = False
  
  
  
  return model_info


# %%
df

# %%


# %%
# # Load the results
# df = pd.read_csv("/app/evalutions_on_other_data/test_results.csv")

# Extract model information
model_info = df['model'].apply(extract_model_info).apply(pd.Series)

# drop the 'model' column
df.drop('model', axis=1, inplace=True)




# add each extracted column to the dataframe
df = pd.concat([model_info, df], axis=1)

df

# %%


# %%
if JUST_ADD_NEW_RESULTS:
    # load old results add the new results
    old_df = pd.read_csv("/app/evalutions_on_other_data/test_results.csv")

    # add the new results
    df = pd.concat([old_df, df], ignore_index=True) 
    
df

# %%
df

# %%

# Defined order list
order = ['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3']

# Function to find the closest category from the list, checking from end to start
def find_closest_category(value, categories):
    for category in reversed(categories):
        if category in value:
            return category
    return value

# Create a new column with the closest category values
df['model_closest'] = df['model'].apply(lambda x: find_closest_category(x, order))

# Create a categorical type with the defined order on the new column
df['model_closest'] = pd.Categorical(df['model_closest'], categories=order, ordered=True)

# Sort the DataFrame by the new column
df = df.sort_values('model_closest')
df = df.drop(columns=['model_closest'])  # Remove the temporary column


# %%
df

# %%

# save the results
df.to_csv("/app/evalutions_on_other_data/test_results.csv", index=False)


