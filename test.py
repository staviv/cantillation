# %%
import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available.")
else:
    print("CUDA is not available. Check if your GPU drivers are properly installed.")


# %%
import os
os.chdir('./')


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
JUST_ADD_NEW_RESULTS = False
FASTTEST = False

# %%
if JUST_ADD_NEW_RESULTS:
    models = []
else:
    models = [
    "cantillation/Teamim-IvritAI-large-v3-turbo-new_WeightDecay-0.005_Augmented_WithSRT_date-23-04-2025",
    "cantillation/Teamim-Large-v3-Turbo_WeightDecay-0.005_Augmented_WithSRT_date-18-04-2025",
    "cantillation/Teamim-small_WeightDecay-0.005_Augmented__date-15-04-2025",
    "cantillation/Teamim-IvritAI-large-v3-turbo_WeightDecay-0.005_Augmented_WithSRT_date-15-04-2025",
    "cantillation/Teamim-large-v3-turbo_WeightDecay-0.005_Augmented_WithSRT_date-15-04-2025",
    "cantillation/Teamim-small_WeightDecay-0.005_Augmented__WithSRT_date-11-04-2025",
    "cantillation/Teamim-tiny_WeightDecay-0.005_Augmented__WithSRT_date-11-04-2025",
    "cantillation/Teamim-tiny_WeightDecay-0.005_Augmented__date-10-04-2025",
    "cantillation/Teamim-medium_WeightDecay-0.005_Augmented__date-08-04-2025",
    "cantillation/Teamim-medium_WeightDecay-0.005_Augmented_WithSRT_date-05-04-2025",
    ]


# %%
# Get list of test subdirectories
def get_test_subdirs():
    test_dir = './test_data'
    if not os.path.exists(test_dir):
        return []
    
    subdirs = [d for d in os.listdir(test_dir) 
               if os.path.isdir(os.path.join(test_dir, d))]
    
    # If no subdirs found, return None to run on the whole test directory
    if not subdirs:
        return [None]
    
    return subdirs

# Test datasets to evaluate
test_datasets = get_test_subdirs()
print(f"Found test datasets: {test_datasets if None not in test_datasets else 'Default test directory'}")

# %%
# Our dataset class needs the processor to check if the length of the audio or the text is too long
# Use a generic processor for initial structure, but we'll recreate datasets with correct processors for each model
processor = WhisperProcessor.from_pretrained(models[0])

# Dictionary to hold test datasets information (paths, etc.) - not the actual loaded datasets
test_data_info = {}

# Get information about each test dataset
for test_subdir in test_datasets:
    subdir_name = test_subdir if test_subdir else "all"
    print(f"Loading test dataset info: {subdir_name}")
    
    # Just store the subdirectory information
    test_data_info[subdir_name] = test_subdir
    
    # Print some info about the dataset location
    test_dir = os.path.join('./test_data', test_subdir) if test_subdir else './test_data'
    if os.path.exists(test_dir):
        print(f"Test data directory: {test_dir}")
        print(f"Found {len([f for f in os.listdir(test_dir) if f.endswith('.srt')])} SRT files")

# %%
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer

def test_model(model_name, test_subdir, dataset_name):
    """
    Test a model on a specific dataset, ensuring the processor matches the model.
    
    Args:
        model_name: The name of the model to test
        test_subdir: The subdirectory in test_data containing the test dataset
        dataset_name: Name for the dataset in reports
    
    Returns:
        Evaluation results
    """
    torch.cuda.empty_cache()
    
    try:
        # Load model and matching processor
        print(f"Loading model and processor: {model_name}")
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        processor = WhisperProcessor.from_pretrained(model_name)
        
        # Create dataset with the correct processor that matches this model
        print(f"Creating dataset with processor matched to this model")
        test_data = parashat_hashavua_dataset(
            new_data="other", 
            few_data=FASTTEST, 
            train=False, 
            validation=False, 
            test=True, 
            random=False, 
            num_of_words_in_sample=1, 
            augment=False, 
            processor=processor, 
            load_srt_data=True,
            test_subdir=test_subdir,
        )
        
        print(f"Loaded {len(test_data.data)} samples for testing")
        
        # Create directory for this specific dataset if it doesn't exist
        output_dir = f"evalutions_on_other_data/{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        training_args = Seq2SeqTrainingArguments(
            output_dir=f"{output_dir}/test_{model_name.split('/')[-1]}",
            predict_with_generate=True,
        )
        
        # Create the data collator using the processor matching this model
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
    
    except Exception as e:
        print(f"Error testing model {model_name} on dataset {dataset_name}: {e}")
        # Return a dictionary with error information
        return {
            "error": str(e),
            "wer": float('nan'),  # NaN values for metrics
            "avg_f1_Exact": float('nan'),
            "avg_precision_Exact": float('nan'),
            "avg_recall_Exact": float('nan')
        }
    
def test_models(models_names, test_data_info_dict):
    all_results = {}
    
    for dataset_name, test_subdir in test_data_info_dict.items():
        print(f"\n=== Testing on dataset: {dataset_name} ===")
        dataset_results = []
        
        for model_name in models_names:
            print(f"Testing model: {model_name}")
            with torch.no_grad():
                result = {"model": model_name, "dataset": dataset_name}
                result.update(test_model(model_name, test_subdir, dataset_name))
                dataset_results.append(result)
                
        all_results[dataset_name] = dataset_results
        
        # Save the results for this dataset
        df_dataset = pd.DataFrame(dataset_results)
        df_dataset.to_csv(f"evalutions_on_other_data/test_results_{dataset_name}.csv", index=False)
        print(f"Saved results for dataset {dataset_name}")
        
    return all_results

# %%
# test the models on all datasets
all_results = test_models(models, test_data_info)

# %%
# Process and combine results from all datasets
combined_results = []
for dataset_name, results in all_results.items():
    for result in results:
        combined_results.append(result)

df_combined = pd.DataFrame(combined_results)

# Each column label with "eval_" replaced with ""
df_combined.columns = df_combined.columns.str.replace("eval_", "")

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

# Add dataset column to priority columns
df_combined = reorder_columns(df_combined, ["dataset", "avg_f1_Exact", "avg_recall_Exact", "avg_precision_Exact", "wer"])
df_combined

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
# Extract model information
model_info = df_combined['model'].apply(extract_model_info).apply(pd.Series)

# drop the 'model' column
df_combined.drop('model', axis=1, inplace=True)

# add each extracted column to the dataframe
df_combined = pd.concat([model_info, df_combined], axis=1)

# %%
# Create a combined results file with dataset information
df_combined.to_csv("./evalutions_on_other_data/test_results_all_datasets.csv", index=False)

# %%
# Create a pivot table to compare models across datasets
pivot_df = df_combined.pivot_table(
    index=['model', 'augmented', 'with_nikud', 'random'], 
    columns=['dataset'], 
    values=['avg_f1_Exact', 'wer']
)

# Flatten the hierarchical column names
pivot_df.columns = [f"{col[0]}_{col[1]}" for col in pivot_df.columns]

# Reset index to make it a regular DataFrame
pivot_df = pivot_df.reset_index()

# Save the comparison table
pivot_df.to_csv("./evalutions_on_other_data/model_comparison_across_datasets.csv", index=False)
pivot_df

# %%
# For backward compatibility, save one consolidated file with all results
if JUST_ADD_NEW_RESULTS:
    # load old results add the new results
    old_df = pd.read_csv("./evalutions_on_other_data/test_results.csv")

    # add the new results
    df = pd.concat([old_df, df_combined], ignore_index=True) 
else:
    df = df_combined.copy()
    
# Save as the original file for backward compatibility
df.to_csv("./evalutions_on_other_data/test_results.csv", index=False)

# %%
# Create a bar chart to visualize model performance across datasets
import matplotlib.pyplot as plt

# Select the best models based on avg_f1_Exact for visualization
top_models = df_combined.groupby('model')['avg_f1_Exact'].mean().nlargest(5).index
top_models_df = df_combined[df_combined['model'].isin(top_models)]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

# Plot avg_f1_Exact
top_models_df.pivot(index='model', columns='dataset', values='avg_f1_Exact').plot(
    kind='bar', ax=ax1, rot=45
)
ax1.set_title('Model Performance (F1 Score) Across Datasets')
ax1.set_ylabel('Avg F1 Exact')
ax1.legend(title='Dataset')
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Plot WER (lower is better)
top_models_df.pivot(index='model', columns='dataset', values='wer').plot(
    kind='bar', ax=ax2, rot=45
)
ax2.set_title('Model Performance (WER) Across Datasets')
ax2.set_ylabel('Word Error Rate (lower is better)')
ax2.legend(title='Dataset')
ax2.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("./evalutions_on_other_data/model_comparison_chart.png", dpi=300)
plt.show()


