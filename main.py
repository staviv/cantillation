# %%
import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available.")
else:
    print("CUDA is not available. Check if your GPU drivers are properly installed.")


# %%
# !pip install datasets>=2.6.1
# !pip install git+https://github.com/huggingface/transformers
# !pip install librosa
# !pip install jiwer
# !pip install evaluate>=0.30
# #!pip install gradio
# !pip install -U accelerate

#from 20.5.2024
# !pip install audiomentations[extras]
# !pip install mutagen
# !pip install srt

# %%
from huggingface_hub import login
# load the token from txt file
with open("./tokens/HF_token.txt", "r") as f:
    HF_TOKEN = f.read().strip() # strip() removes the trailing "\n" if it exists
login(token=HF_TOKEN)


# %%
import os
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

#our libraries
from global_variables.training_vars import *
from global_variables.folders import *
from parashat_hashavua_dataset import *
from nikud_and_teamim import just_teamim,remove_nikud


# %%
processor = WhisperProcessor.from_pretrained("openai/whisper-" + BASE_MODEL_VERSION, language="hebrew", task="transcribe")

# %%
# # Check how many tokens needed for each nikud/taam
# chars = ['֑', '֒', '֓', '֔', '֕', '֖', '֗', '֘', '֙', '֚', '֛', '֜', '֝', '֞', '֟', '֠', '֡', '֢', '֣', '֤', '֥', '֦', '֧', '֨', '֩', '֪', '֫', '֬', '֭', '֮', '֯', 'ְ', 'ֱ', 'ֲ', 'ֳ', 'ִ', 'ֵ', 'ֶ', 'ַ', 'ָ', 'ֹ', 'ֺ', 'ֻ', 'ּ', 'ֽ', '־', 'ֿ', '׀', 'ׁ', 'ׂ', '׃', 'ׄ', 'ׅ', '׆', 'ׇ']
# for c in chars:
#     print(f"{c}: {len(processor.tokenizer.encode(c))-5}")


# %%
tokens_added = (len(processor.tokenizer.encode('֟'))==6) # check if the tokens were already added
if ADDTOKENS and not tokens_added: # add the tokens if they weren't already added
    
    teamim = ['֑', '֒', '֓', '֔', '֕', '֖', '֗', '֘', '֙', '֚', '֛', '֜', '֝', '֞', '֟', '֠', '֡', '֢', '֣', '֤', '֥', '֦', '֧', '֨', '֩', '֪', '֫', '֬', '֭', '֮', 'ֽ']
    if JUST_TEAMIM:
        new_tokens = [BASE_CHAR + c for c in teamim] # add the base char to the teamim (e.g. א֑)
    elif NIKUD:
        new_tokens = ['֑', '֒', '֓', '֔', '֕', '֖', '֗', '֘', '֙', '֚', '֛', '֜', '֝', '֞', '֟', '֠', '֡', '֢', '֣', '֤', '֥', '֦', '֧', '֨', '֩', '֪', '֫', '֬', '֭', '֮', '֯', 'ְ', 'ֱ', 'ֲ', 'ֳ', 'ִ', 'ֵ', 'ֶ', 'ַ', 'ָ', 'ֹ', 'ֺ', 'ֻ', 'ּ', 'ֽ', '־', 'ֿ', '׀', 'ׁ', 'ׂ', '׃', 'ׄ', 'ׅ', '׆', 'ׇ']
    else:
        new_tokens = teamim
    
    processor.tokenizer.add_tokens(new_tokens)

# %%
if COMBINE_DATA and NEWDATA == "other":
    print("COMBINE_DATA is True and NEWDATA is 'other'. This is not supported. Please change one of them.")
    exit()
else:
    if COMBINE_DATA:
        train_data1 = parashat_hashavua_dataset(new_data = True, few_data=FASTTEST, train =True ,validation=False, random=RANDOM, num_of_words_in_sample=4, nusachim=NUSACHIM, augment=AUGMENT, processor=processor)
        train_data2 = parashat_hashavua_dataset(new_data = False, train =True ,validation=False, random=RANDOM, num_of_words_in_sample=13, augment=AUGMENT, prob_for_num_of_parts = [0.05, 0.05, 0.05, 0.05, 0.1, 0.15, 0.15, 0.2, 0.2], processor=processor)
        train_data = ConcatDataset([train_data1, train_data2])
    else:
        if NEWDATA == "other":
            train_data = parashat_hashavua_dataset(new_data = NEWDATA, few_data=FASTTEST, train =True ,validation=False, random=RANDOM, num_of_words_in_sample=1, nusachim=NUSACHIM, augment=AUGMENT, processor=processor)
        elif NEWDATA:
            train_data = parashat_hashavua_dataset(new_data = True, few_data=FASTTEST, train =True ,validation=False, random=RANDOM, num_of_words_in_sample=4, nusachim=NUSACHIM, augment=AUGMENT, processor=processor)

        else:
            train_data = parashat_hashavua_dataset(new_data = False, train =True ,validation=False, random=RANDOM, num_of_words_in_sample=13, augment=AUGMENT, prob_for_num_of_parts = [0.05, 0.05, 0.05, 0.05, 0.1, 0.15, 0.15, 0.2, 0.2], processor=processor)
        
        
print(len(train_data))


# %%
# check the data of Pocket Torah
# train_data.check_failed_files_of_old_data()

# %%
# play random from the train data
# train_data.print_and_play_word_by_index(random.randint(0, len(train_data.data)))


#validation of the data
# train_data.check_the_data() 
# remove the sample index:
# train_data.remove_word_by_index(32487)


# %%
if COMBINE_DATA:
    val_data1 = parashat_hashavua_dataset(new_data = True, few_data=FASTTEST, train=False ,validation=True,  num_of_words_in_sample=4, nusachim=NUSACHIM, processor=processor)
    val_data2 = parashat_hashavua_dataset(new_data = False, train=False ,validation=True, num_of_words_in_sample=13, random=RANDOM, processor=processor)
    val_data = ConcatDataset([val_data1, val_data2])
    
else:
    if NEWDATA == "other":
        val_data = parashat_hashavua_dataset(new_data = NEWDATA, few_data=FASTTEST, train=False ,validation=True,  num_of_words_in_sample=1, nusachim=NUSACHIM, processor=processor)
    elif NEWDATA:
        val_data = parashat_hashavua_dataset(new_data = True, few_data=FASTTEST, train=False ,validation=True,  num_of_words_in_sample=4, nusachim=NUSACHIM, processor=processor)
    else:
        val_data = parashat_hashavua_dataset(new_data = False, train=False ,validation=True, num_of_words_in_sample=13, random=RANDOM, processor=processor)
    

print(len(val_data))
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
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

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
    
    # print
    # best:
    print(f"best f1 for {methods[0]}: {f1_max_exact}\nbest pred: {best_pred}\nbest label: {best_label}\n")
    
    # worst (the worst for each method):
    for i in range(4):
        print(f"worst f1 for {methods[i]}: {f1_min[i]}\nworst pred: {worst_pred[i]}\nworst label: {worst_label[i]}\n")
    
    
    
    print("Time taken for each part:")
    print(f"Decode calculation: {decode_time} seconds")
    print(f"WER calculation: {wer_time} seconds")
    
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

# %%
from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_NAME) # we can add "force_download=True" to download the model again


# model.generation_config.language = "he"




# # initialize the last layer of the model:
# model.proj_out.__init__(model.proj_out.in_features, len(processor.tokenizer))


if ADDTOKENS:
    model.resize_token_embeddings(len(processor.tokenizer))

# save the model config
# model.config.save_pretrained(MODEL_NAME)


# %%
# MODEL_NAME

# %%
from transformers import Seq2SeqTrainingArguments


training_args = Seq2SeqTrainingArguments(
    output_dir= MODEL_NAME,  # change to a repo name of your choice
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=LR, # was 1e-5
    warmup_steps=WARMUP_STEPS, # was 500
    max_steps=MAX_STEPS, # was 4000
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant':False}, # I added that because UserWarning: "The default value of use_reentrant will be updated to be False in the future."
    fp16=torch.cuda.is_available(), # I added that because fp16 can't be use on CPU but on cuda
    eval_strategy="steps",
    per_device_eval_batch_size=32,
    predict_with_generate=True,
    save_steps=SAVE_STEPS, 
    eval_steps=EVAL_STEPS,   
    logging_steps=25, 
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model= "avg_f1_Exact",# "avg_f1_..." like "avg_f1_Exact"
    greater_is_better=True, # if we use f1 score in eval so greater is better
    push_to_hub=True,
    # I added the dataloader_prefetch_factor to support newer versions of torch (now it must be int and not None. and the default is 2).
    dataloader_prefetch_factor=2, # support newer versions of torch
    dataloader_num_workers=1, # parallelize the data loading
    weight_decay=WEIGHT_DECAY,
    run_name=RUN_NAME, # It doesn't work
)


# %%
from transformers import Seq2SeqTrainer, TrainerCallback


class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, model, tokenizer, optimizer, lr_scheduler, train_dataloader, eval_dataloader, **kwargs):
        if state.global_step == 0:
            control.should_evaluate = True


trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
    callbacks = [EvaluateFirstStepCallback()] if EVALUATE_FIRST_STEP else None
)



# %%
processor.save_pretrained(training_args.output_dir)


# %%
def flags_warnings():
    if FASTTEST:
        for i in range(10):
            print("!!!TEST-MODE!!! \t \t to test the code only")

    if not ADDTOKENS:
        print("!!!ADDTOKENS==False!!!")

    if not NEWDATA:
        print("!!!NEWDATA==False!!!")

# %%
flags_warnings()

trainer_state = trainer.train()


# %%
kwargs = {
    # "dataset_args": "config: he, split: test",
    "language": "he",
    "model_name": "he-cantillation",
    "finetuned_from": BASE_MODEL_NAME,
    "tasks": "automatic-speech-recognition-cantillation",
    "tags": "hf-asr-leaderboard",
}

# %%
trainer.save_model()

# %%
trainer.push_to_hub(**kwargs)
# processor.push_to_hub("cantillation" +training_args.output_dir[1:])

# %%
trainer.lr_scheduler.get_lr()

# %%
processor.tokenizer.decode(train_data[26]["labels"])

# %%

from datetime import datetime


def log_training_to_markdown_file(training_args, training_loss, epoch, step, validation_loss, f1, recall, precision, filename="training_log.md"):
    # Get the current date and time
    now = datetime.now()

    # Format the date and time as a string
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")

    with open(filename, 'a') as f:
        f.write(f"| {date_time} | {training_args.output_dir } | {training_args.per_device_train_batch_size} | {training_args.gradient_accumulation_steps} | {training_args.learning_rate} | {training_args.warmup_steps} | {training_args.max_steps} | {training_args.gradient_checkpointing} | {training_args.gradient_checkpointing_kwargs} | {training_args.fp16} | {training_args.eval_strategy} | {training_args.per_device_eval_batch_size} | {training_args.predict_with_generate} | {training_args.generation_max_length} | {training_args.save_steps} | {training_args.eval_steps} | {training_args.logging_steps} | {training_args.report_to} | {training_args.load_best_model_at_end} | {training_args.metric_for_best_model} | {training_args.greater_is_better} | {training_args.push_to_hub} | {training_loss} | {epoch} | {step} | {validation_loss} | {f1} | {recall} | {precision} |\n")

def create_markdown_file_with_headers(filename="./markdown_files/training_log_new.md"):
    with open(filename, 'w') as f:
        f.write("| Date Time | Repo Name | Batch Size | Gradient Accumulation Steps | Learning Rate | Warmup Steps | Max Steps | Gradient Checkpointing | Gradient Checkpointing Kwargs | FP16 | Evaluation Strategy | Eval Batch Size | Predict with Generate | Max Length | Save Steps | Eval Steps | Logging Steps | Report To | Load Best Model at End | Metric for Best Model | Greater is Better | Push to Hub | Training Loss | Epoch | Step | Validation Loss | f1 | recall | precision |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|----|---|---|\n")

# Create the Markdown file with headers
#create_markdown_file_with_headers()
        

def get_logs_with_step(trainer, step = 1500):
    # Initialize an empty dictionary to store the merged logs
    merged_logs_with_step = {}

    # Iterate over the log history
    for log in trainer.state.log_history:
        # Check if the 'step' key exists in the log and if it equals the provided step
        if 'step' in log and log['step'] == step:
            # If it does, merge the log into the merged_logs_with_step dictionary
            merged_logs_with_step.update(log)

    # Return the merged logs
    return merged_logs_with_step


# Get the training loss
training_loss = trainer_state.training_loss
# Get the step and epoch from the TrainerState
step = trainer.state.global_step
epoch = trainer.state.epoch

# Get the log history at the specified step
history = get_logs_with_step(trainer,training_args.max_steps)
# Get the evaluation details from the log history
validation_loss = history['eval_loss']
f1 = history['eval_avg_f1_Exact']
recall = history['eval_avg_recall_Exact']
precision = history['eval_avg_precision_Exact']

# Log the training details
log_training_to_markdown_file(training_args, training_loss, epoch, step, validation_loss, f1, recall, precision, filename="./markdown_files/training_log_new.md")


# # %%
# #load the markdown file
# from IPython.display import Markdown

# # Open the file in read mode
# with open('./markdown_files/training_log_new.md', 'r') as file:
#     # Read the content of the file
#     content = file.read()

# # Display the content as Markdown
# display(Markdown(content))

# # %% [markdown]
# # | Model Name | Model Name | data | steps | lr |
# # |----------|----------|----------|--------|--------|
# # | whisper-medium-he-teamim-base | medium | all | 10,000 | 3e-5 |
# # | whisper-medium-he-teamim-ashkenazi-01 | base | ashkenazi | 9,000 | 1e-6 |
# # 
# # 
# # 

# # %%
# # run a web server to see the tensorboard
# # !tensorboard --logdir ./whisper-medium-he-teamim-aviv-base --port 6006 --bind_all

# # %% [markdown]
# # # Test the model

# # %%
# # load the model
# model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
# processor = WhisperProcessor.from_pretrained(MODEL_NAME)

# # load the test data
# test_data = parashat_hashavua_dataset(few_data=FASTTEST, train=False ,validation=False, test=True,  num_of_words_in_sample=4, nusachim=NUSACHIM, processor=processor)

# # create the data collator
# data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)


# # %%
# from transformers import Seq2SeqTrainer
# trainer = Seq2SeqTrainer(
#     args=training_args,
#     model=model,
#     eval_dataset=test_data,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
#     tokenizer=processor.feature_extractor,
# )

# # evaluate the model
# results = trainer.evaluate() # we use evaluate to get the metrics
# print(results)
# # save the results to a json file
# # create the results file
# with open(f"results_{MODEL_NAME.split('/')[-1]}.json", 'w') as f:
#     json.dump(results, f, indent=4)


