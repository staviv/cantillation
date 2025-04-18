# %%
import os
import torch
import subprocess
import argparse
import pickle
import numpy as np
from torch.utils.data import ConcatDataset
from transformers import WhisperProcessor

# our libraries
from global_variables.training_vars import *
from global_variables.folders import *
from parashat_hashavua_dataset import *
from nikud_and_teamim import TEAMIM

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Whisper model for cantillation')
    parser.add_argument('--gpu', type=str, default=None, help='GPU ID to use (overrides automatic selection)')
    parser.add_argument('--use_cached_data', action='store_true', help='Use preprocessed cached datasets')
    parser.add_argument('--model_version', type=str, default=None, 
                       help='Model version to use (tiny, base, small, medium, large, large-v2, large-v3, large-v3-turbo)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=None, help='Maximum number of training steps')
    # Add arguments for USE_IVRITAI and USE_SRT_DATA
    parser.add_argument('--use_ivritai', action='store_true', help='Use ivrit-ai model')
    parser.add_argument('--no_use_ivritai', action='store_false', dest='use_ivritai', help='Do not use ivrit-ai model')
    parser.add_argument('--use_srt_data', action='store_true', help='Use SRT data for training')
    parser.add_argument('--no_use_srt_data', action='store_false', dest='use_srt_data', help='Do not use SRT data for training')
    
    # Set default values to None to check if arguments were provided
    parser.set_defaults(use_ivritai=None, use_srt_data=None)
    
    return parser.parse_args()

def get_gpu_with_most_free_memory():
    try:
        # Run nvidia-smi to get GPU memory information
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', 
                                         '--format=csv,nounits,noheader'], 
                                         encoding='utf-8')
        
        # Parse the output
        lines = result.strip().split('\n')
        gpu_memory = []
        for line in lines:
            index, free_memory = map(int, line.split(','))
            gpu_memory.append((index, free_memory))

        # Sort by free memory (descending)
        gpu_memory.sort(key=lambda x: x[1], reverse=True)
        
        # If multiple GPUs have similar memory (within 1% of the max),
        # prefer the one with higher index to avoid the first GPU
        max_free_memory = gpu_memory[0][1]
        similar_gpus = [gpu for gpu in gpu_memory if gpu[1] >= max_free_memory * 0.99]
        
        if len(similar_gpus) > 1:
            # Choose the GPU with highest index among those with similar memory
            best_gpu = max(similar_gpus, key=lambda x: x[0])
        else:
            best_gpu = gpu_memory[0]
        
        return str(best_gpu[0])
    except Exception as e:
        print(f"Error getting GPU information: {e}")
        return "0"  # Default to GPU 0 if there's an error

# Parse arguments
args = parse_args()

# Override training variables with command line arguments if provided
if args.model_version:
    if args.model_version in BASE_MODEL_VERSIONS:
        BASE_MODEL_VERSION = args.model_version

# Override USE_IVRITAI if specified in arguments
if args.use_ivritai is not None:
    USE_IVRITAI = args.use_ivritai
    print(f"Use ivrit-ai model: {USE_IVRITAI}")

# Override USE_SRT_DATA if specified in arguments
if args.use_srt_data is not None:
    USE_SRT_DATA = args.use_srt_data
    print(f"Use SRT data: {USE_SRT_DATA}")

# Update BASE_MODEL_NAME based on USE_IVRITAI and BASE_MODEL_VERSION
if USE_IVRITAI and BASE_MODEL_VERSION == "large-v3-turbo":
    BASE_MODEL_NAME = "ivrit-ai/whisper-large-v3-turbo"
    print(f"Using ivrit-ai model: {BASE_MODEL_NAME}")
else:
    BASE_MODEL_NAME = f"openai/whisper-{BASE_MODEL_VERSION}"
    print(f"Using openai model: {BASE_MODEL_NAME}")

if args.batch_size:
    BATCH_SIZE = args.batch_size
    print(f"Using batch size: {BATCH_SIZE}")

if args.lr:
    LR = args.lr
    print(f"Using learning rate: {LR}")

if args.max_steps:
    MAX_STEPS = args.max_steps
    print(f"Using max steps: {MAX_STEPS}")

# Set CUDA_VISIBLE_DEVICES to the specified GPU or automatically select one
if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(f"Using GPU: {args.gpu}")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = get_gpu_with_most_free_memory()
    print(f"Automatically selected GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")

# %%
from huggingface_hub import login
# load the token from txt file
with open("./tokens/HF_token.txt", "r") as f:
    HF_TOKEN = f.read().strip() # strip() removes the trailing "\n" if it exists
login(token=HF_TOKEN)

# %%
# Load data and processor
if args.use_cached_data and os.path.exists("./cantillation/cached_datasets"):
    print("Loading preprocessed datasets from cache...")
    # Load processor
    processor = WhisperProcessor.from_pretrained("./cantillation/cached_datasets/processor")
    
    # Load validation data
    with open("./cantillation/cached_datasets/val_data.pkl", "rb") as f:
        val_data = pickle.load(f)
    print(f"Loaded validation data from cache. Size: {len(val_data)}")
    
    # Load training data
    with open("./cantillation/cached_datasets/train_data_ben13.pkl", "rb") as f:
        train_data_ben13 = pickle.load(f)
    
    # Use SRT data only if USE_SRT_DATA is True
    if USE_SRT_DATA and os.path.exists("./cantillation/cached_datasets/train_data_srt.pkl"):
        with open("./cantillation/cached_datasets/train_data_srt.pkl", "rb") as f:
            train_data_srt = pickle.load(f)
        train_data = ConcatDataset([train_data_ben13, train_data_srt])
        print(f"Loaded training data from cache (with SRT). Size: {len(train_data)}")
    else:
        train_data = train_data_ben13
        print(f"Loaded Ben13 training data from cache (without SRT). Size: {len(train_data)}")
else:
    # Initialize processor
    processor = WhisperProcessor.from_pretrained(BASE_MODEL_NAME, language="hebrew", task="transcribe")
    
    # Add tokens if needed
    tokens_added = (len(processor.tokenizer.encode('֟'))==6) # check if the tokens were already added
    if ADDTOKENS and not tokens_added: # add the tokens if they weren't already added
    
        if JUST_TEAMIM:
            new_tokens = [BASE_CHAR + c for c in TEAMIM] # add the base char to the teamim (e.g. א֑)
        elif NIKUD:
            new_tokens = ['֑', '֒', '֓', '֔', '֕', '֖', '֗', '֘', '֙', '֚', '֛', '֜', '֝', '֞', '֟', '֠', '֡', '֢', '֣', '֤', '֥', '֦', '֧', '֨', '֩', '֪', '֫', '֬', '֭', '֮', '֯', 'ְ', 'ֱ', 'ֲ', 'ֳ', 'ִ', 'ֵ', 'ֶ', 'ַ', 'ָ', 'ֹ', 'ֺ', 'ֻ', 'ּ', 'ֽ', '־', 'ֿ', '׀', 'ׁ', 'ׂ', '׃', 'ׄ', 'ׅ', '׆', 'ׇ']
        else:
            new_tokens = TEAMIM
        
        processor.tokenizer.add_tokens(new_tokens)

    # Load data
    val_data = parashat_hashavua_dataset(new_data=True, processor=processor, load_srt_data=True, num_of_words_in_sample=1, test=True, train=False)
    print("The number of validation data is:", len(val_data))
    
    train_data_ben13 = parashat_hashavua_dataset(new_data=True, few_data=FASTTEST, train=True, validation=False, random=RANDOM, num_of_words_in_sample=4, nusachim=NUSACHIM, augment=AUGMENT, processor=processor)
    
    # Use SRT data only if USE_SRT_DATA is True
    if USE_SRT_DATA:
        train_data_srt = parashat_hashavua_dataset(new_data=True, processor=processor, load_srt_data=True, num_of_words_in_sample=1)
        train_data = ConcatDataset([train_data_ben13, train_data_srt])
        print("The number of training data is (with SRT):", len(train_data))
    else:
        train_data = train_data_ben13
        print("The number of training data is (without SRT):", len(train_data))

# Update the RUN_NAME with the actual model version being used and dataset configuration
model_prefix = "IvritAI-" if USE_IVRITAI else ""
RUN_NAME = model_prefix + BASE_MODEL_VERSION + ("_Random" if RANDOM else "") + (("_DropOut-" + str(DROPOUT)) if DROPOUT else "") \
            + (("_WeightDecay-" + str(WEIGHT_DECAY)) if WEIGHT_DECAY else "")  + "_Augmented"*AUGMENT \
            + "_WithNikud"*NIKUD + "_WithSRT"*USE_SRT_DATA + "_date-" + dt_string
MODEL_NAME = f"./Teamim-{RUN_NAME}"

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


if ADDTOKENS:
    model.resize_token_embeddings(len(processor.tokenizer))

# %%
from transformers import Seq2SeqTrainingArguments


training_args = Seq2SeqTrainingArguments(
    output_dir= MODEL_NAME,  # change to a repo name of your choice
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=LR, # was 1e-5
    warmup_steps=WARMUP_STEPS, # was 500
    max_steps=MAX_STEPS, # was 4000
    gradient_checkpointing=True, # 
    gradient_checkpointing_kwargs={'use_reentrant':False}, # I added that because UserWarning: "The default value of use_reentrant will be updated to be False in the future."
    fp16=torch.cuda.is_available(), # I added that because fp16 can't be use on CPU but on cuda
    eval_strategy="steps",
    per_device_eval_batch_size=2,
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
    dataloader_prefetch_factor=2,
    dataloader_num_workers=1, # parallelize the data loading
    weight_decay=WEIGHT_DECAY,
    run_name=RUN_NAME, # It doesn't work
    generation_max_length=225,
    torch_compile=False,
)


# %%
from transformers import Seq2SeqTrainer, TrainerCallback


class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, model, optimizer, lr_scheduler, train_dataloader, eval_dataloader, **kwargs):
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
    # if the file doesn't exist, create it
    if not os.path.exists(filename):
        create_markdown_file_with_headers(filename)

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

def get_latest_eval_logs(trainer):
    """Get the most recent evaluation logs."""
    latest_eval_logs = {}
    latest_step = -1
    
    for log in trainer.state.log_history:
        if 'eval_loss' in log and 'step' in log and log['step'] > latest_step:
            latest_eval_logs = log
            latest_step = log['step']
    
    return latest_eval_logs

# Get the training loss
training_loss = trainer_state.training_loss
# Get the step and epoch from the TrainerState
step = trainer.state.global_step
epoch = trainer.state.epoch

# First try to get logs for the max_steps
history = get_logs_with_step(trainer, training_args.max_steps)

# If eval_loss is not in the history, get the latest evaluation logs
if not history or 'eval_loss' not in history:
    history = get_latest_eval_logs(trainer)

# Get the evaluation details from the log history with fallback values
validation_loss = history.get('eval_loss', 'N/A')
f1 = history.get('eval_avg_f1_Exact', 'N/A')
recall = history.get('eval_avg_recall_Exact', 'N/A')
precision = history.get('eval_avg_precision_Exact', 'N/A')

# Log the training details
log_training_to_markdown_file(training_args, training_loss, epoch, step, validation_loss, f1, recall, precision, 
                             filename="./cantillation/markdown_files/training_log_new.md")

