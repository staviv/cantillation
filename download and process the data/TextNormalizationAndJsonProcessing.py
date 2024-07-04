import json
import re

def process_text(text):
    text = text.replace(" ׀ ", "׀").replace(" ׀ ", "׀").replace("׀", "׀ ").replace("־", "־ ").replace("[1]", "")
    text = re.sub(r'\s+|\n', ' ', text)  # replace multiple spaces or newline with a single space
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)  # remove space before punctuation
    text = re.sub(r'([.,;:!?])\s+', r'\1 ', text)  # ensure space after punctuation
    return text

def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        if "text" in data and isinstance(data["text"], list):
            data["text"] = [process_text(text) for text in data["text"]]
    
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

files_to_process = [
    "/app/jsons/validation_data_other.json",
    "/app/jsons/train_data_other.json"
]

for file_path in files_to_process:
    process_file(file_path)