import json
import random
def split_data(file_path, train_ratio=0.8):
    # Load the original data
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    
    
    
    # Split the data
    random.seed(42)
    # we want to get use the data as a parts of 50 samples each. so close samples will be close after the split
    num_samples = len(data['text']) // 50 
    # get the index of the split including all the 50 samples of the index
    random_vector = random.sample(range(num_samples), int(num_samples * train_ratio))
    random_vector.sort()
    # add all the indexes of the samples
    random_vector = [i * 50 for i in random_vector]
    random_vector = [i + j for i in random_vector for j in range(50)]
    train_data = {key: [data[key][i] for i in random_vector] for key in data.keys()}
    validation_data = {key: [data[key][i] for i in range(len(data['text'])) if i not in random_vector] for key in data.keys()}
    
    # Save the train and validation data
    with open('train_data.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)
    with open('validation_data.json', 'w', encoding='utf-8') as f:
        json.dump(validation_data, f, ensure_ascii=False, indent=4)

split_data('03_dataset.json', 0.95)