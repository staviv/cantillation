import json
from tqdm import tqdm
# Load the files
with open('03_dataset.json', 'r') as f:
    dataset = json.load(f)["text"]
with open('train_data.json', 'r') as f:
    train_data = json.load(f)["text"]
with open('validation_data.json', 'r') as f:
    validation_data = json.load(f)["text"]
with open('test_data.json', 'r') as f:
    test_data = json.load(f)["text"]

# count the number of equivalent text in the train, validation and test data and find if it is equal to the number of text in the dataset
for text in tqdm(dataset):
    # find the number of times the text appears in the train data
    train_count = train_data.count(text)
    # find the number of times the text appears in the validation data
    validation_count = validation_data.count(text)
    # find the number of times the text appears in the test data
    test_count = test_data.count(text)
    # find the number of times the text appears in the dataset
    dataset_count = dataset.count(text)
    # check if the number of text in the train, validation and test data is equal to the number of text in the dataset
    if train_count + validation_count + test_count != dataset_count:
        print('The train, validation and test data are not split correctly')
        print('The text:', text, 'appears', train_count, 'times in the train data,', validation_count, 'times in the validation data and', test_count, 'times in the test data')
        break
    