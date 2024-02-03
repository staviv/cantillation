import json

def split_data(file_path, train_ratio=0.9):
    # Load the original data
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Initialize the train and validation dictionaries
    train_data = {"text": [], "ashkenazi": [], "maroko": [], "yerushalmi": [], "bavly": []}
    validation_data = {"text": [], "ashkenazi": [], "maroko": [], "yerushalmi": [], "bavly": []}

    nusachim = ["ashkenazi", "maroko", "yerushalmi", "bavly"]
    for nusach in nusachim:
        # Calculate the split index
        split_index = int(len(data[nusach]) * train_ratio)

        # Split the data
        train_data[nusach] = data[nusach][:split_index]
        validation_data[nusach] = data[nusach][split_index:]

    # Split the text data
    split_index = int(len(data['text']) * train_ratio)
    train_data['text'] = data['text'][:split_index]
    validation_data['text'] = data['text'][split_index:]

    # Save the train and validation data
    with open('train_data.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)
    with open('validation_data.json', 'w', encoding='utf-8') as f:
        json.dump(validation_data, f, ensure_ascii=False, indent=4)

split_data('03_dataset.json')