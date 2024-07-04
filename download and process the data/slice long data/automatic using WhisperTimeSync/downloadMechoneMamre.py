import json

books_data = [
    {"name": "Bereshit", "number": 1, "chapters": 50},
    {"name": "Shemot", "number": 2, "chapters": 40},
    {"name": "Vaikra", "number": 3, "chapters": 27},
    {"name": "Bamidbar", "number": 4, "chapters": 36},
    {"name": "Dvarim", "number": 5, "chapters": 34}
]

def generate_link(book_number, chapter_number):
    return f"https://mechon-mamre.org/mp3/t{str(book_number).zfill(2)}{str(chapter_number).zfill(2)}.mp3"

books = []
for book_info in books_data:
    book = {"name": book_info["name"], "number": book_info["number"], "chapters": []}
    for chapter in range(1, book_info["chapters"] + 1):
        link = generate_link(book_info["number"], chapter)
        book["chapters"].append({"chapter_number": chapter, "link": link})
    books.append(book)

result_json = {"books": books}

# Convert the Python dictionary to a JSON string
json_string = json.dumps(result_json, indent=2)

# Save the JSON string to a file
with open("mechon_mamre_links.json", "w") as json_file:
    json_file.write(json_string)

print("JSON file has been generated.")



import os
import requests

def download_mp3(link, destination):
    response = requests.get(link, stream=True)
    if response.status_code == 200:
        with open(destination, 'wb') as file:
            for chunk in response.iter_content(chunk_size=128):
                file.write(chunk)
    else:
        print(f"Failed to download: {link}")

def download_books(books_data, base_directory):
    for book_info in books_data:
        book_directory = os.path.join(base_directory, f"{book_info['number']}_{book_info['name']}")
        os.makedirs(book_directory, exist_ok=True)

        for chapter_info in book_info['chapters']:
            link = chapter_info['link']
            chapter_number = chapter_info['chapter_number']
            destination = os.path.join(book_directory, f"Chapter_{chapter_number}.mp3")
            
            print(f"Downloading {link} to {destination}")
            download_mp3(link, destination)

if __name__ == "__main__":
    base_directory = "bible_mp3"
    os.makedirs(base_directory, exist_ok=True)

    # Replace the placeholder URLs with the actual Mechon Mamre links from the generated JSON
    generated_json_path = "mechon_mamre_links.json"
    with open(generated_json_path, 'r') as json_file:
        generated_json = json.load(json_file)

    download_books(generated_json['books'], base_directory)

    print("MP3 files have been downloaded.")
