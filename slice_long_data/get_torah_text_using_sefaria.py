import requests
import re

books_data = [
    {"name": "Bereshit", "number": 1, "chapters": 50, "sefaria_name": "Genesis"},
    {"name": "Shemot", "number": 2, "chapters": 40, "sefaria_name": "Exodus"},
    {"name": "Vaikra", "number": 3, "chapters": 27, "sefaria_name": "Leviticus"},
    {"name": "Bamidbar", "number": 4, "chapters": 36, "sefaria_name": "Numbers"},
    {"name": "Dvarim", "number": 5, "chapters": 34, "sefaria_name": "Deuteronomy"}
]


def get_chapter_string(book_name, chapter_number):
    chapter_number = str(chapter_number)
    # Check if book_name is a valid sefaria_name
    for book in books_data:
        if book_name == book["name"]:
            sefaria_name = book["sefaria_name"]
            break
    else:
        sefaria_name = book_name

    # Construct the API URL for the specified book and chapter
    api_url = f"https://www.sefaria.org/api/texts/{sefaria_name}.{chapter_number}"

    # Fetch data from the API
    while True:
        response = requests.get(api_url)
        if response.status_code == 200:
            break
        else:
            print(f"Failed to fetch data from the API. The response status code is {response.status_code}. Retrying...")

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()

        # Extract the 'he' key from the data
        he_list = data.get('he', [])

        # Concatenate the list into a single string separated by spaces
        he_string = ' '.join(he_list).replace('  ', ' ').replace("""&nbsp;<span class="mam-spi-pe">{פ}</span><br>""", """ """).replace("""&thinsp;׀&thinsp;""", """׀""")
        # Remove all HTML tags from the string: <...>, </...>, and &...;
        he_string = re.sub(r'<.*?>', '', he_string)
        he_string = re.sub(r'&.*?;', '', he_string)
        
        return he_string
    else:
        return None

if __name__ == "__main__":
    # Example usage
    chapter_string = get_chapter_string("Genesis", 27)
    if chapter_string is not None:
        with open("Genesis.1.txt", "w", encoding="utf-8") as file:
            file.write(chapter_string)
    else:
        print("Failed to fetch data from the API")
        


