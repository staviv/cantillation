## the mainly file to download the data from sefaria
import os
from tqdm import tqdm
Genesis = ["Genesis.1.1-2.3", "Genesis.2.4-2.19", "Genesis.2.20-3.21", "Genesis.3.22-4.18", "Genesis.4.19-4.22", "Genesis.4.23-5.24", "Genesis.5.25-6.8", "Genesis.6.9-6.22", "Genesis.7.1-7.16", "Genesis.7.17-8.14", "Genesis.8.15-9.7", "Genesis.9.8-9.17", "Genesis.9.18-10.32", "Genesis.11.1-11.32", "Genesis.12.1-12.13", "Genesis.12.14-13.4", "Genesis.13.5-13.18", "Genesis.14.1-14.20", "Genesis.14.21-15.6", "Genesis.15.7-17.6", "Genesis.17.7-17.27", "Genesis.18.1-18.14", "Genesis.18.15-18.33", "Genesis.19.1-19.20", "Genesis.19.21-21.4", "Genesis.21.5-21.21", "Genesis.21.22-21.34", "Genesis.22.1-22.24", "Genesis.23.1-23.16", "Genesis.23.17-24.9", "Genesis.24.10-24.26", "Genesis.24.27-24.52", "Genesis.24.53-24.67", "Genesis.25.1-25.11", "Genesis.25.12-25.18", "Genesis.25.19-26.5", "Genesis.26.6-26.12", "Genesis.26.13-26.22", "Genesis.26.23-26.29", "Genesis.26.30-27.27", "Genesis.27.28-28.4", "Genesis.28.5-28.9", "Genesis.28.10-28.22", "Genesis.29.1-29.17", "Genesis.29.18-30.13", "Genesis.30.14-30.27", "Genesis.30.28-31.16", "Genesis.31.17-31.42", "Genesis.31.43-32.3", "Genesis.32.4-32.13", "Genesis.32.14-32.30", "Genesis.32.31-33.5", "Genesis.33.6-33.20", "Genesis.34.1-35.11", "Genesis.35.12-36.19", "Genesis.36.20-36.43", "Genesis.37.1-37.11", "Genesis.37.12-37.22", "Genesis.37.23-37.36", "Genesis.38.1-38.30", "Genesis.39.1-39.6", "Genesis.39.7-39.23", "Genesis.40.1-40.23", "Genesis.41.1-41.14", "Genesis.41.15-41.38", "Genesis.41.39-41.52", "Genesis.41.53-42.18", "Genesis.42.19-43.15", "Genesis.43.16-43.29", "Genesis.43.30-44.17", "Genesis.44.18-44.30", "Genesis.44.31-45.7", "Genesis.45.8-45.18", "Genesis.45.19-45.27", "Genesis.45.28-46.27", "Genesis.46.28-47.10", "Genesis.47.11-47.27", "Genesis.47.28-48.9", "Genesis.48.10-48.16", "Genesis.48.17-48.22", "Genesis.49.1-49.18", "Genesis.49.19-49.26", "Genesis.49.27-50.20", "Genesis.50.21-50.26"]
Exodus = ["Exodus.1.1-1.17", "Exodus.1.18-2.10", "Exodus.2.11-2.25", "Exodus.3.1-3.15", "Exodus.3.16-4.17", "Exodus.4.18-4.31", "Exodus.5.1-6.1", "Exodus.6.2-6.13", "Exodus.6.14-6.28", "Exodus.6.29-7.7", "Exodus.7.8-8.6", "Exodus.8.7-8.18", "Exodus.8.19-9.16", "Exodus.9.17-9.35", "Exodus.10.1-10.11", "Exodus.10.12-10.23", "Exodus.10.24-11.3", "Exodus.11.4-12.20", "Exodus.12.21-12.28", "Exodus.12.29-12.51", "Exodus.13.1-13.16", "Exodus.13.17-14.8", "Exodus.14.9-14.14", "Exodus.14.15-14.25", "Exodus.14.26-15.26", "Exodus.15.27-16.10", "Exodus.16.11-16.36", "Exodus.17.1-17.16", "Exodus.18.1-18.12", "Exodus.18.13-18.23", "Exodus.18.24-18.27", "Exodus.19.1-19.6", "Exodus.19.7-19.19", "Exodus.19.20-20.14", "Exodus.20.15-20.23", "Exodus.21.1-21.19", "Exodus.21.20-22.3", "Exodus.22.4-22.26", "Exodus.22.27-23.5", "Exodus.23.6-23.19", "Exodus.23.20-23.25", "Exodus.23.26-24.18", "Exodus.25.1-25.16", "Exodus.25.17-25.30", "Exodus.25.31-26.14", "Exodus.26.15-26.30", "Exodus.26.31-26.37", "Exodus.27.1-27.8", "Exodus.27.9-27.19", "Exodus.27.20-28.12", "Exodus.28.13-28.30", "Exodus.28.31-28.43", "Exodus.29.1-29.18", "Exodus.29.19-29.37", "Exodus.29.38-29.46", "Exodus.30.1-30.10", "Exodus.30.11-31.17", "Exodus.31.18-33.11", "Exodus.33.12-33.16", "Exodus.33.17-33.23", "Exodus.34.1-34.9", "Exodus.34.10-34.26", "Exodus.34.27-34.35", "Exodus.35.1-35.20", "Exodus.35.21-35.29", "Exodus.35.30-36.7", "Exodus.36.8-36.19", "Exodus.36.20-37.16", "Exodus.37.17-37.29", "Exodus.38.1-38.20", "Exodus.38.21-39.1", "Exodus.39.2-39.21", "Exodus.39.22-39.32", "Exodus.39.33-39.43", "Exodus.40.1-40.16", "Exodus.40.17-40.27", "Exodus.40.28-40.38"]
Leviticus = ["Leviticus.1.1-13", "Leviticus.1.14-2.6", "Leviticus.2.7-16", "Leviticus.3.1-17", "Leviticus.4.1-26", "Leviticus.4.27-5.10", "Leviticus.5.11-26", "Leviticus.6.1-11", "Leviticus.6.12-7.10", "Leviticus.7.11-38", "Leviticus.8.1-13", "Leviticus.8.14-21", "Leviticus.8.22-29", "Leviticus.8.30-36", "Leviticus.9.1-16", "Leviticus.9.17-23", "Leviticus.9.24-10.11", "Leviticus.10.12-15", "Leviticus.10.16-20", "Leviticus.11.1-32", "Leviticus.11.33-47", "Leviticus.12.1-13.5", "Leviticus.13.6-17", "Leviticus.13.18-23", "Leviticus.13.24-28", "Leviticus.13.29-39", "Leviticus.13.40-54", "Leviticus.13.55-59", "Leviticus.14.1-12", "Leviticus.14.13-20", "Leviticus.14.21-32", "Leviticus.14.33-53", "Leviticus.14.54-15.15", "Leviticus.15.16-28", "Leviticus.15.29-33", "Leviticus.16.1-17", "Leviticus.16.18-24", "Leviticus.16.25-34", "Leviticus.17.1-7", "Leviticus.17.8-18.5", "Leviticus.18.6-21", "Leviticus.18.22-30", "Leviticus.19.1-14", "Leviticus.19.15-22", "Leviticus.19.23-32", "Leviticus.19.33-37", "Leviticus.20.1-7", "Leviticus.20.8-22", "Leviticus.20.23-27", "Leviticus.21.1-15", "Leviticus.21.16-22.16", "Leviticus.22.17-33", "Leviticus.23.1-22", "Leviticus.23.23-32", "Leviticus.23.33-44", "Leviticus.24.1-23", "Leviticus.25.1-13", "Leviticus.25.14-18", "Leviticus.25.19-24", "Leviticus.25.25-28", "Leviticus.25.29-38", "Leviticus.25.39-46", "Leviticus.25.47-26.2", "Leviticus.26.3-5", "Leviticus.26.6-9", "Leviticus.26.10-46", "Leviticus.27.1-15", "Leviticus.27.16-21", "Leviticus.27.22-28", "Leviticus.27.29-34"]
Numbers = ["Numbers.1.1-19", "Numbers.1.20-54", "Numbers.2.1-34", "Numbers.3.1-13", "Numbers.3.14-39", "Numbers.3.40-51", "Numbers.4.1-20", "Numbers.4.21-37", "Numbers.4.38-49", "Numbers.5.1-10", "Numbers.5.11-6.27", "Numbers.7.1-41", "Numbers.7.42-71", "Numbers.7.72-89", "Numbers.8.1-14", "Numbers.8.15-26", "Numbers.9.1-14", "Numbers.9.15-10.10", "Numbers.10.11-34", "Numbers.10.35-11.29", "Numbers.11.30-12.16", "Numbers.13.1-20", "Numbers.13.21-14.7", "Numbers.14.8-25", "Numbers.14.26-15.7", "Numbers.15.8-16", "Numbers.15.17-26", "Numbers.15.27-41", "Numbers.16.1-13", "Numbers.16.14-19", "Numbers.16.20-17.8", "Numbers.17.9-15", "Numbers.17.16-24", "Numbers.17.25-18.20", "Numbers.18.21-32", "Numbers.19.1-17", "Numbers.19.18-20.6", "Numbers.20.7-13", "Numbers.20.14-21", "Numbers.20.22-21.9", "Numbers.21.10-20", "Numbers.21.21-22.1", "Numbers.22.2-12", "Numbers.22.13-20", "Numbers.22.21-38", "Numbers.22.39-23.12", "Numbers.23.13-26", "Numbers.23.27-24.13", "Numbers.24.14-25.9", "Numbers.25.10-26.4", "Numbers.26.5-51", "Numbers.26.52-27.5", "Numbers.27.6-23", "Numbers.28.1-15", "Numbers.28.16-29.11", "Numbers.29.12-30.1", "Numbers.30.2-17", "Numbers.31.1-12", "Numbers.31.13-24", "Numbers.31.25-41", "Numbers.31.42-54", "Numbers.32.1-19", "Numbers.32.20-42", "Numbers.33.1-9", "Numbers.33.10-49", "Numbers.33.50-34.15", "Numbers.34.16-29", "Numbers.35.1-8", "Numbers.35.9-34", "Numbers.36.1-13"]
# Numbers = ["Numbers.1.1-19", "Numbers.1.20-54", "Numbers.2.1-34", "Numbers.3.1-13", "Numbers.3.14-39", "Numbers.3.40-51", "Numbers.4.1-20", "Numbers.4.21-37", "Numbers.4.38-49", "Numbers.5.1-10", "Numbers.5.11-6.27", "Numbers.7.1-41", "Numbers.7.42-71", "Numbers.7.72-89", "Numbers.8.1-14", "Numbers.8.15-26", "Numbers.9.1-14", "Numbers.9.15-10.10", "Numbers.10.11-34", "Numbers.10.35-11.29", "Numbers.11.30-12.16", "Numbers.13.1-20", "Numbers.13.21-14.7", "Numbers.14.8-25", "Numbers.14.26-15.7", "Numbers.15.8-16", "Numbers.15.17-26", "Numbers.15.27-41", "Numbers.16.1-13", "Numbers.16.14-19", "Numbers.16.20-17.8", "Numbers.17.9-15", "Numbers.17.16-24", "Numbers.17.25-18.20", "Numbers.18.21-32", "Numbers.19.1-17", "Numbers.19.18-20.6", "Numbers.20.7-13", "Numbers.20.14-21", "Numbers.20.22-21.9", "Numbers.21.10-20", "Numbers.21.21-22.1", "Numbers.22.2-12", "Numbers.22.13-20", "Numbers.22.21-38", "Numbers.22.39-23.12", "Numbers.23.13-26", "Numbers.23.27-24.13", "Numbers.24.14-25.9", "Numbers.25.10-26.4", "Numbers.26.5-51", "Numbers.26.52-27.5", "Numbers.27.6-23", "Numbers.28.1-15", "Numbers.28.16-29.11", "Numbers.29.12-30.1", "Numbers.30.2-17", "Numbers.31.1-12", "Numbers.31.13-24", "Numbers.31.25-41", "Numbers.31.42-54", "Numbers.32.1-19", "Numbers.32.20-42", "Numbers.33.1-10", "Numbers.33.11-49", "Numbers.33.50-34.15", "Numbers.34.16-29", "Numbers.35.1-8", "Numbers.35.9-34", "Numbers.36.1-13"]
Deuteronomy = ["Deuteronomy.1.1-10", "Deuteronomy.1.11-21", "Deuteronomy.1.22-38", "Deuteronomy.1.39-2.1", "Deuteronomy.2.2-30", "Deuteronomy.2.31-3.14", "Deuteronomy.3.15-22", "Deuteronomy.3.23-4.4", "Deuteronomy.4.5-40", "Deuteronomy.4.41-49", "Deuteronomy.5.1-18", "Deuteronomy.5.19-6.3", "Deuteronomy.6.4-25", "Deuteronomy.7.1-11", "Deuteronomy.7.12-8.10", "Deuteronomy.8.11-9.3", "Deuteronomy.9.4-29", "Deuteronomy.10.1-11", "Deuteronomy.10.12-11.9", "Deuteronomy.11.10-21", "Deuteronomy.11.22-25", "Deuteronomy.11.26-12.10", "Deuteronomy.12.11-28", "Deuteronomy.12.29-13.19", "Deuteronomy.14.1-21", "Deuteronomy.14.22-29", "Deuteronomy.15.1-18", "Deuteronomy.15.19-16.17", "Deuteronomy.16.18-17.13", "Deuteronomy.17.14-20", "Deuteronomy.18.1-5", "Deuteronomy.18.6-13", "Deuteronomy.18.14-19.13", "Deuteronomy.19.14-20.9", "Deuteronomy.20.10-21.9", "Deuteronomy.21.10-21", "Deuteronomy.21.22-22.7", "Deuteronomy.22.8-23.7", "Deuteronomy.23.8-24", "Deuteronomy.23.25-24.4", "Deuteronomy.24.5-13", "Deuteronomy.24.14-25.19", "Deuteronomy.26.1-11", "Deuteronomy.26.12-15", "Deuteronomy.26.16-19", "Deuteronomy.27.1-10", "Deuteronomy.27.11-28.6", "Deuteronomy.28.7-69", "Deuteronomy.29.1-8", "Deuteronomy.29.9-11", "Deuteronomy.29.12-14", "Deuteronomy.29.15-28", "Deuteronomy.30.1-6", "Deuteronomy.30.7-10", "Deuteronomy.30.11-14", "Deuteronomy.30.15-20", "Deuteronomy.31.1-3", "Deuteronomy.31.4-6", "Deuteronomy.31.7-9", "Deuteronomy.31.10-13", "Deuteronomy.31.14-19", "Deuteronomy.31.20-24", "Deuteronomy.31.25-30", "Deuteronomy.32.1-6", "Deuteronomy.32.7-12", "Deuteronomy.32.13-18", "Deuteronomy.32.19-28", "Deuteronomy.32.29-39", "Deuteronomy.32.40-43", "Deuteronomy.32.44-52", "Deuteronomy.33.1-7", "Deuteronomy.33.8-12", "Deuteronomy.33.13-17", "Deuteronomy.33.18-21", "Deuteronomy.33.22-26", "Deuteronomy.33.27-29", "Deuteronomy.34.1-12"]

# The commented out line is the real aliyot, but in pokethorah it's different (only one verse difference)


aliot_list = []
# dicts = []
for i in ["Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy"]:
    # dicts.append(grouping(eval(i))) # eval() is used to convert string to variable 
    aliot_list+=eval(i)


import requests
import re



def get_sefaria_text_using_api(command):
    
    # Construct the API URL for the specified book and chapter
    api_url = f"https://www.sefaria.org/api/v3/texts/{command}"

    # Fetch data from the API
    response = requests.get(api_url)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()

        # Extract the 'he' key from the data
        he_list = data.get('versions', [])[0].get('text', [])
        
        # Concatenate the list of lists into a single list
        if type(he_list[0]) == list:
            he_list = [item for sublist in he_list for item in sublist]
        # Concatenate the list into a single string separated by spaces
        he_string = '\n'.join(he_list)
        # Replace some special characters with spaces
        he_string = he_string.replace('  ', ' ').replace("""&nbsp;<span class="mam-spi-pe">{פ}</span><br>""", """ """).replace("""&thinsp;׀&thinsp;""", """׀""")
        # Get the Qere (how it is read) from the string
        # replace the qere. "<span class="mam-kq-q">[תְנִיא֔וּן]</span></span>" -> "תְנִיא֔וּן"
        he_string = re.sub(r'<span class="mam-kq-q">\[(.*?)\]</span></span>', r'\1', he_string)
        # replace the ketiv. "<span class="mam-kq"><span class="mam-kq-k">(תנואון)</span>" -> ""
        he_string = re.sub(r'<span class="mam-kq"><span class="mam-kq-k">\((.*?)\)</span>', r'', he_string)
        # Remove all HTML tags from the string: <...>, </...>, and &...;
        he_string = re.sub(r'<.*?>', '', he_string)
        he_string = re.sub(r'&.*?;', '', he_string)
        # Remove all brackets from the string: [...], {....}, and (....)
        he_string = re.sub(r'\[.*?\]', '', he_string)
        he_string = re.sub(r'\{.*?\}', '', he_string)
        he_string = re.sub(r'\(.*?\)', '', he_string)
        
        
        return he_string
    else:
        return None


parsha_names = ["Bereshit", "Noach", "LechLecha", "Vayera", "ChayeiSara", "Toldot", "Vayetzei", "Vayishlach", "Vayeshev", "Miketz", "Vayigash", "Vayechi", "Shemot", "Vaera", "Bo", "Beshalach", "Yitro", "Mishpatim", "Terumah", "Tetzaveh", "KiTisa", "Vayakhel", "Pekudei", "Vayikra", "Tzav", "Shmini", "Tazria", "Metzora", "AchreiMot", "Kedoshim", "Emor", "Behar", "Bechukotai", "Bamidbar", "Nasso", "Behaalotcha", "Shlach", "Korach", "Chukat", "Balak", "Pinchas", "Matot", "Masei", "Devarim", "Vaethanan", "Eikev", "Reeh", "Shoftim", "KiTeitzei", "KiTavo", "Nitzavim", "Vayeilech", "Haazinu", "VezotHaberakhah"]


# Save the text of the aliyot to files "parsha_name-aliyah_number.txt" (Bereshit-1.txt) in the "aliyot_text" folder
def save_aliyot_text_to_files(aliyot, parsha_names):
    # initialize tqdm progress bar
    bar = tqdm(total=len(parsha_names)*7)
    for i, parsha in enumerate(parsha_names):
        for j in range(1, 8):
            text = get_sefaria_text_using_api(aliyot[i*7+j-1])
            os.makedirs('text', exist_ok=True)
            with open(f"text/{parsha_names[i]}-{j}.txt", "w", encoding="utf-8") as f:
                f.write(text)
            bar.update(1)
    bar.close()
save_aliyot_text_to_files(aliot_list, parsha_names)

