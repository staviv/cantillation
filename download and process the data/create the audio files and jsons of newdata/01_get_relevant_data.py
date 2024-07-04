import json

with open('01_the_full_table_of_data.json', encoding='utf-8') as f:
    data = json.load(f)

new_data = {"audio url": [], "text": []}
for book in data:
    for parasha in book:
        if parasha is not None:
            print("parasha", parasha["1"][0][0]["parasha"])
            for aliya in parasha.values():
                for phrase in aliya[0]:
                    if int(phrase["block"]) >= 1:
                        if phrase["color"] == "6" and aliya[0].index(phrase)+1 < len(aliya[0]) and aliya[0][aliya[0].index(phrase)+1]["color"] == "4":
                            new_data["text"].append(phrase["text"] + " " + aliya[0][aliya[0].index(phrase)+1]["text"])
                            new_data["audio url"].append(phrase["url"].replace(" ", ""))
                            print(phrase["text"] + " " + aliya[0][aliya[0].index(phrase)+1]["text"])
                        else:
                            if phrase["color"] == "4" and aliya[0].index(phrase)-1 >= 0 and aliya[0][aliya[0].index(phrase)-1]["color"] == "6":
                                continue
                            new_data["text"].append(phrase["text"])
                            new_data["audio url"].append(phrase["url"].replace(" ", ""))
                    
with open('relevant_data.json', 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)

print(len(new_data["text"]))
