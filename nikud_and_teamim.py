TEAMIM = ['֑', '֒', '֓', '֔', '֕', '֖', '֗', '֘', '֙', '֚', '֛', '֜', '֝', '֞', '֟', '֠', '֡', '֢', '֣', '֤', '֥', '֦', '֧', '֨', '֩', '֪', '֫', '֬', '֭', '֮', 'ֽ']   
BASE_CHAR = "@"

def remove_nikud(text):
    nikud_list = ["ֱ","ֲ","ֳ","ִ","ֵ","ֶ","ַ","ָ","ׂ","ׁ","ֹ","ּ","ֻ","ְ","ׇ"]
    for nikud in nikud_list:
        text = text.replace(nikud, "")
    return text

def just_teamim(text, base_char = BASE_CHAR):
    new_text = ""
    for char in text:
        if char in TEAMIM:
            new_text += base_char
            new_text += char
        elif char == " ":
            new_text += " "
    return new_text
