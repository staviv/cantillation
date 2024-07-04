# TODO: Use re to remove nikud and teamim
# import re
# text = re.sub(r'[\u0591-\u05C7]', '', text)

# nikiud: HATAF_SEGOL:"ֱ",HATAF_PATAH:"ֲ",HATAF_QAMATZ:"ֳ",HIRIQ:"ִ",TSERE:"ֵ",SEGOL:"ֶ",PATAH:"ַ",QAMATZ:"ָ",SIN_DOT:"ׂ",SHIN_DOT:"ׁ",HOLAM:"ֹ",DAGESH:"ּ",QUBUTZ:"ֻ",SHEVA:"ְ",QAMATZ_QATAN:"ׇ"
def remove_nikud(text):
    nikud_list = ["ֱ","ֲ","ֳ","ִ","ֵ","ֶ","ַ","ָ","ׂ","ׁ","ֹ","ּ","ֻ","ְ","ׇ"]
    for nikud in nikud_list:
        text = text.replace(nikud, "")
    return text

# Teamim: ['֑', '֒', '֓', '֔', '֕', '֖', '֗', '֘', '֙', '֚', '֛', '֜', '֝', '֞', '֟', '֠', '֡', '֢', '֣', '֤', '֥', '֦', '֧', '֨', '֩', '֪', '֫', '֬', '֭', '֮', 'ֽ']
def just_teamim(text, base_char = "@"):
    teamim = ['֑', '֒', '֓', '֔', '֕', '֖', '֗', '֘', '֙', '֚', '֛', '֜', '֝', '֞', '֟', '֠', '֡', '֢', '֣', '֤', '֥', '֦', '֧', '֨', '֩', '֪', '֫', '֬', '֭', '֮', 'ֽ', '׀']
    new_text = ""
    for char in text:
        if char in teamim:
            new_text += base_char
            new_text += char
            
    return new_text


# remove nikud and teamim from a string
def remove_nikud_and_teamim(text):
    nikud_and_teamim_list = ["ֱ","ֲ","ֳ","ִ","ֵ","ֶ","ַ","ָ","ׂ","ׁ","ֹ","ּ","ֻ","ְ","ׇ", '֑', '֒', '֓', '֔', '֕', '֖', '֗', '֘', '֙', '֚', '֛', '֜', '֝', '֞', '֟', '֠', '֡', '֢', '֣', '֤', '֥', '֦', '֧', '֨', '֩', '֪', '֫', '֬', '֭', '֮', 'ֽ','׀']
    for nikud_or_teamim in nikud_and_teamim_list:
        text = text.replace(nikud_or_teamim, "")
    return text

def replace_teamim_with_emphasis(text): # 'ֽ' is the teamim for emphasis in a word
    teamim = ['֑', '֒', '֓', '֔', '֕', '֖', '֗', '֘', '֙', '֚', '֛', '֜', '֝', '֞', '֟', '֠', '֡', '֢', '֣', '֤', '֥', '֦', '֧', '֨', '֩', '֪', '֫', '֬', '֭', '֮', 'ֽ']
    for char in teamim:
        text = text.replace(char,'ֽ').replace("׀", "") # 'ֽ' is the teamim for emphasis in a word. '׀' is the teamim that not located in words
    return text
