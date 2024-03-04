import nikud_and_teamim

TAAM_STYLE = "number" # "with_base_char", "without_base_char" or "number"
class cantilLocationsObject:
    """
        python object for cantillations and their location in the string
        the object will be a list of tuples.
        each tuple will contain 3 values:
        (the cantillation, the number of the word in the whole string, the number of char in the specific word)
        for example for the string:
        "בעמ֤וד ענן֙ לנחת֣ם הד֔רך"
        the object for the cantillation will be:
        [('֤', 0, 2), ('֙', 1, 2), ('֣', 2, 3), ('֔', 3, 1)]
    """
    def __init__(self, string):
        self.string = nikud_and_teamim.remove_nikud(string)
        self.cantilLocations = []
        self.get_cantilLocations()
    
    def get_cantilLocations(self):
        for word_num, word in enumerate(self.string.split()):
            count = 1
            for char_num, char in enumerate(word):
                if char in nikud_and_teamim.TEAMIM:
                    if TAAM_STYLE == "number":
                        taam = nikud_and_teamim.TEAMIM.index(char)
                    elif TAAM_STYLE == "with_base_char":
                        taam = nikud_and_teamim.BASE_CHAR + char
                    else:
                        taam = char
                    self.cantilLocations.append((taam, word_num, char_num-count))
                    count += 1
    
    def __str__(self):
        return str(self.cantilLocations)


