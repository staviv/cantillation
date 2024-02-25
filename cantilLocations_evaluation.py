import cantilLocations
from typing import List, Tuple

cantilLocationsObject = cantilLocations.cantilLocationsObject

def calculate_TP_FP_FN_Exact(cantilLocation_original: cantilLocationsObject, cantilLocation_tested: cantilLocationsObject) -> Tuple[int, int, int]:
    """
    Calculate the true positive, false positive, and false negative for the "cantilLocation" object.
    """
    TP = len(set(cantilLocation_original.cantilLocations) & set(cantilLocation_tested.cantilLocations))
    FP = len(set(cantilLocation_tested.cantilLocations) - set(cantilLocation_original.cantilLocations))
    FN = len(set(cantilLocation_original.cantilLocations) - set(cantilLocation_tested.cantilLocations))
    return TP, FP, FN

def calculate_TP_FP_FN_with_Letter_Shift(cantilLocation_original: cantilLocationsObject, cantilLocation_tested: cantilLocationsObject) -> Tuple[int, int, int]:
    """
    Calculate the true positive, false positive, and false negative for the "cantilLocation" object. the letter location can be shifted by 1
    """
    TP = 0
    FP = 0
    FN = 0
    for taam, word_location, letter_location in cantilLocation_original.cantilLocations:
        for i in range(-1, 2):
            if (taam, word_location, letter_location + i) in cantilLocation_tested.cantilLocations:
                TP += 1
                break
        else:
            FN += 1
    for taam, word_location, letter_location in cantilLocation_tested.cantilLocations:
        for i in range(-1, 2):
            if (taam, word_location, letter_location + i) in cantilLocation_original.cantilLocations:
                break
        else:
            FP += 1
    return TP, FP, FN



def calculate_TP_FP_FN_Word_Level(cantilLocation_original: cantilLocationsObject, cantilLocation_tested: cantilLocationsObject) -> Tuple[int, int, int]:
    """
    Calculate the true positive, false positive, and false negative for the "cantilLocation" object. without the letter location
    """
    # the cantilLocation contains list of (taam, word_location, letter_location) tuples. we want to compare only the taam and word_location without the letter_location
    # so we will create a new list of (taam, word_location) tuples
    cantilLocations_original_word_level = [(taam, word_location) for taam, word_location, letter_location in cantilLocation_original.cantilLocations]
    cantilLocations_tested_word_level = [(taam, word_location) for taam, word_location, letter_location in cantilLocation_tested.cantilLocations]
    TP = len(set(cantilLocations_original_word_level) & set(cantilLocations_tested_word_level))
    FP = len(set(cantilLocations_tested_word_level) - set(cantilLocations_original_word_level))
    FN = len(set(cantilLocations_original_word_level) - set(cantilLocations_tested_word_level))
    return TP, FP, FN

def calculate_TP_FP_FN_Word_Level_with_Word_Shift(cantilLocation_original: cantilLocationsObject, cantilLocation_tested: cantilLocationsObject) -> Tuple[int, int, int]:
    """
    Calculate the true positive, false positive, and false negative for the "cantilLocation" object. without the letter location and the word location can be shifted by 1
    """
    TP = 0
    FP = 0
    FN = 0
    for taam, word_location, letter_location in cantilLocation_original.cantilLocations:
        for i in range(-1, 2):
            if (taam, word_location + i) in [(taam, word_location) for taam, word_location, letter_location in cantilLocation_tested.cantilLocations]:
                TP += 1
                break
        else:
            FN += 1
    for taam, word_location, letter_location in cantilLocation_tested.cantilLocations:
        for i in range(-1, 2):
            if (taam, word_location + i) in [(taam, word_location) for taam, word_location, letter_location in cantilLocation_original.cantilLocations]:
                break
        else:
            FP += 1
    return TP, FP, FN


def calculate_TP_FP_FN(cantilLocation_original: cantilLocationsObject, cantilLocation_tested: cantilLocationsObject, method: str) -> Tuple[int, int, int]:
    """
    Calculate the true positive, false positive, and false negative for the "cantilLocation" object.
    method can be "Exact", "Letter_Shift", "Word_Level", "Word_Shift"
    """
    if method == "Exact":
        return calculate_TP_FP_FN_Exact(cantilLocation_original, cantilLocation_tested)
    elif method == "Letter_Shift":
        return calculate_TP_FP_FN_with_Letter_Shift(cantilLocation_original, cantilLocation_tested)
    elif method == "Word_Level":
        return calculate_TP_FP_FN_Word_Level(cantilLocation_original, cantilLocation_tested)
    elif method == "Word_Shift":
        return calculate_TP_FP_FN_Word_Level_with_Word_Shift(cantilLocation_original, cantilLocation_tested)
    else:
        raise ValueError("method should be one of the following: 'Exact', 'Letter_Shift', 'Word_Level', 'Word_Shift'")
    

def calculate_precision_recall_f1(cantilLocations_original: cantilLocationsObject, cantilLocations_tested: cantilLocationsObject, method: str = "Exact") -> Tuple[float, float, float]:
    """
    calculate the precision, recall and f1 for the "cantilLocation" object
    """
    TP, FP, FN = calculate_TP_FP_FN(cantilLocations_original, cantilLocations_tested, method)
    if TP == 0:
        return 0, 0, 0
    else:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * (precision * recall) / (precision + recall)
        return precision, recall, f1

def calculate_precision_recall_f1_for_string(original_string: str, tested_string: str, method: str = "Exact") -> Tuple[float, float, float]:
    """
    calculate the precision, recall and f1 in cantilLocations for the given strings
    """
    cantilLocation_original = cantilLocationsObject(original_string)
    cantilLocation_tested = cantilLocationsObject(tested_string)
    return calculate_precision_recall_f1(cantilLocation_original, cantilLocation_tested, method)


def calculate_precision_recall_f1_for_string_list(original_string_list: List[str], tested_string_list: List[str], method: str = "Exact") -> Tuple[List[float], List[float], List[float]]:
    """
    calculate the precision, recall and f1 for the given list of strings
    """
    precision_list = []
    recall_list = []
    f1_list = []
    for original_string, tested_string in zip(original_string_list, tested_string_list):
        precision, recall, f1 = calculate_precision_recall_f1_for_string(original_string, tested_string, method)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    return precision_list, recall_list, f1_list

