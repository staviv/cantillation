import cantilLocations
from typing import List, Tuple

cantilLocationsObject = cantilLocations.cantilLocationsObject

def calculate_TP_FP_FN(cantilLocation_original: cantilLocationsObject, cantilLocation_tested: cantilLocationsObject) -> Tuple[int, int, int]:
    """
    Calculate the true positive, false positive, and false negative for the "cantilLocation" object.
    """
    TP = len(set(cantilLocation_original.cantilLocations) & set(cantilLocation_tested.cantilLocations))
    FP = len(set(cantilLocation_tested.cantilLocations) - set(cantilLocation_original.cantilLocations))
    FN = len(set(cantilLocation_original.cantilLocations) - set(cantilLocation_tested.cantilLocations))
    return TP, FP, FN

    
def calculate_precision_recall_f1(cantilLocations_original: cantilLocationsObject, cantilLocations_tested: cantilLocationsObject) -> Tuple[float, float, float]:
    """
    calculate the precision, recall and f1 for the "cantilLocation" object
    """
    TP, FP, FN = calculate_TP_FP_FN(cantilLocations_original, cantilLocations_tested)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1

def calculate_precision_recall_f1_for_string(original_string: str, tested_string: str) -> Tuple[float, float, float]:
    """
    calculate the precision, recall and f1 in canti
    """
    cantilLocation_original = cantilLocationsObject(original_string)
    cantilLocation_tested = cantilLocationsObject(tested_string)
    return calculate_precision_recall_f1(cantilLocation_original, cantilLocation_tested)


def calculate_precision_recall_f1_for_string_list(original_string_list: List[str], tested_string_list: List[str]) -> Tuple[List[float], List[float], List[float]]:
    """
    calculate the precision, recall and f1 for 
    """
    precision_list = []
    recall_list = []
    f1_list = []
    for original_string, tested_string in zip(original_string_list, tested_string_list):
        precision, recall, f1 = calculate_precision_recall_f1_for_string(original_string, tested_string)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    return precision_list, recall_list, f1_list

