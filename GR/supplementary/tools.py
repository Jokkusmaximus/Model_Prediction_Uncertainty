"""
Created on 27.05.24
by: jokkus
"""
import numpy as np
import pandas as pd

def clean_nan_entries(array):
    """
    Removes rows with NaN values OBS: only evaluates the 0th value of each row, will lead to bugs if the entire row is not Nan
    :param array: Array from which NaN walues will be remove
    :return: array without NaN values
    """
    nan_arr = pd.isna(array)
    cleaned_arr = np.delete(array, nan_arr[:, 0], 0)

    print(f"Removed {len(array)-len(cleaned_arr)} NaN values, {len(cleaned_arr)} entries remaining")
    return cleaned_arr