"""
Created on 27.05.24
by: jokkus
"""
import time

import numpy as np
import pandas as pd
from math import floor

from supplementary.visualizer import visualize_PCA, visualize_tSNE, create_plots


def clean_nan_entries(array, verbose=False):
    """
    Removes rows with NaN values OBS: only evaluates the 0th value of each row, will lead to bugs if the entire row is not Nan
    :param array: Array from which NaN walues will be remove
    :return: array without NaN values
    """
    nan_arr = pd.isna(array)
    cleaned_arr = np.delete(array, nan_arr[:, 0], 0)

    if verbose:
        print(f"Removed {len(array) - len(cleaned_arr)} NaN values, {len(cleaned_arr)} entries remaining")
    return cleaned_arr

def visualize_action_logstds(divisions=10):
    """
    Method to load and divide data, before passing it to the visualizers(make plots)
    TODO plot per episode / plot several episodes with different colour per episode
    TODO dynamically get all folders trained at a certain time? without manually declaring suffixes
    :return:
    """
    # where to save pictures
    savepath = "logs/cleanrl_test/rl_model_1717487739.4952843_0.01"
    # loading all data
    # filepath_01 = f"{savepath}_0.1/"
    # filepath_001 = f"{savepath}_0.01/"
    # filepath_0001 = f"{savepath}_0.001/"
    # filepath_00001 = f"{savepath}_0.0001/"
    # filepath_1 = f"{savepath}_1/"
    # filepath_10 = f"{savepath}_10/"
    # filepath_100 = f"{savepath}_100/"
    # filepath_1000 = f"{savepath}_1000/"
    # filepath_10000 = f"{savepath}_10000/"
    # filepaths = {
    #     "0.0001": filepath_00001,
    #     "0.001": filepath_0001,
    #     "0.01": filepath_001,
    #     "0.1": filepath_01,
    #     "1": filepath_1,
    #     "10": filepath_10,
    #     "100": filepath_100,
    #     "1000": filepath_1000,
    #     "10000": filepath_10000,
    # }

    # savepath = "logs/cleanrl_test/rl_model_1717487739.4952843_0.01"
    savepath = "logs/cleanrl_test/rl_model_1717420347.3728597_1"
    # savepath = "logs/cleanrl_test/rl_model_1717421859.5474908_0.01"
    # savepath = "logs/cleanrl_test/rl_model_1717421859.5474908_0.0001"
    filepaths = {
        "0.01": f"{savepath}/",
    }

    savepath = f"{savepath}/"  # add a backslash to ensure pictures are added to a folder
    for key in filepaths.keys():
        filepath = filepaths[key]
        print(f"Filepath: {filepath},   Key: {key}")

        # load data into numpy array
        nparrz = np.load(f"{filepath}data.npz", allow_pickle=True)

        for name in nparrz.files:
            np_arr = nparrz[name]

            print(f"removing: {np_arr[0]}")
            np_arr = np.delete(np_arr, 0, 0)        # TODO: remove when not needed or make dynamic(best would be make array in cleanrl_agent, np.empty

            np_arr = np.vstack(np_arr)

            # array = clean_nan_entries(np_arr)  # remove NaN before continuing
            np_arr = np.squeeze(np_arr)         # Remove axes of length one from np_arr

            start_time = time.time()
            times_divided = 10
            slice_size = floor(np_arr.shape[0]/times_divided)
            for i in range(times_divided):
                temp_array = np_arr[i*slice_size:(i+1)*slice_size]
                create_plots(temp_array, title=f"{name}_action_logstd:{key}_{i}", custom_scaling=name)
                # visualize_PCA(temp_array, dims=2, save_path=savepath, title=f"{name}_action_logstd:{key}_{i}", full_save=False)
                # visualize_tSNE(temp_array, dims=2, save_path=savepath, title=f"{name}_action_logstd:{key}_i", full_save=False)
                print(f"Time spent on calculating division {i}: {time.time()-start_time}, with size {slice_size}")
                start_time = time.time()

            # try:
            # except ValueError:
            #    print(f"The content of nparrz[\"{name}\"], is of wrong shape \n The shape is {nparrz[name].shape}")


def visualize_episodes():
    # TODO visualize per episode, either all episodes in one large figure, or several episodes in 1 plot different color
    pass