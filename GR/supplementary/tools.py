"""
Created on 27.05.24
by: jokkus
"""
import time

import numpy as np
import pandas as pd
from math import floor, ceil

from supplementary.visualizer import visualize_PCA, visualize_tSNE, create_plots


def clean_nan_entries(array, verbose=False):
    """
    Removes rows with NaN values OBS: only evaluates the 0th value of each row, will lead to bugs if the entire row is not Nan
    :param array: Array from which NaN walues will be remove
    :return: array without NaN values
    """
    nan_arr = pd.isna(array)
    cleaned_arr = np.delete(array, nan_arr, 0)

    if verbose:
        print(f"Removed {len(array) - len(cleaned_arr)} NaN values, {len(cleaned_arr)} entries remaining")
    return cleaned_arr


def visualize_action_logstds(times_sliced=10):
    """
    Method to load and divide data, before passing it to the visualizers(make plots)
    TODO plot several rollouts with different colour per rollout
    TODO dynamically set x_lim & y_lim to be equal throughout plot_generation
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
    # savepath = "logs/cleanrl_test/rl_model_1717587951.3226361"
    # savepath = "logs/cleanrl_test/rl_model_1717420347.3728597_1"
    # savepath = "logs/cleanrl_test/rl_model_1717421859.5474908_0.01"
    # savepath = "logs/cleanrl_test/rl_model_1717421859.5474908_0.0001"
    # savepath = "logs/cleanrl_test/rl_model_1717582485.8645725_0.01"
    savepath = "logs/cleanrl_test/rl_model_1718183744.2232614_0.01"
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

            # array cleaning
            np_arr = clean_nan_entries(np_arr)  # remove NaN entries
            np_arr = np.vstack(np_arr)
            np_arr = np.squeeze(np_arr)  # Remove axes of length one from np_arr

            start_time = time.time()
            slice_size = floor(np_arr.shape[0] / times_sliced)
            for i in range(times_sliced):
                temp_array = np_arr[i * slice_size:(i + 1) * slice_size]
                create_plots(temp_array, title=f"{name}_action_logstd:{key}_{i}", custom_scaling=name)
                print(f"Time spent on calculating division {i}: {time.time() - start_time}, with size {slice_size}")
                start_time = time.time()

            # visualize_PCA(temp_array, dims=2, title=f"{name}_action_logstd:{key}_{i}", full_save=False)


def visualize_per_rollout(lim_create_plots=np.inf):
    """

    :param lim_create_plots: TODO: BUG: creates one plot more than the limit
    :return:
    """
    # TODO: create plots between rollout x and y. e.g. final 25 rollouts
    # TODO: figure out if the plots are made exactly per rollout, could be shifted due to rounding error.
    savepath = "logs/cleanrl_test/rl_model_1718183744.2232614_0.01/"
    # nparrz = np.load(f"{savepath}data.npz", allow_pickle=True)
    # nparrz = np.load("logs/cleanrl_test/rl_model_1718200844.9319282/data.npz", allow_pickle=True)
    nparrz = np.load("logs/cleanrl_test/rl_model_1718876308.8511665_0.01/data.npz", allow_pickle=True)

    for name in nparrz.files:
        np_arr = nparrz[name]

        # array cleaning
        np_arr = clean_nan_entries(np_arr)  # remove NaN entries
        np_arr = np.vstack(np_arr)
        np_arr = np.squeeze(np_arr)  # Remove axes of length one from np_arr

        print(name, np_arr.shape)

        slice_size = 2048
        num_rollouts = floor(np_arr.shape[0] / slice_size)
        # mean = np.empty(shape=(num_rollouts, np_arr.shape[1]))
        # print(floor(np_arr.shape[0] / slice_size))
        if lim_create_plots >= num_rollouts:
            for i in range(num_rollouts):
                print(f"num_rollouts: {num_rollouts}, lim_create_plots: {lim_create_plots}",
                      "*****" * 100)
                temp_arr = np_arr[i * slice_size:(i + 1) * slice_size]
                print(temp_arr)
                # print(f"mean: {temp_arr.mean(axis=0)}")
                # print(f"std: {temp_arr.std(axis=0)}")
                # print(f"var: {temp_arr.var(axis=0)}")
                print(f"Mean:{temp_arr.mean():.8f}  std:{temp_arr.std():.8f}  Var:{temp_arr.var():.8f}")
                np.set_printoptions(precision=8)
                print(f"Mean:{temp_arr.mean(axis=0)}  std:{temp_arr.std(axis=0)}  Var:{temp_arr.var(axis=0)}")
                # # create_plots(temp_arr, title=f"{name}_rollout:{i}", save_path=savepath)

        elif isinstance(lim_create_plots, int):
            slice_jump = ceil(
                num_rollouts / lim_create_plots)  # Space between rollouts which are used to create plots
            print(f"num_rollouts: {num_rollouts}, lim_create_plots: {lim_create_plots}, slice_jump:{slice_jump}",
                  "*****" * 100)
            for i in range(lim_create_plots):  # Here "lim_create_plots" should be an int   # TODO shoulnd this be num_rollouts?

                # Creating all plots between 0 & lim_create_plots-1
                temp_arr = np_arr[i * slice_jump * slice_size: ((i * slice_jump) + 1) * slice_size]
                create_plots(temp_arr, title=f"{name}_rollout:{i * slice_jump}", save_path=savepath)

                # print(f"array size: {len(temp_arr)}")
            # Create plot of final rollout
            temp_arr = np_arr[(num_rollouts - 1 )* slice_size: num_rollouts * slice_size]
            print(f"Mean:{temp_arr.mean():.8f}  std:{temp_arr.std():.8f}  Var:{temp_arr.var():.8f}")
            create_plots(temp_arr, title=f"{name}_rollout:{num_rollouts - 1}", save_path=savepath)
            # print(f"array size: {len(temp_arr)}", "::", f"{(num_rollouts - 1) * slice_size}: {num_rollouts * slice_size} : full size {len(np_arr)}")
        else:
            print(f"is \"lim_create_plots\" set, but not an int? type: {type(lim_create_plots)}")
