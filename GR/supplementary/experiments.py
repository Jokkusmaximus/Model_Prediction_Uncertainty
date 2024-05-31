"""
Script to implement methods for conducting experiments.
Created on 24.05.24
by: jokkus
"""

from datetime import datetime
import time
import numpy as np
import matplotlib.pyplot as plt

from rl import cleanrl_agent
from supplementary.settings import (
    rl_config,
    get_path_addition,
    set_current_time,
    set_path_addition,
)
# from supplementary.visualizer import visualize_PCA, visualize_tSNE    # TODO: why broken?
from supplementary.tools import clean_nan_entries


def ex_different_lr(num_tests=10, env=None, lrs=None):
    """
    Conducts several trainings with different learning rates.

    Relevant configurations:    "custom_max_episode_steps": 1024,
                                "custom_total_timesteps": 5000000,

    :param lrs: array containing learning rates
    :param num_tests: number of tests to be conducted, ignored if lrs is not None
    :param env: Optional: The environment which gets passed to the model for training
    :return: None
    """
    lr_min = 0.000001  # 1e-6
    lr_max = 1

    # set up logarithmic spacing of learning rates
    if lrs is None:
        lrs = np.geomspace(lr_min, lr_max, num_tests)
    print(f"Learning rates: {lrs}\n")

    # Logging set-up
    config_name = rl_config[
        "config_name"
    ]  # Configuration name used in folder structure
    current_time = datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )  # Current date and time for unique directory names
    set_current_time(current_time)  # saving to access from other methods

    for i in range(num_tests):
        lr = lrs[i]
        temp_path_additional = f"{current_time}_lr{lr}"
        set_path_addition(temp_path_additional)
        print(f"Training model {i+1} / {num_tests}, Learning rate: {lr}")
        sb3_agent.train_rl_model(
            learning_rate=lr, env=env, path_additional=temp_path_additional
        )

def ex_different_action_logstd():
    logstds = [10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
    current_time = time.time()

    for i in range(len(logstds)):
        print(f"Training model {i+1} / {len(logstds)}")
        path_addtion = f"{current_time}_{logstds[i]}"
        cleanrl_agent.train_rl_model(path_additional=path_addtion, action_std=logstds[i])


def visualize_epsilons():
    # where to save pictures
    savepath = "logs/rl_test/rl_model_20240524-215058"

    # loading all data
    filepath_lr1 = f"{savepath}_lr1.0/data.npz"
    filepath_lr01 = f"{savepath}_lr0.1/data.npz"
    filepath_lr001 = f"{savepath}_lr0.01/data.npz"
    filepath_lr0001 = f"{savepath}_lr0.001/data.npz"
    filepath_lr00001 = f"{savepath}_lr0.0001/data.npz"
    filepath_lr000001 = f"{savepath}_lr1e-05/data.npz"
    filepath_lr0000001 = f"{savepath}_lr1e-06/data.npz"

    filepaths = {
        "1.0": filepath_lr1,
        "0.1": filepath_lr01,
        "0.01": filepath_lr001,
        "0.001": filepath_lr0001,
        "0.0001": filepath_lr00001,
        "1e-05": filepath_lr000001,
        "1e-06": filepath_lr0000001,
    }

    savepath = f"{savepath}/"  # add a backslash to ensure pictures are added to a folder
    for key in filepaths.keys():
        filepath = filepaths[key]
        print(filepath, key)
        nparrz = np.load(filepath, allow_pickle=True)

        for name in nparrz.files:
            array = clean_nan_entries(nparrz[name])  # remove NaN before continuing
            # try:
            visualize_tSNE(array, dims=2, save_path=savepath, title=f"{name}_LR:{key}", full_save=False)
            # except ValueError:
            #    print(f"The content of nparrz[\"{name}\"], is of wrong shape \n The shape is {nparrz[name].shape}")
