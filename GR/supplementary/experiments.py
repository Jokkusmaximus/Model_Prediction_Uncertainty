"""
Script to implement methods for conducting experiments.
Created on 24.05.24
by: jokkus
"""

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from rl import train_rl
from supplementary.settings import (
    rl_config,
    get_path_addition,
    set_current_time,
    set_path_addition,
)


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
        train_rl.train_rl_model(
            learning_rate=lr, env=env, path_additional=temp_path_additional
        )
