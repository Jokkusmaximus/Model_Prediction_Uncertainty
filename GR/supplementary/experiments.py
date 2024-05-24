"""
Script to implement methods for conducting experiments.
Created on 24.05.24
by: jokkus
"""

import numpy as np
import matplotlib.pyplot as plt

from rl import train_rl
from supplementary.settings import rl_config, get_path_addition


def ex_different_lr(num_tests=10, env=None, lrs=None):
    """
    Conducts several trainings with different learning rates.

    Relevant configurations:    "custom_max_episode_steps": 2048,
                                "custom_total_timesteps": 5000000,

    :param lrs: array containing learning rates
    :param num_tests: number of tests to be conducted, ignored if lrs is not None
    :param env: Optional: The environment which gets passed to the model for training
    :return: None
    """
    lr_min = 0.000001  # 1e-6
    lr_max = 1

    if lrs is None:
        lrs = np.geomspace(lr_min, lr_max, num_tests)

    print(f"Learning rates: {lrs}\n")
    for lr in lrs:
        path_addition = f"lr{lr}"
        # train_rl.train_rl_model(
        #     learning_rate=lr, env=env, path_additional=path_addition
        # )

        # TODO: bellow
        print(f"{lr :.6f}")
        config_name = rl_config["config_name"]
        temp_path_additional = get_path_addition()
        path_additional = temp_path_additional + path_addition
        log_path = f"./logs/{config_name}/rl_model_{path_additional}/"
        np.save(f"{log_path}lr.npy", lrs)

