"""
Script to implement methods for conducting experiments.
Created on 24.05.24
by: jokkus
"""
from datetime import datetime
import time
import numpy as np

from rl import cleanrl_agent, sb3_agent
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
        print(f"Training model {i + 1} / {num_tests}, Learning rate: {lr}")
        sb3_agent.train_rl_model(
            learning_rate=lr, env=env, path_additional=temp_path_additional
        )


def ex_different_action_logstd():
    """
    Method which iterates over a hard-coded array of log_stds, and trains new cleanRL models for each log_std
    :return: None
    """
    #logstds = [25, 10, 5, 1, 0.1, 0.01, 0.001, 0.0001]  # Viable range found: max 25(50->ERROR), min 0.0001
    logstds = [0.01]

    timings = np.zeros(shape=(1 + len(logstds)))
    start_time = time.time()

    timings[0] = start_time

    try:
        for i in range(len(logstds)):
            print(f"Training model {i + 1} / {len(logstds)}")
            path_addtion = f"{start_time}_{logstds[i]}"
            cleanrl_agent.train_rl_model(path_additional=path_addtion, action_std=logstds[i])
            timings[i+1] = time.time() - start_time
    except TypeError:
        print(f"Training model")
        path_addtion = f"{start_time}_{logstds}"
        cleanrl_agent.train_rl_model(path_additional=path_addtion, action_std=logstds)
        timings[i + 1] = time.time() - start_time

    for i in range(len(timings), 1, -1):
        timings[i] = timings[i] - timings[i-1]

    print(f"Training time mean: {np.mean(timings)}, max: {np.max(timings)}, min: {np.min(timings)}")


