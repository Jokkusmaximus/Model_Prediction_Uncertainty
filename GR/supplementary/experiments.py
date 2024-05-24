"""
Script to implement methods for conducting experiments.
Created on 24.05.24
by: jokkus
"""

from rl import train_rl


def ex_different_lr(num_tests=10, env=None, lrs=None):
    """
    Method for conducting several trainings with different learning rates.

    Relevant configurations:    "custom_max_episode_steps": 2048,
                                "custom_total_timesteps": 2000000,

    :param num_tests:
    :param env:
    :return:
    """
    lr_min = 0.000001  # 1e-5?
    lr_max = 1

    if lrs is not None:
        for lr in lrs:
            path_addition = f"lr{lr}"
            train_rl.train_rl_model(learning_rate=lr, env=env, path_additional=path_addition)

    else:
        for i in range(num_tests):
            lr = ((lr_max - lr_min) / (num_tests-1)) * i    # TODO make exponential
            path_addition = f"lr{lr}"
            train_rl.train_rl_model(learning_rate=lr, env=env, path_additional=path_addition)
