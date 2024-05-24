"""
Created on 03.05.24
by: jokkus
"""

# Imports
import os
import json
from datetime import datetime

# RL related imports
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    StopTrainingOnNoModelImprovement,
    EvalCallback,
)
from lightning.pytorch.loggers import TensorBoardLogger

# Imports from own config files
from supplementary.settings import (
    PROJECT_ENV,
    rl_config,
    NUM_SAVES,
    set_current_time,
    set_path_addition,
)
from supplementary.progress_bar import ProgressBar
from supplementary.custom_callback import CustomCallback


def train_rl_model(
    env=None,
    load_model=False,  # TODO, if needed can be found at https://github.com/Jokkusmaximus/tum-adlr-9/tree/main
    model_path=None,  # TODO, if needed can be found at https://github.com/Jokkusmaximus/tum-adlr-9/tree/main
    custom_model=False,  # TODO, if needed can be found at https://github.com/Jokkusmaximus/tum-adlr-9/tree/main
    policy_kwargs=None,
    render_mode=None,
    path_additional=None,
    learning_rate=None,
):
    # Model parameters
    policy_type = rl_config["policy_type"]
    max_episode_steps = rl_config["custom_max_episode_steps"]
    total_timesteps = rl_config["custom_total_timesteps"]

    # ** Environment setup **
    if env is None:
        env = gym.make(PROJECT_ENV)
    # Wrappers
    env = Monitor(env)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    # Learning rate set up, fetched from settings.py if not defined when calling train_rl_model
    if learning_rate is None:
        learning_rate = rl_config["learning_rate"]

    # Model Setup
    model = PPO(
        policy=policy_type,
        env=env,
        **rl_config["model_hyperparams"],
        learning_rate=learning_rate,
    )

    # Logging
    if path_additional is None:
        current_time = datetime.now().strftime(
            "%Y%m%d-%H%M%S"
        )  # Current date and time for unique directory names
        set_current_time(current_time)  # saving to access from other methods
        set_path_addition(current_time)  # saving to access from other methods
        path_additional = current_time

    config_name = rl_config[
        "config_name"
    ]  # Configuration name used in folder structure

    # Create and set up directories for logs, checkpoints, and final model TODO: figure out if lightning.logger can use
    log_path = f"./logs/{config_name}/rl_model_{path_additional}/"
    checkpoint_path = f"./logs/{config_name}/rl_model_{path_additional}/checkpoints/"
    final_model_path = f"./logs/{config_name}/rl_model_{path_additional}/ppo_model/"

    # Make folders
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)

    # Write model config to JSON file
    with open(final_model_path + "_config.json", "w") as file:
        json.dump(rl_config, file, indent=4)

    # Configure logger to save data for TensorBoard
    logger = configure(log_path, ["tensorboard"])
    model.set_logger(logger)

    # ** Configure callbacks **
    # Configure progress bar
    progress_bar = ProgressBar(total_timesteps=total_timesteps, verbose=1)
    # Configure Checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=total_timesteps / NUM_SAVES,
        save_path=checkpoint_path,
        name_prefix="checkpoint_model",
    )
    # Call custom callback (24.05-24: saving actions, observations & rewards as npz zip after training
    custom_callback = CustomCallback()
    # Configure Early stopping callback
    # Stop training if there is no improvement after more than 3 evaluations
    stop_train_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=15, min_evals=50, verbose=1
    )
    eval_callback = EvalCallback(
        env, eval_freq=1024, callback_after_eval=stop_train_callback, verbose=0
    )

    # Train model
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, progress_bar, custom_callback, eval_callback],
    )

    model.save(final_model_path)
