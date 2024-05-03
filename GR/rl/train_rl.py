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
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback
from lightning.pytorch.loggers import TensorBoardLogger

# Imports from own config files
from supplementary.settings import PROJECT_ENV, rl_config
from supplementary.progress_bar import ProgressBar


def train_rl_model(
    env=None,
    load_model=False,       # TODO, if needed can be found at https://github.com/Jokkusmaximus/tum-adlr-9/tree/main
    model_path=None,        # TODO, if needed can be found at https://github.com/Jokkusmaximus/tum-adlr-9/tree/main
    custom_model=False,     # TODO, if needed can be found at https://github.com/Jokkusmaximus/tum-adlr-9/tree/main
    policy_kwargs=None,
    render_mode=None,
):
    # model parameters
    policy_type = rl_config["policy_type"]
    max_episode_steps = rl_config["custom_max_episode_steps"]
    total_timesteps = rl_config["custom_total_timesteps"]

    # Environment setup
    if env is None:
        env = gym.make(PROJECT_ENV)

    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    # Model Setup
    model = PPO(
        policy=policy_type, env=env, **rl_config["model_hyperparams"]
    )

    # Logging
    # Current date and time for unique directory names
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    config_name = rl_config[
        "config_name"
    ]  # configuration name used in folder structure

    logger = TensorBoardLogger("logs/rl_test_run1/", name="test")

    # Create directories for logs, checkpoints, and final model TODO: figure out if lightning.logger can be used
    log_path = f"./logs/{config_name}/"
    checkpoint_path = f"./logs/{config_name}/checkpoints/{current_time}/"
    final_model_path = f"./logs/{config_name}/rl_model_{current_time}/ppo_model"

    os.makedirs(log_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)

    # Write to JSON file
    with open(final_model_path + "_config.json", "w") as file:
        json.dump(rl_config, file, indent=4)

    # Configure logger to save data for TensorBoard
    logger = configure(log_path, ["tensorboard"])
    model.set_logger(logger)

    # Configure progress bar
    progress_bar = ProgressBar(total_timesteps=total_timesteps, verbose=1)

    checkpoint_callback = CheckpointCallback(
        save_freq=total_timesteps / 100,
        save_path=checkpoint_path,
        name_prefix="checkpoint_model",
    )

    model.learn(
        total_timesteps=total_timesteps, callback=[checkpoint_callback, progress_bar]
    )

    model.save(final_model_path)
