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
from supplementary.settings import PROJECT_ENV, rl_config, NUM_SAVES, set_current_time
from supplementary.progress_bar import ProgressBar
from supplementary.custom_callback import CustomCallback


def train_rl_model(
    env=None,
    load_model=False,       # TODO, if needed can be found at https://github.com/Jokkusmaximus/tum-adlr-9/tree/main
    model_path=None,        # TODO, if needed can be found at https://github.com/Jokkusmaximus/tum-adlr-9/tree/main
    custom_model=False,     # TODO, if needed can be found at https://github.com/Jokkusmaximus/tum-adlr-9/tree/main
    policy_kwargs=None,
    render_mode=None,
):
    # Model parameters
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
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")     # Current date and time for unique directory names
    set_current_time(current_time)                              # saving to acess from other methods
    config_name = rl_config["config_name"]                      # Configuration name used in folder structure

    # Create directories for logs, checkpoints, and final model TODO: figure out if lightning.logger can be used
    log_path = f"./logs/{config_name}/rl_model_{current_time}/"
    checkpoint_path = f"./logs/{config_name}/rl_model_{current_time}/checkpoints/"
    final_model_path = f"./logs/{config_name}/rl_model_{current_time}/ppo_model/"

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
        save_freq=total_timesteps / NUM_SAVES,
        save_path=checkpoint_path,
        name_prefix="checkpoint_model",
    )

    custom_callback = CustomCallback()

    # Train model
    model.learn(
        total_timesteps=total_timesteps, callback=[checkpoint_callback, progress_bar, custom_callback]
    )

    model.save(final_model_path)
