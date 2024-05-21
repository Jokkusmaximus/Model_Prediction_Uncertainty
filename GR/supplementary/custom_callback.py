"""
Created on 10.05.24
by: jokkus
"""

import os

from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd
import numpy as np
from math import ceil
from torch import Tensor

from supplementary.settings import (
    rl_config,
    get_current_time,
    ACTION_SPACE,
    OBSERVATION_SPACE,
)


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    Used to extract and save actions, observations, rewards from PPO training

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env # type: VecEnv
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # num_timesteps = n_envs * n times env.step() was called
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = {}  # type: Dict[str, Any]
        # self.globals = {}  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger # type: stable_baselines3.common.logger.Logger
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

        # calculate expected steps
        self.total_steps = (
            ceil(rl_config["custom_total_timesteps"] / 2048) * 2048
        )  # assuming the standard 2048 steps per rollout

        self.counter = 0
        self.action_observation_reward = pd.DataFrame()
        self.actions = np.empty(
            shape=(self.total_steps, ACTION_SPACE), dtype=np.ndarray
        )
        self.observations = np.empty(
            shape=(self.total_steps, OBSERVATION_SPACE), dtype=Tensor
        )
        self.new_observations = np.empty(
            shape=(self.total_steps, OBSERVATION_SPACE), dtype=np.ndarray
        )
        self.rewards = np.empty(shape=self.total_steps, dtype=np.float32)

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """

        self.actions[self.counter] = self.locals["actions"]
        self.observations[self.counter] = self.locals["obs_tensor"]
        self.new_observations[self.counter] = self.locals["new_obs"]
        self.rewards[self.counter] = self.locals["rewards"]

        self.counter += 1

        return True  # must be true to enable training

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        # print("rollout ended.", f"steps: {self.counter}")
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        # print("training ended.", f"steps: {self.counter}")

        data = {
            "actions": self.actions.tolist(),
            "observations": self.observations.tolist(),
            "new_obs": self.new_observations.tolist(),
            "rewards": self.rewards,
        }

        df = pd.DataFrame(data)

        config_name = rl_config[
            "config_name"
        ]  # Configuration name used in folder structure
        current_time = get_current_time()
        log_path = f"./logs/{config_name}/rl_model_{current_time}/"

        os.makedirs(log_path, exist_ok=True)

        df.to_csv(f"{log_path}data.csv")

        np.savez_compressed(
            f"{log_path}data.npz",
            actions=self.actions,
            observations=self.observations,
            new_observations=self.new_observations,
            rewards=self.rewards,
        )
        # df.to_json(f"{log_path}data.json", default_handler=str)
