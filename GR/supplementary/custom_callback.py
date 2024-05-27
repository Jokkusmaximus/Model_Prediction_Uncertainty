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
    get_path_addition,
    ACTION_SPACE,
    OBSERVATION_SPACE,
)
from supplementary.tools import clean_nan_entries

class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    Used to extract and save actions, observations, rewards from PPO training

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, verbose: int = 0, save_per_rollout=False):
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

        self.counter = 0
        self.rollout_counter = 0

        # ** Array set-up **
        # Calculate array sizes based on expected length
        self.save_per_rollout = save_per_rollout
        if (
            rl_config["custom_total_timesteps"] > 2000000 or self.save_per_rollout
        ):  # 2million is arbitrarily chosen, but larger means more RAM used
            print("Saving per rollout")
            self.save_per_rollout = True
            self.array_size = 2048  # currently known, TODO: make dynamic: eg. self.locals["n_rollout_steps"]

        else:
            # calculate expected total steps
            self.array_size = (
                ceil(rl_config["custom_total_timesteps"] / 2048) * 2048
            )  # assuming the standard 2048 steps per rollout

        self.actions = np.empty(shape=(self.array_size, ACTION_SPACE), dtype=np.ndarray)
        self.observations = np.empty(
            shape=(self.array_size, OBSERVATION_SPACE), dtype=Tensor
        )
        # self.new_observations = np.empty(
        #     shape=(self.array_size, OBSERVATION_SPACE), dtype=np.ndarray
        # )
        # self.rewards = np.empty(shape=self.array_size, dtype=np.float32)

        # ** Logging set-up **
        self.config_name = rl_config[
            "config_name"
        ]  # Configuration name used in folder structure
        self.path_addition = get_path_addition()
        self.log_path = (
            f"./logs/{self.config_name}/rl_model_{self.path_addition}/"  # TODO shorten?
        )

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        # Make folders used for saving the arrays later
        os.makedirs(self.log_path, exist_ok=True)
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
        # self.new_observations[self.counter] = self.locals["new_obs"]
        # self.rewards[self.counter] = self.locals["rewards"]

        self.counter += 1

        return True  # must be true to enable training

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        # print("rollout ended.", f"steps: {self.counter}")
        if self.save_per_rollout:
            # update counters
            self.rollout_counter += 1
            self.counter = 0
            # save arrays
            np.savez_compressed(
                f"{self.log_path}data_rollout{self.rollout_counter}.npz",
                actions=self.actions,
                observations=self.observations,
                # new_observations=self.new_observations,
                # rewards=self.rewards,
            )

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        # print("training ended.", f"steps: {self.counter}")
        if not self.save_per_rollout:

            # removing NaN entries due to creating to large arrays
            self.actions = clean_nan_entries(self.actions)
            self.observations = clean_nan_entries(self.observations)

            np.savez_compressed(
                f"{self.log_path}data.npz",
                actions=self.actions,
                observations=self.observations,
                # new_observations=self.new_observations,
                # rewards=self.rewards,
            )
        if self.save_per_rollout:
            self.rollout_counter += 1
            np.savez_compressed(
                f"{self.log_path}data_rollout{self.rollout_counter}.npz",
                actions=self.actions,
                observations=self.observations,
                # new_observations=self.new_observations,
                # rewards=self.rewards,
            )
