"""
Written by Javier Muinelo Monteagudo <javimuinelo1@gmail.com>
during WS2324, as part of lecture project for advanced deep learning for robotics
"""

from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from tqdm import tqdm


class ProgressBar(EventCallback):
    def __init__(self, total_timesteps, verbose=0):
        super(ProgressBar, self).__init__(verbose=verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self, **kwargs) -> None:
        self.pbar = tqdm(total=self.total_timesteps)

    def _on_step(self) -> bool:
        self.pbar.update(1)
        return True

    def _on_training_end(self, **kwargs) -> None:
        self.pbar.close()
