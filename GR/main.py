'''
Created on 05.04.24
by: jokkus
'''
import random
import numpy as np

import torch
import gymnasium as gym
from lightning.pytorch import seed_everything

from rl import train_rl
from supplementary.settings import SEED, PROJECT_ENV
from supplementary.experiments import ex_different_lr


# seeds and generators
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
g = torch.Generator()
g.manual_seed(SEED)
seed_everything(SEED, workers=True)

# create env
# NOTE setting up mujoco is troublesome
# You can switch to any env you like
env = gym.make(PROJECT_ENV)
env.action_space.seed(SEED)

# ex_different_lr(10, env)

train_rl.train_rl_model(env=env)

