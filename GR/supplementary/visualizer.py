'''
Created on 29.04.24
by: jokkus
'''
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import RolloutBuffer

from world_model.model_net import ModelNetwork

# create env
env = gym.make("HalfCheetah-v4", render_mode='human')

# Sets an initial state, gets observations
obs, _ = env.reset()
obs = torch.tensor(obs)
obs = obs.type(torch.FloatTensor)
print(obs.type())

callback = BaseCallback()
rollout_buffer = RolloutBuffer()

# load models
# world_model = ModelNetwork.load_from_checkpoint("logs/model_test_run_1/test/version_0/checkpoints/epoch=24-step=23450.ckpt")

rl_model = PPO.load("logs/rl_test/rl_model_20240508-154616/checkpoints/checkpoint_model_100000_steps.zip")
rl_model.collect_rollouts(env, callback, rollout_buffer,10)

# eval mode? should set to training mode when using dropout, otherwise the model is run without dropout
# x = obs + action

# render 1200 times
# for _ in range(1000):
#     # render env
#     env.render()
#
#     obs, _ = env.reset()
#     action = rl_model(obs)
#     # random action                 # TODO: make AI take action
#     env.step(action)
#     obs, _, _, _, _ = env.step(action)
#     x = torch.concat(obs, action)
# env.close()
