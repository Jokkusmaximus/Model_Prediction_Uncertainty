'''
Created on 29.04.24
by: jokkus
'''
import gymnasium as gym
import torch

from world_model.model_net import ModelNetwork

# create env
env = gym.make("HalfCheetah-v4", render_mode='human')

# Sets an initial state, gets observations
obs, _ = env.reset()
obs = torch.tensor(obs)
obs = obs.type(torch.FloatTensor)
print(obs.type())

# load model
model = ModelNetwork.load_from_checkpoint("test_run_1/test/version_0/checkpoints/epoch=24-step=23450.ckpt")

# eval mode?
model.eval()
# x = obs + action

# render 1200 times
for _ in range(1000):
    # render env
    env.render()
    action = model(x)
    # random action                 # TODO: make AI take action
    env.step(action)
    obs, _, _, _, _ = env.step(action)
    x = torch.concat(obs, action)
env.close()
