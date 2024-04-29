'''
Created on 29.04.24
by: jokkus
'''
import gymnasium as gym

# create env
env = gym.make("HalfCheetah-v4", render_mode='human')

# Sets an initial state
env.reset()

# render 1200 times
for _ in range(1000):
    #render env
    env.render()
    # random action                 # TODO: make AI take action
    env.step(env.action_space.sample())

env.close()
