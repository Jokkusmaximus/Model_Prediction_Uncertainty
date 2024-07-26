'''
Created on 05.04.24
by: jokkus
'''
from supplementary.experiments import ex_different_lr, ex_different_action_logstd
from supplementary.settings import set_seed, SEED, get_seed

# seeds and generators
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
# random.seed(SEED)
# np.random.seed(SEED)
# g = torch.Generator()
# g.manual_seed(SEED)
# seed_everything(SEED, workers=True)
#
# # create env
# # NOTE setting up mujoco is troublesome
# # You can switch to any env you like
# env = gym.make(PROJECT_ENV)
# env.action_space.seed(SEED)

print(SEED)
ex_different_action_logstd()





# ex_different_action_logstd(logstds=[2, 1.5,  1, 0.1, 0.05, 0.01, 0.005])
# set_seed(17)
# ex_different_action_logstd(logstds=[2, 1.5,  1, 0.1, 0.05, 0.01, 0.005])

# train_rl_model(env=env)

