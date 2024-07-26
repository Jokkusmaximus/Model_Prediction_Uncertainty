'''
Created on 05.04.24
by: jokkus
'''
from supplementary.experiments import ex_different_action_logstd, ex_different_seed_logstd_mod
from supplementary.settings import set_seed, get_seed, set_rl_config, get_rl_config

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


# Train baseline (actor_logstd_grad = True, => no manual changing of actor_logstd), also train missing init:1.5
rl_config_new = {
    "policy_type": "MlpPolicy",
    "config_name": "seeds",
    "run_name": "rl_model",
    "custom_max_episode_steps": 1024,
    # two episodes per policy update, standard is 1000 (seems like it cannot be more than 1000 in env)
    "custom_total_timesteps": 2000000,
    # "learning_rate": 1e-3,
    "model_hyperparams": {},
    "description": "Add description",
}
set_rl_config(rl_config_new)

print(get_rl_config())

set_seed(1)
ex_different_action_logstd([1.5])
set_seed(11)
ex_different_action_logstd()
set_seed(17)
ex_different_action_logstd([1.5])
set_seed(24)
ex_different_action_logstd([1.5])
set_seed(41)
ex_different_action_logstd()


rl_config_new = {
    "policy_type": "MlpPolicy",
    "config_name": "logstd_mod",
    "run_name": "logstd_mod",
    "custom_max_episode_steps": 1024,
    # two episodes per policy update, standard is 1000 (seems like it cannot be more than 1000 in env)
    "custom_total_timesteps": 2000000,
    # "learning_rate": 1e-3,
    "model_hyperparams": {},
    "description": "Add description",
}
set_rl_config(rl_config_new)
ex_different_seed_logstd_mod()
