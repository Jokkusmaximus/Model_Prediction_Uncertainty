"""
Created on 01.05.24
by: jokkus
"""
# project parameters
PROJECT_ENV = "HalfCheetah-v4"


# world model parameters
NUM_ENV_SAMPLES = 300_000
TRAIN_TEST_RELATIVE = 0.8
N_EPOCHS = 25
BATCH_SIZE = 256
SEED = 40
LR = 0.001
N_NEURONS_HIDDEN = 256

wm_config = {
    "config_name": "wm_test"
}

# RL parameters
rl_config = {
    "policy_type": "MlpPolicy",
    "config_name": "rl_test",
    "custom_max_episode_steps": 3000,
    "custom_total_timesteps": 100000,
    "model_hyperparams": {},
    "description": "Add description",
}
