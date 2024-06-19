"""
Created on 01.05.24
by: jokkus
"""
# project parameters
PROJECT_ENV = "HalfCheetah-v4"
ACTION_SPACE = 6  # taken from env config
OBSERVATION_SPACE = 17  # taken from env config

# world model parameters
NUM_ENV_SAMPLES = 300_000
TRAIN_TEST_RELATIVE = 0.8
N_EPOCHS = 25
BATCH_SIZE = 256
SEED = 42
N_NEURONS_HIDDEN = 256
LR = 0.001

wm_config = {
    "config_name": "wm_test"
}

# RL parameters
rl_config = {
    "policy_type": "MlpPolicy",
    "config_name": "cleanrl_test",
    "custom_max_episode_steps": 1024,  # two episodes per policy update, standard is 1000 (seems like it cannot be more than 1000 in env)
    "custom_total_timesteps": 1000000,
    # "learning_rate": 1e-3,
    "model_hyperparams": {},
    "description": "Add description",
}

# imformation used when saving model and other relevant data
current_time = 0
path_addition = ""

NUM_SAVES = 10

# ** getters & setters **
def set_current_time(datetime):
    global current_time
    current_time = datetime
    print(f"Current time set to: {current_time}")


def get_current_time():
    return current_time


def set_path_addition(path):
    global path_addition
    path_addition = path
    # print(f"Path addition is {path_addition}")


def get_path_addition():
    return path_addition
