import datetime
import random

import gymnasium as gym

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import seed_everything

# NOTE recommended
# wandb
# pytorch_lightning
# conda
# hydra
# (tensorboard)

# TODO
# use val and test set --> system models easily overfit
# possibly: use some regularization to prevent overfit

# parameters
NUM_ENV_SAMPLES = 300_000
TRAIN_TEST_RELATIVE = 0.8
N_EPOCHS = 25
BATCH_SIZE = 256
SEED = 40
LR = 0.001
N_NEURONS_HIDDEN = 256

# device; not required in pytorch lightning
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using {device} device")
# print(f"Using {torch.cuda.get_device_name(0)} device")

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
env = gym.make("HalfCheetah-v4")
env.action_space.seed(SEED)


# define network
class ModelNetwork(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(
            env.observation_space.shape[0] + env.action_space.shape[0], N_NEURONS_HIDDEN
        )
        self.layer2 = nn.Linear(N_NEURONS_HIDDEN, N_NEURONS_HIDDEN)
        self.layer3 = nn.Linear(N_NEURONS_HIDDEN, env.observation_space.shape[0])

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log("train_loss", loss, on_step=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log("test_loss", loss, on_step=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=LR)
        return optimizer


# sample train env interactions
# NOTE Sampling random actions. This is a start, but obvisously only explores a small part of the state space. Better to train a DRL policy from scratch until convergence, and then use different policies saved during training to sample env interactions.
x = np.zeros(
    shape=(
        NUM_ENV_SAMPLES,
        env.observation_space.shape[0] + env.action_space.shape[0],
    ),
    dtype=np.float32,
)
y = np.zeros(shape=(NUM_ENV_SAMPLES, env.observation_space.shape[0]), dtype=np.float32)
obs, info = env.reset(seed=SEED)
for i in range(NUM_ENV_SAMPLES):
    action = env.action_space.sample()
    x[i, 0 : env.observation_space.shape[0]] = obs
    x[i, env.observation_space.shape[0] :] = action
    prev_obs = obs
    obs, rew, term, trunc, _ = env.step(action)
    # target is difference between consecutive states
    # do this, because When the change is small, this method prevents the    dynamic model from memorizing the input state
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9195341
    y[i] = obs - prev_obs
    if term or trunc:
        obs, info = env.reset(seed=SEED + i)
env.close()

# Train test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=TRAIN_TEST_RELATIVE, random_state=SEED, shuffle=True
)

# Normalize in and output so that network doesn't overfit to specific states
x_scaler = StandardScaler()
x_scaler.fit(x_train)
x_train, x_test = x_scaler.transform(x_train), x_scaler.transform(x_test)

y_scaler = StandardScaler()
y_scaler.fit(y_train)
y_train, y_test = y_scaler.transform(y_train), y_scaler.transform(y_test)

# Dataloaders
x_train, y_train, x_test, y_test = (
    torch.tensor(x_train),
    torch.tensor(y_train),
    torch.tensor(x_test),
    torch.tensor(y_test),
)

traindataset, testdataset = TensorDataset(x_train, y_train), TensorDataset(
    x_test, y_test
)
trainloader = DataLoader(
    traindataset, batch_size=BATCH_SIZE, num_workers=15, generator=g
)
testloader = DataLoader(testdataset, batch_size=BATCH_SIZE, num_workers=15, generator=g)

# Logging
# timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
logger = TensorBoardLogger("test_run_1/", name="test")

# init network
model = ModelNetwork()

# Train and test
trainer = L.Trainer(
    max_epochs=N_EPOCHS, logger=logger, deterministic=True, accelerator="auto"
)  # auto-chooses gpu; deterministic=True for reproducability
trainer.fit(model, trainloader)
trainer.test(model, testloader)

print("Training finished")