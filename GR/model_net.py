'''
Created on 01.05.24
by: jokkus
'''
import torch
from torch import nn
import lightning as L

from settings import N_NEURONS_HIDDEN, LR


# define network
class ModelNetwork(L.LightningModule):
    """
    The Network which predicts the next observation / state
    !also gets env as input/init, to determine net size!
    """
    def __init__(self, env):
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
