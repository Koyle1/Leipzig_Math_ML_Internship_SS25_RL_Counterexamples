import gym
import numpy as np
import torch
import torch.nn as nn

import model.Model
import base.ModelType

from callbacks.modelCallback import ModelCallback
from wandb.integration.sb3 import WandbCallback

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from collections import deque, defaultdict

@ModelType.register("ExploRS")
class ExploRSmodel(Model):
    def __init__(model: str = "PPO"):
        self.model = Model.create(PPO)

        #initialise the intrinsic model
        intrinsic_model = IntrinsicRewardModel(obs_dim, 1).to("cpu")
        optimizer = torch.optim.Adam(intrinsic_model.parameters(), lr=1e-3)
        bonus_tracker = BonusTracker()

        #initialise Bonus Tracker
        bonus_tracker = BonusTracker()

    @classmethod 
    def create(cls, name, *args, **kwargs):
        return ExploRSmodel(model=name, *args, **kwargs)

    def train():
        pass

# Define Intrinsic Reward Model
class IntrinsicRewardModel(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=1)
        return self.net(x)

# Bonus tracker: state visit counts 
class BonusTracker:
    def __init__(self):
        self.state_counts = defaultdict(int)

    def update(self, states):
        for s in states:
            key = tuple(np.round(s, 2))  # Discretize to prevent float mismatch
            self.state_counts[key] += 1

    def get_bonus(self, state):
        key = tuple(np.round(state, 2))
        return 1.0 / np.sqrt(self.state_counts[key] + 1)