import torch
import torch.nn as nn
import torch.nn.functional as F

class SurrogateRewardModel(nn.Module):
    def __init__(self, input_dim=OBSERVATION_SPACE, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output: scalar reward
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # shape: (batch,)