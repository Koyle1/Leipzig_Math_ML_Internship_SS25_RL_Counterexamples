import gymnasium as gym
from stable_baselines3 import PPO, DQN, A2C
import torch
from modelCallback import modelCallback

class BaseModel:
    def load_weights(self, source: str):
        self.policy.load_state_dict(torch.load(source))

    def model_train(self, save_freq=1000, save_path="model.pth", timesteps=1_000_000, threshhold=0.01):
        callback = modelCallback(save_freq, save_path, threshhold)
        self.learn(total_timesteps=timesteps, callback=callback)


class Model_PPO(PPO, BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Model_DQN(DQN, BaseModel):
    def __init__(self, *args, **kwargs):
        # DQN (no DDQN): optimize_memory_usage=False
        super().__init__(*args, optimize_memory_usage=False, **kwargs)


class Model_DDQN(DQN, BaseModel):
    def __init__(self, *args, **kwargs):
        # Double DQN: optimize_memory_usage=True
        super().__init__(*args, optimize_memory_usage=True, **kwargs)


class Model_A2C(A2C, BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
