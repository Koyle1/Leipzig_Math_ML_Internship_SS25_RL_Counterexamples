import gymnasium as gym

from stable_baselines3 import PPO
import torch
from modelCallback import modelCallback 

class model(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_weights(self, source: str = "ppo_weights.pth"):
        #weights has to point to a file.pth (source)
        self.policy.load_state_dict(torch.load(source))

    def model_train(self, save_freq: int =1000, save_path: str ="ppo_weights.pth", timesteps: int = 1000000000000, threshhold : int = 0.01):
        callback = modelCallback(save_freq, save_path, threshhold)
        self.learn(timesteps, callback=callback)

    def model_evaluate():
        pass

