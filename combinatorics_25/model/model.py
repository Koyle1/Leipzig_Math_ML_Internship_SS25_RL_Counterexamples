#Class implementing the logic for the initialising the sb3 algorithms
#Using a string, the algorithm which will be used for counterexample search cna be determined

from stable_baselines3 import PPO, DQN, A2C # Base algortihms from Stablebaselines3
from stable_baselines3.common.callbacks import CallbackList

from callbacks.modelCallback import ModelCallback # Custom callback used during model training
from wandb.integration.sb3 import WandbCallback

#import base.ModelType

#@ModelType.register("Normal")
class Model:
    #Shared logic across all algorithms
    
    registry = {}

    def model_train(self, save_freq=1000, save_path="model.pth", timesteps=100_000_000, threshhold=0.01):
        
        callback = ModelCallback(save_freq, save_path, threshhold)
        callbacks = CallbackList([callback,
                                 WandbCallback(
                                     gradient_save_freq=100,
                                     model_save_path="./models/",
                                     verbose=2)])
        self.learn(total_timesteps=timesteps, callback=callbacks)

    @classmethod
    def register(cls, name):
        def inner(subclass):
            cls.registry[name.lower()] = subclass
            return subclass
        return inner

    @classmethod
    def create(cls, name, *args, **kwargs):
        subclass = cls.registry.get(name.lower())
        if subclass is None:
            raise ValueError(f"Unknown model name: {name}")
        return subclass(*args, **kwargs)

# Proximity Policy Optimizer (PPO)
@Model.register("PPO")
class Model_PPO(PPO, Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

# Deep Q Network
@Model.register("DQN")
class Model_DQN(DQN, Model):
    def __init__(self, *args, **kwargs):
        # DQN (no DDQN): optimize_memory_usage=False
        super().__init__(*args, optimize_memory_usage=False, **kwargs)

# Double Deep Q Network
@Model.register("DDQN")
class Model_DDQN(DQN, Model):
    def __init__(self, *args, **kwargs):
        # Double DQN: optimize_memory_usage=True
        super().__init__(*args, optimize_memory_usage=True, **kwargs)

# Advantage Actor Critic
@Model.register("A2C")
class Model_A2C(A2C, Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
