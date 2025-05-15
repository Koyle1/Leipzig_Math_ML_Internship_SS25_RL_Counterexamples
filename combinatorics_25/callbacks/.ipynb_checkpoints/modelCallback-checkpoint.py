from stable_baselines3.common.callbacks import BaseCallback
import torch

class modelCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, threshhold : int):
        super().__init__()
        self.save_freq = save_freq
        self.save_path = save_path
        self.threshhold = threshhold

    def _on_step(self) -> bool:
        #To be updated for interrupting when counterexample is found
        if self.n_calls % self.save_freq == 0:
            torch.save(self.model.policy.state_dict(), self.save_path)
            print(f"Reward: {self.locals.get('rewards')}")
            print(f"Dones: {self.locals.get('dones')}")
            print(f"Info: {self.locals.get('infos')}")
            print(f"Saved weights at step {self.n_calls}")
            value_loss = self.model.logger.name_to_value.get("train/value_loss")
            #Change for non sequential enviorment
            
        return True