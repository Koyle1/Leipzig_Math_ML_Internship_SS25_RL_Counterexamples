from stable_baselines3.common.callbacks import BaseCallback
import torch

class modelCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, threshhold : int):
        super().__init__()
        self.save_freq = save_freq
        self.save_path = save_path
        self.threshhold = threshhold
        self.episode_count = 0

    def _on_step(self) -> bool:
        #To be updated for interrupting when counterexample is found
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_count += 1
        
        if self.n_calls % self.save_freq == 0:
            torch.save(self.model.policy.state_dict(), self.save_path)
            print(f"Saved weights at step {self.n_calls}")
            value_loss = self.model.logger.name_to_value.get("train/value_loss")
            #Change for non sequential enviorment
            
        return True

    def _on_training_end(self) -> None:
        print(f"\n[Callback] Total episodes completed: {self.episode_count}")