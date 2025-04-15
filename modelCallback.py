from stable_baselines3.common.callbacks import BaseCallback
import torch

class modelCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, threshhold : int):
        super().__init__()
        self.save_freq = save_freq
        self.save_path = save_path
        self.threshhold = threshhold

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            torch.save(self.model.policy.state_dict(), self.save_path)
            print(f"Reward: {self.locals.get('rewards')}")
            print(f"Dones: {self.locals.get('dones')}")
            print(f"Info: {self.locals.get('infos')}")
            print(f"Gewichte bei Schritt {self.n_calls} gespeichert.")
            value_loss = self.model.logger.name_to_value.get("train/value_loss")
            if value_loss is not None and value_loss < self.threshhold:
                print(f"Stop, Value_loss: {value_loss}")
                return False
            #Change for non sequential enviorment
            if self.locals.get('rewards') is not None and self.locals.get('rewards').any() > 0:
                print("Counterexample found!")
                return False
        return True