from stable_baselines3.common.callbacks import BaseCallback
import torch

class ModelCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, threshold: float = None, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.threshold = threshold
        self.episode_count = 0
        self.found_proof = False

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_count += 1
                reward = info["episode"]["r"]

                if self.threshold is not None and reward >= self.threshold:
                    print(f"\n[Callback] ✅ Counterexample / proof found! Reward = {reward}")
                    print(f"[Callback] 💾 Saving final model to {self.save_path}")
                    torch.save(self.model.policy.state_dict(), self.save_path)
                    self.found_proof = True
                    return False  # Stops training

        if self.n_calls % self.save_freq == 0:
            torch.save(self.model.policy.state_dict(), self.save_path)
            print(f"[Callback] 💾 Periodic save at step {self.n_calls}")
            value_loss = self.model.logger.name_to_value.get("train/value_loss")
            if value_loss is not None:
                print(f"[Callback] ℹ️ Value loss: {value_loss:.4f}")
        return True

    def _on_training_end(self) -> None:
        print(f"\n[Callback] 🏁 Training ended.")
        print(f"[Callback] 📊 Total episodes: {self.episode_count}")
        if self.found_proof:
            print(f"[Callback] 🎉 Training stopped early due to counterexample / proof.")
        else:
            print(f"[Callback] 💤 No proof found during training.")
