import os
import torch
import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback

class ModelCallback(BaseCallback):
    def __init__(
        self,
        save_freq: int,
        save_path: str,
        threshold: float = None,
        verbose: int = 0,
        state_save_dir: str = "./saved_states"
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.threshold = threshold
        self.episode_count = 0
        self.found_proof = False
        self.state_save_dir = state_save_dir
        os.makedirs(self.state_save_dir, exist_ok=True)
        self.best_graph = -10_000

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        observations = self.locals.get("obs")

        for i, info in enumerate(infos):
            if "episode" in info:
                self.episode_count += 1
                reward = info["episode"]["r"]
                length = info["episode"].get("l", None)
                if reward > self.best_graph:
                    self.best_graph = reward

                # Log to wandb
                wandb.log({
                    "episode_reward": reward,
                    "episode_length": length,
                    "best_graph": self.best_graph
                }, step=self.episode_count)

                #Save state if reward is positive
                if reward > 0 and observations is not None:
                    state = observations[i] if isinstance(observations, (list, np.ndarray)) else observations
                    state_path = os.path.join(
                        self.state_save_dir,
                        f"state_ep{self.episode_count}_rew{reward:.2f}.npy"
                    )
                    np.save(state_path, state)
                    if self.verbose:
                        print(f"[Callback] Saved state with reward {reward:.2f} to {state_path}")

                #Stop training if reward meets threshold
                if self.threshold is not None and reward >= self.threshold:
                    print(f"\n[Callback] Counterexample / proof found! Reward = {reward}")
                    print(f"[Callback] Saving model to {self.save_path}")
                    torch.save(self.model.policy.state_dict(), self.save_path)
                    self.found_proof = True
                    return False

        #Periodic model saving
        if self.n_calls % self.save_freq == 0:
            torch.save(self.model.policy.state_dict(), self.save_path)
            print(f"[Callback] Periodic save at step {self.n_calls}")
            print(f"[Callback] Current episode: {self.episode_count}")

        return True

    def _on_training_end(self) -> None:
        print(f"\n[Callback] Training ended.")
        print(f"[Callback] Total episodes: {self.episode_count}")
        if self.found_proof:
            print(f"[Callback] Training stopped early due to counterexample / proof.")
        else:
            print(f"[Callback] No proof found during training.")
