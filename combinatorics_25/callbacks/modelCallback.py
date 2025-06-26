import os
import torch
import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback

class ModelCallback(BaseCallback):
    def __init__(
        self,
        save_freq: int = 1000,
        threshold: float = None,
        verbose: int = 0,
        save_dir: str = "./training_run_logs"
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.threshold = threshold
        self.episode_count = 0
        self.found_proof = False
        model_dir = os.path.join(save_dir, "models")
        self.state_dir = os.path.join(save_dir, "states")
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(self.state_dir, exist_ok=True)
        self.model_path = os.path.join(
                        model_dir,
                        "model.pth"
                    )

        self.best_graph_cumulative = -10_000
        self.best_graph_final_step = -10_000

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        observations = self.locals.get("new_obs")

        for i, info in enumerate(infos):
            if "episode" in info:
                self.episode_count += 1

                # Cumulative episode reward from SB3
                cumulative_reward = info["episode"]["r"]
                # Final step reward explicitly added by environment
                final_step_reward = info.get("final_step_reward", None)
                final_graph_reward = info.get("final_graph_reward", None)
                alpha = info.get("alpha", 0.0)
                beta = info.get("beta", 0.0)
                true_reward = info.get("true_reward", 0.0)

                length = info["episode"].get("l", None)

                # Track best cumulative reward
                if cumulative_reward > self.best_graph_cumulative:
                    self.best_graph_cumulative = cumulative_reward

                # Track best final step reward
                if final_step_reward is not None and final_step_reward > self.best_graph_final_step:
                    self.best_graph_final_step = final_step_reward

                # Log metrics to wandb
                log_dict = {
                    "episode_reward_cumulative": cumulative_reward,
                    "episode_length": length,
                    "best_graph_cumulative": self.best_graph_cumulative,
                    "best_graph_final_step": self.best_graph_final_step,
                }
                if final_step_reward is not None:
                    log_dict["episode_reward_final_step"] = final_step_reward

                if final_graph_reward is not None:
                    log_dict["episode_reward_final_step"] = final_step_reward

                if alpha > 0.0:
                    log_dict["alpha"] = alpha

                if beta > 0.0:
                    log_dict["beta"] = beta

                log_dict["true_reward"] = true_reward

                

                wandb.log(log_dict, step=self.episode_count)

                # Save state if final step reward is positive and best so far
                if (
                    final_step_reward is not None 
                    and final_step_reward > 0 
                    and final_step_reward >= self.best_graph_final_step
                    and observations is not None
                ):
                    state = observations[i] if isinstance(observations, (list, np.ndarray)) else observations
                    state_path = os.path.join(
                        self.state_dir,
                        f"state_ep{self.episode_count}_finalrew{final_step_reward:.2f}.npy"
                    )
                    np.save(state_path, state)
                    if self.verbose:
                        print(f"[Callback] Saved state with final step reward {final_step_reward:.2f} to {state_path}")

                # Stop training if threshold met on final step reward
                if self.threshold is not None and final_step_reward is not None and final_step_reward >= self.threshold:
                    print(f"\n[Callback] Counterexample / proof found! Final step reward = {final_step_reward}")
                    print(f"[Callback] Saving model to {self.model_path}")

                    torch.save(self.model.policy.state_dict(), self.model_path)
                    self.found_proof = True
                    return False

        # Periodic model saving
        if self.n_calls % self.save_freq == 0:
            torch.save(self.model.policy.state_dict(), self.model_path)
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
