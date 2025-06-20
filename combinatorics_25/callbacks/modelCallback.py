import os
import torch
import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from collections import deque                       


class ModelCallback(BaseCallback):
    def __init__(
        self,
        save_freq: int = 1000,
        save_path: str = "model.pth",
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

        self.best_graph_cumulative = -10_000
        self.best_graph_final_step = -10_000

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        observations = self.locals.get("obs")

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
                        self.state_save_dir,
                        f"state_ep{self.episode_count}_finalrew{final_step_reward:.2f}.npy"
                    )
                    np.save(state_path, state)
                    if self.verbose:
                        print(f"[Callback] Saved state with final step reward {final_step_reward:.2f} to {state_path}")

                # Stop training if threshold met on final step reward
                if self.threshold is not None and final_step_reward is not None and final_step_reward >= self.threshold:
                    print(f"\n[Callback] Counterexample / proof found! Final step reward = {final_step_reward}")
                    print(f"[Callback] Saving model to {self.save_path}")
                    torch.save(self.model.policy.state_dict(), self.save_path)
                    self.found_proof = True
                    return False

        # Periodic model saving
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




class LagrangeMultiplierCallback(ModelCallback):
    """
    Adaptive Lagrange multiplier (dual‑gradient ascent) + all bookkeeping of ModelCallback.
    Keeps the constraint  E[P] ≤ p_max  satisfied online.
    """

    def __init__(
        self,
        p_max: float = 0.05,
        lr_lambda: float = 0.05,
        window: int = 1024,
        lambda_update_freq: int = 1_000,
        # --- pass‑through kwargs for ModelCallback ---
        save_freq: int = 1000,
        save_path: str = "model.pth",
        threshold: float | None = None,
        verbose: int = 0,
        state_save_dir: str = "./saved_states",
    ):
        # let the parent initialise everything it needs
        super().__init__(
            save_freq=save_freq,
            save_path=save_path,
            threshold=threshold,
            verbose=verbose,
            state_save_dir=state_save_dir,
        )

        # parameters specific to the dual update
        self.p_max               = p_max
        self.lr_lambda           = lr_lambda
        self.lambda_update_freq  = lambda_update_freq
        self.running_P           = deque(maxlen=window)  # short moving window

    # --------------------------------------------------------------------- #
    # Every env step:   1) update λ  2) run the usual ModelCallback logic
    # --------------------------------------------------------------------- #
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        # ------------------------------------------------------------
        # ❶  Collect P only when the node is finished AND the graph is
        #    disconnected (signalled by lambda_2 == 0 or an explicit flag).
        #    Otherwise treat P as zero for the dual update.
        # ------------------------------------------------------------
        for info in infos:
            if info.get("end_of_note"):                     # finished node
                # If your env already gives a boolean flag, use it:
                #   disconnected = not info.get("is_connected", True)
                # Otherwise infer from λ₂ :
                disconnected = info.get("lambda_2", 0.0) < 1e-12

                if disconnected and "P" in info:
                    self.running_P.append(info["P"])
                else:
                    self.running_P.append(0.0)              # no penalty

        # ------------------------------------------------------------
        # ❷  Dual‑gradient ascent on λ (unchanged)
        # ------------------------------------------------------------
        if self.n_calls % self.lambda_update_freq == 0 and self.running_P:
            mean_P = float(np.mean(self.running_P))

            envs = getattr(self, "training_env", None)
            if envs is not None:
                cur_lambda = envs.envs[0].unwrapped.lambda_dual
                new_lambda = max(
                    0.0,
                    cur_lambda + self.lr_lambda * (mean_P - self.p_max),
                )
                for e in envs.envs:
                    e.unwrapped.lambda_dual = new_lambda

                if self.verbose:
                    print(
                        f"[Lagrange] step={self.n_calls:>8}  "
                        f"mean_P={mean_P:.4f}  λ→{new_lambda:.4f}"
                    )

        # ------------------------------------------------------------
        # ❸  Call parent callback and propagate stop signal
        # ------------------------------------------------------------
        continue_training = super()._on_step()
        return continue_training
