import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os

from collections import defaultdict, deque

from model.model import Model  # Custom model factory for policy (e.g., PPO)

from stable_baselines3.common.callbacks import CallbackList

from callbacks.modelCallback import ModelCallback
from wandb.integration.sb3 import WandbCallback # Optional logging via Weights & Biases

# --- Intrinsic Reward Model ---
class IntrinsicRewardModel(nn.Module):
     """
        A neural network that learns to predict an intrinsic reward based on state-action pairs.
        Used to drive exploration beyond extrinsic environment rewards.
    """
    
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=1)
        return self.net(x)

# --- Bonus Tracker ---
class BonusTracker:
    """
        Keeps track of how often each state has been visited.
        Provides a novelty bonus inversely proportional to visit count.
    """
    def __init__(self):
        self.state_counts = defaultdict(int)

    def update(self, states):
        for s in states:
            key = tuple(np.round(s, 2))
            self.state_counts[key] += 1

    def get_bonus(self, state, done):
        key = tuple(np.round(state, 2))
        if self.state_counts[key] > 0 and done:
            return -10000.0
        elif done:
            return 0
        else:
            return 1.0 / np.sqrt(self.state_counts[key] + 1)


# --- Explorer Model with EXPLORS logic ---
class ExplorerModel:
    """
        Wraps the RL policy model and integrates intrinsic reward modeling, novelty bonus tracking,
        and adaptive policy/intrinsic model updates.
    """
    def __init__(self, env, seed, model_name="PPO", buffer_size=50, stop_on_solution=False, log_dir="./training_run_logs"):
        self.env = env
        self.horizon = 171
        self.buffer = deque(maxlen=buffer_size)  # FIFO buffer
        self.step_counter = 0
        self.stop_on_solution = stop_on_solution

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n

        self.model = Model.create(model_name, "MlpPolicy", n_steps=171*4, batch_size=19,  seed=seed, env=env, verbose=0)

        self.intrinsic_model = IntrinsicRewardModel(obs_dim, act_dim).to("cpu")
        self.optimizer = optim.Adam(self.intrinsic_model.parameters(), lr=1e-3)
        self.bonus_tracker = BonusTracker()

        self.callback = ModelCallback(
            threshold=9.9,
            save_dir=log_dir
        )
        self.callbacks = CallbackList([self.callback])
                                #  WandbCallback(
                                #      gradient_save_freq=100,
                                #      model_save_path= os.path.join(log_dir, "models"),
                                #      verbose=2)])

    def compute_intrinsic_reward(self, obs, action):
        """
            Predicts intrinsic reward for the given state-action pair using the learned model.
        """
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action_tensor = torch.nn.functional.one_hot(torch.tensor(action), num_classes=self.env.action_space.n).float().unsqueeze(0)
        with torch.no_grad():
            return self.intrinsic_model(obs_tensor, action_tensor).item()

    def model_train(self, total_timesteps=1_000_000_000, intrinsic_coeff=0.2, update_policy_every=342, update_intrinsic_every=171):
        """
            Main training loop.
            Alternates between policy learning and intrinsic model training, guided by intrinsic bonuses and rewards.
        """
        obs = self.env.reset()
        trajectory = []

        for step in range(1, total_timesteps + 1):
            action, _ = self.model.predict(obs, deterministic=False)
            new_obs, reward, done, info = self.env.step(action)
            intrinsic_bonus = self.bonus_tracker.get_bonus(new_obs[0],done[0])
            intrinsic_reward = self.compute_intrinsic_reward(obs[0], action[0])
            total_reward = reward[0] + intrinsic_coeff * (intrinsic_reward + intrinsic_bonus)

            self.bonus_tracker.update([new_obs[0]])

            # Collect transition for trajectory buffer
            trajectory.append((obs[0], action[0], total_reward, new_obs[0]))

            # Reset environment if done
            obs = new_obs
            if done[0] or len(trajectory) >= self.horizon:
                self.buffer.append(trajectory)
                trajectory = []
                obs = self.env.reset()

            # --- Policy Update ---
            if step % update_policy_every == 0:
                print(f"Step {step}: PPO Policy Update")
                self.model.learn(total_timesteps=update_policy_every, callback=self.callbacks)

            # --- Intrinsic Reward Model Update ---
            if step % update_intrinsic_every == 0:
                print(f"Step {step}: Intrinsic Model Update")
                self.update_intrinsic_model()

            # --- Stop training on solution found ---
            if self.callback.found_proof and self.stop_on_solution: 
                print("Solution found. Stopping training.")
                break
            # Log
            #print(f"[{step}] R: {reward[0]:.2f} | IntR: {intrinsic_reward:.4f} | Bonus: {intrinsic_bonus:.4f} | Total: {total_reward:.4f}")

    def update_intrinsic_model(self, gamma=0.99, discounted: bool = False):
        """
            Trains the intrinsic reward model using stored trajectories.
            The model tries to regress the cumulative reward from state-action pairs.
        """
        if len(self.buffer) == 0:
            return

        states, actions, returns = [], [], []

        for trajectory in self.buffer:
            G = 0
            discounted_returns = []
            # Trajektorie rückwärts durchlaufen für Discounting
            if discounted:
                for (_, _, reward, _) in reversed(trajectory):
                    G = reward + gamma * G
                    discounted_returns.insert(0, G)
            else:
                discounted_returns = [reward for (_, _, reward, _) in trajectory]

            for idx, (s, a, _, _) in enumerate(trajectory):
                states.append(s)
                actions.append(a)
                returns.append(discounted_returns[idx])

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions)
        actions_one_hot = torch.nn.functional.one_hot(actions, num_classes=self.env.action_space.n).float()
        returns = torch.tensor(returns, dtype=torch.float32)

        self.intrinsic_model.train()
        pred_rewards = self.intrinsic_model(states, actions_one_hot).squeeze()
        loss = nn.MSELoss()(pred_rewards, returns)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        print(f"Intrinsic Reward Model Loss: {loss.item():.4f}")
