import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from stable_baselines3 import PPO
from collections import defaultdict, deque

from model.model import Model  # Custom model factory

# --- Intrinsic Reward Model ---
class IntrinsicRewardModel(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=1)
        return self.net(x)

# --- Bonus Tracker ---
class BonusTracker:
    def __init__(self):
        self.state_counts = defaultdict(int)

    def update(self, states):
        for s in states:
            key = tuple(np.round(s, 2))
            self.state_counts[key] += 1

    def get_bonus(self, state):
        key = tuple(np.round(state, 2))
        if self.state_counts[key] > 0:
            return -10000.0
        else:
            return 0


# --- Explorer Model with EXPLORS logic ---
class ExplorerModel:
    def __init__(self, env, seed, model_name="PPO", buffer_size=50, horizon=128):
        self.env = env
        self.horizon = env.MYN
        self.buffer = deque(maxlen=buffer_size)  # FIFO buffer
        self.step_counter = 0

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n

        self.model = Model.create(model_name, "MlpPolicy", seed=seed, env=env, verbose=0)

        self.intrinsic_model = IntrinsicRewardModel(obs_dim, act_dim).to("cpu")
        self.optimizer = optim.Adam(self.intrinsic_model.parameters(), lr=1e-3)
        self.bonus_tracker = BonusTracker()

    def compute_intrinsic_reward(self, obs, action):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action_tensor = torch.nn.functional.one_hot(torch.tensor(action), num_classes=self.env.action_space.n).float().unsqueeze(0)
        with torch.no_grad():
            return self.intrinsic_model(obs_tensor, action_tensor).item()

    def train(self, total_timesteps=10000, intrinsic_coeff=100, update_policy_every=512, update_intrinsic_every=256):
        obs = self.env.reset()
        trajectory = []

        for step in range(1, total_timesteps + 1):
            action, _ = self.model.predict(obs, deterministic=False)
            new_obs, reward, done, info = self.env.step(action)
            intrinsic_bonus = 0
            if done[0] or len(trajectory) >= self.horizon:
                intrinsic_bonus = self.bonus_tracker.get_bonus(new_obs[0])
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
                self.model.learn(total_timesteps=update_policy_every)

            # --- Intrinsic Reward Model Update ---
            if step % update_intrinsic_every == 0:
                print(f"Step {step}: Intrinsic Model Update")
                self.update_intrinsic_model()

            # Log
            print(f"[{step}] R: {reward[0]:.2f} | IntR: {intrinsic_reward:.4f} | Bonus: {intrinsic_bonus:.4f} | Total: {total_reward:.4f}")

    def update_intrinsic_model(self):
        if len(self.buffer) == 0:
            return

        batch = []
        for traj in self.buffer:
            batch.extend(traj)

        states, actions, rewards, next_states = zip(*batch)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions)
        actions_one_hot = torch.nn.functional.one_hot(actions, num_classes=self.env.action_space.n).float()
        rewards = torch.tensor(rewards, dtype=torch.float32)

        self.intrinsic_model.train()
        pred_rewards = self.intrinsic_model(states, actions_one_hot).squeeze()
        loss = nn.MSELoss()(pred_rewards, rewards)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        print(f"Intrinsic Reward Model Loss: {loss.item():.4f}")
