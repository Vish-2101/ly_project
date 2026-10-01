"""
reward_model.py

A small ensemble of MLPs that predicts reward (mean + variance) from
(state, action). This is the entire "world model" needed for this project:
DynamicPricingEnv's next-state transitions are exogenous (the next row is
sampled independently of the chosen action), so only the reward function
needs to be learned.

Each ensemble member outputs (mean, log_var) and is trained with Gaussian
negative log-likelihood, so the ensemble captures the uncertainty coming
from demand_noise and fx_shock rather than just regressing to the average
reward.

Usage:
    python reward_model.py
(trains on reward_model_transitions.csv, saves reward_ensemble.pt)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class RewardNet(nn.Module):
    """Single ensemble member: (state, action) -> (mean, log_var)."""

    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, 1)
        self.logvar_head = nn.Linear(hidden, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        features = self.net(x)
        mean = self.mean_head(features)
        # clamp for numerical stability
        log_var = torch.clamp(self.logvar_head(features), -10.0, 5.0)
        return mean, log_var


class RewardEnsemble(nn.Module):
    """Ensemble of RewardNets. Prediction = mean/variance across members."""

    def __init__(self, obs_dim: int, num_members: int = 5, hidden: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_members = num_members
        self.members = nn.ModuleList(
            [RewardNet(obs_dim, hidden) for _ in range(num_members)]
        )

    def forward(self, state, action):
        means, log_vars = [], []
        for member in self.members:
            m, lv = member(state, action)
            means.append(m)
            log_vars.append(lv)
        means = torch.stack(means, dim=0)       # (num_members, batch, 1)
        log_vars = torch.stack(log_vars, dim=0)  # (num_members, batch, 1)
        return means, log_vars

    @torch.no_grad()
    def predict(self, state, action):
        """
        Returns (reward_mean, reward_std) aggregated across the ensemble.
        reward_std reflects both learned per-member uncertainty AND
        disagreement between members (epistemic + aleatoric combined).
        """
        means, log_vars = self.forward(state, action)
        ensemble_mean = means.mean(dim=0)  # (batch, 1)

        aleatoric_var = torch.exp(log_vars).mean(dim=0)
        epistemic_var = means.var(dim=0)
        total_std = torch.sqrt(aleatoric_var + epistemic_var + 1e-8)

        return ensemble_mean, total_std

    def sample(self, state, action):
        """Sample a reward, useful when the surrogate env wants a stochastic
        reward rather than the deterministic mean."""
        mean, std = self.predict(state, action)
        return mean + std * torch.randn_like(mean)


def gaussian_nll_loss(mean, log_var, target):
    inv_var = torch.exp(-log_var)
    return (0.5 * (target - mean) ** 2 * inv_var + 0.5 * log_var).mean()


def train_reward_ensemble(
    csv_path: str = "reward_model_transitions.csv",
    output_path: str = "reward_ensemble.pt",
    obs_dim: int = 13,
    num_members: int = 5,
    hidden: int = 128,
    epochs: int = 30,
    batch_size: int = 512,
    lr: float = 1e-3,
    val_fraction: float = 0.1,
    seed: int = 42,
    max_rows: int = 100_000,
    device: str = None,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    df = pd.read_csv(csv_path)

    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
        print(f"Subsampled to {max_rows} rows (from a larger file)")

    obs_cols = [f"obs_{i}" for i in range(obs_dim)]
    states = df[obs_cols].values.astype(np.float32)
    actions = df["action"].values.astype(np.float32).reshape(-1, 1)
    rewards = df["reward"].values.astype(np.float32).reshape(-1, 1)

    n = len(df)
    idx = np.random.permutation(n)
    n_val = int(n * val_fraction)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    def to_tensor(arr):
        return torch.tensor(arr, dtype=torch.float32)

    train_ds = TensorDataset(
        to_tensor(states[train_idx]),
        to_tensor(actions[train_idx]),
        to_tensor(rewards[train_idx]),
    )
    val_states = to_tensor(states[val_idx])
    val_actions = to_tensor(actions[val_idx])
    val_rewards = to_tensor(rewards[val_idx])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        pin_memory=(device == "cuda"),
    )

    ensemble = RewardEnsemble(obs_dim=obs_dim, num_members=num_members, hidden=hidden)
    ensemble.to(device)
    val_states = val_states.to(device)
    val_actions = val_actions.to(device)
    val_rewards = val_rewards.to(device)
    optimizer = torch.optim.Adam(ensemble.parameters(), lr=lr)

    for epoch in range(epochs):
        ensemble.train()
        epoch_loss = 0.0

        for state_b, action_b, reward_b in train_loader:
            state_b = state_b.to(device, non_blocking=True)
            action_b = action_b.to(device, non_blocking=True)
            reward_b = reward_b.to(device, non_blocking=True)

            means, log_vars = ensemble(state_b, action_b)
            # each member sees the same batch; sum losses across members
            loss = sum(
                gaussian_nll_loss(means[i], log_vars[i], reward_b)
                for i in range(num_members)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            ensemble.eval()
            with torch.no_grad():
                pred_mean, _ = ensemble.predict(val_states, val_actions)
                val_mae = (pred_mean - val_rewards).abs().mean().item()
            print(
                f"Epoch {epoch + 1}/{epochs} | train_loss={epoch_loss:.3f} | "
                f"val_MAE={val_mae:.3f}"
            )

    torch.save(
        {
            "state_dict": ensemble.state_dict(),
            "obs_dim": obs_dim,
            "num_members": num_members,
            "hidden": hidden,
        },
        output_path,
    )
    print(f"\nSaved reward ensemble to {output_path}")

    return ensemble


def load_reward_ensemble(path: str = "reward_ensemble.pt") -> RewardEnsemble:
    checkpoint = torch.load(path, map_location="cpu")
    ensemble = RewardEnsemble(
        obs_dim=checkpoint["obs_dim"],
        num_members=checkpoint["num_members"],
        hidden=checkpoint["hidden"],
    )
    ensemble.load_state_dict(checkpoint["state_dict"])
    ensemble.eval()
    return ensemble


if __name__ == "__main__":
    train_reward_ensemble()