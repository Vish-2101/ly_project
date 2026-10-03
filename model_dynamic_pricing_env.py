"""
model_dynamic_pricing_env.py

A surrogate environment with the exact same Gym interface as
DynamicPricingEnv (same observation_space, action_space, reset(), step()),
so it can be dropped directly into sac_train.py's make_env() in place of
the real environment.

The only thing that changes: reward comes from the trained RewardEnsemble
instead of the real demand-simulator formula. Next-state sampling is
IDENTICAL to the real env (pull another real row from the dataset) — this
is valid because DynamicPricingEnv's next-state is exogenous to the agent's
action in the first place, so there is nothing extra to "learn" there.

This env wraps a real DynamicPricingEnv internally purely to reuse its data
loading, normalization, and state-sampling logic — it never calls the real
env's reward calculation.
"""

from typing import Optional

import numpy as np
import torch

from env.dynamic_pricing_env import DynamicPricingEnv, EnvConfig
from reward_model import RewardEnsemble, load_reward_ensemble


class ModelDynamicPricingEnv(DynamicPricingEnv):
    """
    Subclasses DynamicPricingEnv to reuse all of its data loading,
    normalization, and observation-building logic unchanged. Only the
    reward computation inside step() is overridden.
    """

    def __init__(
        self,
        data_path: str,
        reward_ensemble: Optional[RewardEnsemble] = None,
        reward_ensemble_path: str = "reward_ensemble.pt",
        stochastic_reward: bool = True,
        config: Optional[EnvConfig] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__(data_path=data_path, config=config, render_mode=render_mode)

        if reward_ensemble is not None:
            self.reward_ensemble = reward_ensemble
        else:
            self.reward_ensemble = load_reward_ensemble(reward_ensemble_path)

        self.stochastic_reward = stochastic_reward

    def step(self, action):
        # Grab the row/obs BEFORE stepping — this is the state the reward
        # model needs, matching what collect_transitions.py logged.
        row = self._get_row()
        state_before = self._build_observation(row)

        raw_action = float(np.asarray(action).reshape(-1)[0])

        if self.config.action_mode == "multiplier":
            multiplier = float(
                np.clip(
                    raw_action,
                    self.config.min_multiplier,
                    self.config.max_multiplier,
                )
            )
        else:
            multiplier = raw_action  # not used by the reward model directly

        # ----------------------------------------------------
        # PREDICTED REWARD (this is the model-based part)
        # ----------------------------------------------------

        state_tensor = torch.tensor(
            state_before, dtype=torch.float32
        ).unsqueeze(0)
        action_tensor = torch.tensor(
            [[multiplier]], dtype=torch.float32
        )

        if self.stochastic_reward:
            reward_tensor = self.reward_ensemble.sample(
                state_tensor, action_tensor
            )
        else:
            reward_tensor, _ = self.reward_ensemble.predict(
                state_tensor, action_tensor
            )

        reward = float(reward_tensor.item())

        # ----------------------------------------------------
        # ADVANCE TO NEXT STATE — identical logic to the real env.
        # Next-state is exogenous (next randomly-sampled row), so this
        # part does not need a learned model at all.
        # ----------------------------------------------------

        self.current_idx += 1
        self.current_step += 1

        terminated = self.current_idx >= len(self.episode_indices)
        truncated = False

        if not terminated:
            next_row = self._get_row()
            observation = self._build_observation(next_row)
        else:
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)

        info = {
            "date": row["date"],
            "drug_name": row["drug_name"],
            "category": row["category"],
            "price_multiplier": multiplier,
            "predicted_reward": reward,
        }

        self.last_info = info

        return observation, reward, terminated, truncated, info


if __name__ == "__main__":
    # Quick smoke test — requires reward_ensemble.pt and the dataset to exist.
    env = ModelDynamicPricingEnv(
        data_path="data/dynamic_pricing_noisy.xlsx",
        reward_ensemble_path="reward_ensemble.pt",
        config=EnvConfig(max_steps=10, episode_mode="random", action_mode="multiplier"),
    )
    obs, info = env.reset(seed=42)
    print("Observation shape:", obs.shape)
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"reward={reward:.3f} info={info}")
        if terminated or truncated:
            break
