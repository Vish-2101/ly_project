"""
collect_transitions.py

Runs random-action rollouts through the REAL DynamicPricingEnv and logs
(state, action, reward) tuples for training the reward model.

This mirrors main.py's structure, with one important addition: main.py only
saves the `info` dict, which does not contain the raw 13-dim observation
vector the reward model needs as input. This script saves that vector too.

Random actions are used (not a trained policy) to keep memory usage low —
no model loading, no VecNormalize, no VecEnv wrapping. Just a single
DynamicPricingEnv instance, same as main.py.
"""

from env.dynamic_pricing_env import DynamicPricingEnv, EnvConfig
import numpy as np
import pandas as pd


def collect(
    data_path: str = "data/dynamic_pricing_noisy.xlsx",
    num_episodes: int = 200,
    max_steps: int = 2000,
    random_seed: int = 42,
    output_path: str = "reward_model_transitions.csv",
):
    env = DynamicPricingEnv(
        data_path=data_path,
        config=EnvConfig(
            max_steps=max_steps,
            episode_mode="random",
            action_mode="multiplier",
            random_seed=random_seed,
        ),
    )

    obs_dim = env.observation_space.shape[0]
    rows = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False

        while not done:
            action = env.action_space.sample()

            # obs BEFORE the step is the state the action was taken in —
            # this is what the reward model needs to learn from.
            state_before = obs.copy()

            obs, reward, terminated, truncated, info = env.step(action)

            row = {f"obs_{i}": float(state_before[i]) for i in range(obs_dim)}
            row["action"] = float(np.asarray(action).reshape(-1)[0])
            row["reward"] = float(reward)
            row["episode"] = ep

            rows.append(row)

            done = terminated or truncated

        if (ep + 1) % 20 == 0:
            print(f"Episode {ep + 1}/{num_episodes} done")

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    print(f"\nSaved {len(df)} transitions to {output_path}")
    print(f"Observation dim: {obs_dim}, action dim: 1")

    return df


if __name__ == "__main__":
    collect()
