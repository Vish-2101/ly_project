from env.dynamic_pricing_env import DynamicPricingEnv, EnvConfig
from stable_baselines3 import PPO
import pandas as pd

env = DynamicPricingEnv(
    data_path="data/dynamic_pricing.csv.xlsx",
    config=EnvConfig(
        max_steps=50000,
        episode_mode="random",
        action_mode="multiplier"
    )
)

model = PPO.load("ppo_dynamic_pricing")

results = []

obs, info = env.reset()
done = False

while not done:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    results.append(info)

    done = terminated or truncated

# ✅ Save RL-generated data
df = pd.DataFrame(results)
df.to_csv("rl_simulation_output.csv", index=False)

print("✅ RL simulation saved!")