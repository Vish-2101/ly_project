from env.dynamic_pricing_env import DynamicPricingEnv, EnvConfig
import pandas as pd

env = DynamicPricingEnv(
    data_path="data/dynamic_pricing.csv.xlsx",
    config=EnvConfig(max_steps=10, action_mode="multiplier", random_seed=42)
)

obs, info = env.reset()
done = False

results = []

while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    results.append(info)

    done = terminated or truncated

# Save to CSV
df = pd.DataFrame(results)
df.to_csv("simulation_output.csv", index=False)

print("✅ Saved simulation_output.csv")