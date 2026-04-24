from env.dynamic_pricing_env import DynamicPricingEnv, EnvConfig
import pandas as pd

# ✅ Create environment
env = DynamicPricingEnv(
    data_path="data/dynamic_pricing.csv.xlsx",
    config=EnvConfig(
        max_steps=2000,
        episode_mode="random",   # 👈 enables unlimited sampling
        action_mode="multiplier",
        random_seed=42
    )
)

results = []

num_episodes = 200   # 👈 change this to control total data

for ep in range(num_episodes):
    obs, info = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()   # random policy (replace with RL later)
        obs, reward, terminated, truncated, info = env.step(action)

        # ✅ store extra info
        info["episode"] = ep

        results.append(info)

        done = terminated or truncated

    print(f"✅ Episode {ep+1} completed")

# ✅ Convert to DataFrame
df = pd.DataFrame(results)

# ✅ Save CSV
df.to_csv("simulation_output.csv", index=False)

print("\n🎉 Simulation complete!")
print(f"Total rows generated: {len(df)}")
print("Saved to simulation_output.csv")