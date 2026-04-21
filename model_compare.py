from env.dynamic_pricing_env import DynamicPricingEnv, EnvConfig
from stable_baselines3 import PPO, SAC, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd

# ✅ Create environment
def make_env():
    return DynamicPricingEnv(
        data_path="data/dynamic_pricing.csv.xlsx",
        config=EnvConfig(
            max_steps=500,
            episode_mode="random",
            action_mode="multiplier",
            random_seed=42
        )
    )

env = DummyVecEnv([make_env])

# 🚀 Models dictionary
models = {
    "PPO": PPO("MlpPolicy", env, verbose=0),
    "SAC": SAC("MlpPolicy", env, verbose=0),
    "A2C": A2C("MlpPolicy", env, verbose=0),
    
}

results = {}

# ============================
# TRAIN + EVALUATE EACH MODEL
# ============================

for name, model in models.items():
    print(f"\n🚀 Training {name}...")

    model.learn(total_timesteps=50000)

    # Evaluate
    test_env = make_env()
    obs, info = test_env.reset()
    done = False

    profits = []

    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = test_env.step(action)

        profits.append(info["profit"])
        done = terminated or truncated

    results[name] = {
        "avg_profit": sum(profits) / len(profits),
        "total_profit": sum(profits)
    }

# ============================
# SAVE RESULTS
# ============================

df = pd.DataFrame(results).T
df.to_csv("rl_model_comparison.csv")

print("\n📊 Comparison Results:")
print(df)