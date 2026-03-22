from env.dynamic_pricing_env import DynamicPricingEnv, EnvConfig
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# ✅ Create environment (THIS generates data dynamically)
def make_env():
    return DynamicPricingEnv(
        data_path="data/dynamic_pricing.csv.xlsx",
        config=EnvConfig(
            max_steps=200,
            episode_mode="random",   # 🔥 important
            action_mode="multiplier",
            random_seed=42
        )
    )

env = DummyVecEnv([make_env])

# 🚀 PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=64,
    gamma=0.99,
)

# ✅ Train (this generates its own data)
model.learn(total_timesteps=500000)

# ✅ Save model
model.save("ppo_dynamic_pricing")

print("✅ RL training complete!")