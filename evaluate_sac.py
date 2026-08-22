from env.dynamic_pricing_env import DynamicPricingEnv, EnvConfig
from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import pandas as pd

# ============================
# ENV
# ============================

def make_env():
    return DynamicPricingEnv(
        data_path="data/dynamic_pricing_noisy.xlsx",
        config=EnvConfig(
            max_steps=50000,
            episode_mode="random",
            action_mode="multiplier"
        )
    )

env = DummyVecEnv([make_env])

# 🔥 LOAD CORRECT NORMALIZATION
env = VecNormalize.load("sac_vecnormalize.pkl", env)

env.training = False
env.norm_reward = False

# ============================
# LOAD MODEL
# ============================

# model = A2C.load("a2c_dynamic_pricing")
# model = PPO.load("ppo_dynamic_pricing_improved")
model = SAC.load("sac_dynamic_pricing")

# ============================
# EVALUATION
# ============================

num_episodes = 10
all_results = []

for ep in range(num_episodes):
    obs = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        episode_reward += reward[0]          # ✅ FIXED
        all_results.append(info[0])          # ✅ FIXED

        done = done[0]

    print(f"Episode {ep+1} Reward: {episode_reward}")

# ============================
# SAVE
# ============================

df = pd.DataFrame(all_results)
df.to_csv("sac_rl_simulation_output.csv", index=False)

print("✅ RL simulation saved!")