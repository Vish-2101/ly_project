from env.dynamic_pricing_env import DynamicPricingEnv, EnvConfig
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

# ============================
# CREATE ENV FUNCTION
# ============================

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

# ============================
# VECTORIZED ENV (FASTER TRAINING)
# ============================

env = DummyVecEnv([make_env for _ in range(4)])   # 🔥 4 parallel envs

# ============================
# EVALUATION ENV
# ============================

eval_env = DummyVecEnv([make_env])

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/",
    log_path="./logs/",
    eval_freq=10000,
    deterministic=True,
    render=False
)

# ============================
# POLICY NETWORK (DEEPER)
# ============================

policy_kwargs = dict(net_arch=[256, 256])

# ============================
# PPO MODEL (IMPROVED)
# ============================

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=1e-4,      # 🔽 better convergence
    n_steps=2048,            # 🔼 stability
    batch_size=128,
    gamma=0.995,             # 🔼 long-term rewards
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,           # 🔥 exploration
    vf_coef=0.5,
    policy_kwargs=policy_kwargs,
)

# ============================
# TRAIN MODEL
# ============================

model.learn(
    total_timesteps=800000,   # 🔥 increased training
    callback=eval_callback
)

# ============================
# SAVE MODEL
# ============================

model.save("ppo_dynamic_pricing_improved")

print("✅ Improved PPO training complete!")