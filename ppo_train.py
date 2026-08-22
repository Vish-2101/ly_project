from env.dynamic_pricing_env import DynamicPricingEnv, EnvConfig
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback

# ============================
# CREATE ENV FUNCTION
# ============================

def make_env():
    return DynamicPricingEnv(
        data_path="data/dynamic_pricing_noisy.xlsx",
        config=EnvConfig(
            max_steps=200,
            episode_mode="random",
            action_mode="multiplier",
            random_seed=42
        )
    )

# ============================
# TRAINING ENV (WITH NORMALIZATION)
# ============================

env = DummyVecEnv([make_env for _ in range(4)])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# ============================
# EVAL ENV (IMPORTANT: NO REWARD NORMALIZATION)
# ============================

eval_env = DummyVecEnv([make_env])
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)

# Sync stats between train & eval
eval_env.obs_rms = env.obs_rms

# ============================
# EVAL CALLBACK
# ============================

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/",
    log_path="./logs/",
    eval_freq=10000,
    deterministic=True,
    render=False
)

# ============================
# POLICY NETWORK
# ============================

policy_kwargs = dict(net_arch=[256, 256])

# ============================
# PPO MODEL (FAIRLY TUNED)
# ============================

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,

    # ⚖️ Balanced tuning
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=256,

    gamma=0.99,
    gae_lambda=0.95,

    clip_range=0.2,
    ent_coef=0.01,

    vf_coef=0.5,
    max_grad_norm=0.5,

    policy_kwargs=policy_kwargs,
)

# ============================
# TRAIN
# ============================

model.learn(
    total_timesteps=800000,
    callback=eval_callback
)

# ============================
# SAVE MODEL + NORMALIZATION
# ============================

model.save("ppo_dynamic_pricing_improved")
env.save("ppo_vecnormalize.pkl")   # 🔥 IMPORTANT

print("✅ PPO training complete!")