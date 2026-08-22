from env.dynamic_pricing_env import DynamicPricingEnv, EnvConfig
from stable_baselines3 import SAC
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
# TRAIN ENV (NORMALIZED)
# ============================

env = DummyVecEnv([make_env for _ in range(4)])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# ============================
# EVAL ENV (IMPORTANT)
# ============================

eval_env = DummyVecEnv([make_env])
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)

# 🔥 CRITICAL: sync normalization stats
eval_env.obs_rms = env.obs_rms

# ============================
# CALLBACK
# ============================

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs_sac/",
    log_path="./logs_sac/",
    eval_freq=10000,
    deterministic=True,
    render=False
)

# ============================
# POLICY
# ============================

policy_kwargs = dict(net_arch=[256, 256])

# ============================
# SAC MODEL
# ============================

model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    buffer_size=1_000_000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    ent_coef="auto",
    train_freq=1,
    gradient_steps=1,
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

model.save("sac_dynamic_pricing")
env.save("sac_vecnormalize.pkl")   # 🔥 VERY IMPORTANT

print("✅ SAC training complete!")