from env.dynamic_pricing_env import DynamicPricingEnv, EnvConfig
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback

# ============================
# CREATE ENV FUNCTION
# ============================

def make_env():
    return DynamicPricingEnv(
        data_path="data/dynamic_pricing.csv.xlsx",
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

# 🔥 CRITICAL: sync stats
eval_env.obs_rms = env.obs_rms

# ============================
# CALLBACK
# ============================

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs_a2c/",
    log_path="./logs_a2c/",
    eval_freq=10000,
    deterministic=True,
    render=False
)

# ============================
# POLICY
# ============================

policy_kwargs = dict(net_arch=[256, 256])

# ============================
# A2C MODEL
# ============================

model = A2C(
    "MlpPolicy",
    env,
    verbose=1,

    learning_rate=7e-4,
    n_steps=5,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    vf_coef=0.5,

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

model.save("a2c_dynamic_pricing")
env.save("a2c_vecnormalize.pkl")   # 🔥 IMPORTANT

print("✅ A2C training complete!")