"""
sac_train_mbpo.py

Mirrors sac_train.py almost line-for-line. The ONLY difference: make_env()
builds a ModelDynamicPricingEnv (reward comes from the learned ensemble)
instead of the real DynamicPricingEnv.

Everything else — SAC hyperparameters, VecNormalize setup, EvalCallback —
is intentionally kept identical to sac_train.py, so that if the results
differ, the difference can be attributed to the reward source and not to
some other confounding change in training setup.

IMPORTANT: the EvalCallback below evaluates on the REAL environment, not
the surrogate one. This matters — if eval also used the surrogate's
predicted rewards, a good score would only prove the policy learned to
exploit the reward model, not that it actually performs well in the real
market simulation. Evaluating on the real env is what makes the final
comparison against PPO/SAC/A2C fair.

Run collect_transitions.py and reward_model.py before this script.
"""

from env.dynamic_pricing_env import DynamicPricingEnv, EnvConfig
from model_dynamic_pricing_env import ModelDynamicPricingEnv
from reward_model import load_reward_ensemble
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback

# ============================
# SHARED REWARD ENSEMBLE
# (load once, share across parallel envs — avoids reloading the model
#  4 times and keeps memory usage down)
# ============================

reward_ensemble = load_reward_ensemble("reward_ensemble.pt")

# ============================
# TRAIN ENV FUNCTION (MODEL-BASED)
# ============================

def make_model_env():
    return ModelDynamicPricingEnv(
        data_path="data/dynamic_pricing_noisy.xlsx",
        reward_ensemble=reward_ensemble,
        stochastic_reward=True,
        config=EnvConfig(
            max_steps=200,
            episode_mode="random",
            action_mode="multiplier",
            random_seed=42,
        ),
    )

# ============================
# EVAL ENV FUNCTION (REAL — unchanged from sac_train.py)
# ============================

def make_real_env():
    return DynamicPricingEnv(
        data_path="data/dynamic_pricing_noisy.xlsx",
        config=EnvConfig(
            max_steps=200,
            episode_mode="random",
            action_mode="multiplier",
            random_seed=42,
        ),
    )

# ============================
# TRAIN ENV (NORMALIZED, MODEL-BASED)
# ============================

env = DummyVecEnv([make_model_env for _ in range(4)])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# ============================
# EVAL ENV (REAL ENVIRONMENT — CRITICAL)
# ============================

eval_env = DummyVecEnv([make_real_env])
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)

# sync normalization stats, same as sac_train.py
eval_env.obs_rms = env.obs_rms

# ============================
# CALLBACK
# ============================

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs_sac_mbpo/",
    log_path="./logs_sac_mbpo/",
    eval_freq=10000,
    deterministic=True,
    render=False,
)

# ============================
# POLICY (identical to sac_train.py)
# ============================

policy_kwargs = dict(net_arch=[256, 256])

# ============================
# SAC MODEL (identical hyperparameters to sac_train.py)
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
#
# NOTE: consider starting with fewer timesteps (e.g. 200_000) for a first
# sanity-check run before committing to the full 800_000 to match
# sac_train.py — the surrogate env is cheaper per-step (no real demand
# simulation), so total wall-clock should be similar or faster, but it's
# worth confirming the pipeline works end-to-end first.
# ============================

model.learn(
    total_timesteps=800000,
    callback=eval_callback,
)

# ============================
# SAVE MODEL + NORMALIZATION
# ============================

model.save("sac_mbpo_dynamic_pricing")
env.save("sac_mbpo_vecnormalize.pkl")

print("SAC (model-based) training complete!")
