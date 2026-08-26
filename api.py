from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

import numpy as np
import pandas as pd

from env.dynamic_pricing_env import DynamicPricingEnv, EnvConfig

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Dynamic Pricing API",
    description="A reinforcement learning API that recommends a price using an A2C model.",
    version="1.0.0"
)


# ============================================================
# CREATE ENVIRONMENT
# ============================================================

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


# ============================================================
# LOAD ENVIRONMENT + NORMALIZATION
# ============================================================

base_env = DummyVecEnv([make_env])

env = VecNormalize.load(
    "a2c_vecnormalize.pkl",
    base_env
)

# Inference mode
env.training = False
env.norm_reward = False


# ============================================================
# LOAD A2C MODEL
# ============================================================

model = A2C.load(
    "a2c_dynamic_pricing.zip",
    env=env
)


# Get access to the original environment
pricing_env = base_env.envs[0]


# ============================================================
# REQUEST MODEL
# ============================================================

class PricingRequest(BaseModel):

    exchange_rate: float = Field(
        ...,
        description="Current currency exchange rate"
    )

    import_tax_percent: float = Field(
        ...,
        description="Import tax percentage"
    )

    base_cost_usd: float = Field(
        ...,
        description="Base product cost in USD"
    )

    shipping_cost_usd: float = Field(
        ...,
        description="Shipping cost in USD"
    )

    demand_index: float = Field(
        ...,
        description="Current market demand index"
    )

    competitor_price_usd: float = Field(
        ...,
        description="Competitor price in USD"
    )

    reference_price: float = Field(
        ...,
        description="Current/reference selling price"
    )

    landed_cost_usd: float = Field(
        ...,
        description="Total landed cost in USD"
    )

    fx_shock: float = Field(
        ...,
        description="Current foreign exchange shock"
    )

    category: str = Field(
        ...,
        description="Product category"
    )

    month: int = Field(
        ...,
        ge=1,
        le=12,
        description="Month from 1 to 12"
    )


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "A2C Dynamic Pricing Model"
    }


# ============================================================
# MODEL INFO ENDPOINT
# ============================================================

@app.get("/model-info")
def model_info():

    return {
        "model": "A2C",
        "action_type": "price multiplier",
        "minimum_multiplier": 0.80,
        "maximum_multiplier": 1.20,
        "observation_features": 13
    }


# ============================================================
# AVAILABLE CATEGORIES
# ============================================================

@app.get("/categories")
def get_categories():

    return {
        "categories": list(
            pricing_env.category_to_id.keys()
        )
    }


# ============================================================
# PREDICT OPTIMAL PRICE
# ============================================================

@app.post("/predict")
def predict(request: PricingRequest):

    # --------------------------------------------------------
    # VALIDATE CATEGORY
    # --------------------------------------------------------

    if request.category not in pricing_env.category_to_id:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown category: {request.category}. "
                f"Available categories: "
                f"{list(pricing_env.category_to_id.keys())}"
            )
        )

    # --------------------------------------------------------
    # CALCULATE DERIVED FEATURES
    # --------------------------------------------------------

    competitor_gap = (
        request.competitor_price_usd
        - request.reference_price
    ) / (
        request.reference_price + 1e-8
    )

    month_sin = np.sin(
        2 * np.pi * request.month / 12.0
    )

    month_cos = np.cos(
        2 * np.pi * request.month / 12.0
    )


    # --------------------------------------------------------
    # CREATE ROW IN THE SAME FORMAT AS TRAINING ENVIRONMENT
    # --------------------------------------------------------

    row = pd.Series({

        "exchange_rate":
            request.exchange_rate,

        "import_tax_percent":
            request.import_tax_percent,

        "base_cost_usd":
            request.base_cost_usd,

        "shipping_cost_usd":
            request.shipping_cost_usd,

        "demand_index":
            request.demand_index,

        "competitor_price_usd":
            request.competitor_price_usd,

        "reference_price":
            request.reference_price,

        "landed_cost_usd":
            request.landed_cost_usd,

        "fx_shock":
            request.fx_shock,

        "competitor_gap":
            competitor_gap,

        "category":
            request.category,

        "month_sin":
            month_sin,

        "month_cos":
            month_cos
    })


    # --------------------------------------------------------
    # BUILD THE 13-FEATURE OBSERVATION
    #
    # Uses the SAME function that was used during training
    # --------------------------------------------------------

    observation = pricing_env._build_observation(
        row
    )


    # Add vectorized environment dimension
    observation = observation.reshape(1, -1)


    # --------------------------------------------------------
    # APPLY VECNORMALIZE
    # --------------------------------------------------------

    normalized_observation = env.normalize_obs(
        observation
    )


    # --------------------------------------------------------
    # GET A2C PREDICTION
    # --------------------------------------------------------

    action, _ = model.predict(
        normalized_observation,
        deterministic=True
    )


    multiplier = float(action[0][0])


    # --------------------------------------------------------
    # CALCULATE RECOMMENDED PRICE
    # --------------------------------------------------------

    recommended_price = (
        request.reference_price
        * multiplier
    )


    # --------------------------------------------------------
    # RETURN RESULT
    # --------------------------------------------------------

    return {

        "recommended_multiplier":
            round(multiplier, 4),

        "reference_price":
            round(request.reference_price, 2),

        "recommended_price":
            round(recommended_price, 2),

        "competitor_price":
            round(
                request.competitor_price_usd,
                2
            ),

        "competitor_gap":
            round(
                competitor_gap,
                4
            )
    }