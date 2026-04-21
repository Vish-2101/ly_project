import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


@dataclass
class EnvConfig:
    """Configuration for the dynamic pricing environment."""
    max_steps: Optional[int] = None
    action_mode: str = "multiplier"   # "multiplier" or "absolute"
    min_multiplier: float = 0.80
    max_multiplier: float = 1.20
    max_price: float = 500.0
    elasticity_scale: float = 1.15
    inventory_penalty: float = 0.0
    margin_penalty: float = 0.05
    episode_mode: str = "sequential"  # "sequential" or "random"
    random_seed: Optional[int] = None


class DynamicPricingEnv(gym.Env):
    """
    Gymnasium environment for dynamic pricing under demand and exchange-rate uncertainty.

    Dataset columns expected:
        date, currency_pair, exchange_rate, drug_name, category,
        import_tax_percent, base_cost_usd, shipping_cost_usd,
        demand_index, competitor_price_usd, final_price_usd

    Action space:
        Box(1,) representing either:
        - a price multiplier over the dataset reference price, or
        - an absolute price in USD

    Observation space (normalized float vector):
        [
            exchange_rate,
            import_tax_percent,
            base_cost_usd,
            shipping_cost_usd,
            demand_index,
            competitor_price_usd,
            reference_price,
            landed_cost,
            fx_shock,
            competitor_gap,
            category_id,
            month_sin,
            month_cos
        ]
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data_path: str,
        config: Optional[EnvConfig] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.config = config or EnvConfig()
        self.render_mode = render_mode
        self.rng = np.random.default_rng(self.config.random_seed)

        self.df = self._load_and_prepare_data(data_path)
        self.feature_mins, self.feature_maxs = self._fit_normalizers(self.df)

        self.category_to_id = {
            c: i for i, c in enumerate(sorted(self.df["category"].unique()))
        }
        self.drug_to_id = {
            d: i for i, d in enumerate(sorted(self.df["drug_name"].unique()))
        }

        if self.config.action_mode == "multiplier":
            self.action_space = spaces.Box(
                low=np.array([self.config.min_multiplier], dtype=np.float32),
                high=np.array([self.config.max_multiplier], dtype=np.float32),
                dtype=np.float32,
            )
        elif self.config.action_mode == "absolute":
            self.action_space = spaces.Box(
                low=np.array([0.0], dtype=np.float32),
                high=np.array([self.config.max_price], dtype=np.float32),
                dtype=np.float32,
            )
        else:
            raise ValueError("action_mode must be either 'multiplier' or 'absolute'")

        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(13,),
            dtype=np.float32,
        )

        self.current_step = 0
        self.current_idx = 0
        self.episode_indices = None
        self.last_info: Dict[str, Any] = {}

    def _load_and_prepare_data(self, data_path: str) -> pd.DataFrame:
        df = pd.read_excel(data_path)
        required = {
            "date", "currency_pair", "exchange_rate", "drug_name", "category",
            "import_tax_percent", "base_cost_usd", "shipping_cost_usd",
            "demand_index", "competitor_price_usd", "final_price_usd"
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["date", "drug_name"]).reset_index(drop=True)

        df["landed_cost_usd"] = (
            (df["base_cost_usd"] + df["shipping_cost_usd"])
            * (1.0 + df["import_tax_percent"] / 100.0)
        )
        df["month"] = df["date"].dt.month
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
        df["fx_ma_7"] = (
            df.groupby("currency_pair")["exchange_rate"]
            .transform(lambda s: s.rolling(7, min_periods=1).mean())
        )
        df["fx_shock"] = (df["exchange_rate"] - df["fx_ma_7"]) / (df["fx_ma_7"] + 1e-8)
        df["reference_price"] = df["final_price_usd"].astype(float)
        df["competitor_gap"] = (
            (df["competitor_price_usd"] - df["reference_price"])
            / (df["reference_price"] + 1e-8)
        )
        return df

    def _fit_normalizers(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        cols = [
            "exchange_rate", "import_tax_percent", "base_cost_usd", "shipping_cost_usd",
            "demand_index", "competitor_price_usd", "reference_price", "landed_cost_usd",
            "fx_shock", "competitor_gap"
        ]
        mins = df[cols].min()
        maxs = df[cols].max()
        return mins, maxs

    def _normalize(self, value: float, col: str) -> float:
        lo = float(self.feature_mins[col])
        hi = float(self.feature_maxs[col])
        if math.isclose(lo, hi):
            return 0.0
        return ((value - lo) / (hi - lo)) * 2.0 - 1.0

    def _build_observation(self, row: pd.Series) -> np.ndarray:
        category_id = self.category_to_id[row["category"]]
        category_norm = category_id / max(1, len(self.category_to_id) - 1)

        obs = np.array([
            self._normalize(row["exchange_rate"], "exchange_rate"),
            self._normalize(row["import_tax_percent"], "import_tax_percent"),
            self._normalize(row["base_cost_usd"], "base_cost_usd"),
            self._normalize(row["shipping_cost_usd"], "shipping_cost_usd"),
            self._normalize(row["demand_index"], "demand_index"),
            self._normalize(row["competitor_price_usd"], "competitor_price_usd"),
            self._normalize(row["reference_price"], "reference_price"),
            self._normalize(row["landed_cost_usd"], "landed_cost_usd"),
            self._normalize(row["fx_shock"], "fx_shock"),
            self._normalize(row["competitor_gap"], "competitor_gap"),
            category_norm * 2.0 - 1.0,
            float(row["month_sin"]),
            float(row["month_cos"]),
        ], dtype=np.float32)
        return obs

    def _get_row(self) -> pd.Series:
        idx = self.episode_indices[self.current_idx]
        return self.df.iloc[idx]

    def _sample_indices(self) -> np.ndarray:
        if self.config.episode_mode == "random":
            n = self.config.max_steps or 1000
            return self.rng.integers(0, len(self.df), size=n)
        n = self.config.max_steps or len(self.df)
        start_max = max(0, len(self.df) - n)
        start = 0 if start_max == 0 else int(self.rng.integers(0, start_max + 1))
        return np.arange(start, start + n)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.episode_indices = self._sample_indices()
        self.current_idx = 0
        self.current_step = 0

        row = self._get_row()
        obs = self._build_observation(row)
        info = {
            "date": row["date"],
            "drug_name": row["drug_name"],
            "category": row["category"],
        }
        self.last_info = info
        return obs, info

    def step(self, action):
        row = self._get_row()
        raw_action = float(np.asarray(action).reshape(-1)[0])

        reference_price = float(row["reference_price"])
        if self.config.action_mode == "multiplier":
            multiplier = float(np.clip(raw_action, self.config.min_multiplier, self.config.max_multiplier))
            chosen_price = reference_price * multiplier
        else:
            chosen_price = float(np.clip(raw_action, 0.0, self.config.max_price))
            multiplier = chosen_price / max(reference_price, 1e-8)

        landed_cost = float(row["landed_cost_usd"])
        demand_index = float(row["demand_index"])
        competitor_price = float(row["competitor_price_usd"])
        exchange_rate = float(row["exchange_rate"])
        fx_shock = float(row["fx_shock"])

        price_gap = (chosen_price - competitor_price) / max(competitor_price, 1e-8)
        margin = chosen_price - landed_cost
        base_units = max(demand_index, 1.0)

        # Demand simulator: higher price lowers units, stronger FX shock increases uncertainty.
        deterministic_units = base_units * np.exp(
            -self.config.elasticity_scale * price_gap
            - 0.15 * max(fx_shock, 0.0)
        )
        uncertainty = self.rng.normal(loc=0.0, scale=0.08 + 0.12 * abs(fx_shock))
        realized_units = max(0.0, deterministic_units * (1.0 + uncertainty))

        revenue = chosen_price * realized_units
        cost = landed_cost * realized_units
        profit = revenue - cost

        reward = profit / 100.0
        if margin < 0:
            reward -= abs(margin) * self.config.margin_penalty * max(realized_units, 1.0)
        reward -= self.config.inventory_penalty * realized_units

        self.current_idx += 1
        self.current_step += 1

        terminated = self.current_idx >= len(self.episode_indices)
        truncated = False

        if not terminated:
            next_row = self._get_row()
            obs = self._build_observation(next_row)
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        info = {
            "date": row["date"],
            "drug_name": row["drug_name"],
            "category": row["category"],
            "exchange_rate": exchange_rate,
            "reference_price": reference_price,
            "chosen_price": chosen_price,
            "price_multiplier": multiplier,
            "competitor_price": competitor_price,
            "landed_cost": landed_cost,
            "demand_index": demand_index,
            "realized_units": realized_units,
            "revenue": revenue,
            "cost": cost,
            "profit": profit,
            "reward": reward,
        }
        self.last_info = info

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, info

    def render(self):
        if not self.last_info:
            print("Environment not stepped yet.")
            return
        print(
            f"{self.last_info['date'].date()} | {self.last_info['drug_name']} | "
            f"Price: {self.last_info['chosen_price']:.2f} | "
            f"Units: {self.last_info['realized_units']:.2f} | "
            f"Profit: {self.last_info['profit']:.2f}"
        )


if __name__ == "__main__":
    env = DynamicPricingEnv(
        data_path="data/dynamic_pricing.csv.xlsx",
        config=EnvConfig(max_steps=10, action_mode="multiplier", random_seed=42),
        render_mode="human",
    )

    obs, info = env.reset(seed=42)
    print("Initial observation shape:", obs.shape)
    print("Initial info:", info)

    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
