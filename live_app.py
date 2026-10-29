"""
Live dynamic-pricing dashboard: one web page per trained RL model.

    /ppo   /sac   /a2c

Each page streams one pricing decision per tick (default 1 s). A background thread
per model steps the SAME DynamicPricingEnv used in training/evaluate.py, asks the
trained model for a price multiplier, and publishes the result. Browsers poll
/api/<model>/ticks.

Run:   python live_app.py        ->  http://localhost:5000

The dashboard requires the trained model artifacts and never substitutes a
demo or heuristic policy.
"""
import os
import json
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request, redirect

TICK_SECONDS = float(os.environ.get("TICK_SECONDS", 1.0))
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = os.environ.get(
    "DATA_PATH",
    str(PROJECT_ROOT / "deterministic_pricing_dataset.csv"),
)
HISTORY = 500  # ticks kept in memory per model
STATE_PATH = PROJECT_ROOT / os.environ.get("STATE_PATH", ".live_state.json")
HISTORY_PATH = PROJECT_ROOT / os.environ.get("HISTORY_PATH", ".live_history.json")
STATE_LOCK = threading.RLock()

# To add SAC-MBPO later: add an entry here (same file naming as sac_train_mbpo.py).
MODELS = {
    "ppo": dict(name="PPO", cls="PPO", model="ppo_dynamic_pricing_improved.zip", vec="ppo_vecnormalize.pkl"),
    "sac": dict(name="SAC", cls="SAC", model="sac_dynamic_pricing.zip", vec="sac_vecnormalize.pkl"),
    "a2c": dict(name="A2C", cls="A2C", model="a2c_dynamic_pricing.zip", vec="a2c_vecnormalize.pkl"),
}


def _clean(info):
    """Make an env info dict JSON-safe."""
    keys = ["date", "drug_name", "category", "exchange_rate", "reference_price", "chosen_price",
            "price_multiplier", "competitor_price", "landed_cost", "demand_index",
            "realized_units", "revenue", "cost", "profit", "reward"]
    out = {}
    for k in keys:
        v = info[k]
        if k == "date":
            out[k] = str(v)[:10]
        elif k in ("drug_name", "category"):
            out[k] = str(v)
        else:
            out[k] = round(float(v), 4)
    return out


def _resolve_path(path):
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _load_state():
    try:
        with STATE_PATH.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_state(state):
    with STATE_LOCK:
        temporary = STATE_PATH.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2)
        temporary.replace(STATE_PATH)


def _load_history():
    try:
        with HISTORY_PATH.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_history(history):
    with STATE_LOCK:
        temporary = HISTORY_PATH.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(history, handle)
        temporary.replace(HISTORY_PATH)


# ------------------------------------------------------------------ real runner
class RealRunner:
    mode = "live"

    def __init__(self, cfg, saved_state=None):
        from env.dynamic_pricing_env import DynamicPricingEnv, EnvConfig
        from stable_baselines3 import PPO, SAC, A2C
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        def make_env():
            return DynamicPricingEnv(
                data_path=str(_resolve_path(DATA_PATH)),
                config=EnvConfig(max_steps=200, episode_mode="random", action_mode="multiplier",
                                 random_seed=int(time.time()) % 100000),
            )

        venv = VecNormalize.load(str(_resolve_path(cfg["vec"])), DummyVecEnv([make_env]))
        venv.training = False
        venv.norm_reward = False
        self.env = venv
        self.base_env = venv.venv.envs[0]
        self.model = {"PPO": PPO, "SAC": SAC, "A2C": A2C}[cfg["cls"]].load(
            str(_resolve_path(cfg["model"]))
        )
        self.items = sorted(self.base_env.df["drug_name"].unique().tolist())
        self.selected_item = self.items[0]
        self.item_positions = {
            item: int((saved_state or {}).get(item, {}).get("position", 0))
            for item in self.items
        }
        self.last_values = dict(saved_state or {})
        self._select_item(self.selected_item)

    def _select_item(self, item):
        if item not in self.items:
            raise ValueError(f"Unknown item: {item}")
        indices = self.base_env.df.index[
            self.base_env.df["drug_name"] == item
        ].to_numpy()
        self.base_env.episode_indices = indices
        self.base_env.current_idx = self.item_positions.get(item, 0) % len(indices)
        self.base_env.current_step = 0
        raw_observation = self.base_env._build_observation(
            self.base_env._get_row()
        )
        self.obs = self.env.normalize_obs(raw_observation[np.newaxis, :])
        self.selected_item = item

    def select_item(self, item):
        self._select_item(item)

    def reset_selected_item(self):
        self._select_item(self.selected_item)

    def step(self):
        if self.base_env.current_idx >= len(self.base_env.episode_indices):
            self.item_positions[self.selected_item] = 0
            self.reset_selected_item()
        action, _ = self.model.predict(self.obs, deterministic=True)
        self.obs, _, _, infos = self.env.step(action)
        self.item_positions[self.selected_item] = self.base_env.current_idx
        return _clean(infos[0])


# ------------------------------------------------------------------ stream state
class Stream:
    def __init__(self, key, cfg):
        self.key, self.cfg = key, cfg
        self.ticks = deque(maxlen=HISTORY)
        self.n = 0
        self.total_profit = 0.0
        self.total_revenue = 0.0
        self.paused = False
        self.error = None
        self.lock = threading.RLock()
        self.state = _load_state()
        self.history = _load_history()
        self.model_history = self.history.setdefault(key, {})
        self.model_state = self.state.setdefault(key, {})
        self.selected_item = None
        self.price_history = {}
        self.price_totals = {}
        self.price_counts = {}
        try:
            for f in (cfg["model"], cfg["vec"], DATA_PATH):
                resolved = _resolve_path(f)
                if not resolved.exists():
                    raise FileNotFoundError(str(resolved))
            self.runner = RealRunner(cfg, self.model_state.get("items"))
            saved_item = self.model_state.get("selected_item")
            if saved_item in self.runner.items:
                self.runner.select_item(saved_item)
            self.selected_item = self.runner.selected_item
            for item in self.runner.items:
                self.price_history[item] = deque(maxlen=20)
                self.price_totals[item] = 0.0
                self.price_counts[item] = 0
        except (FileNotFoundError, ImportError, OSError, ValueError) as e:
            self.runner = None
            self.error = f"{type(e).__name__}: {e}"
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            if self.runner is not None and not self.paused:
                try:
                    with self.lock:
                        t = self.runner.step()
                        self.n += 1
                        t["id"] = self.n
                        item = t["drug_name"]
                        self.price_history[item].append(t["chosen_price"])
                        self.price_totals[item] += t["chosen_price"]
                        self.price_counts[item] += 1
                        t["moving_average_price"] = round(
                            sum(self.price_history[item]) / len(self.price_history[item]), 4
                        )
                        t["average_price"] = round(
                            self.price_totals[item] / self.price_counts[item], 4
                        )
                        self.model_history.setdefault(item, []).append({
                            "id": t["id"],
                            "date": t["date"],
                            "chosen_price": t["chosen_price"],
                            "moving_average_price": t["moving_average_price"],
                            "average_price": t["average_price"],
                            "price_multiplier": t["price_multiplier"],
                            "profit": t["profit"],
                        })
                        self.model_state.setdefault("items", {})[item] = {
                            "position": self.runner.item_positions[item],
                            "last_date": t["date"],
                            "last_price": t["chosen_price"],
                        }
                        _save_state(self.state)
                        _save_history(self.history)
                    self.total_profit += t["profit"]
                    self.total_revenue += t["revenue"]
                    t["cum_profit"] = round(self.total_profit, 2)
                    self.ticks.append(t)
                except Exception as e:
                    self.error = f"step failed: {e}"
            time.sleep(TICK_SECONDS)


app = Flask(__name__)
streams = {k: Stream(k, c) for k, c in MODELS.items()}


@app.route("/")
def index():
    return redirect("/ppo")


@app.route("/<key>")
def page(key):
    if key not in MODELS:
        return "Unknown model", 404
    s = streams[key]
    return render_template("model.html", key=key, models=MODELS, name=MODELS[key]["name"],
                           mode="live" if s.runner else "unavailable",
                           reason=s.error, tick_ms=int(TICK_SECONDS * 1000))


@app.route("/api/<key>/items")
def items(key):
    if key not in streams:
        return jsonify(error="unknown model"), 404
    s = streams[key]
    if s.runner is None:
        return jsonify(error="model unavailable", detail=s.error), 503
    saved = s.model_state.get("items", {})
    return jsonify(
        items=s.runner.items,
        selected=s.selected_item,
        last=saved.get(s.selected_item),
    )


@app.route("/api/<key>/select", methods=["POST"])
def select_item(key):
    if key not in streams:
        return jsonify(error="unknown model"), 404
    s = streams[key]
    if s.runner is None:
        return jsonify(error="model unavailable", detail=s.error), 503
    payload = request.get_json(silent=True) or {}
    item = payload.get("item")
    if not isinstance(item, str) or not item:
        return jsonify(error="item must be a non-empty string"), 400
    try:
        with s.lock:
            s.runner.select_item(item)
            s.selected_item = item
            s.model_state["selected_item"] = item
            _save_state(s.state)
    except ValueError as e:
        return jsonify(error=str(e)), 400
    return jsonify(selected=item)


@app.route("/api/<key>/ticks")
def ticks(key):
    if key not in streams:
        return jsonify(error="unknown model"), 404
    s = streams[key]
    if s.runner is None:
        return jsonify(error="model unavailable", detail=s.error), 503
    after = int(request.args.get("after", 0))
    new = [
        t for t in list(s.ticks)
        if t["id"] > after and t["drug_name"] == s.selected_item
    ]
    n = max(s.n, 1)
    selected_ticks = [t for t in list(s.ticks) if t["drug_name"] == s.selected_item]
    latest = selected_ticks[-1] if selected_ticks else None
    return jsonify(ticks=new, paused=s.paused, mode=s.runner.mode,
                   selected=s.selected_item,
                   total_profit=round(s.total_profit, 2), total_revenue=round(s.total_revenue, 2),
                   avg_profit=round(s.total_profit / n, 2), count=s.n,
                   average_price=latest["average_price"] if latest else None)


@app.route("/api/<key>/history")
def history(key):
    if key not in streams:
        return jsonify(error="unknown model"), 404
    s = streams[key]
    if s.runner is None:
        return jsonify(error="model unavailable", detail=s.error), 503
    item = request.args.get("item", s.selected_item)
    if item not in s.runner.items:
        return jsonify(error=f"unknown item: {item}"), 400
    return jsonify(item=item, history=s.model_history.get(item, []))


@app.route("/api/<key>/pause", methods=["POST"])
def pause(key):
    if key not in streams:
        return jsonify(error="unknown model"), 404
    s = streams[key]
    s.paused = not s.paused
    return jsonify(paused=s.paused)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False, threaded=True)
