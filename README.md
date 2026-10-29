# Dynamic Pricing Using Reinforcement Learning

This project implements a reinforcement learning based dynamic pricing system. It uses simulated environments and various RL algorithms (PPO, SAC, A2C) from Stable Baselines3 to optimize pricing strategies and maximize revenue.

## 🚀 Project Overview

The core of this project is a custom Gym environment (`env/dynamic_pricing_env.py`) that simulates market dynamics based on historical data. Three distinct RL agents are trained to observe current market states (like demand, opponent prices, inventory, etc.) and take continuous actions to adjust pricing multipliers to maximize reward.

A baseline model and random action simulations (`main.py`) are also included for comparison. 

## 📂 Project Structure

```
├── README.md                           # Project Documentation
├── requirements.txt                    # Python dependencies
├── deterministic_pricing_dataset.csv   # Deterministic price and market data
├── env/                                # Custom Environment
│   └── dynamic_pricing_env.py          # Gym environment for the simulation
├── baseline_model.py                   # Baseline static pricing model
├── main.py                             # Random action simulation script
├── ppo_train.py                        # Script to train PPO agent
├── sac_train.py                        # Script to train SAC agent
├── a2c_train.py                        # Script to train A2C agent
├── evaluate.py                         # Evaluation script using trained models
├── comparision.ipynb                   # Notebook comparing different approaches
└── model_compare.ipynb                 # Notebook comparing different approaches
```

## Live Dashboard

The Flask dashboard uses `deterministic_pricing_dataset.csv` by default and
provides one page per model:

```bash
python live_app.py
```

Open `http://localhost:5000/ppo`, `http://localhost:5000/sac`, or
`http://localhost:5000/a2c`. The dataset is an Excel workbook distributed with
a `.csv` filename; the application detects and loads it correctly. The live
dashboard requires the corresponding trained model and `VecNormalize` artifacts;
it never substitutes a demo or heuristic policy. Select a product from the
dashboard dropdown to keep predictions focused on that product; the chart also
shows a moving-average price to smooth record-to-record variation. The app
persists each model/product cursor and last date/price in `.live_state.json`,
so switching products or restarting the app resumes from the previous record.
Prediction history is persisted in `.live_history.json`; use the chart's
history slider to browse earlier trends, or press `Live` to follow new ticks.

Set `DATA_PATH` to use another compatible dataset, or `PORT` and `TICK_SECONDS`
to change the server port and update interval.

## 🛠 Prerequisites

Ensure you have Python 3.8+ installed. Install the required dependencies using:

```bash
pip install -r requirements.txt
```
*(Note: Key libraries include `stable-baselines3`, `pandas`, `numpy`, and `gymnasium`)*

## 🧠 Training the Models

You can train any of the three provided state-of-the-art models (PPO, SAC, A2C). The training scripts will automatically save the trained model zip files and their corresponding `VecNormalize` statistics.

To train the Proximal Policy Optimization (PPO) agent:
```bash
python ppo_train.py
```

To train the Soft Actor-Critic (SAC) agent:
```bash
python sac_train.py
```

To train the Advantage Actor Critic (A2C) agent:
```bash
python a2c_train.py
```

The trained models will be saved as `[model_name]_dynamic_pricing_improved.zip` along with stats like `[model_name]_vecnormalize.pkl` in the root directory.

## 📊 Evaluation & Simulation

To evaluate a trained model, use the `evaluate.py` script. By default, you can uncomment the desired model loading function inside the script and run:

```bash
python evaluate.py
```
This script will output the simulation results to a CSV file (e.g., `ppo_rl_simulation_output.csv`).

## 📈 Baseline & Comparison

To run the random action simulation baseline:
```bash
python main.py
```

To run a static statistical baseline model:
```bash
python baseline_model.py
```

You can view the comprehensive performance comparisons by running Jupyter Notebook and exploring `comparision.ipynb` and `model_compare.ipynb`.
