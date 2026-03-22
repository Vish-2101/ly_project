import pandas as pd
import matplotlib.pyplot as plt

# ✅ Load data
rl = pd.read_csv("rl_simulation_output.csv")
baseline = pd.read_csv("simulation_output.csv")

print("RL shape:", rl.shape)
print("Baseline shape:", baseline.shape)

# ===============================
# 📊 BASIC METRICS
# ===============================

print("\n📊 ===== BASIC COMPARISON =====")

print("RL Avg Profit:", rl["profit"].mean())
print("Baseline Avg Profit:", baseline["profit"].mean())

print("\nRL Total Profit:", rl["profit"].sum())
print("Baseline Total Profit:", baseline["profit"].sum())

print("\nRL Avg Revenue:", rl["revenue"].mean())
print("Baseline Avg Revenue:", baseline["revenue"].mean())

print("\nRL Avg Price:", rl["chosen_price"].mean())
print("Baseline Avg Price:", baseline["chosen_price"].mean())

# ===============================
# 📈 IMPROVEMENT %
# ===============================

profit_improvement = (
    (rl["profit"].mean() - baseline["profit"].mean())
    / baseline["profit"].mean()
) * 100

print("\n🚀 Profit Improvement (%):", profit_improvement)

# ===============================
# 📊 VISUALIZATIONS
# ===============================

# 1. Profit over time
plt.figure()
plt.plot(rl["profit"].values, label="RL")
plt.plot(baseline["profit"].values, label="Random")
plt.legend()
plt.title("Profit Comparison Over Time")
plt.xlabel("Steps")
plt.ylabel("Profit")
plt.show()

# 2. Price comparison
plt.figure()
plt.plot(rl["chosen_price"].values, label="RL Price")
plt.plot(baseline["chosen_price"].values, label="Random Price")
plt.legend()
plt.title("Pricing Strategy Comparison")
plt.xlabel("Steps")
plt.ylabel("Price")
plt.show()

# 3. Histogram of profits
plt.figure()
plt.hist(rl["profit"], bins=30, alpha=0.5, label="RL")
plt.hist(baseline["profit"], bins=30, alpha=0.5, label="Random")
plt.legend()
plt.title("Profit Distribution")
plt.xlabel("Profit")
plt.ylabel("Frequency")
plt.show()

# 4. Revenue comparison
plt.figure()
plt.plot(rl["revenue"].values, label="RL Revenue")
plt.plot(baseline["revenue"].values, label="Random Revenue")
plt.legend()
plt.title("Revenue Comparison")
plt.xlabel("Steps")
plt.ylabel("Revenue")
plt.show()

# ===============================
# 📉 STABILITY (VERY IMPORTANT)
# ===============================

print("\n📉 ===== STABILITY =====")

print("RL Profit Std Dev:", rl["profit"].std())
print("Baseline Profit Std Dev:", baseline["profit"].std())

# ===============================
# 📊 SUMMARY TABLE
# ===============================

summary = pd.DataFrame({
    "Metric": ["Avg Profit", "Total Profit", "Avg Revenue", "Avg Price"],
    "RL": [
        rl["profit"].mean(),
        rl["profit"].sum(),
        rl["revenue"].mean(),
        rl["chosen_price"].mean()
    ],
    "Baseline": [
        baseline["profit"].mean(),
        baseline["profit"].sum(),
        baseline["revenue"].mean(),
        baseline["chosen_price"].mean()
    ]
})

print("\n📊 ===== SUMMARY TABLE =====")
print(summary)