import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ===============================
# LOAD DATA
# ===============================

df = pd.read_csv("simulation_output.csv")

print("Dataset shape:", df.shape)

# ===============================
# FEATURE ENGINEERING (IMPORTANT)
# ===============================

df["price_gap"] = df["chosen_price"] - df["competitor_price"]
df["margin"] = df["chosen_price"] - df["landed_cost"]
df["price_ratio"] = df["chosen_price"] / (df["competitor_price"] + 1e-6)

# ===============================
# FEATURES & TARGET
# ===============================

target = "profit"

features = [
    "exchange_rate",
    "reference_price",
    "competitor_price",
    "landed_cost",
    "demand_index",
    "price_multiplier",
    "price_gap",        # 🔥 new
    "margin",           # 🔥 new
    "price_ratio"       # 🔥 new
]

X = df[features]
y = df[target]

# ===============================
# TRAIN-TEST SPLIT (USE MORE DATA)
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42   # 🔥 more training data
)

# ===============================
# MODEL 1: LINEAR REGRESSION
# ===============================

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

print("\n📊 Linear Regression:")
print("MAE:", mean_absolute_error(y_test, lr_pred))
print("R2:", r2_score(y_test, lr_pred))

# ===============================
# MODEL 2: RANDOM FOREST (BETTER)
# ===============================

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\n🌳 Random Forest:")
print("MAE:", mean_absolute_error(y_test, rf_pred))
print("R2:", r2_score(y_test, rf_pred))

# ===============================
# FEATURE IMPORTANCE (RF)
# ===============================

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\n📈 Feature Importance (Random Forest):")
print(importance_df)

# ===============================
# SAVE BASELINE OUTPUT (IMPORTANT)
# ===============================

# ===============================
# SAVE LARGE BASELINE OUTPUT (~400K)
# ===============================

# 🔥 Sample large dataset (adjust if needed)
sample_size = min(400000, len(df))

df_sample = df.sample(n=sample_size, random_state=42)

# Recreate features for sample
X_sample = df_sample[features]

# Predict using trained RF model
predicted_profit = rf_model.predict(X_sample)

# Create baseline dataframe
baseline_df = df_sample.copy()

baseline_df["predicted_profit"] = predicted_profit

# Recalculate revenue (consistent with your env logic)
baseline_df["revenue"] = baseline_df["predicted_profit"] + baseline_df["landed_cost"]

# Ensure chosen price exists
baseline_df["chosen_price"] = baseline_df["reference_price"] * baseline_df["price_multiplier"]

# Save CSV
baseline_df.to_csv("baseline_model_output.csv", index=False)

print(f"\n✅ Baseline results saved with {len(baseline_df)} rows")