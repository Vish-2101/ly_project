import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("simulation_output.csv")

print("Dataset shape:", df.shape)
print(df.head())

target = "profit"  

features = [
    "exchange_rate",
    "reference_price",
    "competitor_price",
    "landed_cost",
    "demand_index",
    "price_multiplier"
]

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Performance:")
print("MAE:", mae)
print("R2 Score:", r2)

coeff_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
})

print("\n📈 Feature Importance:")
print(coeff_df)