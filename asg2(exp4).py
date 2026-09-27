import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load Real-Estate Dataset
df = pd.read_csv("real_estate.csv")

print(df.head())

# 2. Select Multiple Features and Target
X = df[["Area", "Bedrooms"]]
y = df["Price"]

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 4. Model Initialization & Training
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Prediction
y_pred = model.predict(X_test)

# 6. Metrics Calculation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# 7. Model Coefficients
print("\n--- Model Coefficients ---")
print("Coefficient of Area:", model.coef_[0])
print("Coefficient of Bedrooms:", model.coef_[1])
print("Intercept:", model.intercept_)

# 8. Evaluation Metrics
print("\n--- Evaluation Metrics ---")
print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.4f}")

# 9. Actual vs Predicted Values
print("\n--- Actual vs Predicted Prices ---")

for actual, predicted in zip(y_test, y_pred):
    print(f"Actual: {actual:.2f}   Predicted: {predicted:.2f}")

# 10. Plotting
plt.figure(figsize=(7, 5))

plt.scatter(
    y_test,
    y_pred,
    color="blue",
    label="Actual vs Predicted"
)

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Multiple Linear Regression")
plt.legend()
plt.show()




