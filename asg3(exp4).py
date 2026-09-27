import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 1. Load Real-Estate Dataset
df = pd.read_csv("real_estate.csv")

print(df.head())

# 2. Select Feature and Target
X = df[["Area"]]
y = df["Price"]

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 4. Standard Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Prediction
y_pred_linear = linear_model.predict(X_test)

# R2 Score
r2_linear = r2_score(y_test, y_pred_linear)

print("\n--- Linear Regression ---")
print("Slope (b1):", linear_model.coef_[0])
print("Intercept (b0):", linear_model.intercept_)
print(f"R2 Score : {r2_linear:.4f}")


# 5. Polynomial Regression
# Create polynomial features of degree 2
poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train Polynomial Regression Model
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

# Prediction
y_pred_poly = poly_model.predict(X_test_poly)

# R2 Score
r2_poly = r2_score(y_test, y_pred_poly)

print("\n--- Polynomial Regression ---")
print("Coefficients:", poly_model.coef_)
print("Intercept:", poly_model.intercept_)
print(f"R2 Score : {r2_poly:.4f}")


# 6. Compare R2 Scores
print("\n--- R2 Score Comparison ---")
print(f"Linear Regression    : {r2_linear:.4f}")
print(f"Polynomial Regression: {r2_poly:.4f}")

if r2_poly > r2_linear:
    print("Polynomial Regression has a higher R2 score.")
elif r2_poly < r2_linear:
    print("Linear Regression has a higher R2 score.")
else:
    print("Both models have the same R2 score.")


# 7. Plotting
plt.figure(figsize=(7, 5))

plt.scatter(
    X,
    y,
    color="blue",
    label="Actual Data"
)

# Sort X for a smooth polynomial curve
X_sorted = np.sort(X.values, axis=0)
X_sorted_poly = poly.transform(X_sorted)

plt.plot(
    X_sorted,
    linear_model.predict(X_sorted),
    color="red",
    linewidth=2,
    label="Linear Regression"
)

plt.plot(
    X_sorted,
    poly_model.predict(X_sorted_poly),
    color="green",
    linewidth=2,
    label="Polynomial Regression"
)

plt.xlabel("House Area (sq. ft.)")
plt.ylabel("House Price")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.show()
