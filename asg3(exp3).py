import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

# 1. Create Dataset
data = {
    "Age": [20, 21, None, 23, 24],
    "Salary": [25000, 30000, 28000, None, 40000],
    "Department": ["IT", "HR", "IT", "Finance", None],
    "Years of Experience": [1, 2, None, 3, 4]
}

df = pd.DataFrame(data)

# 2. Select Numerical Features
numeric_features = ["Age", "Salary", "Years of Experience"]

X = df[numeric_features]

# 3. Handle Missing Values
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# 4. Apply StandardScaler
standard_scaler = StandardScaler()
X_standard = standard_scaler.fit_transform(X_imputed)

# 5. Apply MinMaxScaler
minmax_scaler = MinMaxScaler()
X_minmax = minmax_scaler.fit_transform(X_imputed)

# 6. Print Original Data
print("--- Original Numerical Data ---")
print(X)

# 7. Print StandardScaler Results
print("\n--- StandardScaler Transformed Data ---")
print(X_standard)

# 8. Print MinMaxScaler Results
print("\n--- MinMaxScaler Transformed Data ---")
print(X_minmax)

# 9. Print Numerical Ranges
print("\n--- Numerical Ranges ---")

print("\nStandardScaler:")
print("Minimum value:", X_standard.min())
print("Maximum value:", X_standard.max())

print("\nMinMaxScaler:")
print("Minimum value:", X_minmax.min())
print("Maximum value:", X_minmax.max())



