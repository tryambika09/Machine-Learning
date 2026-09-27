import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

# 1. Load dataset
wine = load_wine()

df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["target"] = wine.target

# 2. Correlation matrix
print("\n--- Correlation Matrix ---")
correlation = df.corr(numeric_only=True)
print(correlation)

# 3. Correlation Heatmap
plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap of Wine Dataset")
plt.show()

# 4. Find strongest positive correlation
correlation_matrix = df.drop(columns=["target"]).corr()

# Remove self-correlation
correlation_matrix = correlation_matrix.where(
    correlation_matrix < 1
)

strongest_pair = correlation_matrix.stack().idxmax()
strongest_value = correlation_matrix.stack().max()

print("\n--- Strongest Positive Correlation ---")
print("Feature 1:", strongest_pair[0])
print("Feature 2:", strongest_pair[1])
print("Correlation:", round(strongest_value, 2))




