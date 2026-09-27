import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

# 1. Load dataset
wine = load_wine()

df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["target"] = wine.target

# 2. Basic Data Exploration

print("--- First Five Rows ---")
print(df.head())

print("\n--- Dataset Information ---")
print(df.info())

print("\n--- Statistical Summary ---")
print(df.describe())

print("\n--- Missing Values ---")
print(df.isnull().sum())

# 3. Correlation matrix

print("\n--- Correlation Matrix ---")
print(df.corr(numeric_only=True))

# 4. Scatter plot: Alcohol vs Color Intensity

plt.figure(figsize=(7, 5))

sns.scatterplot(
    data=df,
    x="alcohol",
    y="color_intensity",
    hue="target",
    palette="viridis"
)

plt.title("Alcohol vs Color Intensity by Class")
plt.show()

# 5. Histogram: Distribution of Alcohol

plt.figure(figsize=(7, 5))

sns.histplot(
    df["alcohol"],
    kde=True,
    color="blue"
)

plt.title("Distribution of Alcohol")
plt.show()




