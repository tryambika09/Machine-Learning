import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

# 1. Load dataset
wine = load_wine()

df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["target"] = wine.target

# 2. Boxplots for all numerical attributes

plt.figure(figsize=(15, 8))

sns.boxplot(data=df.drop(columns=["target"]))

plt.title("Boxplots of All Numerical Attributes in Wine Dataset")
plt.xticks(rotation=90)
plt.xlabel("Features")
plt.ylabel("Values")

plt.show()