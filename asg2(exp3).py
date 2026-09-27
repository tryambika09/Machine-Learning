import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# 1. Create Synthetic Dataset with Missing Values
data = {
    "Age": [20, 21, None, 23, 24],
    "Salary": [25000, 30000, 28000, None, 40000],
    "Department": ["IT", "HR", "IT", "Finance", None],
    "Years of Experience": [1, 2, None, 3, 4]
}

df = pd.DataFrame(data)

# Separate Features
X = df

# Define numerical and categorical features
numeric_features = ["Age", "Salary", "Years of Experience"]
categorical_features = ["Department"]

# 2. Define Preprocessing Pipelines

# Numerical Pipeline
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])

# Categorical Pipeline
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# 3. Combine using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Apply Transformations
X_processed = preprocessor.fit_transform(X)

# 4. Train-Test Split
X_train, X_test = train_test_split(
    X_processed, test_size=0.2, random_state=42
)

# Display Results
print("--- Original Dataset ---")
print(df)

print("\n--- Processed Feature Matrix Shape ---")
print(X_processed.shape)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])