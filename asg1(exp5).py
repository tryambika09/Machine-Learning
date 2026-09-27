import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Create Custom Student Dataset
# Study Hours, Attendance, Pass/Fail
data = np.array([
    [2, 60, 0],
    [3, 65, 0],
    [4, 70, 0],
    [5, 75, 1],
    [6, 80, 1],
    [7, 85, 1],
    [8, 90, 1],
    [1, 50, 0],
    [3, 55, 0],
    [6, 70, 1],
    [4, 60, 0],
    [7, 75, 1],
    [8, 80, 1],
    [2, 55, 0],
    [5, 85, 1]
])

# Separate Features and Target
X = data[:, 0:2]       # Study Hours and Attendance
y = data[:, 2]         # Pass/Fail

# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Model Training
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# 5. Prediction
y_pred = model.predict(X_test)

# 6. Evaluation Metrics
print("--- Logistic Regression Performance ---")
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score  : {f1_score(y_test, y_pred):.4f}")



