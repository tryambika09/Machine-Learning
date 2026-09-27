import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

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

# 2. Separate Features and Target
X = data[:, 0:2]       # Study Hours and Attendance
y = data[:, 2]         # Pass/Fail

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Model Training
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# 6. Predict Probability of Pass
y_prob = model.predict_proba(X_test)[:, 1]

# 7. Calculate ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# 8. Calculate AUC Score
auc_score = roc_auc_score(y_test, y_prob)

print("--- ROC and AUC Performance ---")
print(f"AUC Score : {auc_score:.4f}")

# 9. Plot ROC Curve
plt.figure(figsize=(7, 5))

plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc_score:.4f})")

# Random classifier line
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Student Pass/Fail Prediction")
plt.legend()
plt.grid()

plt.show()


