import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score

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
X = data[:, 0:2]
y = data[:, 2]

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

# 6. Get Probability of Pass (Class 1)
y_prob = model.predict_proba(X_test)[:, 1]

# 7. Test Different Decision Thresholds
thresholds = [0.3, 0.5, 0.7]

print("--- Precision and Recall at Different Thresholds ---")

for threshold in thresholds:

    # Predict Pass if probability >= threshold
    y_pred = (y_prob >= threshold).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    print(f"\nThreshold : {threshold}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}") 


