"""
Train a digit classifier and save it.
Used during Docker build — runs in ~2 seconds.

MODEL_TYPE env var controls the algorithm:
  v1 = LogisticRegression (faster, ~97% accuracy)
  v2 = RandomForestClassifier (heavier, ~98% accuracy)

This demonstrates model versioning: same service, different model.
"""

import os
import joblib
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

MODEL_TYPE = os.getenv("MODEL_TYPE", "v1")

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42
)

if MODEL_TYPE == "v2":
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    print(f"Training RandomForest (v2)...")
else:
    model = LogisticRegression(max_iter=5000, random_state=42)
    print(f"Training LogisticRegression (v1)...")

model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Test accuracy: {accuracy:.4f}")

joblib.dump(model, "model.joblib")
print("Model saved to model.joblib")
