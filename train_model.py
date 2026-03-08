import pandas as pd
import os
import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, precision_score
import joblib

from preprocess import preprocess_data

print("========== MLOps Training Pipeline Started ==========")

# Load dataset
print("Loading dataset...")
df = pd.read_csv("Telco_customer_churn.csv")

print("Dataset Shape:", df.shape)

# Preprocess data
print("Preprocessing data...")
X, y = preprocess_data(df)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Random Forest Model...")

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Model Training Completed!")

# Predictions
preds = model.predict(X_test)

# Probability handling
probs = model.predict_proba(X_test)

if probs.shape[1] > 1:
    probs = probs[:, 1]
else:
    probs = probs[:, 0]

# Evaluation Metrics
print("========== Model Metrics ==========")

f1 = f1_score(y_test, preds)
roc_auc = roc_auc_score(y_test, probs)
precision = precision_score(y_test, preds)

print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")
print(f"Precision Score: {precision:.4f}")

print("====================================")

# ---------- Model Versioning ----------

os.makedirs("models", exist_ok=True)

version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

model_path = f"models/model_{version}.pkl"

joblib.dump(model, model_path)

print("Model saved at:", model_path)

print("========== Training Pipeline Finished ==========")
