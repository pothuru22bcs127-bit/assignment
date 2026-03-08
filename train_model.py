import pandas as pd
import os
import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, precision_score
import joblib

from preprocess import preprocess_data

# load dataset
df = pd.read_csv("Telco_customer_churn.csv")

# preprocess
X, y = preprocess_data(df)

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# predictions
preds = model.predict(X_test)
probs = model.predict_proba(X_test)

if probs.shape[1] > 1:
    probs = probs[:, 1]
else:
    probs = probs[:, 0]

# evaluation
print("F1:", f1_score(y_test, preds))
print("ROC-AUC:", roc_auc_score(y_test, probs))
print("Precision:", precision_score(y_test, preds))

# ---------- MLOps Addition (Model Versioning) ----------

# create models folder if not exists
os.makedirs("models", exist_ok=True)

# create version name
version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

model_path = f"models/model_{version}.pkl"

# save model
joblib.dump(model, model_path)

print(f"Model saved at {model_path}")
