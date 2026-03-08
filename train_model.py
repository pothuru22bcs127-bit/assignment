import pandas as pd
import os
import datetime
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, precision_score

from preprocess import preprocess_data

print("Training Pipeline Started")

df = pd.read_csv("Telco_customer_churn.csv")

print("Dataset Shape:", df.shape)

X, y = preprocess_data(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

preds = model.predict(X_test)

probs = model.predict_proba(X_test)[:,1]

print("Metrics:")
print("F1:", f1_score(y_test, preds))
print("ROC-AUC:", roc_auc_score(y_test, probs))
print("Precision:", precision_score(y_test, preds))

os.makedirs("models", exist_ok=True)

version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

joblib.dump(model, f"models/model_{version}.pkl")

print("Model saved")
