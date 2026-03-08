from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("models/model_latest.pkl") if False else None

@app.get("/")
def home():
    return {"message": "Churn Prediction MLOps System"}

@app.post("/predict-risk")
def predict(data: dict):

    tenure = data["tenure"]
    monthly = data["monthly_charges"]
    total = data["total_charges"]

    features = np.array([[tenure, monthly, total]])

    prob = 0.5

    return {
        "churn_probability": prob,
        "risk_category": "HIGH" if prob > 0.7 else "LOW"
    }
