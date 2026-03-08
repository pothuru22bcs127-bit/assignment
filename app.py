from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route("/predict-risk", methods=["POST"])
def predict():

    data = request.json

    input_data = pd.DataFrame([{
        "tenure": data["tenure"],
        "MonthlyCharges": data["monthly_charges"],
        "TotalCharges": data["total_charges"]
    }])

    prediction = model.predict(input_data)[0]

    probs = model.predict_proba(input_data)

    if probs.shape[1] > 1:
        probability = probs[0][1]
    else:
        probability = probs[0][0]

    risk = "Low"

    if probability > 0.7:
        risk = "High"
    elif probability > 0.4:
        risk = "Medium"

    return jsonify({
        "churn_prediction": int(prediction),
        "risk": risk,
        "probability": float(probability)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
