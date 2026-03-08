
from flask import Flask, request, jsonify
from rules import predict_risk

app = Flask(__name__)

@app.route("/predict-risk", methods=["POST"])
def risk():

    data = request.json

    result = predict_risk(
        data["tickets_last_30_days"],
        data["monthly_charge_increase"],
        data["contract_type"],
        data["complaint_ticket"]
    )

    return jsonify({"risk": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
