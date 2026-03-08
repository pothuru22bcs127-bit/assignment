from rules import predict_risk

def test_high_risk():
    result = predict_risk(6, False, "Two year", False)
    assert result == "HIGH"
