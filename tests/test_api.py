import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from rules import predict_risk

def test_high_risk():
    result = predict_risk(6, False, "Two year", False)
    assert result == "HIGH"
