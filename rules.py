def predict_risk(tickets_last_30_days, monthly_charge_increase, contract_type, complaint_ticket):

    if tickets_last_30_days > 5:
        return "HIGH"

    if monthly_charge_increase and tickets_last_30_days >= 3:
        return "MEDIUM"

    if contract_type == "Month-to-Month" and complaint_ticket:
        return "HIGH"

    return "LOW"
