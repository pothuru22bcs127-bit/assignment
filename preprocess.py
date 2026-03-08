import pandas as pd

def preprocess_data(df):

    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    features = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    X = df[features]
    y = df["Churn"]

    return X, y
