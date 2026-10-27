"""Leakage-free, row-wise feature engineering shared by the notebook and the Streamlit app."""
import numpy as np
import pandas as pd

SERVICES = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"]
NUM_FEATS = ["tenure", "MonthlyCharges", "TotalCharges", "num_services", "avg_monthly_spend", "charge_delta"]


def engineer(frame: pd.DataFrame) -> pd.DataFrame:
    """Row-wise transformations only (no statistics learned from data -> safe before the split)."""
    f = frame.copy()
    f["TotalCharges"] = pd.to_numeric(f["TotalCharges"].astype(str).str.strip().replace("", np.nan), errors="coerce")
    f["TotalCharges"] = f["TotalCharges"].fillna(0.0)  # blank only when tenure == 0 -> nothing billed yet
    f["SeniorCitizen"] = f["SeniorCitizen"].map({0: "No", 1: "Yes", "0": "No", "1": "Yes",
                                                 "No": "No", "Yes": "Yes"}).fillna("No")
    f["num_services"] = (f[SERVICES] == "Yes").sum(axis=1) + (f["InternetService"] != "No").astype(int)
    f["avg_monthly_spend"] = np.where(f["tenure"] > 0, f["TotalCharges"] / f["tenure"].clip(lower=1),
                                      f["MonthlyCharges"])
    f["charge_delta"] = f["MonthlyCharges"] - f["avg_monthly_spend"]  # recent price increase signal
    f["auto_pay"] = np.where(f["PaymentMethod"].str.contains("automatic"), "Yes", "No")
    f["tenure_bucket"] = pd.cut(f["tenure"], [-1, 6, 12, 24, 48, 1000],
                                labels=["0-6", "7-12", "13-24", "25-48", "49+"]).astype(str)
    return f.drop(columns=["customerID", "Churn", "churn"], errors="ignore")
