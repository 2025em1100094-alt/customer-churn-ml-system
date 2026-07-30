from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "Churn"
ID_COLUMN = "customerID"


def clean_telco(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw Telco records using rules that also work at inference."""
    out = df.copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    out["TotalCharges"] = out["TotalCharges"].fillna(out["MonthlyCharges"] * out["tenure"])
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Shared offline/online feature definitions to prevent training-serving skew."""
    out = clean_telco(df)
    out["avg_charge_per_month"] = out["TotalCharges"] / out["tenure"].clip(lower=1)
    out["charge_gap"] = out["MonthlyCharges"] - out["avg_charge_per_month"]
    out["tenure_years"] = out["tenure"] / 12.0
    out["is_new_customer"] = (out["tenure"] <= 6).astype(int)
    out["is_month_to_month"] = (out["Contract"] == "Month-to-month").astype(int)
    out["has_auto_payment"] = (
        out["PaymentMethod"]
        .str.contains("automatic", case=False, na=False)
        .astype(int)
    )
    protection_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
    ]
    out["support_services_count"] = (out[protection_cols] == "Yes").sum(axis=1)
    streaming_cols = ["StreamingTV", "StreamingMovies"]
    out["streaming_services_count"] = (out[streaming_cols] == "Yes").sum(axis=1)
    out["estimated_ltv"] = out["MonthlyCharges"] * (out["tenure"] + 1)
    return out.replace([np.inf, -np.inf], np.nan)


def split_xy(df: pd.DataFrame):
    featured = engineer_features(df)
    y = featured[TARGET].map({"No": 0, "Yes": 1}).astype(int)
    X = featured.drop(columns=[TARGET, ID_COLUMN], errors="ignore")
    return X, y
