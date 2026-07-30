import pandas as pd
from churn_ml.features import engineer_features


def test_engineered_features_are_shared_and_finite():
    row = pd.DataFrame(
        [
            {
                "tenure": 12,
                "MonthlyCharges": 50.0,
                "TotalCharges": 600.0,
                "Contract": "Month-to-month",
                "PaymentMethod": "Bank transfer (automatic)",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "Yes",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
            }
        ]
    )
    out = engineer_features(row)
    assert out.loc[0, "support_services_count"] == 2
    assert out.loc[0, "streaming_services_count"] == 1
    assert out.loc[0, "has_auto_payment"] == 1
    assert out.loc[0, "avg_charge_per_month"] == 50.0
