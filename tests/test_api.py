import os
from pathlib import Path

os.environ["MODEL_PATH"] = str(Path(__file__).resolve().parents[1] / "models/model.joblib")

from fastapi.testclient import TestClient
from churn_ml.api import app


def test_predict_contract():
    payload = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    }
    response = TestClient(app).post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["model_version"] == "churn-rf-1.0.0"


def test_health_reports_loaded_model():
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_version": "churn-rf-1.0.0",
    }
