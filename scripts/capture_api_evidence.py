import json
from pathlib import Path

from fastapi.testclient import TestClient

from churn_ml.api import app


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/evidence/api_execution.json"

customer = {
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

client = TestClient(app)
health_response = client.get("/health")
prediction_response = client.post("/predict", json=customer)
health_response.raise_for_status()
prediction_response.raise_for_status()

evidence = {
    "method": "FastAPI TestClient",
    "health": {
        "status_code": health_response.status_code,
        "response": health_response.json(),
    },
    "prediction": {
        "status_code": prediction_response.status_code,
        "request": customer,
        "response": prediction_response.json(),
    },
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(evidence, indent=2))
print(json.dumps(evidence, indent=2))
