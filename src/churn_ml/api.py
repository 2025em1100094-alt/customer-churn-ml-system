from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .features import engineer_features

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.joblib"))


class Customer(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=100)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


@lru_cache
def load_model():
    """Load the trained pipeline once and reuse it for later requests."""
    return joblib.load(MODEL_PATH)


app = FastAPI(title="Telco Churn Prediction API", version="1.0.0")


@app.get("/health")
def health():
    bundle = load_model()
    return {"status": "ok", "model_version": bundle["model_version"]}


@app.post("/predict")
def predict(customer: Customer):
    bundle = load_model()
    frame = pd.DataFrame([customer.model_dump()])
    features = engineer_features(frame)
    probability = float(
        bundle["pipeline"].predict_proba(features)[:, 1][0]
    )
    prediction = int(probability >= bundle["threshold"])

    return {
        "prediction": prediction,
        "churn_probability": round(probability, 4),
        "model_version": bundle["model_version"],
    }
