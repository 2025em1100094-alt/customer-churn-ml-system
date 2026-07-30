# Telco Customer Churn - Mini Production ML System

This repository predicts whether a telecom customer will churn and demonstrates a reproducible production-style ML lifecycle: ingestion, shared feature engineering, baseline/candidate evaluation, model promotion, online serving, performance measurement, monitoring, and retraining triggers.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
PYTHONPATH=src .venv/bin/python scripts/train.py
PYTHONPATH=src .venv/bin/pytest -q
PYTHONPATH=src .venv/bin/uvicorn churn_ml.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs`, or POST a customer JSON record to `/predict`. The response contains the binary prediction, churn probability, and model version.

## Main artifacts

- `data/raw/telco_churn.csv`: public IBM Telco dataset (7,043 records)
- `src/churn_ml/features.py`: shared offline/online feature definitions
- `scripts/train.py`: repeatable training and promotion pipeline
- `src/churn_ml/api.py`: FastAPI online inference service
- `scripts/ingest.py`: idempotent micro-batch ingestion
- `scripts/check_drift.py`: data-quality and mean-shift checks
- `scripts/benchmark.py`: average/p95 latency and throughput
- `tests/`: feature, API, ingestion, and retraining tests
- `artifacts/evidence/`: recorded outputs from the latest verified execution
- `docs/`: architecture and design report source

The verified assignment run completed five automated tests and evaluated 250 API requests. The exact outputs are retained under `artifacts/evidence/` and `artifacts/eval/`.

## Reproducibility

Random seeds and promotion thresholds are versioned in `configs/model.json`. Generated model and evaluation reports are saved under `models/` and `artifacts/`. The raw data SHA-256 is `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`.

Dataset source: IBM, Telco Customer Churn sample, https://github.com/IBM/telco-customer-churn-on-icp4d

Assignment repository: https://github.com/2025em1100094-alt/customer-churn-ml-system
