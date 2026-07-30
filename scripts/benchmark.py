import json
import statistics
import time
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient
from churn_ml.api import app

ROOT = Path(__file__).resolve().parents[1]
rows = (
    pd.read_csv(ROOT / "data/raw/telco_churn.csv")
    .drop(columns=["customerID", "Churn"])
    .head(250)
    .to_dict("records")
)
client = TestClient(app)
latencies = []
for row in rows:
    start = time.perf_counter()
    response = client.post("/predict", json=row)
    response.raise_for_status()
    latencies.append((time.perf_counter() - start) * 1000)
latencies.sort()
report = {
    "requests": len(latencies),
    "average_latency_ms": round(statistics.mean(latencies), 3),
    "p95_latency_ms": round(
        latencies[int(0.95 * len(latencies)) - 1],
        3,
    ),
    "throughput_requests_per_second": round(
        len(latencies) / (sum(latencies) / 1000),
        2,
    ),
}
(ROOT / "artifacts/eval/benchmark.json").write_text(json.dumps(report, indent=2))
print(report)
