from pathlib import Path

import pandas as pd

from churn_ml.ingestion import ingest
from churn_ml.monitoring import should_retrain


def test_ingestion_deduplicates(tmp_path: Path):
    source = tmp_path / "new.csv"
    dest = tmp_path / "all.csv"
    pd.DataFrame(
        [
            {"customerID": "A", "x": 1},
            {"customerID": "A", "x": 2},
        ]
    ).to_csv(source, index=False)
    event = ingest(source, dest, tmp_path / "log.jsonl")
    assert event["rows_after_merge"] == 1
    assert pd.read_csv(dest).loc[0, "x"] == 2


def test_retraining_trigger():
    result = should_retrain(
        new_days=10,
        recent_auc=0.74,
        baseline_auc=0.80,
        max_drift_std=0.2,
    )
    assert result["retrain"] is True
    assert result["signals"]["performance_drop"] is True
