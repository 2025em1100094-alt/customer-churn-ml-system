from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .features import engineer_features


def check_batch(
    batch_path: Path,
    reference_path: Path,
    output_path: Path,
    z_threshold: float = 1.0,
) -> dict:
    """Check missing values and standardised mean shifts in a recent batch."""
    batch = engineer_features(pd.read_csv(batch_path))
    reference = json.loads(reference_path.read_text())
    warnings = []
    checks = {}
    for column in ["MonthlyCharges", "TotalCharges", "tenure", "avg_charge_per_month"]:
        current_mean = float(batch[column].mean())
        ref = reference["numeric"][column]
        z_shift = abs(current_mean - ref["mean"]) / max(ref["std"], 1e-9)
        missing_rate = float(batch[column].isna().mean())
        checks[column] = {
            "current_mean": round(current_mean, 4),
            "reference_mean": round(ref["mean"], 4),
            "mean_shift_std": round(z_shift, 4),
            "missing_rate": round(missing_rate, 4),
        }
        if z_shift > z_threshold:
            warnings.append(f"{column}: mean shifted by {z_shift:.2f} standard deviations")
        if missing_rate > 0.05:
            warnings.append(f"{column}: missing rate {missing_rate:.1%} exceeds 5%")
    result = {
        "status": "warning" if warnings else "ok",
        "rows_checked": len(batch),
        "warnings": warnings,
        "checks": checks,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    return result


def should_retrain(
    new_days: int,
    recent_auc: float,
    baseline_auc: float,
    max_drift_std: float,
) -> dict:
    """Evaluate the three documented retraining signals."""
    signals = {
        "scheduled_data_window": new_days >= 30,
        "performance_drop": recent_auc < baseline_auc - 0.03,
        "feature_drift": max_drift_std > 1.0,
    }
    return {"retrain": any(signals.values()), "signals": signals}
