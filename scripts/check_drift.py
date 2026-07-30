from pathlib import Path
from churn_ml.monitoring import check_batch

ROOT = Path(__file__).resolve().parents[1]
result = check_batch(
    ROOT / "data/incoming/recent_batch.csv",
    ROOT / "models/reference_stats.json",
    ROOT / "artifacts/monitoring/drift_report.json",
)
print(result)
