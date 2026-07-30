from pathlib import Path
from churn_ml.training import train

ROOT = Path(__file__).resolve().parents[1]
report = train(
    ROOT / "data/raw/telco_churn.csv",
    ROOT / "configs/model.json",
    ROOT / "models",
    ROOT / "artifacts/eval",
)
print(report)
