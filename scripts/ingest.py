import argparse
from pathlib import Path
from churn_ml.ingestion import ingest

parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path)
parser.add_argument("--destination", type=Path, default=Path("data/processed/training_data.csv"))
parser.add_argument("--log", type=Path, default=Path("artifacts/monitoring/ingestion.jsonl"))
args = parser.parse_args()
print(ingest(args.source, args.destination, args.log))
