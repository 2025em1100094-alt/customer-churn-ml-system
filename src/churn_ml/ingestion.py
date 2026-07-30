from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def ingest(source: Path, destination: Path, log_path: Path) -> dict:
    """Merge a micro-batch into the training table and write an audit event."""
    incoming = pd.read_csv(source)
    existing = (
        pd.read_csv(destination)
        if destination.exists()
        else pd.DataFrame()
    )
    combined = pd.concat([existing, incoming], ignore_index=True)
    if "customerID" in combined:
        combined = combined.drop_duplicates(subset=["customerID"], keep="last")
    destination.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(destination, index=False)
    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "rows_read": len(incoming),
        "rows_after_merge": len(combined),
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as handle:
        handle.write(json.dumps(event) + "\n")
    return event
