from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import engineer_features, split_xy


def load_config(path: Path) -> dict:
    return json.loads(path.read_text())


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing fitted only on the training split."""
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                categorical,
            ),
        ]
    )


def score(model, X, y) -> dict:
    """Calculate threshold-independent and threshold-based test metrics."""
    probability = model.predict_proba(X)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y, probability)), 4),
        "accuracy": round(float(accuracy_score(y, prediction)), 4),
        "precision": round(float(precision_score(y, prediction, zero_division=0)), 4),
        "recall": round(float(recall_score(y, prediction, zero_division=0)), 4),
        "f1": round(float(f1_score(y, prediction, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y, prediction).tolist(),
    }


def train(data_path: Path, config_path: Path, model_dir: Path, eval_dir: Path) -> dict:
    """Train baseline and candidate pipelines and save the promoted artifact."""
    cfg = load_config(config_path)
    raw = pd.read_csv(data_path)
    X, y = split_xy(raw)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["test_size"], stratify=y, random_state=cfg["random_state"]
    )
    baseline = Pipeline(
        [
            ("preprocessor", make_preprocessor(X_train)),
            (
                "model",
                LogisticRegression(
                    max_iter=1500,
                    class_weight="balanced",
                    random_state=cfg["random_state"],
                ),
            ),
        ]
    )
    candidate = Pipeline(
        [
            ("preprocessor", make_preprocessor(X_train)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=cfg["candidate_n_estimators"],
                    max_depth=cfg["candidate_max_depth"],
                    min_samples_leaf=cfg["candidate_min_samples_leaf"],
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=cfg["random_state"],
                ),
            ),
        ]
    )
    baseline.fit(X_train, y_train)
    candidate.fit(X_train, y_train)
    baseline_metrics = score(baseline, X_test, y_test)
    candidate_metrics = score(candidate, X_test, y_test)
    promote = (
        candidate_metrics["roc_auc"] >= cfg["promotion_min_auc"]
        and candidate_metrics["roc_auc"]
        >= baseline_metrics["roc_auc"] - cfg["promotion_max_auc_drop"]
    )
    promoted = candidate if promote else baseline
    promoted_name = "candidate_random_forest" if promote else "baseline_logistic_regression"
    model_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    model_bundle = {
        "pipeline": promoted,
        "model_version": cfg["model_version"],
        "model_name": promoted_name,
        "threshold": cfg["decision_threshold"],
    }
    joblib.dump(model_bundle, model_dir / "model.joblib")
    reference = engineer_features(raw).drop(columns=["Churn", "customerID"], errors="ignore")
    numeric_ref = {
        column: {
            "mean": float(reference[column].mean()),
            "std": float(reference[column].std(ddof=0)),
            "missing_rate": float(reference[column].isna().mean()),
        }
        for column in reference.select_dtypes(include="number").columns
    }
    reference_report = {
        "row_count": len(reference),
        "numeric": numeric_ref,
    }
    (model_dir / "reference_stats.json").write_text(
        json.dumps(reference_report, indent=2)
    )
    report = {
        "data_rows": len(raw),
        "test_rows": len(X_test),
        "positive_rate": round(float(y.mean()), 4),
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "promotion_rule": {
            "minimum_auc": cfg["promotion_min_auc"],
            "maximum_drop_vs_baseline": cfg["promotion_max_auc_drop"],
        },
        "promoted": promoted_name,
        "model_version": cfg["model_version"],
    }
    (eval_dir / "evaluation.json").write_text(json.dumps(report, indent=2))
    return report
