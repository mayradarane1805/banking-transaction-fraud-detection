"""Pure helpers for the investigation queue and live-style demo stream."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from fraud_detection import FEATURE_COLUMNS

REVIEW_STATUSES = (
    "New",
    "Investigating",
    "Confirmed fraud",
    "False positive",
    "Closed",
)


def risk_level(probability: float) -> str:
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Risk probability must be finite and between 0 and 1.")
    if probability >= 0.75:
        return "High"
    if probability >= 0.30:
        return "Medium"
    return "Low"


def build_historical_alerts(
    results: dict[str, Any],
    model_name: str,
    threshold: float,
) -> pd.DataFrame:
    """Create alerts from the actual held-out test predictions."""
    if model_name not in results["probabilities"]:
        raise ValueError(f"No held-out predictions are available for {model_name}.")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("The alert threshold must be between 0 and 1.")
    scores = np.asarray(results["probabilities"][model_name], dtype=np.float64)
    if len(scores) != len(results["X_test"]):
        raise ValueError("Held-out probabilities are not aligned with test transactions.")
    selected = np.flatnonzero(scores >= threshold)
    if not len(selected):
        return pd.DataFrame(
            columns=[
                "Transaction ID",
                "Source",
                "Amount",
                "Elapsed seconds",
                "Fraud probability",
                "Risk",
                "Model",
                "Review status",
            ]
        )
    rows = results["X_test"].iloc[selected]
    return pd.DataFrame(
        {
            "Transaction ID": [f"HIST-{index}" for index in rows.index],
            "Source": "Historical holdout",
            "Amount": rows["Amount"].to_numpy(dtype=np.float64),
            "Elapsed seconds": rows["Time"].to_numpy(dtype=np.float64),
            "Fraud probability": scores[selected],
            "Risk": [risk_level(float(value)) for value in scores[selected]],
            "Model": model_name,
            "Review status": "New",
        }
    ).sort_values("Fraud probability", ascending=False, ignore_index=True)


def create_demo_event(
    results: dict[str, Any],
    model_name: str,
    cursor: int,
    threshold: float,
) -> tuple[dict[str, Any], int]:
    """Score one held-out real row and label the stream event as demonstration data."""
    if model_name not in results["models"]:
        raise ValueError(f"No trained model is available for {model_name}.")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("The simulator threshold must be between 0 and 1.")
    test_rows = results["X_test"]
    if test_rows.empty:
        raise ValueError("The held-out transaction set is empty.")
    if cursor < 0:
        raise ValueError("The simulator cursor cannot be negative.")
    row_position = cursor % len(test_rows)
    transaction = test_rows.iloc[[row_position]].loc[:, FEATURE_COLUMNS]
    probability = float(results["models"][model_name].predict_proba(transaction)[0, 1])
    event_number = cursor + 1
    event = {
        "Transaction ID": f"DEMO-{event_number:06d}",
        "Timestamp (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "Source": "DEMO DATA",
        "Amount": float(transaction["Amount"].iloc[0]),
        "Elapsed seconds": float(transaction["Time"].iloc[0]),
        "Fraud probability": probability,
        "Predicted class": "Fraud" if probability >= threshold else "Normal",
        "Risk": risk_level(probability),
        "Model": model_name,
        "Review status": "New" if probability >= threshold else "Not flagged",
    }
    return event, event_number


def apply_review_status(
    alerts: pd.DataFrame,
    transaction_id: str,
    new_status: str,
) -> pd.DataFrame:
    """Return a copy with one validated investigation status updated."""
    if new_status not in REVIEW_STATUSES:
        raise ValueError(f"Unknown investigation status: {new_status}")
    if "Transaction ID" not in alerts.columns or transaction_id not in set(
        alerts["Transaction ID"].astype(str)
    ):
        raise ValueError(f"Transaction {transaction_id} is not in the current alert queue.")
    updated = alerts.copy()
    updated.loc[updated["Transaction ID"] == transaction_id, "Review status"] = new_status
    return updated
