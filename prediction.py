"""Single-row scoring and model-level explanation helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd

from data_processing import validate_transaction_features
from fraud_detection import predict_transaction
from model_training import feature_importance


def score_transaction(
    model: Any,
    transaction: pd.DataFrame,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Validate a single row and return its immediate probability/risk response."""
    if not 0 <= threshold <= 1:
        raise ValueError("The classification threshold must be between 0 and 1.")
    features = validate_transaction_features(transaction, max_rows=1)
    if len(features) != 1:
        raise ValueError("Single-transaction scoring requires exactly one row.")
    probability = predict_transaction(model, features)
    prediction = probability >= threshold
    risk = "High" if probability >= 0.75 else "Medium" if probability >= 0.30 else "Low"
    return {
        "fraud_probability": probability,
        "predicted_class": "Fraud" if prediction else "Normal",
        "risk_level": risk,
        "threshold": float(threshold),
        "leading_global_features": feature_importance(model, list(features.columns)).head(5),
        "explanation_note": (
            "Global model influence, not a local causal explanation. "
            "V1–V28 are anonymized PCA features."
        ),
    }
