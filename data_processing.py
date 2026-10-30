"""Reusable validation and preprocessing helpers for transaction data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_detection import FEATURE_COLUMNS, TARGET, validate_dataset

MAX_BATCH_ROWS = 10_000


def validate_transaction_features(
    transactions: pd.DataFrame,
    max_rows: int = MAX_BATCH_ROWS,
) -> pd.DataFrame:
    """Validate feature-only input and return finite, ordered numeric columns."""
    if transactions.columns.has_duplicates:
        raise ValueError("The uploaded CSV contains duplicate column names.")
    missing = [column for column in FEATURE_COLUMNS if column not in transactions.columns]
    extra = [column for column in transactions.columns if column not in FEATURE_COLUMNS]
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing columns: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected columns: {', '.join(extra)}")
        raise ValueError("CSV schema does not match the template (" + "; ".join(details) + ").")
    if transactions.empty:
        raise ValueError("The uploaded CSV contains no transaction rows.")
    if max_rows < 1 or len(transactions) > max_rows:
        raise ValueError(f"Upload at most {max_rows:,} transaction rows at a time.")

    try:
        features = transactions.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("All transaction features must contain numeric values.") from exc

    values = features.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Transaction features must contain finite numeric values.")
    if (features[["Time", "Amount"]] < 0).any().any():
        raise ValueError("Transaction time and amount must be non-negative.")
    return features.astype("float64")


def preprocess_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Validate, select, and order the documented dataset columns."""
    return validate_dataset(data)


def dataset_quality_report(data: pd.DataFrame) -> dict[str, int | float]:
    """Return descriptive input-quality facts without silently imputing values."""
    return {
        "rows": int(len(data)),
        "columns": int(len(data.columns)),
        "missing_values": int(data.isna().sum().sum()),
        "duplicate_rows": int(data.duplicated().sum()),
        "fraud_rows": int((data[TARGET] == 1).sum()),
        "normal_rows": int((data[TARGET] == 0).sum()),
        "fraud_rate": float(data[TARGET].mean()),
    }
