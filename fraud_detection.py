"""Data access and machine-learning utilities for the fraud detection dashboard."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "creditcard.parquet"
DATASET_URL = "https://data.openml.org/datasets/0000/1597/dataset_1597.pq"
TARGET = "Class"
FEATURE_COLUMNS = ["Time", "Amount", *(f"V{i}" for i in range(1, 29))]
EXPECTED_COLUMNS = [*FEATURE_COLUMNS, TARGET]
RANDOM_STATE = 42


def _download_dataset(path: Path) -> None:
    """Download the public OpenML parquet file once and replace it atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.download")
    request = urllib.request.Request(
        DATASET_URL,
        headers={"User-Agent": "Banking-Fraud-Detection-Student-Project/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with temporary_path.open("wb") as destination:
                while chunk := response.read(1024 * 1024):
                    destination.write(chunk)
    except Exception as exc:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            "The public dataset could not be downloaded from OpenML. "
            "Check your internet connection and try again."
        ) from exc

    os.replace(temporary_path, path)


def validate_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Check that the dataset has the known schema and usable fraud labels."""
    missing_columns = [column for column in EXPECTED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")

    clean_data = data.loc[:, EXPECTED_COLUMNS].copy()
    if clean_data.empty:
        raise ValueError("The dataset has no transactions.")
    if clean_data.isna().any().any():
        raise ValueError("The dataset contains missing values in required columns.")

    try:
        clean_data[TARGET] = pd.to_numeric(clean_data[TARGET], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("The Class column must contain numeric fraud labels 0 and 1.") from exc
    labels = set(clean_data[TARGET].unique())
    if not labels.issubset({0, 1}) or labels != {0, 1}:
        raise ValueError("The Class column must contain both 0 (normal) and 1 (fraud).")
    if not np.isfinite(clean_data.to_numpy(dtype=np.float64)).all():
        raise ValueError("The dataset contains non-finite numeric values.")

    clean_data[TARGET] = clean_data[TARGET].astype("int8")
    return clean_data


@st.cache_resource(show_spinner=False)
def load_dataset(path: str = str(DATA_PATH)) -> pd.DataFrame:
    """Load and validate the real OpenML dataset, caching it for the app session."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        _download_dataset(dataset_path)

    try:
        data = pd.read_parquet(dataset_path)
    except (OSError, ValueError, ImportError) as exc:
        raise RuntimeError(
            f"Could not read the dataset at {dataset_path}. Remove that file and "
            "restart the app to download a fresh copy."
        ) from exc

    return validate_dataset(data)


def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Calculate metrics from real labels and probabilities at a chosen threshold."""
    actual = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(probabilities, dtype=np.float64)
    if actual.ndim != 1 or scores.ndim != 1 or len(actual) != len(scores):
        raise ValueError("Labels and prediction probabilities must be matching 1D arrays.")
    if not 0 <= threshold <= 1:
        raise ValueError("The classification threshold must be between 0 and 1.")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Prediction probabilities must be finite values between 0 and 1.")

    predicted = (scores >= threshold).astype(np.int8)
    return {
        "accuracy": accuracy_score(actual, predicted),
        "precision": precision_score(actual, predicted, zero_division=0),
        "recall": recall_score(actual, predicted, zero_division=0),
        "f1": f1_score(actual, predicted, zero_division=0),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=[0, 1]),
        "roc_auc": roc_auc_score(actual, scores),
        "average_precision": average_precision_score(actual, scores),
        "roc_curve": roc_curve(actual, scores),
        "precision_recall_curve": precision_recall_curve(actual, scores),
    }


def train_models(
    path: str = str(DATA_PATH),
    training_rows: int = 120_000,
    forest_trees: int = 100,
    max_depth: int = 16,
) -> dict[str, Any]:
    """Train and evaluate a decision tree and random forest on real transactions."""
    data = load_dataset(path)
    features = data[FEATURE_COLUMNS]
    labels = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    if training_rows < 1_000:
        raise ValueError("Choose at least 1,000 training transactions.")
    if training_rows < len(X_train):
        X_train, _, y_train, _ = train_test_split(
            X_train,
            y_train,
            train_size=training_rows,
            random_state=RANDOM_STATE,
            stratify=y_train,
        )

    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=max_depth,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=forest_trees,
            max_depth=max_depth,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }

    probabilities: dict[str, np.ndarray] = {}
    for model in models.values():
        model.fit(X_train, y_train)
    for name, model in models.items():
        probabilities[name] = model.predict_proba(X_test)[:, 1]

    return {
        "models": models,
        "probabilities": probabilities,
        "X_test": X_test,
        "y_test": y_test,
        "training_rows": len(X_train),
        "training_fraud_rows": int(y_train.sum()),
        "test_rows": len(X_test),
        "test_fraud_rows": int(y_test.sum()),
        "forest_trees": forest_trees,
        "max_depth": max_depth,
    }


def predict_transaction(model: Any, transaction: pd.DataFrame) -> float:
    """Return the fraud probability after enforcing the expected feature schema."""
    if list(transaction.columns) != FEATURE_COLUMNS:
        raise ValueError("Prediction input columns do not match the dataset feature schema.")
    if len(transaction) != 1:
        raise ValueError("Prediction requires exactly one transaction.")
    values = transaction.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("All transaction features must be finite numeric values.")
    if transaction["Time"].iloc[0] < 0 or transaction["Amount"].iloc[0] < 0:
        raise ValueError("Transaction time and amount must be non-negative.")
    return float(model.predict_proba(transaction)[:, 1][0])
