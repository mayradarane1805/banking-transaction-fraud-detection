"""Advanced model validation, selection, and threshold utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_processing import validate_transaction_features
from fraud_detection import FEATURE_COLUMNS, RANDOM_STATE, load_dataset

MAX_THRESHOLD = float(np.nextafter(1.0, np.inf))


def build_advanced_models() -> dict[str, Any]:
    """Create diverse, class-imbalance-aware models for advanced comparison."""
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=1_000,
                random_state=RANDOM_STATE,
            ),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=16,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "Histogram Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.1,
            max_leaf_nodes=31,
            class_weight="balanced",
            early_stopping=True,
            random_state=RANDOM_STATE,
        ),
    }


def train_advanced_models(training_rows: int = 40_000) -> dict[str, Any]:
    """Cross-validate candidates and preserve independent validation and test sets."""
    data = load_dataset()
    features = data[FEATURE_COLUMNS]
    labels = data["Class"]

    X_development, X_test, y_development, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_development,
        y_development,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_development,
    )
    if training_rows < 1_000:
        raise ValueError("Choose at least 1,000 training transactions.")
    if training_rows > len(X_train):
        raise ValueError(f"Training size cannot exceed {len(X_train):,} rows.")
    if training_rows < len(X_train):
        X_train, _, y_train, _ = train_test_split(
            X_train,
            y_train,
            train_size=training_rows,
            random_state=RANDOM_STATE,
            stratify=y_train,
        )

    scoring = {
        "average_precision": "average_precision",
        "roc_auc": "roc_auc",
        "balanced_accuracy": "balanced_accuracy",
    }
    cross_validator = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    candidates: dict[str, Any] = {}
    cross_validation: dict[str, dict[str, float]] = {}
    validation_probabilities: dict[str, np.ndarray] = {}
    test_probabilities: dict[str, np.ndarray] = {}

    for name, model in build_advanced_models().items():
        fold_results = cross_validate(
            model,
            X_train,
            y_train,
            scoring=scoring,
            cv=cross_validator,
            n_jobs=1,
            error_score="raise",
        )
        cross_validation[name] = {
            metric: float(np.mean(fold_results[f"test_{metric}"]))
            for metric in scoring
        } | {
            f"{metric}_std": float(np.std(fold_results[f"test_{metric}"], ddof=1))
            for metric in scoring
        }

        model.fit(X_train, y_train)
        candidates[name] = model
        validation_probabilities[name] = model.predict_proba(X_validation)[:, 1]
        test_probabilities[name] = model.predict_proba(X_test)[:, 1]

    selected_model_name = max(
        cross_validation,
        key=lambda name: cross_validation[name]["average_precision"],
    )
    return {
        "models": candidates,
        "cross_validation": cross_validation,
        "selected_model_name": selected_model_name,
        "validation_probabilities": validation_probabilities,
        "test_probabilities": test_probabilities,
        "X_test": X_test,
        "y_validation": y_validation,
        "y_test": y_test,
        "train_rows": len(X_train),
        "train_fraud_rows": int(y_train.sum()),
        "validation_rows": len(X_validation),
        "validation_fraud_rows": int(y_validation.sum()),
        "test_rows": len(X_test),
        "test_fraud_rows": int(y_test.sum()),
    }


def select_cost_threshold(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    false_positive_cost: float,
    false_negative_cost: float,
) -> dict[str, float | int]:
    """Choose the validation threshold with lowest explicitly configured error cost."""
    actual = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(probabilities, dtype=np.float64)
    if actual.ndim != 1 or scores.ndim != 1 or len(actual) != len(scores) or not len(actual):
        raise ValueError("Labels and scores must be non-empty matching 1D arrays.")
    if set(np.unique(actual)) != {0, 1}:
        raise ValueError("Threshold selection requires both normal and fraud labels.")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Prediction probabilities must be finite values between 0 and 1.")
    if (
        not np.isfinite(false_positive_cost)
        or not np.isfinite(false_negative_cost)
        or false_positive_cost <= 0
        or false_negative_cost <= 0
    ):
        raise ValueError("Both false-positive and false-negative costs must be positive.")

    sorted_indices = np.argsort(scores, kind="stable")
    sorted_scores = scores[sorted_indices]
    sorted_labels = actual[sorted_indices]
    cumulative_fraud = np.cumsum(sorted_labels, dtype=np.int64)
    thresholds = np.unique(
        np.concatenate(([0.0], scores, [MAX_THRESHOLD]))
    )
    first_predicted_index = np.searchsorted(sorted_scores, thresholds, side="left")
    fraud_below_threshold = np.where(
        first_predicted_index > 0,
        cumulative_fraud[np.maximum(first_predicted_index - 1, 0)],
        0,
    )
    total_fraud = int(actual.sum())
    false_negatives = fraud_below_threshold
    true_positives = total_fraud - false_negatives
    predicted_positive = len(actual) - first_predicted_index
    false_positives = predicted_positive - true_positives
    costs = false_positives * false_positive_cost + false_negatives * false_negative_cost
    best_index = int(np.argmin(costs))

    return {
        "threshold": float(thresholds[best_index]),
        "false_positives": int(false_positives[best_index]),
        "false_negatives": int(false_negatives[best_index]),
        "total_cost": float(costs[best_index]),
    }


def evaluate_advanced_model(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    false_positive_cost: float,
    false_negative_cost: float,
) -> dict[str, float | int]:
    """Summarize final test-set ranking, calibration, and cost at a fixed threshold."""
    actual = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(probabilities, dtype=np.float64)
    if actual.ndim != 1 or scores.ndim != 1 or len(actual) != len(scores) or not len(actual):
        raise ValueError("Labels and scores must be non-empty matching 1D arrays.")
    if set(np.unique(actual)) != {0, 1}:
        raise ValueError("Model evaluation requires both normal and fraud labels.")
    if not 0 <= threshold <= MAX_THRESHOLD:
        raise ValueError("The classification threshold must be between 0 and 1.")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Prediction probabilities must be finite values between 0 and 1.")
    if (
        not np.isfinite(false_positive_cost)
        or not np.isfinite(false_negative_cost)
        or false_positive_cost <= 0
        or false_negative_cost <= 0
    ):
        raise ValueError("Both false-positive and false-negative costs must be positive.")

    predicted = scores >= threshold
    actual_fraud = actual == 1
    true_positives = int(np.count_nonzero(predicted & actual_fraud))
    false_positives = int(np.count_nonzero(predicted & ~actual_fraud))
    false_negatives = int(np.count_nonzero(~predicted & actual_fraud))
    return {
        "average_precision": float(average_precision_score(actual, scores)),
        "roc_auc": float(roc_auc_score(actual, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "brier_score": float(brier_score_loss(actual, scores)),
        "precision": float(true_positives / (true_positives + false_positives))
        if true_positives + false_positives
        else 0.0,
        "recall": float(true_positives / (true_positives + false_negatives))
        if true_positives + false_negatives
        else 0.0,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "total_cost": float(
            false_positives * false_positive_cost
            + false_negatives * false_negative_cost
        ),
    }


def score_batch_transactions(
    model: Any,
    batch: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Validate feature-only transaction rows and append model scores and labels."""
    if not 0 <= threshold <= MAX_THRESHOLD:
        raise ValueError("The classification threshold must be between 0 and 1.")
    transactions = validate_transaction_features(batch)

    probabilities = np.asarray(model.predict_proba(transactions)[:, 1], dtype=np.float64)
    if len(probabilities) != len(transactions) or not np.isfinite(probabilities).all():
        raise ValueError("The model returned invalid prediction probabilities.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("The model returned probabilities outside the 0-to-1 range.")

    scored = transactions.copy()
    scored["fraud_probability"] = probabilities
    scored["predicted_class"] = np.where(probabilities >= threshold, "Fraud", "Normal")
    return scored
