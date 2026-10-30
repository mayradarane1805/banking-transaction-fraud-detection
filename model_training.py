"""Syllabus-informed fraud-model comparisons on real OpenML transactions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from fraud_detection import FEATURE_COLUMNS, RANDOM_STATE, load_dataset

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

CLASSIFICATION_ALGORITHMS = (
    "Logistic Regression",
    "Logistic Regression · selected features",
    "k-NN · Euclidean",
    "k-NN · Manhattan",
    "SVM · linear",
    "SVM · polynomial",
    "SVM · RBF",
    "Perceptron",
    "Decision Tree",
    "Random Forest",
    "Bagging",
    "AdaBoost",
    "Gradient Boosting",
    "Histogram Gradient Boosting",
    "MLP · ReLU",
    "MLP · tanh",
    "MLP · sigmoid",
    "Stacking",
    "Blending",
    "Voting",
    "Self-training",
    "XGBoost",
)

SCALED_ALGORITHMS = {
    "Logistic Regression",
    "Logistic Regression · selected features",
    "k-NN · Euclidean",
    "k-NN · Manhattan",
    "SVM · linear",
    "SVM · polynomial",
    "SVM · RBF",
    "Perceptron",
    "MLP · ReLU",
    "MLP · tanh",
    "MLP · sigmoid",
}


def build_classifier(name: str, neighbors: int = 11) -> Any:
    """Create one deterministic classifier with imbalance-aware settings."""
    if name not in CLASSIFICATION_ALGORITHMS:
        raise ValueError(f"Unknown classification algorithm: {name}")
    if name == "Logistic Regression":
        estimator: Any = LogisticRegression(
            class_weight="balanced",
            max_iter=1_000,
            random_state=RANDOM_STATE,
        )
    elif name == "Logistic Regression · selected features":
        return make_pipeline(
            StandardScaler(),
            SelectKBest(score_func=f_classif, k=15),
            LogisticRegression(class_weight="balanced", max_iter=1_000, random_state=RANDOM_STATE),
        )
    elif name.startswith("k-NN"):
        estimator = KNeighborsClassifier(
            n_neighbors=neighbors,
            metric="manhattan" if name.endswith("Manhattan") else "euclidean",
            weights="distance",
            n_jobs=-1,
        )
    elif name.startswith("SVM"):
        kernel = {
            "SVM · linear": "linear",
            "SVM · polynomial": "poly",
            "SVM · RBF": "rbf",
        }[name]
        estimator = SVC(
            kernel=kernel,
            class_weight="balanced",
            probability=True,
            cache_size=512,
            random_state=RANDOM_STATE,
        )
    elif name == "Perceptron":
        estimator = CalibratedClassifierCV(
            estimator=Perceptron(
                class_weight="balanced",
                max_iter=1_000,
                random_state=RANDOM_STATE,
            ),
            cv=3,
        )
    elif name == "Decision Tree":
        return DecisionTreeClassifier(
            max_depth=12,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    elif name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=16,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    elif name == "Bagging":
        return BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
            n_estimators=40,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    elif name == "AdaBoost":
        return AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE),
            n_estimators=60,
            learning_rate=0.5,
            random_state=RANDOM_STATE,
        )
    elif name == "Gradient Boosting":
        return GradientBoostingClassifier(
            n_estimators=70,
            learning_rate=0.05,
            max_depth=2,
            random_state=RANDOM_STATE,
        )
    elif name == "Histogram Gradient Boosting":
        return HistGradientBoostingClassifier(
            max_iter=100,
            max_leaf_nodes=21,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    elif name.startswith("MLP"):
        estimator = MLPClassifier(
            hidden_layer_sizes=(48, 24),
            activation={
                "MLP · ReLU": "relu",
                "MLP · tanh": "tanh",
                "MLP · sigmoid": "logistic",
            }[name],
            early_stopping=True,
            max_iter=100,
            random_state=RANDOM_STATE,
        )
    elif name in {"Stacking", "Voting"}:
        estimators = [
            (
                "logistic",
                make_pipeline(
                    StandardScaler(),
                    LogisticRegression(class_weight="balanced", max_iter=1_000),
                ),
            ),
            (
                "forest",
                RandomForestClassifier(
                    n_estimators=60,
                    max_depth=12,
                    min_samples_leaf=3,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
        if name == "Stacking":
            return StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(class_weight="balanced"),
                cv=3,
                n_jobs=1,
            )
        return VotingClassifier(estimators=estimators, voting="soft", n_jobs=1)
    elif name == "Blending":
        raise ValueError("Blending is fitted with a dedicated holdout meta-model.")
    elif name == "Self-training":
        base = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1_000),
        )
        return SelfTrainingClassifier(estimator=base, threshold=0.9, max_iter=10)
    else:
        if XGBClassifier is None:
            raise RuntimeError("XGBoost is unavailable; install the requirements.txt dependencies.")
        return XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

    return make_pipeline(StandardScaler(), estimator) if name in SCALED_ALGORITHMS else estimator


def _sample_training_rows(
    features: pd.DataFrame,
    labels: pd.Series,
    rows: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if rows < 1_000:
        raise ValueError("Use at least 1,000 training rows for the comparison.")
    if rows >= len(features):
        return features, labels
    sampled_x, _, sampled_y, _ = train_test_split(
        features,
        labels,
        train_size=rows,
        stratify=labels,
        random_state=RANDOM_STATE,
    )
    return sampled_x, sampled_y


def compare_classifiers(
    algorithms: list[str] | tuple[str, ...],
    training_rows: int = 8_000,
    test_rows: int = 10_000,
    neighbors: int = 11,
) -> dict[str, Any]:
    """Fit selected syllabus classifiers and score the same untouched test sample."""
    if not algorithms:
        raise ValueError("Select at least one classifier.")
    unknown = sorted(set(algorithms) - set(CLASSIFICATION_ALGORITHMS))
    if unknown:
        raise ValueError(f"Unknown classification algorithms: {', '.join(unknown)}")
    if len(set(algorithms)) != len(algorithms):
        raise ValueError("Choose each classification algorithm only once.")
    if not 3 <= neighbors <= 31:
        raise ValueError("The k-NN neighbor count must be between 3 and 31.")

    data = load_dataset()
    features = data[FEATURE_COLUMNS]
    labels = data["Class"]
    X_development, X_test, y_development, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=RANDOM_STATE,
    )
    if test_rows < 500 or test_rows > len(X_test):
        raise ValueError(f"Test sample size must be from 500 to {len(X_test):,}.")
    X_train, y_train = _sample_training_rows(X_development, y_development, training_rows)
    if test_rows < len(X_test):
        X_test, _, y_test, _ = train_test_split(
            X_test,
            y_test,
            train_size=test_rows,
            stratify=y_test,
            random_state=RANDOM_STATE,
        )

    results: dict[str, dict[str, Any]] = {}
    models: dict[str, Any] = {}
    for name in algorithms:
        if name.startswith("SVM") and len(X_train) > 6_000:
            raise ValueError("SVM probability experiments are capped at 6,000 training rows.")
        if name.startswith("k-NN") and len(X_train) > 12_000:
            raise ValueError("k-NN experiments are capped at 12,000 training rows.")
        if name == "Self-training":
            rng = np.random.default_rng(RANDOM_STATE)
            pseudo_labels = y_train.to_numpy(dtype=np.int8).copy()
            for class_label in (0, 1):
                class_rows = np.flatnonzero(pseudo_labels == class_label)
                hide_count = min(
                    int(len(class_rows) * 0.3),
                    max(0, len(class_rows) - 1),
                )
                if hide_count:
                    hidden = rng.choice(class_rows, size=hide_count, replace=False)
                    pseudo_labels[hidden] = -1
            fit_labels = pseudo_labels
        else:
            fit_labels = y_train

        if name == "Blending":
            X_base, X_meta, y_base, y_meta = train_test_split(
                X_train,
                y_train,
                test_size=0.2,
                stratify=y_train,
                random_state=RANDOM_STATE,
            )
            base_models = [
                build_classifier("Logistic Regression"),
                build_classifier("Random Forest"),
            ]
            for base_model in base_models:
                base_model.fit(X_base, y_base)
            meta_features = np.column_stack(
                [base_model.predict_proba(X_meta)[:, 1] for base_model in base_models]
            )
            meta_model = LogisticRegression(class_weight="balanced", max_iter=1_000)
            meta_model.fit(meta_features, y_meta)
            test_meta = np.column_stack(
                [base_model.predict_proba(X_test)[:, 1] for base_model in base_models]
            )
            probabilities = meta_model.predict_proba(test_meta)[:, 1]
            model = (base_models, meta_model)
        else:
            model = build_classifier(name, neighbors=neighbors)
        if name == "XGBoost":
            positive_count = max(1, int(np.count_nonzero(np.asarray(fit_labels) == 1)))
            negative_count = int(np.count_nonzero(np.asarray(fit_labels) == 0))
            model.set_params(scale_pos_weight=negative_count / positive_count)
        if name != "Blending":
            model.fit(X_train, fit_labels)
            probabilities = np.asarray(model.predict_proba(X_test)[:, 1], dtype=np.float64)
        predictions = probabilities >= 0.5
        results[name] = {
            "probabilities": probabilities,
            "metrics": {
                "accuracy": float(accuracy_score(y_test, predictions)),
                "precision": float(precision_score(y_test, predictions, zero_division=0)),
                "recall": float(recall_score(y_test, predictions, zero_division=0)),
                "f1": float(f1_score(y_test, predictions, zero_division=0)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
                "average_precision": float(average_precision_score(y_test, probabilities)),
                "roc_auc": float(roc_auc_score(y_test, probabilities)),
                "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]),
            },
        }
        models[name] = model

    return {
        "algorithms": list(algorithms),
        "models": models,
        "results": results,
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "test_fraud_rows": int(y_test.sum()),
        "y_test": y_test,
        "X_test": X_test,
        "notes": {
            "Self-training": "30% of training labels were hidden from the learner; test labels remained untouched.",
            "SVM · linear": "SVM fits are capped at 6,000 rows for interactive runtime.",
            "SVM · RBF": "SVM fits are capped at 6,000 rows for interactive runtime.",
        },
    }


def feature_importance(model: Any, feature_names: list[str]) -> pd.DataFrame:
    """Return global model importance where the fitted estimator exposes it."""
    estimator = model
    names = feature_names
    if hasattr(model, "named_steps"):
        selector = model.named_steps.get("selectkbest")
        if selector is not None:
            names = [
                feature
                for feature, keep in zip(feature_names, selector.get_support())
                if keep
            ]
        estimator = list(model.named_steps.values())[-1]
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=np.float64)
    elif hasattr(estimator, "coef_"):
        values = np.mean(np.abs(np.asarray(estimator.coef_, dtype=np.float64)), axis=0)
    else:
        return pd.DataFrame(columns=["Feature", "Importance"])
    return (
        pd.DataFrame({"Feature": names, "Importance": values})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
