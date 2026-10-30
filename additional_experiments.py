"""Syllabus-fit regression and unsupervised experiments on real transaction rows."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import (
    AgglomerativeClustering,
    Birch,
    DBSCAN,
    KMeans,
    SpectralClustering,
)
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    homogeneity_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from fraud_detection import RANDOM_STATE, load_dataset

CLUSTERING_ALGORITHMS = (
    "K-Means",
    "K-Medoids",
    "Hierarchical · Ward",
    "Hierarchical · Average",
    "MST · single-linkage",
    "DBSCAN",
    "Spectral",
    "BIRCH",
    "CURE",
    "LOF · outlier detection",
)


def train_amount_regressors(training_rows: int = 8_000) -> dict[str, Any]:
    """Compare real-amount regression models with an untouched test split."""
    data = load_dataset()
    features = data[[*(f"V{i}" for i in range(1, 29)), "Time"]]
    target = np.log1p(data["Amount"].to_numpy(dtype=np.float64))
    X_train_all, X_test, y_train_all, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    if training_rows < 1_000:
        raise ValueError("Choose at least 1,000 training rows for regression.")
    if training_rows < len(X_train_all):
        X_train, _, y_train, _ = train_test_split(
            X_train_all,
            y_train_all,
            train_size=training_rows,
            random_state=RANDOM_STATE,
        )
    else:
        X_train, y_train = X_train_all, y_train_all

    models = {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Polynomial Regression (degree 2)": make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=2, include_bias=False),
            StandardScaler(),
            Ridge(alpha=1.0),
        ),
        "Ridge Regression": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "Lasso Regression": make_pipeline(
            StandardScaler(),
            Lasso(alpha=0.005, max_iter=5_000, random_state=RANDOM_STATE),
        ),
    }
    actual = np.expm1(y_test).clip(min=0)
    predictions: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, float]] = {}
    maximum_log_amount = float(y_train.max())
    for name, model in models.items():
        model.fit(X_train, y_train)
        predicted_log_amount = np.clip(model.predict(X_test), 0.0, maximum_log_amount)
        prediction = np.expm1(predicted_log_amount)
        predictions[name] = prediction
        metrics[name] = {
            "MAE": float(mean_absolute_error(actual, prediction)),
            "RMSE": float(np.sqrt(mean_squared_error(actual, prediction))),
            "R2": float(r2_score(actual, prediction)),
        }
    return {
        "models": models,
        "predictions": predictions,
        "actual_amount": actual,
        "metrics": metrics,
        "training_rows": len(X_train),
        "test_rows": len(X_test),
    }


def _kmedoids(values: np.ndarray, clusters: int) -> np.ndarray:
    from sklearn.metrics import pairwise_distances

    distances = pairwise_distances(values)
    rng = np.random.default_rng(RANDOM_STATE)
    medoids = rng.choice(len(values), size=clusters, replace=False)
    for _ in range(30):
        labels = distances[:, medoids].argmin(axis=1)
        updated = medoids.copy()
        for cluster in range(clusters):
            members = np.flatnonzero(labels == cluster)
            if len(members):
                costs = distances[np.ix_(members, members)].sum(axis=1)
                updated[cluster] = members[int(costs.argmin())]
        if np.array_equal(updated, medoids):
            break
        medoids = updated
    return distances[:, medoids].argmin(axis=1)


def _cure(values: np.ndarray, clusters: int, representatives: int = 4) -> np.ndarray:
    """Small-sample CURE-style representative clustering with centroid shrinkage."""
    if len(values) > 250:
        raise ValueError("CURE's teaching implementation is limited to 250 rows.")
    active: list[dict[str, Any]] = [
        {"members": [index], "representatives": values[[index]]}
        for index in range(len(values))
    ]
    while len(active) > clusters:
        best: tuple[float, int, int] | None = None
        for left in range(len(active)):
            for right in range(left + 1, len(active)):
                pairwise = np.linalg.norm(
                    active[left]["representatives"][:, None, :]
                    - active[right]["representatives"][None, :, :],
                    axis=2,
                )
                candidate = (float(pairwise.min()), left, right)
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            raise RuntimeError("CURE could not find a cluster pair to merge.")
        _, left, right = best
        merged_members = active[left]["members"] + active[right]["members"]
        points = values[merged_members]
        center = points.mean(axis=0)
        representatives_indices = [int(np.argmax(np.linalg.norm(points - center, axis=1)))]
        while len(representatives_indices) < min(representatives, len(points)):
            distances = np.linalg.norm(
                points[:, None, :] - points[representatives_indices][None, :, :],
                axis=2,
            ).min(axis=1)
            distances[representatives_indices] = -1
            representatives_indices.append(int(np.argmax(distances)))
        representative_points = points[representatives_indices]
        shrunk = center + 0.5 * (representative_points - center)
        active[left] = {"members": merged_members, "representatives": shrunk}
        active.pop(right)
    labels = np.empty(len(values), dtype=np.int16)
    for label, cluster in enumerate(active):
        labels[cluster["members"]] = label
    return labels


def cluster_real_transactions(
    algorithm: str,
    sample_rows: int = 500,
    clusters: int = 4,
) -> dict[str, Any]:
    """Fit feature-only clusters, then compare them with labels post-fit."""
    if algorithm not in CLUSTERING_ALGORITHMS:
        raise ValueError(f"Unknown clustering algorithm: {algorithm}")
    maximum = 250 if algorithm == "CURE" else 2_000
    if not 200 <= sample_rows <= maximum:
        raise ValueError(f"{algorithm} supports samples from 200 to {maximum:,} rows.")
    if not 2 <= clusters < sample_rows:
        raise ValueError("Choose at least 2 clusters and fewer than the sample size.")
    data = load_dataset().sample(n=sample_rows, random_state=RANDOM_STATE)
    raw = data[[*(f"V{i}" for i in range(1, 9))]].to_numpy(dtype=np.float64)
    scaled = StandardScaler().fit_transform(raw)
    values = PCA(n_components=5, random_state=RANDOM_STATE).fit_transform(scaled)
    fraud_labels = data["Class"].to_numpy(dtype=np.int8)

    if algorithm == "K-Means":
        labels = KMeans(n_clusters=clusters, n_init=10, random_state=RANDOM_STATE).fit_predict(values)
    elif algorithm == "K-Medoids":
        labels = _kmedoids(values, clusters)
    elif algorithm == "Hierarchical · Ward":
        labels = AgglomerativeClustering(n_clusters=clusters, linkage="ward").fit_predict(values)
    elif algorithm == "Hierarchical · Average":
        labels = AgglomerativeClustering(n_clusters=clusters, linkage="average").fit_predict(values)
    elif algorithm == "MST · single-linkage":
        labels = AgglomerativeClustering(n_clusters=clusters, linkage="single").fit_predict(values)
    elif algorithm == "DBSCAN":
        labels = DBSCAN(eps=0.8, min_samples=10).fit_predict(values)
    elif algorithm == "Spectral":
        labels = SpectralClustering(
            n_clusters=clusters,
            affinity="nearest_neighbors",
            n_neighbors=min(10, sample_rows - 1),
            assign_labels="kmeans",
            random_state=RANDOM_STATE,
        ).fit_predict(values)
    elif algorithm == "BIRCH":
        labels = Birch(n_clusters=clusters, threshold=0.7).fit_predict(values)
    elif algorithm == "CURE":
        labels = _cure(values, clusters)
    else:
        labels = np.where(
            LocalOutlierFactor(
                n_neighbors=min(20, sample_rows - 1),
                contamination=0.01,
            ).fit_predict(values)
            == -1,
            1,
            0,
        )

    if algorithm == "LOF · outlier detection":
        anomaly = labels == 1
        metrics: dict[str, float | int | None] = {
            "outliers": int(anomaly.sum()),
            "precision": float(
                np.count_nonzero(anomaly & (fraud_labels == 1)) / max(int(anomaly.sum()), 1)
            ),
            "recall": float(
                np.count_nonzero(anomaly & (fraud_labels == 1))
                / max(int((fraud_labels == 1).sum()), 1)
            ),
        }
    else:
        valid = labels != -1
        valid_labels = labels[valid]
        valid_values = values[valid]
        unique = np.unique(valid_labels)
        internal = 2 <= len(unique) < len(valid_labels)
        external = len(np.unique(fraud_labels[valid])) == 2 and valid.any()
        metrics = {
            "silhouette": float(silhouette_score(valid_values, valid_labels)) if internal else None,
            "davies_bouldin": float(davies_bouldin_score(valid_values, valid_labels)) if internal else None,
            "calinski_harabasz": float(calinski_harabasz_score(valid_values, valid_labels)) if internal else None,
            "homogeneity": float(homogeneity_score(fraud_labels[valid], valid_labels)) if external else None,
            "completeness": float(completeness_score(fraud_labels[valid], valid_labels)) if external else None,
            "adjusted_rand_index": float(adjusted_rand_score(fraud_labels[valid], valid_labels)) if external else None,
            "noise_rows": int((labels == -1).sum()),
        }
    return {
        "algorithm": algorithm,
        "rows": sample_rows,
        "labels": labels,
        "projection": values[:, :2],
        "fraud_labels": fraud_labels,
        "cluster_count": int(len(np.unique(labels[labels != -1]))),
        "metrics": metrics,
    }
