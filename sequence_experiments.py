"""Time-ordered HMM and linear-chain CRF teaching comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fraud_detection import FEATURE_COLUMNS, RANDOM_STATE, load_dataset


def _categorical_observations(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    amount_cut = float(train["Amount"].median())
    component_cut = float(train["V1"].median())

    def encode(frame: pd.DataFrame) -> np.ndarray:
        return (
            (frame["Amount"].to_numpy() > amount_cut).astype(np.int8) * 2
            + (frame["V1"].to_numpy() > component_cut).astype(np.int8)
        )

    return encode(train), encode(test)


def _fit_categorical_hmm(
    observations: np.ndarray,
    labels: np.ndarray,
    states: int = 2,
    symbols: int = 4,
    iterations: int = 6,
    topology: str = "ergodic",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a small discrete HMM with scaled Baum-Welch and map states post-fit."""
    if topology not in {"ergodic", "left-to-right"}:
        raise ValueError("HMM topology must be ergodic or left-to-right.")
    rng = np.random.default_rng(RANDOM_STATE)
    start = np.full(states, 1 / states, dtype=np.float64)
    transition = np.full((states, states), 0.15 / (states - 1), dtype=np.float64)
    np.fill_diagonal(transition, 0.85)
    allowed = np.ones((states, states), dtype=bool)
    if topology == "left-to-right":
        allowed = np.triu(np.ones((states, states), dtype=bool))
        transition = np.where(allowed, transition, 0.0)
        transition /= transition.sum(axis=1, keepdims=True)
        start = np.zeros(states, dtype=np.float64)
        start[0] = 1.0
    emission = rng.uniform(0.5, 1.5, size=(states, symbols))
    emission /= emission.sum(axis=1, keepdims=True)
    sequence_length = len(observations)

    for _ in range(iterations):
        forward = np.zeros((sequence_length, states), dtype=np.float64)
        scales = np.zeros(sequence_length, dtype=np.float64)
        forward[0] = start * emission[:, observations[0]]
        scales[0] = max(float(forward[0].sum()), 1e-300)
        forward[0] /= scales[0]
        for time_index in range(1, sequence_length):
            forward[time_index] = (
                forward[time_index - 1] @ transition
            ) * emission[:, observations[time_index]]
            scales[time_index] = max(float(forward[time_index].sum()), 1e-300)
            forward[time_index] /= scales[time_index]

        backward = np.ones((sequence_length, states), dtype=np.float64)
        for time_index in range(sequence_length - 2, -1, -1):
            backward[time_index] = transition @ (
                emission[:, observations[time_index + 1]] * backward[time_index + 1]
            )
            backward[time_index] /= max(scales[time_index + 1], 1e-300)

        posterior = forward * backward
        posterior /= np.maximum(posterior.sum(axis=1, keepdims=True), 1e-300)
        transition_counts = np.where(allowed, 1e-2, 0.0)
        for time_index in range(sequence_length - 1):
            next_observation = observations[time_index + 1]
            pair = (
                forward[time_index, :, None]
                * transition
                * (emission[:, next_observation] * backward[time_index + 1])[None, :]
            )
            transition_counts += pair / max(float(pair.sum()), 1e-300)
        transition = transition_counts / transition_counts.sum(axis=1, keepdims=True)

        emission_counts = np.full((states, symbols), 1e-2, dtype=np.float64)
        for symbol in range(symbols):
            emission_counts[:, symbol] += posterior[observations == symbol].sum(axis=0)
        emission = emission_counts / emission_counts.sum(axis=1, keepdims=True)
        if topology == "ergodic":
            start = posterior[0] + 1e-2
            start /= start.sum()

    state_fraud_rate = (
        posterior.T @ labels.astype(np.float64) + 1.0
    ) / (posterior.sum(axis=0) + 2.0)
    return start, transition, emission, state_fraud_rate


def _viterbi(
    observations: np.ndarray,
    start: np.ndarray,
    transition: np.ndarray,
    emission: np.ndarray,
) -> np.ndarray:
    """Decode the most likely hidden-state path using the Viterbi algorithm."""
    log_start = np.log(np.maximum(start, 1e-300))
    log_transition = np.log(np.maximum(transition, 1e-300))
    log_emission = np.log(np.maximum(emission, 1e-300))
    scores = np.empty((len(observations), len(start)), dtype=np.float64)
    backpointers = np.zeros((len(observations), len(start)), dtype=np.int16)
    scores[0] = log_start + log_emission[:, observations[0]]
    for time_index in range(1, len(observations)):
        candidates = scores[time_index - 1, :, None] + log_transition
        backpointers[time_index] = np.argmax(candidates, axis=0)
        scores[time_index] = (
            candidates[backpointers[time_index], np.arange(len(start))]
            + log_emission[:, observations[time_index]]
        )
    path = np.empty(len(observations), dtype=np.int16)
    path[-1] = int(np.argmax(scores[-1]))
    for time_index in range(len(observations) - 2, -1, -1):
        path[time_index] = backpointers[time_index + 1, path[time_index + 1]]
    return path


def _filter_hmm(
    observations: np.ndarray,
    start: np.ndarray,
    transition: np.ndarray,
    emission: np.ndarray,
    state_fraud_rate: np.ndarray,
) -> np.ndarray:
    alpha = start * emission[:, observations[0]]
    alpha /= max(float(alpha.sum()), 1e-300)
    probabilities = np.empty(len(observations), dtype=np.float64)
    probabilities[0] = alpha @ state_fraud_rate
    for time_index in range(1, len(observations)):
        alpha = (alpha @ transition) * emission[:, observations[time_index]]
        alpha /= max(float(alpha.sum()), 1e-300)
        probabilities[time_index] = alpha @ state_fraud_rate
    return probabilities


def _crf_transition_objective(
    flattened: np.ndarray,
    emissions: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, np.ndarray]:
    transitions = flattened.reshape(2, 2)
    length = len(labels)
    forward = np.empty((length, 2), dtype=np.float64)
    forward[0] = emissions[0]
    for index in range(1, length):
        previous = forward[index - 1]
        forward[index, 0] = emissions[index, 0] + np.logaddexp(
            previous[0] + transitions[0, 0],
            previous[1] + transitions[1, 0],
        )
        forward[index, 1] = emissions[index, 1] + np.logaddexp(
            previous[0] + transitions[0, 1],
            previous[1] + transitions[1, 1],
        )
    log_partition = float(logsumexp(forward[-1]))

    backward = np.zeros((length, 2), dtype=np.float64)
    for index in range(length - 2, -1, -1):
        following = emissions[index + 1] + backward[index + 1]
        backward[index, 0] = np.logaddexp(
            transitions[0, 0] + following[0],
            transitions[0, 1] + following[1],
        )
        backward[index, 1] = np.logaddexp(
            transitions[1, 0] + following[0],
            transitions[1, 1] + following[1],
        )
    log_pairs = (
        forward[:-1, :, None]
        + transitions[None, :, :]
        + emissions[1:, None, :]
        + backward[1:, None, :]
        - log_partition
    )
    expected = np.exp(log_pairs).sum(axis=0)

    empirical = np.zeros((2, 2), dtype=np.float64)
    np.add.at(empirical, (labels[:-1], labels[1:]), 1.0)
    score = float(emissions[np.arange(length), labels].sum())
    score += float(transitions[labels[:-1], labels[1:]].sum())
    regularization = 1e-2
    loss = log_partition - score + 0.5 * regularization * float(np.square(transitions).sum())
    gradient = expected - empirical + regularization * transitions
    return loss, gradient.ravel()


def _filter_crf(emissions: np.ndarray, transitions: np.ndarray) -> np.ndarray:
    alpha = emissions[0].copy()
    alpha -= float(logsumexp(alpha))
    probabilities = np.empty(len(emissions), dtype=np.float64)
    probabilities[0] = np.exp(alpha[1])
    for index in range(1, len(emissions)):
        previous = alpha
        alpha = emissions[index] + np.array(
            [
                np.logaddexp(
                    previous[0] + transitions[0, 0],
                    previous[1] + transitions[1, 0],
                ),
                np.logaddexp(
                    previous[0] + transitions[0, 1],
                    previous[1] + transitions[1, 1],
                ),
            ]
        )
        alpha -= float(logsumexp(alpha))
        probabilities[index] = np.exp(alpha[1])
    return probabilities


def run_sequence_comparison(max_transactions: int = 40_000) -> dict[str, Any]:
    """Compare logistic, HMM, and CRF outputs on a chronological held-out segment."""
    if max_transactions < 10_000:
        raise ValueError("Choose at least 10,000 transactions for a sequence experiment.")
    data = load_dataset().sort_values("Time", kind="stable").head(max_transactions)
    split = int(len(data) * 0.8)
    train = data.iloc[:split]
    test = data.iloc[split:]
    X_train = train[FEATURE_COLUMNS]
    X_test = test[FEATURE_COLUMNS]
    y_train = train["Class"].to_numpy(dtype=np.int8)
    y_test = test["Class"].to_numpy(dtype=np.int8)
    if set(np.unique(y_train)) != {0, 1} or set(np.unique(y_test)) != {0, 1}:
        raise ValueError("The chronological split needs both classes in train and test.")

    emission_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1_000, random_state=RANDOM_STATE),
    )
    emission_model.fit(X_train, y_train)
    baseline = emission_model.predict_proba(X_test)[:, 1]
    train_emission_probabilities = emission_model.predict_proba(X_train)[:, 1]

    train_obs, test_obs = _categorical_observations(train, test)
    hmm_models: dict[str, np.ndarray] = {}
    hmm_paths: dict[str, np.ndarray] = {}
    for topology in ("ergodic", "left-to-right"):
        start, transition, emission, state_rates = _fit_categorical_hmm(
            train_obs,
            y_train,
            topology=topology,
        )
        name = f"HMM · {topology}"
        hmm_models[name] = _filter_hmm(
            test_obs,
            start,
            transition,
            emission,
            state_rates,
        )
        hmm_paths[name] = _viterbi(test_obs, start, transition, emission)

    crf_training_rows = min(len(y_train), 1_000)
    clipped_train = np.clip(
        train_emission_probabilities[:crf_training_rows],
        1e-8,
        1 - 1e-8,
    )
    crf_train_labels = y_train[:crf_training_rows]
    unary_train = np.column_stack(
        [np.zeros(len(clipped_train)), np.log(clipped_train / (1 - clipped_train))]
    )
    optimized = minimize(
        _crf_transition_objective,
        np.zeros(4, dtype=np.float64),
        args=(unary_train, crf_train_labels),
        method="L-BFGS-B",
        jac=True,
        bounds=[(-5.0, 5.0)] * 4,
        options={"maxiter": 10, "maxfun": 30, "maxls": 8},
    )
    if not optimized.success and not np.isfinite(optimized.fun):
        raise RuntimeError(f"CRF transition fitting failed: {optimized.message}")
    transitions = optimized.x.reshape(2, 2)
    clipped_test = np.clip(baseline, 1e-8, 1 - 1e-8)
    unary_test = np.column_stack(
        [np.zeros(len(clipped_test)), np.log(clipped_test / (1 - clipped_test))]
    )
    crf_probabilities = _filter_crf(unary_test, transitions)

    model_scores = {"Logistic baseline": baseline, **hmm_models, "Linear-chain CRF": crf_probabilities}
    metrics: dict[str, dict[str, float]] = {}
    for name, probabilities in model_scores.items():
        metrics[name] = {
            "average_precision": float(average_precision_score(y_test, probabilities)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
        }
    return {
        "metrics": metrics,
        "probabilities": model_scores,
        "viterbi_paths": hmm_paths,
        "test_labels": y_test,
        "training_rows": len(train),
        "test_rows": len(test),
        "test_fraud_rows": int(y_test.sum()),
        "crf_training_rows": crf_training_rows,
        "crf_optimizer_message": str(optimized.message),
        "sequence_definition": "Transactions ordered by elapsed Time; no account-level sequence is available.",
    }
