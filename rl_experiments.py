"""Explicit-assumption offline Q-learning and DQN policy demonstrations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score


def _reward(label: int, action: int, false_positive_cost: float, false_negative_cost: float) -> float:
    if label == 1 and action == 0:
        return -false_negative_cost
    if label == 0 and action == 1:
        return -false_positive_cost
    return 0.0


def _state_values(scores: np.ndarray) -> np.ndarray:
    return np.digitize(scores, bins=[0.1, 0.5]).astype(np.int8)


def _one_hot(states: np.ndarray) -> np.ndarray:
    return np.eye(3, dtype=np.float64)[states]


def _train_dqn(
    states: np.ndarray,
    labels: np.ndarray,
    false_positive_cost: float,
    false_negative_cost: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    learning_rate = 0.01
    discount = 0.9
    input_size, hidden_size, action_count = 3, 8, 2
    weights_1 = rng.normal(0, 0.3, size=(input_size, hidden_size))
    bias_1 = np.zeros(hidden_size)
    weights_2 = rng.normal(0, 0.3, size=(hidden_size, action_count))
    bias_2 = np.zeros(action_count)
    next_states = np.roll(states, -1)
    epsilon = 0.2
    experiences: list[tuple[int, int, float, int]] = []

    def forward(input_states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hidden = np.maximum(_one_hot(input_states) @ weights_1 + bias_1, 0)
        return hidden, hidden @ weights_2 + bias_2

    for index in range(len(states) - 1):
        _, q_values = forward(states[index : index + 1])
        action = (
            int(rng.integers(action_count))
            if rng.random() < epsilon
            else int(np.argmax(q_values[0]))
        )
        reward = _reward(
            int(labels[index]),
            action,
            false_positive_cost,
            false_negative_cost,
        )
        experiences.append((int(states[index]), action, reward, int(next_states[index])))

    batch_size = min(64, len(experiences))
    for _ in range(20):
        order = rng.permutation(len(experiences))
        for start in range(0, len(order), batch_size):
            batch = [experiences[int(index)] for index in order[start : start + batch_size]]
            batch_states = np.array([item[0] for item in batch], dtype=np.int8)
            actions = np.array([item[1] for item in batch], dtype=np.int8)
            rewards = np.array([item[2] for item in batch], dtype=np.float64)
            following = np.array([item[3] for item in batch], dtype=np.int8)
            hidden, q_values = forward(batch_states)
            _, next_q_values = forward(following)
            targets = rewards + discount * next_q_values.max(axis=1)
            output_gradient = np.zeros_like(q_values)
            output_gradient[np.arange(len(batch)), actions] = (
                q_values[np.arange(len(batch)), actions] - targets
            ) / len(batch)
            gradient_w2 = hidden.T @ output_gradient
            gradient_b2 = output_gradient.sum(axis=0)
            hidden_gradient = (output_gradient @ weights_2.T) * (hidden > 0)
            gradient_w1 = _one_hot(batch_states).T @ hidden_gradient
            gradient_b1 = hidden_gradient.sum(axis=0)
            weights_2 -= learning_rate * gradient_w2
            bias_2 -= learning_rate * gradient_b2
            weights_1 -= learning_rate * gradient_w1
            bias_1 -= learning_rate * gradient_b1

    _, q_table = forward(np.arange(3, dtype=np.int8))
    return q_table, weights_1, bias_1, np.concatenate([weights_2.ravel(), bias_2])


def _evaluate_policy(
    actions: np.ndarray,
    labels: np.ndarray,
    false_positive_cost: float,
    false_negative_cost: float,
) -> dict[str, float]:
    flagged = actions == 1
    return {
        "review_rate": float(flagged.mean()),
        "fraud_capture_recall": float(recall_score(labels, flagged, zero_division=0)),
        "review_precision": float(precision_score(labels, flagged, zero_division=0)),
        "relative_mean_reward": float(
            np.mean(
                [
                    _reward(int(label), int(action), false_positive_cost, false_negative_cost)
                    for label, action in zip(labels, actions)
                ]
            )
        ),
    }


def run_offline_policy_experiment(
    probabilities: np.ndarray,
    labels: np.ndarray,
    elapsed_time: np.ndarray,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 10.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare Q-learning and DQN on an assumed-reward, held-out transaction stream."""
    scores = np.asarray(probabilities, dtype=np.float64)
    actual = np.asarray(labels, dtype=np.int8)
    times = np.asarray(elapsed_time, dtype=np.float64)
    if scores.ndim != 1 or actual.ndim != 1 or times.ndim != 1:
        raise ValueError("Scores, labels, and elapsed time must be one-dimensional.")
    if len(scores) != len(actual) or len(scores) != len(times) or len(scores) < 1_000:
        raise ValueError("Provide at least 1,000 aligned held-out transactions.")
    if set(np.unique(actual)) != {0, 1}:
        raise ValueError("The offline policy experiment needs normal and fraud labels.")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Scores must be finite probabilities between 0 and 1.")
    if not np.isfinite(times).all():
        raise ValueError("Elapsed times must be finite.")
    if (
        not np.isfinite(false_positive_cost)
        or not np.isfinite(false_negative_cost)
        or false_positive_cost <= 0
        or false_negative_cost <= 0
    ):
        raise ValueError("Relative costs must be finite and positive.")

    order = np.argsort(times, kind="stable")
    scores = scores[order]
    actual = actual[order]
    states = _state_values(scores)
    split = int(len(states) * 0.7)
    train_states, test_states = states[:split], states[split:]
    train_labels, test_labels = actual[:split], actual[split:]
    if set(np.unique(train_labels)) != {0, 1} or set(np.unique(test_labels)) != {0, 1}:
        raise ValueError("Both classes must occur in policy-train and policy-test segments.")

    rng = np.random.default_rng(seed)
    q_values = np.zeros((3, 2), dtype=np.float64)
    next_train_states = train_states[1:]
    for _ in range(12):
        for index in range(len(train_states) - 1):
            state = int(train_states[index])
            action = (
                int(rng.integers(2))
                if rng.random() < 0.15
                else int(np.argmax(q_values[state]))
            )
            reward = _reward(
                int(train_labels[index]),
                action,
                false_positive_cost,
                false_negative_cost,
            )
            next_state = int(next_train_states[index])
            target = reward + 0.9 * float(np.max(q_values[next_state]))
            q_values[state, action] += 0.1 * (target - q_values[state, action])

    dqn_values, weights_1, bias_1, packed_output = _train_dqn(
        train_states,
        train_labels,
        false_positive_cost,
        false_negative_cost,
        seed,
    )
    weights_2 = packed_output[: 8 * 2].reshape(8, 2)
    bias_2 = packed_output[8 * 2 :]
    dqn_hidden = np.maximum(_one_hot(test_states) @ weights_1 + bias_1, 0)
    dqn_test_q = dqn_hidden @ weights_2 + bias_2
    q_actions = np.argmax(q_values[test_states], axis=1)
    dqn_actions = np.argmax(dqn_test_q, axis=1)
    threshold_actions = (scores[split:] >= 0.5).astype(np.int8)

    policies = {
        "Q-learning": _evaluate_policy(q_actions, test_labels, false_positive_cost, false_negative_cost),
        "DQN (small neural Q model)": _evaluate_policy(
            dqn_actions,
            test_labels,
            false_positive_cost,
            false_negative_cost,
        ),
        "Fixed 0.50 threshold": _evaluate_policy(
            threshold_actions,
            test_labels,
            false_positive_cost,
            false_negative_cost,
        ),
    }
    return {
        "policies": policies,
        "training_rows": split,
        "test_rows": len(test_states),
        "test_fraud_rows": int(test_labels.sum()),
        "assumption": (
            "The state is a model-score bucket; actions are allow/review; rewards use "
            "user-selected relative error costs. Consecutive market-wide transactions "
            "are exogenous transitions, so this is a teaching simulator, not bank policy."
        ),
        "q_values": q_values,
        "dqn_values": dqn_values,
    }
