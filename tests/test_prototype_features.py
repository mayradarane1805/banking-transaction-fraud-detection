import unittest

import numpy as np
import pandas as pd

from data_processing import validate_transaction_features
from fraud_detection import FEATURE_COLUMNS
from prototype_utils import (
    apply_review_status,
    build_historical_alerts,
    create_demo_event,
    risk_level,
)
from rl_experiments import run_offline_policy_experiment
from sequence_experiments import (
    _crf_transition_objective,
    _filter_hmm,
    _fit_categorical_hmm,
    _viterbi,
)


class FixedProbabilityModel:
    def predict_proba(self, features):
        return np.tile([0.2, 0.8], (len(features), 1))


class PrototypeFeatureTests(unittest.TestCase):
    def setUp(self):
        self.features = pd.DataFrame(
            np.zeros((2, len(FEATURE_COLUMNS))),
            columns=FEATURE_COLUMNS,
        )

    def test_feature_validation_orders_columns_and_rejects_bad_values(self):
        reversed_features = self.features.loc[:, list(reversed(FEATURE_COLUMNS))]
        validated = validate_transaction_features(reversed_features)
        self.assertEqual(list(validated.columns), FEATURE_COLUMNS)
        self.assertEqual(validated.shape, (2, len(FEATURE_COLUMNS)))

        with self.assertRaisesRegex(ValueError, "missing columns"):
            validate_transaction_features(self.features.drop(columns=["Amount"]))

        invalid = self.features.copy()
        invalid.loc[0, "Amount"] = -1
        with self.assertRaisesRegex(ValueError, "non-negative"):
            validate_transaction_features(invalid)

        invalid = self.features.copy()
        invalid.loc[0, "V1"] = np.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_transaction_features(invalid)

    def test_risk_bands_validate_probability(self):
        self.assertEqual(risk_level(0.10), "Low")
        self.assertEqual(risk_level(0.30), "Medium")
        self.assertEqual(risk_level(0.75), "High")
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            risk_level(float("nan"))

    def test_demo_event_and_historical_alerts_are_labeled(self):
        test_rows = self.features.iloc[[0]].copy()
        results = {
            "models": {"Fixture model": FixedProbabilityModel()},
            "X_test": test_rows,
            "probabilities": {"Fixture model": np.array([0.8])},
        }
        event, next_cursor = create_demo_event(results, "Fixture model", 0, 0.5)
        self.assertEqual(event["Source"], "DEMO DATA")
        self.assertEqual(event["Predicted class"], "Fraud")
        self.assertEqual(next_cursor, 1)

        alerts = build_historical_alerts(results, "Fixture model", 0.5)
        self.assertEqual(alerts.iloc[0]["Source"], "Historical holdout")
        self.assertEqual(alerts.iloc[0]["Review status"], "New")

    def test_review_status_update_is_validated_and_non_mutating(self):
        alerts = pd.DataFrame(
            [{"Transaction ID": "HIST-1", "Review status": "New"}]
        )
        updated = apply_review_status(alerts, "HIST-1", "Investigating")
        self.assertEqual(updated.iloc[0]["Review status"], "Investigating")
        self.assertEqual(alerts.iloc[0]["Review status"], "New")
        with self.assertRaisesRegex(ValueError, "Unknown investigation status"):
            apply_review_status(alerts, "HIST-1", "Approved")

    def test_hmm_training_filtering_and_viterbi_return_finite_values(self):
        observations = np.array([0, 1, 1, 2, 3, 2, 1, 0], dtype=np.int8)
        labels = np.array([0, 0, 1, 0, 1, 1, 0, 0], dtype=np.int8)
        start, transition, emission, state_rates = _fit_categorical_hmm(
            observations,
            labels,
            iterations=3,
            topology="left-to-right",
        )
        probabilities = _filter_hmm(
            observations,
            start,
            transition,
            emission,
            state_rates,
        )
        path = _viterbi(observations, start, transition, emission)
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
        self.assertEqual(path.shape, observations.shape)
        self.assertAlmostEqual(float(start.sum()), 1.0)
        self.assertTrue(np.allclose(transition.sum(axis=1), 1.0))

    def test_crf_objective_gradient_matches_finite_differences(self):
        emissions = np.array(
            [[0.0, -0.2], [0.0, 0.6], [0.0, -0.4], [0.0, 0.3]],
            dtype=np.float64,
        )
        labels = np.array([0, 1, 0, 1], dtype=np.int8)
        parameters = np.array([0.1, -0.2, 0.3, 0.05], dtype=np.float64)
        _, analytic = _crf_transition_objective(parameters, emissions, labels)
        epsilon = 1e-6
        numeric = np.empty_like(parameters)
        for index in range(len(parameters)):
            step = np.zeros_like(parameters)
            step[index] = epsilon
            plus = _crf_transition_objective(parameters + step, emissions, labels)[0]
            minus = _crf_transition_objective(parameters - step, emissions, labels)[0]
            numeric[index] = (plus - minus) / (2 * epsilon)
        np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-5)

    def test_offline_policy_experiment_checks_split_and_returns_metrics(self):
        labels = np.tile(np.array([0, 1], dtype=np.int8), 500)
        scores = np.where(labels == 1, 0.9, 0.05)
        elapsed = np.arange(len(labels), dtype=np.float64)
        result = run_offline_policy_experiment(scores, labels, elapsed)
        self.assertEqual(result["training_rows"], 700)
        self.assertEqual(result["test_rows"], 300)
        self.assertEqual(result["test_fraud_rows"], 150)
        self.assertEqual(
            set(result["policies"]),
            {"Q-learning", "DQN (small neural Q model)", "Fixed 0.50 threshold"},
        )


if __name__ == "__main__":
    unittest.main()
