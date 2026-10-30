import unittest

import numpy as np

from fraud_detection import (
    DATA_PATH,
    FEATURE_COLUMNS,
    EXPECTED_COLUMNS,
    evaluate_predictions,
    load_dataset,
    predict_transaction,
    train_models,
)
from advanced_models import (
    evaluate_advanced_model,
    score_batch_transactions,
    select_cost_threshold,
    train_advanced_models,
)


class FraudDetectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_dataset()

    def test_public_dataset_has_expected_columns_and_real_classes(self):
        self.assertEqual(list(self.data.columns), EXPECTED_COLUMNS)
        self.assertGreater(len(self.data), 200_000)
        self.assertEqual(set(self.data["Class"].unique()), {0, 1})
        self.assertGreater(int(self.data["Class"].sum()), 0)
        self.assertEqual(self.data.isna().sum().sum(), 0)

    def test_metrics_use_both_real_test_classes(self):
        results = train_models(
            path=str(DATA_PATH),
            training_rows=10_000,
            forest_trees=5,
            max_depth=8,
        )
        self.assertGreater(results["test_fraud_rows"], 0)
        for name, model in results["models"].items():
            probabilities = results["probabilities"][name]
            self.assertEqual(len(probabilities), results["test_rows"])
            self.assertTrue(np.isfinite(probabilities).all())
            metrics = evaluate_predictions(results["y_test"], probabilities)
            self.assertEqual(metrics["confusion_matrix"].shape, (2, 2))
            self.assertGreaterEqual(metrics["recall"], 0)
            self.assertLessEqual(metrics["recall"], 1)
            transaction = results["X_test"].iloc[[0]][FEATURE_COLUMNS]
            probability = predict_transaction(model, transaction)
            self.assertGreaterEqual(probability, 0)
            self.assertLessEqual(probability, 1)

    def test_cost_threshold_is_selected_from_supplied_validation_scores(self):
        labels = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.05, 0.2, 0.4, 0.45, 0.7, 0.95])
        result = select_cost_threshold(
            labels,
            scores,
            false_positive_cost=1,
            false_negative_cost=10,
        )
        self.assertIn(result["threshold"], scores)
        self.assertLess(result["false_negatives"], len(labels))
        metrics = evaluate_advanced_model(
            labels,
            scores,
            float(result["threshold"]),
            false_positive_cost=1,
            false_negative_cost=10,
        )
        self.assertIn("brier_score", metrics)
        self.assertGreaterEqual(metrics["brier_score"], 0)
        self.assertLessEqual(metrics["brier_score"], 1)

    def test_cost_threshold_can_choose_flag_all_or_flag_none(self):
        labels = np.array([1, 0, 1, 0])
        scores = np.array([0.1, 0.2, 0.6, 0.9])
        flag_all = select_cost_threshold(
            labels,
            scores,
            false_positive_cost=0.01,
            false_negative_cost=10,
        )
        flag_none = select_cost_threshold(
            labels,
            scores,
            false_positive_cost=10,
            false_negative_cost=0.01,
        )
        self.assertEqual(flag_all["threshold"], 0.0)
        self.assertEqual(flag_all["false_negatives"], 0)
        self.assertEqual(flag_none["false_positives"], 0)

    def test_advanced_cross_validation_keeps_independent_test_set(self):
        results = train_advanced_models(training_rows=10_000)
        self.assertEqual(len(results["cross_validation"]), 3)
        self.assertGreater(results["validation_fraud_rows"], 0)
        self.assertGreater(results["test_fraud_rows"], 0)
        self.assertEqual(results["test_rows"], len(results["X_test"]))
        self.assertIn(results["selected_model_name"], results["models"])
        for probabilities in results["test_probabilities"].values():
            self.assertEqual(len(probabilities), results["test_rows"])
            self.assertTrue(np.isfinite(probabilities).all())
        template = results["X_test"].loc[:, FEATURE_COLUMNS].head(5)
        scored = score_batch_transactions(
            results["models"][results["selected_model_name"]],
            template,
            threshold=0.5,
        )
        self.assertEqual(len(scored), 5)
        self.assertIn("fraud_probability", scored.columns)
        self.assertIn("predicted_class", scored.columns)
        with self.assertRaisesRegex(ValueError, "missing columns"):
            score_batch_transactions(
                results["models"][results["selected_model_name"]],
                template.drop(columns=["Amount"]),
                threshold=0.5,
            )


if __name__ == "__main__":
    unittest.main()
