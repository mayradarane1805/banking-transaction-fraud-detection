import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

from advanced_models import (
    evaluate_advanced_model,
    select_cost_threshold,
    score_batch_transactions,
    train_advanced_models,
)
from fraud_detection import FEATURE_COLUMNS
from sequence_experiments import run_sequence_comparison

st.title("Advanced fraud detection lab")
st.caption(
    "Cross-validate candidate models, tune decisions on validation data, and "
    "report final performance on a separate untouched test set."
)

st.info(
    "This workflow keeps the test set out of model selection. Cross-validation and "
    "threshold choice use development data only; final test metrics are calculated "
    "after the selected model and cost policy are fixed."
)

with st.form("advanced_training_settings"):
    training_rows = st.slider(
        "Rows used for cross-validation and fitting",
        min_value=10_000,
        max_value=170_000,
        value=40_000,
        step=10_000,
        help="Higher values take longer but use more real transactions in each CV fold.",
    )
    train_submitted = st.form_submit_button("Run advanced model comparison", type="primary")

if train_submitted:
    with st.spinner("Running 3-fold validation for three models…"):
        st.session_state["advanced_fraud_results"] = train_advanced_models(training_rows)

results = st.session_state.get("advanced_fraud_results")
if results is None:
    st.info("Run the advanced comparison to train and compare the candidate models.")
    st.write(
        "Candidates include standardized Logistic Regression, a class-weighted Random "
        "Forest, and class-weighted Histogram Gradient Boosting. The best candidate is "
        "selected by mean cross-validated average precision."
    )
    st.stop()

st.header("Data split")
split_columns = st.columns(3)
split_columns[0].metric("Training rows", f"{results['train_rows']:,}")
split_columns[1].metric("Validation rows", f"{results['validation_rows']:,}")
split_columns[2].metric("Untouched test rows", f"{results['test_rows']:,}")
st.caption(
    f"Fraud cases: {results['train_fraud_rows']:,} train · "
    f"{results['validation_fraud_rows']:,} validation · "
    f"{results['test_fraud_rows']:,} test."
)

st.header("Cross-validated model comparison")
st.write(
    "Each candidate is evaluated with the same stratified three-fold cross-validation. "
    "Average precision is the selection metric because the fraud class is rare."
)
cv_table = pd.DataFrame.from_dict(results["cross_validation"], orient="index")
cv_table.index.name = "Model"
cv_table = cv_table.reset_index().rename(
    columns={
        "average_precision": "Mean average precision",
        "average_precision_std": "AP standard deviation",
        "roc_auc": "Mean ROC AUC",
        "roc_auc_std": "ROC AUC standard deviation",
        "balanced_accuracy": "Mean balanced accuracy",
        "balanced_accuracy_std": "Balanced accuracy standard deviation",
    }
)
st.dataframe(
    cv_table,
    hide_index=True,
    width="stretch",
    column_config={
        column: st.column_config.NumberColumn(format="%.4f")
        for column in cv_table.columns[1:]
    },
)
selected_name = results["selected_model_name"]
st.success(f"Selected from cross-validation: {selected_name}")

st.header("Cost-sensitive operating threshold")
cost_left, cost_right = st.columns(2)
false_positive_cost = cost_left.number_input(
    "Relative cost of a false alarm",
    min_value=0.01,
    value=1.0,
    step=0.5,
    help="A legitimate transaction incorrectly flagged as fraud.",
)
false_negative_cost = cost_right.number_input(
    "Relative cost of missed fraud",
    min_value=0.01,
    value=10.0,
    step=1.0,
    help="A fraudulent transaction incorrectly treated as normal.",
)
validation_thresholds = {
    name: select_cost_threshold(
        results["y_validation"],
        scores,
        false_positive_cost,
        false_negative_cost,
    )
    for name, scores in results["validation_probabilities"].items()
}
threshold_result = validation_thresholds[selected_name]
threshold = float(threshold_result["threshold"])
st.write(
    f"Validation-selected threshold for {selected_name}: **{threshold:.5f}**. "
    f"Validation false alarms: {threshold_result['false_positives']:,}; "
    f"missed fraud cases: {threshold_result['false_negatives']:,}; "
    f"relative validation cost: {threshold_result['total_cost']:,.2f}."
)
st.caption(
    "These relative costs are an educational assumption, not measured monetary loss. "
    "Each model gets its own threshold from the same validation split. Changing the "
    "cost inputs changes decision thresholds, not the fitted models."
)

test_rows = []
test_metrics_by_model = {}
for name, scores in results["test_probabilities"].items():
    model_threshold = float(validation_thresholds[name]["threshold"])
    metrics = evaluate_advanced_model(
        results["y_test"],
        scores,
        model_threshold,
        false_positive_cost,
        false_negative_cost,
    )
    test_metrics_by_model[name] = metrics
    test_rows.append(
        {
            "Model": name,
            "Validation threshold": model_threshold,
            "Average precision": metrics["average_precision"],
            "ROC AUC": metrics["roc_auc"],
            "Balanced accuracy": metrics["balanced_accuracy"],
            "Brier score": metrics["brier_score"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "False alarms": metrics["false_positives"],
            "Missed fraud": metrics["false_negatives"],
            "Relative cost": metrics["total_cost"],
        }
    )

st.header("Final held-out test results")
st.dataframe(
    test_rows,
    hide_index=True,
    width="stretch",
    column_config={
        column: st.column_config.NumberColumn(format="%.4f")
        for column in (
            "Validation threshold",
            "Average precision",
            "ROC AUC",
            "Balanced accuracy",
            "Brier score",
            "Precision",
            "Recall",
            "Relative cost",
        )
    },
)
st.caption(
    "Each threshold was selected on validation cases, not this test set. "
    "Brier score measures probability error (lower is better)."
)

st.header("Precision–recall and calibration")
pr_figure = go.Figure()
calibration_figure = go.Figure()
for name, scores in results["test_probabilities"].items():
    precision_values, recall_values, _ = precision_recall_curve(
        results["y_test"],
        scores,
    )
    pr_figure.add_trace(
        go.Scatter(
            x=recall_values,
            y=precision_values,
            name=f"{name} (AP {test_metrics_by_model[name]['average_precision']:.3f})",
            mode="lines",
        )
    )
    observed_rate, mean_probability = calibration_curve(
        results["y_test"],
        scores,
        n_bins=10,
        strategy="quantile",
    )
    calibration_figure.add_trace(
        go.Scatter(
            x=mean_probability,
            y=observed_rate,
            mode="lines+markers",
            name=name,
        )
    )
pr_figure.update_layout(
    title="Test-set precision–recall curve",
    xaxis_title="Recall",
    yaxis_title="Precision",
    template="plotly_white",
)
calibration_figure.add_trace(
    go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        line={"dash": "dash", "color": "#999999"},
        name="Perfect calibration",
    )
)
calibration_figure.update_layout(
    title="Test-set probability calibration",
    xaxis_title="Mean predicted fraud probability",
    yaxis_title="Observed fraud frequency",
    template="plotly_white",
)
pr_column, calibration_column = st.columns(2)
with pr_column:
    st.plotly_chart(pr_figure, width="stretch")
with calibration_column:
    st.plotly_chart(calibration_figure, width="stretch")

st.header("Batch scoring")
st.write(
    "Upload a CSV with the exact 30 feature columns (`Time`, `Amount`, and `V1`–`V28`). "
    "The file may contain up to 10,000 rows. Download the template below to check the "
    "required header names."
)
template = results["X_test"].loc[:, FEATURE_COLUMNS].head(5)
st.download_button(
    "Download CSV template",
    data=template.to_csv(index=False).encode("utf-8"),
    file_name="fraud_prediction_template.csv",
    mime="text/csv",
)
uploaded_file = st.file_uploader("Upload transaction features", type=["csv"])
if uploaded_file is not None:
    try:
        uploaded_file.seek(0)
        batch = pd.read_csv(uploaded_file)
        output = score_batch_transactions(
            results["models"][selected_name],
            batch,
            threshold,
        )
        st.dataframe(output.head(100), hide_index=True, width="stretch")
        st.download_button(
            "Download scored transactions",
            data=output.to_csv(index=False).encode("utf-8"),
            file_name="fraud_scored_transactions.csv",
            mime="text/csv",
        )
        st.caption(
            f"Scored {len(output):,} rows with {selected_name} at threshold {threshold:.5f}."
        )
    except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        st.error(str(exc), title="Could not score this CSV")

report = {
    "selected_model": selected_name,
    "selection_metric": "three-fold cross-validated average precision",
    "training_rows": results["train_rows"],
    "validation_rows": results["validation_rows"],
    "test_rows": results["test_rows"],
    "test_fraud_rows": results["test_fraud_rows"],
    "threshold": threshold,
    "validation_thresholds": validation_thresholds,
    "false_positive_cost": false_positive_cost,
    "false_negative_cost": false_negative_cost,
    "cross_validation": results["cross_validation"],
    "held_out_test_metrics": test_metrics_by_model,
}
st.download_button(
    "Download advanced evaluation report",
    data=json.dumps(report, indent=2, allow_nan=False).encode("utf-8"),
    file_name="advanced_fraud_evaluation.json",
    mime="application/json",
)

st.divider()
with st.expander("Sequence-model experiment: HMM and CRF", expanded=False):
    st.warning(
        "This dataset has no customer/account IDs. The experiment treats records as "
        "one aggregate transaction stream ordered by elapsed Time; it is not a "
        "customer-level sequence classifier."
    )
    st.write(
        "The HMM learns two latent regimes from coarse transaction observations and "
        "maps them to fraud prevalence using training labels after fitting. The CRF "
        "uses logistic feature emissions and learns pairwise sequence transitions. "
        "A chronological final segment is held out. The dataset has no customer IDs, "
        "and a short test segment may contain few fraud cases, so those metrics are "
        "illustrative rather than reliable estimates of future performance."
    )
    sequence_rows = st.select_slider(
        "Transactions in time-ordered experiment",
        options=[10_000, 20_000, 80_000],
        value=20_000,
        key="sequence_experiment_rows",
    )
    if st.button("Run HMM / CRF comparison", key="run_sequence_experiment"):
        try:
            with st.spinner("Fitting sequence models and scoring a future time segment..."):
                st.session_state["sequence_results"] = run_sequence_comparison(sequence_rows)
                st.session_state["sequence_settings"] = sequence_rows
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc), title="Could not run sequence experiment")

    sequence_result = st.session_state.get("sequence_results")
    if sequence_result and st.session_state.get("sequence_settings") == sequence_rows:
        sequence_table = pd.DataFrame(sequence_result["metrics"]).T.rename_axis("Model").reset_index()
        st.caption(
            f"{sequence_result['training_rows']:,} earlier transactions for training · "
            f"{sequence_result['test_rows']:,} later transactions for test · "
            f"{sequence_result['test_fraud_rows']:,} fraud cases in test · "
            f"{sequence_result['crf_training_rows']:,} training rows for CRF transitions."
        )
        st.dataframe(
            sequence_table,
            hide_index=True,
            width="stretch",
            column_config={
                "average_precision": st.column_config.NumberColumn("Average precision", format="%.4f"),
                "roc_auc": st.column_config.NumberColumn("ROC AUC", format="%.4f"),
            },
        )
        viterbi_summary = pd.DataFrame(
            [
                {
                    "Model": name,
                    "State 0 transactions": int(np.count_nonzero(path == 0)),
                    "State 1 transactions": int(np.count_nonzero(path == 1)),
                    "State 1 share": float(np.mean(path == 1)),
                }
                for name, path in sequence_result["viterbi_paths"].items()
            ]
        )
        st.subheader("HMM Viterbi path")
        st.dataframe(
            viterbi_summary,
            hide_index=True,
            width="stretch",
            column_config={
                "State 1 share": st.column_config.NumberColumn(format="%.3%")
            },
        )
        st.caption(
            "Viterbi reports the most likely hidden-state path; latent state numbers "
            "are not fraud labels. These are computed experiment results on an aggregate transaction sequence. "
            "They do not demonstrate customer histories, and the short historical period "
            "does not establish future banking performance."
        )
