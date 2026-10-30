import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from fraud_detection import DATA_PATH, evaluate_predictions, load_dataset, train_models
from additional_experiments import train_amount_regressors
from model_training import CLASSIFICATION_ALGORITHMS, compare_classifiers, feature_importance

st.title("Model training and evaluation")
st.caption(
    "Train on real transactions, keep an untouched stratified test set, and compare "
    "the measured results."
)

data = load_dataset()
maximum_training_rows = int(len(data) * 0.8)
default_training_rows = min(120_000, maximum_training_rows)

with st.form("training_settings"):
    training_rows = st.slider(
        "Training transactions",
        min_value=20_000,
        max_value=maximum_training_rows,
        value=default_training_rows,
        step=10_000,
        help="A smaller training sample is faster. Sampling preserves the fraud ratio.",
    )
    left, right = st.columns(2)
    forest_trees = left.slider("Random Forest trees", 50, 200, 100, step=50)
    max_depth = right.slider("Maximum tree depth", 4, 24, 16, step=2)
    submitted = st.form_submit_button("Train and compare models", type="primary")

if submitted:
    with st.spinner("Training both models and evaluating them on held-out transactions…"):
        st.session_state["fraud_model_results"] = train_models(
            path=str(DATA_PATH),
            training_rows=training_rows,
            forest_trees=forest_trees,
            max_depth=max_depth,
        )
        st.session_state.pop("fraud_prediction_pending", None)

results = st.session_state.get("fraud_model_results")
if results is None:
    st.info(
        "No evaluation is shown yet. Choose the settings above and train the models to "
        "calculate genuine results from the downloaded dataset."
    )
    st.write(
        "**Models:** a class-weighted Decision Tree and Random Forest. "
        "**Evaluation:** the final 20% of transactions is held out with stratified "
        "sampling and is never used to fit either model."
    )
    st.stop()

st.header("Training setup")
details = st.columns(4)
details[0].metric("Training rows", f"{results['training_rows']:,}")
details[1].metric("Fraud in training", f"{results['training_fraud_rows']:,}")
details[2].metric("Held-out rows", f"{results['test_rows']:,}")
details[3].metric("Fraud in test set", f"{results['test_fraud_rows']:,}")
st.write(
    "Class weights are used to make missed fraud cases more costly during tree training. "
    "The untouched test set keeps the dataset's original class distribution; it is not "
    "oversampled. This avoids letting duplicated minority-class records leak into "
    "evaluation."
)

threshold = st.slider(
    "Classification threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.05,
    help="A lower threshold usually catches more fraud but can also create more false alarms.",
)
st.session_state["fraud_prediction_threshold"] = threshold

model_metrics = {
    name: evaluate_predictions(results["y_test"], probabilities, threshold)
    for name, probabilities in results["probabilities"].items()
}

st.header("Decision Tree and Random Forest")
metric_rows = []
for name, metrics in model_metrics.items():
    metric_rows.append(
        {
            "Model": name,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1-score": metrics["f1"],
            "ROC AUC": metrics["roc_auc"],
            "Average precision": metrics["average_precision"],
        }
    )
st.dataframe(
    metric_rows,
    hide_index=True,
    width="stretch",
    column_config={
        column: st.column_config.NumberColumn(format="%.3f")
        for column in ("Accuracy", "Precision", "Recall", "F1-score", "ROC AUC", "Average precision")
    },
)
st.caption(
    "Precision answers: of the transactions flagged as fraud, how many were fraud? "
    "Recall answers: of all actual fraud cases, how many did the model catch? "
    "F1-score balances precision and recall. Accuracy is included for context, but "
    "it can be misleading on highly imbalanced data."
)

st.header("Confusion matrix")
matrix_figure = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=tuple(model_metrics),
    horizontal_spacing=0.16,
)
for column_number, (name, metrics) in enumerate(model_metrics.items(), start=1):
    matrix = metrics["confusion_matrix"]
    matrix_figure.add_trace(
        go.Heatmap(
            z=matrix,
            x=["Predicted normal", "Predicted fraud"],
            y=["Actual normal", "Actual fraud"],
            text=matrix,
            texttemplate="%{text:,}",
            colorscale=[[0, "#EEF3F8"], [1, "#368F8B"]],
            showscale=False,
        ),
        row=1,
        col=column_number,
    )
matrix_figure.update_layout(
    title=f"Test-set predictions at threshold {threshold:.2f}",
    template="plotly_white",
    height=380,
)
st.plotly_chart(matrix_figure, width="stretch")

st.header("ROC and precision–recall curves")
curve_left, curve_right = st.columns(2)
roc_figure = go.Figure()
pr_figure = go.Figure()
colors = {"Decision Tree": "#234E70", "Random Forest": "#368F8B"}
for name, metrics in model_metrics.items():
    false_positive_rate, true_positive_rate, _ = metrics["roc_curve"]
    precision, recall, _ = metrics["precision_recall_curve"]
    roc_figure.add_trace(
        go.Scatter(
            x=false_positive_rate,
            y=true_positive_rate,
            name=f"{name} (AUC {metrics['roc_auc']:.3f})",
            line={"color": colors[name]},
        )
    )
    pr_figure.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            name=f"{name} (AP {metrics['average_precision']:.3f})",
            line={"color": colors[name]},
        )
    )
roc_figure.add_trace(
    go.Scatter(x=[0, 1], y=[0, 1], name="Random baseline", line={"dash": "dash", "color": "#999999"})
)
roc_figure.update_layout(
    title="ROC curve",
    xaxis_title="False positive rate",
    yaxis_title="True positive rate (recall)",
    template="plotly_white",
)
pr_figure.update_layout(
    title="Precision–recall curve",
    xaxis_title="Recall",
    yaxis_title="Precision",
    template="plotly_white",
)
with curve_left:
    st.plotly_chart(roc_figure, width="stretch")
with curve_right:
    st.plotly_chart(pr_figure, width="stretch")
st.caption(
    "Average precision summarizes the precision–recall curve. It is often more informative "
    "than ROC AUC when fraud is extremely rare."
)

st.header("Random Forest feature importance")
forest = results["models"]["Random Forest"]
importances = sorted(
    zip(forest.feature_names_in_, forest.feature_importances_),
    key=lambda item: item[1],
    reverse=True,
)[:12]
importance_figure = go.Figure(
    go.Bar(
        x=[importance for _, importance in reversed(importances)],
        y=[feature for feature, _ in reversed(importances)],
        orientation="h",
        marker_color="#368F8B",
    )
)
importance_figure.update_layout(
    title="Top 12 features by Random Forest impurity importance",
    xaxis_title="Relative importance",
    yaxis_title="Feature",
    template="plotly_white",
)
st.plotly_chart(importance_figure, width="stretch")
st.caption(
    "Feature importance shows which input variables the fitted forest used most. "
    "Importance is not causation, and the anonymized PCA feature names have no direct "
    "business interpretation."
)

st.divider()
st.header("Syllabus-informed algorithm comparison")
st.write(
    "Compare selected classification algorithms on the same stratified held-out sample. "
    "Pipelines scale features only for algorithms that need it. Self-training hides "
    "30% of training labels; test labels are never hidden from evaluation or passed to fitting."
)
with st.expander("Choose algorithms and run a common comparison", expanded=False):
    with st.form("syllabus_classifier_comparison"):
        comparison_algorithms = st.multiselect(
            "Algorithms to compare",
            options=list(CLASSIFICATION_ALGORITHMS),
            default=["Logistic Regression", "Random Forest", "Histogram Gradient Boosting"],
            help="SVM models are capped at 6,000 training rows; k-NN at 12,000.",
        )
        comparison_columns = st.columns(3)
        comparison_train_rows = comparison_columns[0].select_slider(
            "Training rows",
            options=[2_000, 4_000, 6_000, 8_000, 12_000, 20_000],
            value=8_000,
        )
        comparison_test_rows = comparison_columns[1].select_slider(
            "Common test rows",
            options=[2_000, 5_000, 10_000, 20_000],
            value=10_000,
        )
        comparison_neighbors = comparison_columns[2].slider(
            "k for k-NN",
            min_value=3,
            max_value=31,
            value=11,
            step=2,
        )
        compare_submitted = st.form_submit_button("Run algorithm comparison", type="primary")

    if compare_submitted:
        try:
            with st.spinner("Training selected algorithms on real transactions..."):
                st.session_state["syllabus_comparison_results"] = compare_classifiers(
                    comparison_algorithms,
                    training_rows=comparison_train_rows,
                    test_rows=comparison_test_rows,
                    neighbors=comparison_neighbors,
                )
                st.session_state["syllabus_comparison_settings"] = (
                    tuple(comparison_algorithms),
                    comparison_train_rows,
                    comparison_test_rows,
                    comparison_neighbors,
                )
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc), title="Could not run algorithm comparison")

    comparison = st.session_state.get("syllabus_comparison_results")
    expected_settings = (
        tuple(comparison_algorithms),
        comparison_train_rows,
        comparison_test_rows,
        comparison_neighbors,
    )
    if comparison and st.session_state.get("syllabus_comparison_settings") != expected_settings:
        comparison = None

    if comparison:
        score_rows = [
            {"Model": name, **values["metrics"]}
            for name, values in comparison["results"].items()
        ]
        score_table = pd.DataFrame(score_rows)
        st.caption(
            f"Common test sample: {comparison['test_rows']:,} transactions, "
            f"including {comparison['test_fraud_rows']:,} recorded fraud cases. "
            "These are measured results for this run."
        )
        st.dataframe(
            score_table.drop(columns=["confusion_matrix"]),
            hide_index=True,
            width="stretch",
            column_config={
                column: st.column_config.NumberColumn(format="%.4f")
                for column in score_table.columns
                if column != "Model"
            },
        )
        selected_comparison = st.selectbox(
            "Inspect confusion matrix",
            list(comparison["results"]),
            key="comparison_confusion_model",
        )
        matrix = comparison["results"][selected_comparison]["metrics"]["confusion_matrix"]
        figure = go.Figure(
            go.Heatmap(
                z=matrix,
                x=["Predicted normal", "Predicted fraud"],
                y=["Actual normal", "Actual fraud"],
                text=matrix,
                texttemplate="%{text:,}",
                colorscale=[[0, "#EEF3F8"], [1, "#368F8B"]],
                showscale=False,
            )
        )
        figure.update_layout(
            title=f"{selected_comparison} · common held-out sample",
            template="plotly_white",
            height=330,
        )
        st.plotly_chart(
            figure,
            width="stretch",
            alt=f"Confusion matrix for {selected_comparison}.",
        )
        importance = feature_importance(
            comparison["models"][selected_comparison],
            list(comparison["X_test"].columns),
        )
        if not importance.empty:
            st.subheader("Global feature importance")
            st.dataframe(importance.head(12), hide_index=True, width="stretch")
            st.caption(
                "Global model importance, not a local causal explanation for a single transaction."
            )
        st.info(
            "For a smaller model-selection estimate, use the Advanced model lab's "
            "three-fold cross-validation workflow. This panel compares selected "
            "syllabus algorithms on one common held-out sample."
        )

st.divider()
with st.expander("Regression experiment: estimate transaction Amount", expanded=False):
    st.write(
        "This separate regression task predicts the real `Amount` field using "
        "Linear, Polynomial, Ridge, and Lasso Regression. It is included to exercise "
        "the syllabus regression techniques; it is not the fraud classifier."
    )
    regression_rows = st.select_slider(
        "Regression training rows",
        options=[2_000, 4_000, 8_000, 12_000, 20_000],
        value=8_000,
        key="amount_regression_rows",
    )
    if st.button("Run amount regression comparison", key="run_amount_regression"):
        try:
            with st.spinner("Fitting regression models on real transaction amounts..."):
                st.session_state["amount_regression_results"] = train_amount_regressors(
                    regression_rows
                )
                st.session_state["amount_regression_settings"] = regression_rows
        except ValueError as exc:
            st.error(str(exc), title="Could not run regression comparison")
    regression = st.session_state.get("amount_regression_results")
    if regression and st.session_state.get("amount_regression_settings") == regression_rows:
        regression_metrics = pd.DataFrame(regression["metrics"]).T.rename_axis("Model").reset_index()
        st.caption(
            f"Fit on {regression['training_rows']:,} real transactions; evaluated on "
            f"{regression['test_rows']:,} separate test transactions."
        )
        st.dataframe(regression_metrics, hide_index=True, width="stretch")
        regression_model_name = st.selectbox(
            "Regression predictions to inspect",
            list(regression["predictions"]),
            key="amount_regression_model",
        )
        regression_plot = go.Figure(
            go.Scatter(
                x=regression["actual_amount"],
                y=regression["predictions"][regression_model_name],
                mode="markers",
                opacity=0.4,
                name="Held-out predictions",
            )
        )
        maximum_amount = float(np.quantile(regression["actual_amount"], 0.99))
        regression_plot.add_trace(
            go.Scatter(
                x=[0, maximum_amount],
                y=[0, maximum_amount],
                mode="lines",
                name="Ideal prediction",
                line={"dash": "dash"},
            )
        )
        regression_plot.update_layout(
            title=f"{regression_model_name} · held-out actual vs predicted amount",
            xaxis_title="Actual amount",
            yaxis_title="Predicted amount",
            template="plotly_white",
            height=390,
        )
        st.plotly_chart(
            regression_plot,
            width="stretch",
            alt=f"Actual versus predicted transaction amount for {regression_model_name}.",
        )
