import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from fraud_detection import evaluate_predictions

st.title("Model performance monitoring")
st.caption("Inspect measured held-out performance and separately track prototype DEMO DATA scores.")

results = st.session_state.get("fraud_model_results")
if results is None:
    st.info("Train the models on the Model evaluation page to view measured monitoring metrics.")
    st.stop()

model_name = st.selectbox("Model to monitor", list(results["probabilities"]))
threshold = st.slider(
    "Operating threshold",
    min_value=0.05,
    max_value=0.95,
    value=float(st.session_state.get("fraud_prediction_threshold", 0.5)),
    step=0.05,
    key="monitoring_threshold",
)
metrics = evaluate_predictions(results["y_test"], results["probabilities"][model_name], threshold)
cards = st.columns(6)
for card, (label, key) in zip(
    cards,
    (
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1-score", "f1"),
        ("ROC AUC", "roc_auc"),
        ("Average precision", "average_precision"),
    ),
):
    card.metric(label, f"{metrics[key]:.3f}", border=True)
st.caption(
    f"Computed on {results['test_rows']:,} held-out transactions, including "
    f"{results['test_fraud_rows']:,} recorded fraud cases."
)

test_rows = results["X_test"].copy()
test_rows["Recorded class"] = np.asarray(results["y_test"])
test_rows["Fraud probability"] = results["probabilities"][model_name]
test_rows["Time cohort"] = pd.qcut(test_rows["Time"], q=4, duplicates="drop")
cohort = (
    test_rows.groupby("Time cohort", observed=True)
    .apply(
        lambda group: pd.Series(
            {
                "Transactions": len(group),
                "Fraud cases": int(group["Recorded class"].sum()),
                "Fraud recall": float(
                    (
                        (group["Fraud probability"] >= threshold)
                        & (group["Recorded class"] == 1)
                    ).sum()
                    / max(int(group["Recorded class"].sum()), 1)
                ),
                "Average precision": float(
                    __import__("sklearn.metrics", fromlist=["average_precision_score"])
                    .average_precision_score(group["Recorded class"], group["Fraud probability"])
                )
                if group["Recorded class"].nunique() == 2
                else np.nan,
            }
        ),
        include_groups=False,
    )
    .reset_index()
)
cohort["Time cohort"] = cohort["Time cohort"].astype(str)
st.header("Performance across elapsed-time cohorts")
st.write(
    "This retrospective breakdown checks consistency over the recorded two-day "
    "window. It is not a production drift alert or a guarantee of future performance."
)
st.dataframe(cohort, hide_index=True, width="stretch")
curve = px.line(
    cohort,
    x="Time cohort",
    y="Fraud recall",
    markers=True,
    title="Held-out recall by time cohort",
    template="plotly_white",
)
curve.update_layout(yaxis_range=[0, 1], height=330)
st.plotly_chart(curve, width="stretch", alt="Held-out fraud recall across elapsed-time cohorts.")

demo = pd.DataFrame(st.session_state.get("demo_events", []))
st.header("DEMO DATA stream health")
if demo.empty:
    st.info("No simulator events are available. Start the Live-style transaction simulator to populate demo metrics.")
else:
    flagged = int((demo["Predicted class"] == "Fraud").sum())
    demo_cards = st.columns(3)
    demo_cards[0].metric("Scored demo rows", f"{len(demo):,}")
    demo_cards[1].metric("Flagged demo rows", f"{flagged:,}")
    demo_cards[2].metric("Flag rate", f"{flagged / len(demo):.1%}")
    st.caption("Simulator metrics describe demonstration events only; they are not current bank traffic.")
