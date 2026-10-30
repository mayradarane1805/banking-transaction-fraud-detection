import pandas as pd
import streamlit as st

from prototype_utils import REVIEW_STATUSES, build_historical_alerts

st.title("Fraud alerts and investigation queue")
st.caption("Search, prioritize, and track model-flagged examples from the held-out dataset and DEMO DATA stream.")

results = st.session_state.get("fraud_model_results")
if results is None:
    st.info("Train the models on the Model evaluation page to generate scored alerts.")
    st.stop()

model_names = list(results["probabilities"])
selected_model = st.selectbox("Alert model", model_names, key="investigation_model")
threshold = st.slider(
    "Alert threshold",
    min_value=0.05,
    max_value=0.95,
    value=float(st.session_state.get("fraud_prediction_threshold", 0.5)),
    step=0.05,
    key="investigation_threshold",
)
alerts = build_historical_alerts(results, selected_model, threshold)
demo_events = pd.DataFrame(st.session_state.get("demo_events", []))
if not demo_events.empty:
    demo_alerts = demo_events.loc[
        demo_events["Fraud probability"] >= threshold,
        [
            "Transaction ID",
            "Source",
            "Amount",
            "Elapsed seconds",
            "Fraud probability",
            "Risk",
            "Model",
            "Review status",
        ],
    ]
    alerts = pd.concat([alerts, demo_alerts], ignore_index=True)

review_state = st.session_state.setdefault("investigation_status", {})
if not alerts.empty:
    alerts["Review status"] = alerts.apply(
        lambda row: review_state.get(row["Transaction ID"], row["Review status"]),
        axis=1,
    )

if alerts.empty:
    st.success("No transactions meet this alert threshold.")
    st.stop()

filter_columns = st.columns(3)
query = filter_columns[0].text_input("Search transaction ID", type="search")
risk_filter = filter_columns[1].multiselect("Risk level", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
status_filter = filter_columns[2].multiselect("Review status", list(REVIEW_STATUSES), default=list(REVIEW_STATUSES))
filtered = alerts.loc[
    alerts["Risk"].isin(risk_filter) & alerts["Review status"].isin(status_filter)
].copy()
if query:
    filtered = filtered.loc[
        filtered["Transaction ID"].str.contains(query, case=False, regex=False)
    ]

high = int((filtered["Risk"] == "High").sum())
medium = int((filtered["Risk"] == "Medium").sum())
metrics = st.columns(3)
metrics[0].metric("Open flagged transactions", f"{len(filtered):,}", border=True)
metrics[1].metric("High risk", f"{high:,}", border=True)
metrics[2].metric("Medium risk", f"{medium:,}", border=True)
st.dataframe(
    filtered.sort_values("Fraud probability", ascending=False),
    hide_index=True,
    width="stretch",
    column_config={
        "Amount": st.column_config.NumberColumn(format="%.2f"),
        "Elapsed seconds": st.column_config.NumberColumn(format="%.0f"),
        "Fraud probability": st.column_config.NumberColumn(format="%.3%"),
    },
)
st.caption(
    "Historical holdout rows have known labels for retrospective evaluation, but the "
    "queue intentionally displays model scores and review state only. DEMO DATA rows "
    "are simulator examples, not current bank transactions."
)

if not filtered.empty:
    with st.form("investigation_status_update"):
        selected_id = st.selectbox(
            "Transaction to update",
            filtered["Transaction ID"].tolist(),
        )
        new_status = st.selectbox("Set review status", list(REVIEW_STATUSES)[1:])
        save = st.form_submit_button("Save investigation status", type="primary")
    if save:
        review_state[selected_id] = new_status
        st.success(f"{selected_id} marked as {new_status}.")
        st.rerun()
