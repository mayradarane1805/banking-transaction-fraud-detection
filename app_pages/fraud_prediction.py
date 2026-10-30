import numpy as np
import pandas as pd
import streamlit as st

from fraud_detection import FEATURE_COLUMNS, TARGET, load_dataset
from prediction import score_transaction

st.title("Real-time fraud scoring prototype")
st.caption(
    "Submit transaction features and receive an immediate fraud score from the trained model."
)

results = st.session_state.get("fraud_model_results")
if results is None:
    st.info("Train the Decision Tree and Random Forest on the Model training page first.")
    st.stop()

st.write(
    "This prototype demonstrates the scoring step in a real-time fraud workflow. "
    "The form starts with a genuine held-out transaction from the real dataset; edit "
    "the feature values and submit to score immediately. `V1`–`V28` are anonymized "
    "PCA components, so they are not intuitive banking controls. The app does not "
    "receive live transactions from a bank."
)

data = load_dataset()
candidate_ids = results["X_test"].index[:20].tolist()
selected_id = st.selectbox(
    "Held-out transaction row",
    options=candidate_ids,
    format_func=lambda row_id: f"Dataset row {row_id:,}",
)
original_transaction = results["X_test"].loc[[selected_id], FEATURE_COLUMNS]

with st.form("transaction_prediction"):
    selected_model = st.selectbox(
        "Model for this prediction",
        options=list(results["models"]),
    )
    edited_transaction = st.data_editor(
        original_transaction,
        hide_index=True,
        num_rows="fixed",
        width="stretch",
        height=120,
        key=f"transaction_editor_{selected_id}",
        column_config={
            "Time": st.column_config.NumberColumn(help="Seconds since the first recorded transaction."),
            "Amount": st.column_config.NumberColumn(help="Recorded transaction amount."),
        },
    )
    submitted = st.form_submit_button("Check transaction", type="primary")

if submitted:
    edited_transaction = edited_transaction.loc[:, FEATURE_COLUMNS].astype("float64")
    st.session_state["fraud_prediction_pending"] = (
        selected_id,
        edited_transaction,
        selected_model,
    )

pending_prediction = st.session_state.get("fraud_prediction_pending")
if pending_prediction is not None:
    pending_id, transaction, selected_model = pending_prediction
    if pending_id == selected_id:
        threshold = st.session_state.get("fraud_prediction_threshold", 0.5)
        prediction = score_transaction(
            results["models"][selected_model],
            transaction,
            threshold,
        )
        probability = prediction["fraud_probability"]
        with st.container(border=True):
            st.subheader("Immediate model response")
            result_columns = st.columns(2)
            result_columns[0].metric("Estimated fraud probability", f"{probability:.1%}")
            result_columns[1].metric(
                "Predicted class",
                prediction["predicted_class"],
            )
            st.metric("Prototype risk level", prediction["risk_level"])
            st.caption(
                f"Model: {selected_model}. Display threshold: {threshold:.2f}. "
                "Prototype output only; do not use it to authorize or decline a payment."
            )
            if not prediction["leading_global_features"].empty:
                st.write("**Leading global model features**")
                st.dataframe(
                    prediction["leading_global_features"],
                    hide_index=True,
                    width="stretch",
                )
            st.caption(prediction["explanation_note"])
        baseline = original_transaction.to_numpy(dtype=np.float64)
        submitted_values = transaction.to_numpy(dtype=np.float64)
        if np.array_equal(baseline, submitted_values):
            actual_label = int(data.loc[pending_id, TARGET])
            actual_name = "Fraud" if actual_label == 1 else "Normal"
            st.info(
                f"Because the inputs are unchanged, the dataset's recorded label for this "
                f"held-out example is **{actual_name}**."
            )
        else:
            st.caption("The edited input is not the original transaction; no known label is shown.")
