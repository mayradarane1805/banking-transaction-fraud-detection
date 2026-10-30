import pandas as pd
import plotly.express as px
import streamlit as st

from prototype_utils import create_demo_event
from rl_experiments import run_offline_policy_experiment

st.title("Live-style transaction simulator")
st.caption("An automated scoring demonstration using historical rows, clearly marked DEMO DATA.")
st.warning(
    "This simulator does not connect to a bank, card network, or payment gateway. "
    "It replays held-out research transactions and assigns current demo timestamps."
)

results = st.session_state.get("fraud_model_results")
if results is None:
    st.info("Train the Decision Tree and Random Forest before starting the simulator.")
    st.stop()

st.session_state.setdefault("simulator_running", False)
st.session_state.setdefault("simulator_cursor", 0)
st.session_state.setdefault("demo_events", [])
model_name = st.selectbox("Scoring model", list(results["models"]), key="simulator_model")
threshold = float(st.session_state.get("fraud_prediction_threshold", 0.5))
start_column, stop_column, reset_column = st.columns(3)
if start_column.button("Start demo stream", type="primary", disabled=st.session_state["simulator_running"]):
    st.session_state["simulator_running"] = True
    st.rerun()
if stop_column.button("Stop demo stream", disabled=not st.session_state["simulator_running"]):
    st.session_state["simulator_running"] = False
    st.rerun()
if reset_column.button("Clear demo history"):
    st.session_state["demo_events"] = []
    st.session_state["simulator_cursor"] = 0

st.metric(
    "Stream status",
    "RUNNING · DEMO DATA" if st.session_state["simulator_running"] else "STOPPED",
    border=True,
)

@st.fragment(run_every="2s" if st.session_state["simulator_running"] else None)
def render_demo_stream() -> None:
    if st.session_state["simulator_running"]:
        event, next_cursor = create_demo_event(
            results,
            model_name,
            st.session_state["simulator_cursor"],
            threshold,
        )
        st.session_state["simulator_cursor"] = next_cursor
        st.session_state["demo_events"].append(event)
        st.session_state["demo_events"] = st.session_state["demo_events"][-200:]
        if event["Predicted class"] == "Fraud":
            st.session_state.setdefault("investigation_status", {})[event["Transaction ID"]] = "New"

    events = pd.DataFrame(st.session_state["demo_events"])
    if events.empty:
        st.info("Start the stream to see scored demonstration transactions.")
        return
    latest = events.iloc[-1]
    with st.container(border=True):
        st.subheader("Most recent model response")
        metric_columns = st.columns(4)
        metric_columns[0].metric("DEMO transaction", latest["Transaction ID"])
        metric_columns[1].metric("Fraud probability", f"{latest['Fraud probability']:.1%}")
        metric_columns[2].metric("Predicted class", latest["Predicted class"])
        metric_columns[3].metric("Amount", f"{latest['Amount']:.2f}")
    chart = px.line(
        events,
        x="Timestamp (UTC)",
        y="Fraud probability",
        color="Risk",
        markers=True,
        title="Prototype scoring stream",
        template="plotly_white",
    )
    chart.update_layout(height=340, yaxis_tickformat=".0%")
    st.plotly_chart(chart, width="stretch", alt="Fraud probability over the DEMO DATA scoring stream.")
    st.dataframe(
        events.sort_values("Timestamp (UTC)", ascending=False).head(25),
        hide_index=True,
        width="stretch",
        column_config={
            "Fraud probability": st.column_config.NumberColumn(format="%.3%"),
            "Amount": st.column_config.NumberColumn(format="%.2f"),
        },
    )

render_demo_stream()

st.divider()
with st.expander("Reinforcement-learning policy experiment", expanded=False):
    st.warning(
        "This is an offline teaching experiment, not an operational fraud policy. "
        "The dataset does not record bank actions, customer responses, or real rewards."
    )
    st.write(
        "A model-score bucket is the state; Allow and Review are actions. The Q-learning "
        "and small DQN policies train on an earlier portion of held-out examples and "
        "are measured on a later portion. Reward costs below are explicit relative "
        "assumptions, not real bank losses."
    )
    policy_columns = st.columns(2)
    fp_cost = policy_columns[0].number_input(
        "Relative cost of reviewing a normal transaction",
        min_value=0.1,
        value=1.0,
        step=0.5,
        key="rl_fp_cost",
    )
    fn_cost = policy_columns[1].number_input(
        "Relative cost of allowing a fraud transaction",
        min_value=0.1,
        value=10.0,
        step=1.0,
        key="rl_fn_cost",
    )
    if st.button("Run Q-learning and DQN experiment", key="run_rl_experiment"):
        try:
            with st.spinner("Training policies on an earlier held-out segment..."):
                st.session_state["rl_policy_results"] = run_offline_policy_experiment(
                    results["probabilities"][model_name],
                    results["y_test"].to_numpy(),
                    results["X_test"]["Time"].to_numpy(),
                    false_positive_cost=fp_cost,
                    false_negative_cost=fn_cost,
                )
                st.session_state["rl_policy_settings"] = (model_name, fp_cost, fn_cost)
        except ValueError as exc:
            st.error(str(exc), title="Could not run policy experiment")

    policy_result = st.session_state.get("rl_policy_results")
    if policy_result and st.session_state.get("rl_policy_settings") == (
        model_name,
        fp_cost,
        fn_cost,
    ):
        policy_table = pd.DataFrame(policy_result["policies"]).T.rename_axis("Policy").reset_index()
        st.caption(
            f"{policy_result['training_rows']:,} earlier held-out examples used for "
            f"policy learning · {policy_result['test_rows']:,} later examples for "
            f"policy comparison · {policy_result['test_fraud_rows']:,} recorded fraud."
        )
        st.dataframe(policy_table, hide_index=True, width="stretch")
        st.info(policy_result["assumption"])
