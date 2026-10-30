import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fraud_detection import DATASET_URL, load_dataset

NAVY = "#16324F"
CORAL = "#D45D5D"

data = load_dataset()
dataset_hours = max(1, math.ceil(float(data["Time"].max()) / 3600))
maximum_amount = float(data["Amount"].quantile(0.99))
median_amount = float(data["Amount"].median())

st.title("Transaction risk intelligence")
st.caption(
    "An executive view of transaction patterns, rare-event exposure, and model readiness."
)
st.markdown(
    "### Banking Transaction Fraud Detection"
    "  \n**Portfolio case study · Machine learning · Interactive risk analytics**"
)

with st.container(border=True):
    st.markdown(
        "**Executive brief**  \n"
        "This dashboard analyzes a real, highly imbalanced card-transaction dataset. "
        "It turns the analysis into a reviewable workflow: explore activity, evaluate "
        "models on held-out transactions, and inspect explainability and threshold trade-offs."
    )
    st.caption(
        "Real-time scoring prototype using an interactive model and historical, "
        "anonymized research data. It demonstrates immediate inference, not a connected "
        "bank payment feed or production fraud control."
    )

st.header("Portfolio snapshot")
fraud_rows = int(data["Class"].sum())
total_rows = len(data)
all_fraud_amount = float(data.loc[data["Class"] == 1, "Amount"].sum())
all_fraud_share = fraud_rows / total_rows
dataset_columns = st.columns(4)
dataset_columns[0].metric(
    "Transactions", f"{total_rows:,}", border=True
)
dataset_columns[1].metric(
    "Fraud cases",
    f"{fraud_rows:,}",
    delta=f"{all_fraud_share:.3%} of transactions",
    delta_color="inverse",
    border=True,
)
dataset_columns[2].metric(
    "Fraud-class amount",
    f"{all_fraud_amount:,.2f}",
    border=True,
    help="Sum of Amount values for rows labeled as fraud; this is not confirmed or recovered financial loss.",
)

trained_results = st.session_state.get("advanced_fraud_results")
basic_results = st.session_state.get("fraud_model_results")
if trained_results:
    selected_model = trained_results["selected_model_name"]
    average_precision = trained_results["cross_validation"][selected_model]["average_precision"]
    dataset_columns[3].metric(
        "Model status",
        "Trained",
        delta=f"CV AP {average_precision:.3f}",
        border=True,
    )
elif basic_results:
    dataset_columns[3].metric(
        "Model status",
        "Trained",
        delta="Test metrics available",
        border=True,
    )
else:
    dataset_columns[3].metric(
        "Model status",
        "Not trained",
        delta="Run the Advanced model lab",
        delta_color="off",
        border=True,
    )

st.header("Interactive transaction monitor")
st.write(
    "Filter the observed transactions by relative time and transaction-amount cohort. "
    "The figures and summary cards below recalculate from the selected rows."
)
with st.container(border=True):
    filter_columns = st.columns([1.2, 1])
    selected_hours = filter_columns[0].slider(
        "Time from first recorded transaction (hours)",
        min_value=0,
        max_value=dataset_hours,
        value=(0, dataset_hours),
        step=1,
        help="The source records cover approximately two days and use elapsed time, not calendar timestamps.",
    )
    amount_cohort = filter_columns[1].selectbox(
        "Transaction amount cohort",
        options=[
            "All transactions",
            f"Lower half (≤ {median_amount:,.2f})",
            f"Upper half (> {median_amount:,.2f})",
            f"Top 1% (≥ {maximum_amount:,.2f})",
        ],
        help="Amount cohorts use observed dataset percentiles. Amount units are not specified by the source.",
    )

filtered = data.loc[
    (data["Time"] >= selected_hours[0] * 3600)
    & (data["Time"] < selected_hours[1] * 3600)
].copy()
if amount_cohort.startswith("Lower half"):
    filtered = filtered.loc[filtered["Amount"] <= median_amount]
elif amount_cohort.startswith("Upper half"):
    filtered = filtered.loc[filtered["Amount"] > median_amount]
elif amount_cohort.startswith("Top 1%"):
    filtered = filtered.loc[filtered["Amount"] >= maximum_amount]

if filtered.empty:
    st.warning(
        "No transactions match the selected time and amount filters. Widen the time "
        "range or choose a broader amount cohort."
    )
    st.stop()

filtered_fraud = filtered.loc[filtered["Class"] == 1]
filtered_fraud_count = len(filtered_fraud)
filtered_fraud_rate = filtered_fraud_count / len(filtered)
kpis = st.columns(4)
kpis[0].metric("Filtered txns", f"{len(filtered):,}", border=True)
kpis[1].metric("Fraud cases", f"{filtered_fraud_count:,}", border=True)
kpis[2].metric("Fraud rate", f"{filtered_fraud_rate:.3%}", border=True)
kpis[3].metric(
    "Fraud amount",
    f"{filtered_fraud['Amount'].sum():,.2f}",
    border=True,
    help="Transaction amount attached to fraud-labeled records, not verified loss.",
)

st.header("Activity and fraud-rate trend")
st.write(
    "Hourly transaction volume and fraud rate are plotted on separate axes. "
    "Rates with few observations can be volatile; these are descriptive historical data."
)
hourly = filtered.assign(
    Hour=filtered["Time"].floordiv(3600).astype("int16")
).groupby("Hour", observed=True)["Class"].agg(
    transactions="size",
    fraud_cases="sum",
)
hourly["fraud_rate_pct"] = (
    hourly["fraud_cases"].div(hourly["transactions"].replace(0, np.nan)) * 100
)
timeline = go.Figure()
timeline.add_trace(
    go.Bar(
        x=hourly.index,
        y=hourly["transactions"],
        name="Transactions",
        marker_color="#DCE6EF",
        hovertemplate="Hour %{x}<br>Transactions %{y:,}<extra></extra>",
    )
)
timeline.add_trace(
    go.Scatter(
        x=hourly.index,
        y=hourly["fraud_rate_pct"],
        name="Fraud rate",
        mode="lines+markers",
        line={"color": CORAL, "width": 3},
        marker={"size": 6},
        yaxis="y2",
        hovertemplate="Hour %{x}<br>Fraud rate %{y:.3f}%<extra></extra>",
    )
)
timeline.update_layout(
    height=390,
    template="plotly_white",
    margin={"l": 20, "r": 30, "t": 30, "b": 20},
    xaxis={"title": "Elapsed hour", "dtick": 4, "gridcolor": "#EDF1F5"},
    yaxis={"title": "Transactions", "gridcolor": "#EDF1F5"},
    yaxis2={
        "title": "Fraud rate (%)",
        "overlaying": "y",
        "side": "right",
        "showgrid": False,
    },
    legend={"orientation": "h", "y": 1.12, "x": 0},
    hovermode="x unified",
)
st.plotly_chart(
    timeline,
    width="stretch",
    alt="Hourly transaction volume and labeled fraud rate for the selected filters.",
    config={"displaylogo": False, "responsive": True},
)

chart_left, chart_right = st.columns([0.85, 1.15])
with chart_left:
    st.subheader("Class distribution")
    st.caption("Fraud is rare; accuracy alone can hide missed fraud.")
    counts = filtered["Class"].value_counts().reindex([0, 1], fill_value=0)
    donut = go.Figure(
        go.Pie(
            labels=["Normal", "Fraud"],
            values=counts.tolist(),
            hole=0.68,
            sort=False,
            marker={"colors": [NAVY, CORAL], "line": {"color": "white", "width": 3}},
            textinfo="label+percent",
            hovertemplate="%{label}<br>%{value:,} transactions<br>%{percent}<extra></extra>",
        )
    )
    donut.update_layout(
        height=330,
        template="plotly_white",
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        showlegend=False,
        annotations=[
            {
                "text": f"{filtered_fraud_rate:.2%}<br><span style='font-size:12px'>fraud</span>",
                "showarrow": False,
                "font": {"size": 24, "color": NAVY},
            }
        ],
    )
    st.plotly_chart(
        donut,
        width="stretch",
        alt="Donut chart showing the normal and fraud class proportions in the selected transactions.",
        config={"displaylogo": False, "responsive": True},
    )

with chart_right:
    st.subheader("Amount profile by outcome")
    st.caption(
        "Box plots use a log-scaled amount axis to make differences in the long-tailed "
        "transaction values readable."
    )
    amount_data = filtered.loc[filtered["Amount"] > 0, ["Amount", "Class"]].copy()
    amount_data["Recorded outcome"] = amount_data["Class"].map(
        {0: "Normal", 1: "Fraud"}
    )
    amount_profile = go.Figure()
    for label, color in (("Normal", NAVY), ("Fraud", CORAL)):
        values = amount_data.loc[
            amount_data["Recorded outcome"] == label, "Amount"
        ]
        if not values.empty:
            amount_profile.add_trace(
                go.Box(
                    y=values,
                    name=label,
                    marker_color=color,
                    boxpoints=False,
                    hovertemplate=f"{label}<br>Amount %{{y:,.2f}}<extra></extra>",
                )
            )
    amount_profile.update_layout(
        height=330,
        template="plotly_white",
        margin={"l": 20, "r": 20, "t": 10, "b": 20},
        yaxis={"title": "Transaction amount (log scale)", "type": "log", "gridcolor": "#EDF1F5"},
        xaxis={"title": ""},
        showlegend=False,
    )
    st.plotly_chart(
        amount_profile,
        width="stretch",
        alt="Log-scale box plots comparing recorded transaction amounts for normal and fraud labels.",
        config={"displaylogo": False, "responsive": True},
    )

st.header("Model readiness and next steps")
if trained_results:
    selected_name = trained_results["selected_model_name"]
    selected_ap = trained_results["cross_validation"][selected_name]["average_precision"]
    st.success(
        f"Advanced model comparison is available in this session. "
        f"Selected model: **{selected_name}** · Cross-validated average precision: "
        f"**{selected_ap:.3f}**."
    )
else:
    st.info(
        "No advanced model has been trained in this browser session yet. "
        "Run the Advanced model lab to compare three algorithms and see final "
        "held-out results."
    )
st.caption("Next: open **Advanced model lab** in the sidebar to train and compare models.")

with st.expander("Dataset and interpretation notes"):
    st.write(
        "This public research dataset contains 284,807 European card transactions "
        "recorded over approximately two days; it includes 492 fraud labels. `Time` "
        "is elapsed seconds since the first record. `Amount` units are not documented "
        "here. The `V1`–`V28` columns are anonymized principal components, so they do "
        "not map to merchant, account, or customer attributes."
    )
    st.markdown(f"Dataset source: [OpenML dataset 1597]({DATASET_URL})")
    st.warning(
        "A fraud-class amount sum is not confirmed customer loss, recovered funds, "
        "or bank exposure. Model outputs are educational and must not drive real "
        "financial decisions."
    )
