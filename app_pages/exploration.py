import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from additional_experiments import CLUSTERING_ALGORITHMS, cluster_real_transactions
from data_processing import dataset_quality_report
from fraud_detection import EXPECTED_COLUMNS, FEATURE_COLUMNS, TARGET, load_dataset

NAVY = "#16324F"
TEAL = "#167D75"
CORAL = "#D45D5D"

data = load_dataset()
quality = dataset_quality_report(data)
fraud = data.loc[data["Class"] == 1]
normal = data.loc[data["Class"] == 0]
fraud_share = len(fraud) / len(data)

st.title("Transaction explorer")
st.caption(
    "Investigate the labeled transactions, compare distributions, and understand "
    "why rare-event evaluation needs more than accuracy."
)

st.header("Dataset overview and preprocessing")
overview_columns = st.columns(4)
overview_columns[0].metric("Rows", f"{quality['rows']:,}", border=True)
overview_columns[1].metric("Input features", f"{len(FEATURE_COLUMNS):,}", border=True)
overview_columns[2].metric("Missing values", f"{quality['missing_values']:,}", border=True)
overview_columns[3].metric("Duplicate rows", f"{quality['duplicate_rows']:,}", border=True)
st.dataframe(
    pd.DataFrame(
        [
            {"Column group": "Transaction context", "Columns": "Time, Amount", "Purpose": "Elapsed seconds and recorded transaction amount"},
            {"Column group": "Anonymized features", "Columns": "V1–V28", "Purpose": "PCA-transformed input features; original business meanings unavailable"},
            {"Column group": "Target", "Columns": TARGET, "Purpose": "0 = normal; 1 = recorded fraud"},
        ]
    ),
    hide_index=True,
    width="stretch",
)
with st.expander("Preprocessing and leakage controls"):
    st.markdown(
        "- Required columns and binary labels are validated; malformed or non-finite data raises an error.\n"
        "- The source table is already numeric and has no missing required values, so no imputation or categorical encoding is invented.\n"
        "- Stratified sampling preserves rare fraud cases in each split; class weights are applied in selected classifiers.\n"
        "- Scaling is fitted inside model pipelines for distance- and margin-based learners.\n"
        "- Threshold selection uses validation data in the advanced workflow; the held-out test data is for final evaluation only."
    )

kpis = st.columns(4)
kpis[0].metric("Transactions", f"{len(data):,}", border=True)
kpis[1].metric("Normal", f"{len(normal):,}", border=True)
kpis[2].metric("Fraud", f"{len(fraud):,}", border=True)
kpis[3].metric("Fraud prevalence", f"{fraud_share:.3%}", border=True)

st.warning(
    f"Only about 1 transaction in {1 / fraud_share:,.0f} is labeled fraud. "
    "A model that predicts every transaction as normal could still report high accuracy."
)

st.header("Amount patterns by recorded outcome")
st.write(
    "The plot compares the distribution shape within each outcome rather than raw "
    "counts, so the rare fraud class remains visible. Amount values use the units "
    "provided by the source dataset; the currency is not specified here."
)
amount_figure = go.Figure()
logged_amounts = np.log1p(data["Amount"].to_numpy(dtype=np.float64))
amount_bin_edges = np.linspace(logged_amounts.min(), logged_amounts.max(), 46)
for label, frame, color in (
    ("Normal", normal, NAVY),
    ("Fraud", fraud, CORAL),
):
    counts, _ = np.histogram(
        np.log1p(frame["Amount"].to_numpy(dtype=np.float64)),
        bins=amount_bin_edges,
    )
    amount_figure.add_trace(
        go.Bar(
            x=(amount_bin_edges[:-1] + amount_bin_edges[1:]) / 2,
            y=counts / max(int(counts.sum()), 1),
            width=np.diff(amount_bin_edges),
            name=label,
            opacity=0.68,
            marker_color=color,
            hovertemplate="log(1 + amount): %{x:.2f}<br>Share: %{y:.2%}<extra>%{fullData.name}</extra>",
        )
    )
amount_figure.update_layout(
    barmode="overlay",
    height=390,
    template="plotly_white",
    margin={"l": 20, "r": 20, "t": 35, "b": 20},
    xaxis={"title": "log(1 + recorded transaction amount)", "gridcolor": "#EDF1F5"},
    yaxis={"title": "Share within outcome", "gridcolor": "#EDF1F5", "tickformat": ".0%"},
    legend={"orientation": "h", "y": 1.12, "x": 0},
)
st.plotly_chart(
    amount_figure,
    width="stretch",
    alt="Overlaid, normalized distributions of logged transaction amount for normal and fraud labels.",
    config={"displaylogo": False, "responsive": True},
)

st.header("Hourly activity")
st.write(
    "The dataset records elapsed seconds, not calendar timestamps. This view bins "
    "the approximately two-day observation window into one-hour intervals."
)
hourly = data.assign(Hour=data["Time"].floordiv(3600).astype("int16")).groupby(
    "Hour", observed=True
)["Class"].agg(
    transactions="size",
    fraud_cases="sum",
)
hourly["fraud_rate_pct"] = (
    hourly["fraud_cases"].div(hourly["transactions"].replace(0, np.nan)) * 100
)
timeline = go.Figure()
timeline.add_trace(
    go.Scatter(
        x=hourly.index,
        y=hourly["transactions"],
        mode="lines",
        name="Transactions",
        line={"color": NAVY, "width": 2},
        fill="tozeroy",
        fillcolor="rgba(22, 50, 79, 0.08)",
        hovertemplate="Hour %{x}<br>Transactions %{y:,}<extra></extra>",
    )
)
timeline.add_trace(
    go.Scatter(
        x=hourly.index,
        y=hourly["fraud_rate_pct"],
        mode="lines+markers",
        name="Fraud rate",
        line={"color": CORAL, "width": 2},
        marker={"size": 5},
        yaxis="y2",
        hovertemplate="Hour %{x}<br>Fraud rate %{y:.3f}%<extra></extra>",
    )
)
timeline.update_layout(
    height=380,
    template="plotly_white",
    margin={"l": 20, "r": 30, "t": 35, "b": 20},
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
    alt="Transaction counts and labeled fraud rate by elapsed hour.",
    config={"displaylogo": False, "responsive": True},
)

analysis_left, analysis_right = st.columns([1.05, 0.95])
with analysis_left:
    st.header("Fraud rate by amount segment")
    amount_buckets = pd.qcut(
        data["Amount"],
        q=5,
        duplicates="drop",
    )
    segments = (
        data.assign(AmountSegment=amount_buckets)
        .groupby("AmountSegment", observed=True)["Class"]
        .agg(transactions="size", fraud_cases="sum", fraud_rate="mean")
        .reset_index()
    )
    segments["segment_label"] = segments["AmountSegment"].map(
        lambda bounds: f"{bounds.left:,.0f}–{bounds.right:,.0f}"
    )
    segment_figure = go.Figure(
        go.Bar(
            x=segments["segment_label"],
            y=segments["fraud_rate"] * 100,
            marker_color=[NAVY, "#456A83", "#53899B", TEAL, CORAL][: len(segments)],
            text=[
                f"{rate:.2f}%<br>n={count:,}"
                for rate, count in zip(
                    segments["fraud_rate"] * 100,
                    segments["transactions"],
                )
            ],
            textposition="outside",
            hovertemplate=(
                "Amount range: %{x}<br>Fraud rate: %{y:.3f}%"
                "<extra></extra>"
            ),
        )
    )
    segment_figure.update_layout(
        height=370,
        template="plotly_white",
        margin={"l": 20, "r": 20, "t": 35, "b": 50},
        xaxis={"title": "Recorded amount quintile", "gridcolor": "#EDF1F5"},
        yaxis={"title": "Fraud rate (%)", "gridcolor": "#EDF1F5"},
        showlegend=False,
    )
    st.plotly_chart(
        segment_figure,
        width="stretch",
        alt="Observed fraud rate and sample counts across five transaction amount quantiles.",
        config={"displaylogo": False, "responsive": True},
    )
    st.caption(
        "Amount segments are descriptive groups of equal-frequency transactions. "
        "They are not model rules or evidence that amount causes fraud."
    )

    st.header("Transaction search and history")
    st.write(
        "Search historical source rows by their dataset row number, amount range, "
        "and recorded class. This is a research-data lookup, not a customer's banking history."
    )
    search_columns = st.columns(3)
    query = search_columns[0].text_input(
        "Search dataset row number",
        type="search",
        placeholder="For example: 12345",
    )
    class_filter = search_columns[1].selectbox(
        "Recorded outcome",
        options=["All", "Normal", "Fraud"],
    )
    max_results = search_columns[2].selectbox(
        "Maximum rows to display",
        options=[25, 50, 100, 250],
        index=1,
    )
    history = data.rename_axis("Dataset row").reset_index()
    if query:
        if not query.isdecimal():
            st.error("Enter a numeric dataset row number.")
            history = history.iloc[0:0]
        else:
            history = history.loc[history["Dataset row"].astype(str).str.contains(query, regex=False)]
    if class_filter == "Normal":
        history = history.loc[history[TARGET] == 0]
    elif class_filter == "Fraud":
        history = history.loc[history[TARGET] == 1]
    st.dataframe(
        history.head(max_results),
        hide_index=True,
        width="stretch",
        column_config={
            "Dataset row": st.column_config.NumberColumn(format="%d"),
            "Time": st.column_config.NumberColumn(format="%.0f"),
            "Amount": st.column_config.NumberColumn(format="%.2f"),
            TARGET: st.column_config.NumberColumn("Recorded label", format="%d"),
        },
    )
    st.caption(
        f"Showing {min(len(history), max_results):,} of {len(history):,} matching "
        f"historical records. Required schema: {', '.join(EXPECTED_COLUMNS)}."
    )

with analysis_right:
    st.header("Feature associations")
    correlations = (
        data.corr(numeric_only=True)["Class"]
        .drop("Class")
        .sort_values(key=lambda values: values.abs(), ascending=False)
        .head(10)
        .sort_values()
    )
    correlation_figure = go.Figure(
        go.Bar(
            x=correlations.values,
            y=correlations.index,
            orientation="h",
            marker_color=[
                CORAL if value > 0 else TEAL for value in correlations.values
            ],
            hovertemplate="%{y}<br>Correlation: %{x:.4f}<extra></extra>",
        )
    )
    correlation_figure.update_layout(
        height=370,
        template="plotly_white",
        margin={"l": 20, "r": 20, "t": 35, "b": 20},
        xaxis={"title": "Pearson correlation with Class", "gridcolor": "#EDF1F5"},
        yaxis={"title": ""},
        showlegend=False,
    )
    st.plotly_chart(
        correlation_figure,
        width="stretch",
        alt="Ten features with the largest absolute Pearson correlation to the fraud label.",
        config={"displaylogo": False, "responsive": True},
    )

st.header("Outcome summary")
summary = (
    data.groupby("Class", observed=True)
    .agg(
        transactions=("Amount", "size"),
        mean_amount=("Amount", "mean"),
        median_amount=("Amount", "median"),
        total_amount=("Amount", "sum"),
    )
    .rename(index={0: "Normal", 1: "Fraud"})
    .rename_axis("Recorded outcome")
    .reset_index()
)
st.dataframe(
    summary,
    hide_index=True,
    width="stretch",
    column_config={
        "transactions": st.column_config.NumberColumn("Transactions", format="%d"),
        "mean_amount": st.column_config.NumberColumn("Mean amount", format="%.2f"),
        "median_amount": st.column_config.NumberColumn("Median amount", format="%.2f"),
        "total_amount": st.column_config.NumberColumn("Sum of amounts", format="%.2f"),
    },
)
st.caption(
    "The fraud-class amount sum is a descriptive transaction total, not confirmed "
    "financial loss. V1–V28 are anonymized principal components, so their original "
    "banking meanings are unavailable."
)

st.divider()
with st.expander("Unsupervised clustering and outlier experiment", expanded=False):
    st.write(
        "Fit clustering on transaction features only; fraud labels are compared after "
        "fitting. This gives a practical comparison for the syllabus clustering unit "
        "without treating clusters as fraud decisions."
    )
    cluster_columns = st.columns(3)
    cluster_algorithm = cluster_columns[0].selectbox(
        "Clustering / outlier method",
        list(CLUSTERING_ALGORITHMS),
        key="eda_cluster_algorithm",
    )
    max_cluster_rows = 250 if cluster_algorithm == "CURE" else 2_000
    cluster_rows = cluster_columns[1].slider(
        "Sample rows",
        min_value=200,
        max_value=max_cluster_rows,
        value=min(500, max_cluster_rows),
        step=50,
        key="eda_cluster_rows",
    )
    cluster_count = cluster_columns[2].slider(
        "Requested clusters",
        min_value=2,
        max_value=10,
        value=4,
        key="eda_cluster_count",
    )
    cluster_key = (
        cluster_algorithm,
        cluster_rows,
        cluster_count,
    )
    if st.button("Run unsupervised experiment", key="run_eda_clustering"):
        try:
            with st.spinner(f"Fitting {cluster_algorithm} on real transaction features..."):
                st.session_state["eda_clustering_results"] = cluster_real_transactions(
                    cluster_algorithm,
                    sample_rows=cluster_rows,
                    clusters=cluster_count,
                )
                st.session_state["eda_clustering_settings"] = cluster_key
        except ValueError as exc:
            st.error(str(exc), title="Could not run clustering experiment")
    cluster_result = st.session_state.get("eda_clustering_results")
    if cluster_result and st.session_state.get("eda_clustering_settings") == cluster_key:
        cluster_metrics = cluster_result["metrics"]
        st.caption(
            f"{cluster_result['algorithm']} · {cluster_result['rows']:,} real rows · "
            f"{cluster_result['cluster_count']:,} clusters found."
        )
        if cluster_algorithm == "LOF · outlier detection":
            st.dataframe(
                pd.DataFrame([cluster_metrics]),
                hide_index=True,
                width="stretch",
            )
        else:
            st.dataframe(
                pd.DataFrame([cluster_metrics]),
                hide_index=True,
                width="stretch",
            )
        plot_data = pd.DataFrame(
            {
                "Component 1": cluster_result["projection"][:, 0],
                "Component 2": cluster_result["projection"][:, 1],
                "Cluster": cluster_result["labels"].astype(str),
                "Recorded label (post-fit)": np.where(
                    cluster_result["fraud_labels"] == 1, "Fraud", "Normal"
                ),
            }
        )
        cluster_plot = px.scatter(
            plot_data,
            x="Component 1",
            y="Component 2",
            color="Cluster",
            symbol="Recorded label (post-fit)",
            template="plotly_white",
            title=f"{cluster_algorithm} · PCA projection",
        )
        cluster_plot.update_layout(height=440)
        st.plotly_chart(
            cluster_plot,
            width="stretch",
            alt=f"Post-fit PCA view of {cluster_algorithm} clusters and recorded labels.",
        )
        st.caption(
            "CURE is bounded to 250 rows for interactive runtime. MST single-linkage "
            "is represented by its minimum-spanning-tree-equivalent hierarchy. "
            "Unsupervised external scores may be unavailable when the random sample "
            "contains only one recorded class."
        )
