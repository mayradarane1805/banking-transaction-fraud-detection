import streamlit as st

st.title("System architecture and workflow")
st.caption("A prototype architecture: historical data in, validated model scores out, with DEMO DATA separated from real feeds.")

st.graphviz_chart(
    """
    digraph fraud_prototype {
      graph [rankdir=LR, bgcolor="transparent", pad="0.25", nodesep="0.45", ranksep="0.55"];
      node [shape=box, style="rounded,filled", fillcolor="#F1F5F9", color="#16324F", fontname="Arial", fontsize=10];
      edge [color="#167D75", fontname="Arial", fontsize=9];
      data [label="OpenML research dataset\\n284,807 rows · 492 fraud"];
      validate [label="Schema validation\\nquality checks · preprocessing"];
      split [label="Stratified train / validation / test\\nno test leakage"];
      models [label="Model training\\nDT · RF · Logistic · ensembles"];
      evaluate [label="Evaluation\\nPR · ROC · confusion matrix"];
      ui [label="Streamlit prototype\\nEDA · monitoring · investigation"];
      score [label="Immediate scoring\\nsingle transaction / batch CSV"];
      demo [label="DEMO DATA stream\\nreplayed held-out examples"];
      alerts [label="Review queue\\nsearch · status tracking"];
      data -> validate -> split -> models -> evaluate -> ui;
      ui -> score;
      score -> alerts;
      demo -> ui [style=dashed, label="simulated only"];
      demo -> alerts [style=dashed];
    }
    """
)

st.header("Request workflow")
for step in (
    "Load the public dataset or provide feature-only CSV input.",
    "Validate exact columns, numeric values, row limits, and non-negative Time/Amount.",
    "Fit preprocessing and models using training data only; preserve held-out rows.",
    "Calculate scores immediately and compare them with the selected threshold.",
    "Record flagged events in a session-scoped investigation queue.",
    "Use the simulator only as DEMO DATA; it does not subscribe to a payment feed.",
):
    st.markdown(f"- {step}")

st.warning(
    "The prototype does not include bank authentication, payment authorization, "
    "production API hosting, persistent enterprise case management, or a real-time "
    "financial data connection."
)
