import pandas as pd
import streamlit as st

from data_processing import dataset_quality_report
from fraud_detection import DATASET_URL, load_dataset

data = load_dataset()
quality = dataset_quality_report(data)

st.title("Project introduction and objectives")
st.caption("AI-Powered Banking Transaction Fraud Detection and Risk Monitoring System")

with st.container(border=True):
    st.subheader("Executive summary")
    st.write(
        "This project is a real-time scoring prototype for rare-event transaction "
        "fraud detection. It trains classifiers on documented historical transactions, "
        "returns an immediate score for a submitted row, ranks alerts for review, and "
        "provides a clearly labeled DEMO DATA stream. It is not connected to a bank."
    )

st.header("Problem statement")
st.write(
    "Fraud is a very small share of the records. A system that predicts every row as "
    "normal can have high accuracy while finding no fraud. The prototype therefore "
    "compares recall, precision, F1-score, average precision, ROC AUC, and the actual "
    "confusion matrix, alongside accuracy."
)

st.header("Project objectives")
objectives = [
    "Inspect and validate the real dataset before modeling.",
    "Prevent leakage by keeping training and held-out examples separate.",
    "Compare syllabus-relevant algorithms with consistent test rows.",
    "Provide immediate single-row and batch transaction scoring.",
    "Make suspected cases searchable and track investigation status.",
    "Demonstrate an automated stream using visibly labeled DEMO DATA.",
    "Monitor measured model performance and communicate dataset limitations.",
]
for objective in objectives:
    st.markdown(f"- {objective}")

st.header("Dataset snapshot")
columns = st.columns(4)
columns[0].metric("Real transactions", f"{quality['rows']:,}", border=True)
columns[1].metric("Recorded fraud", f"{quality['fraud_rows']:,}", border=True)
columns[2].metric("Fraud prevalence", f"{quality['fraud_rate']:.3%}", border=True)
columns[3].metric("Missing required values", f"{quality['missing_values']:,}", border=True)
st.markdown(f"Dataset source: [OpenML Credit Card Fraud Detection, ID 1597]({DATASET_URL})")

st.header("Syllabus coverage at a glance")
coverage = [
    ("Unit I · Foundations", "Preprocessing/validation, class balance, feature handling, holdout splits, stratified CV, and threshold/cost analysis."),
    ("Unit II · Regression", "Linear, Polynomial, Ridge, and Lasso models predict real transaction Amount as a separate task, not the fraud label."),
    ("Unit III · Classification", "Decision Tree, Random Forest, Logistic Regression, k-NN distance choices, SVM kernels, and additional classifiers are trained and compared."),
    ("Unit IV · HMM/CRF", "Sequence methods need an honest ordered sequence. This dataset has no customer IDs; global time order is only an aggregate-stream teaching experiment."),
    ("Unit V · Clustering", "Unsupervised discovery/outlier comparisons use features only; fraud labels are used after fitting for evaluation."),
    ("Unit VI · Advanced methods", "Bagging/boosting/ensembles, Perceptron/MLP variants, and Q-learning/DQN are implemented with their dataset limitations stated."),
]
st.dataframe(
    pd.DataFrame(coverage, columns=["Syllabus unit", "Application in this project"]),
    hide_index=True,
    width="stretch",
)
st.info(
    "The syllabus guides algorithm selection and project explanations; it is not "
    "a separate syllabus-only dashboard module. Topics that the dataset cannot "
    "support as a valid fraud experiment are labeled and bounded in the coverage guide."
)
