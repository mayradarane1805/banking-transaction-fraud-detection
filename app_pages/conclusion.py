import streamlit as st

st.title("Conclusion and future scope")

st.header("What this project demonstrates")
st.write(
    "Fraud detection is a rare-event classification problem: the normal class dominates, "
    "so accuracy alone cannot describe how useful a detector is. The Model training page "
    "shows actual test-set precision, recall, F1-score, confusion matrices, and curves "
    "after you train the models."
)
st.write(
    "The primary workflow compares Decision Tree and Random Forest models. The "
    "syllabus-informed comparison also implements Logistic Regression, selected-feature "
    "classification, k-NN distance choices, SVM kernels, Perceptron, MLP activations, "
    "bagging, boosting, voting, stacking, blending, self-training, and XGBoost. "
    "Separate experiments demonstrate amount regression, unsupervised clustering, "
    "aggregate-stream HMM/CRF methods, and an explicitly assumed-reward RL exercise."
)
st.write(
    "The Advanced lab adds stratified cross-validation, a separate validation split "
    "for cost-sensitive threshold selection, calibration analysis, batch CSV scoring, "
    "and sequence-model comparisons. Test data remains separate from training and "
    "threshold selection in the primary evaluation workflows."
)
st.write(
    "The Transaction scoring page demonstrates the real-time prototype: a submitted "
    "transaction feature row is passed to the trained model and a fraud probability "
    "and threshold-based result are returned immediately. A production deployment "
    "would additionally require a secure live payment feed, service/API hosting, "
    "monitoring, and approved operational controls."
)

st.header("Future scope")
for item in (
    "Validate the candidate models on newer, independently sourced transaction data.",
    "Use customer/account histories and event timestamps for meaningful per-customer sequence models.",
    "Replace assumed threshold and reinforcement-learning costs with approved operational evidence.",
    "Add local explanations, fairness analysis, audit logging, access controls, and persistent case management.",
    "Monitor data drift and latency in a properly governed production environment before any real-world use.",
):
    st.markdown(f"- {item}")

st.warning(
    "This is a real-time scoring prototype, not a connected bank payment system. "
    "Its dataset is historical, anonymized, and limited to two recorded days; results "
    "may not generalize to current transactions or support real payment decisions."
)
