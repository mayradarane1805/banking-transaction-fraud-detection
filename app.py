import streamlit as st

st.set_page_config(
    page_title="AI-Powered Banking Transaction Fraud Detection",
    page_icon=":material/account_balance:",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = {
    "Overview": [
        st.Page(
            "app_pages/overview.py",
            title="Executive dashboard",
            icon=":material/dashboard:",
            default=True,
        ),
        st.Page(
            "app_pages/project_overview.py",
            title="Introduction and objectives",
            icon=":material/info:",
        ),
    ],
    "Data and models": [
        st.Page(
            "app_pages/exploration.py",
            title="Dataset, EDA and search",
            icon=":material/query_stats:",
        ),
        st.Page(
            "app_pages/models.py",
            title="Training and evaluation",
            icon=":material/model_training:",
        ),
        st.Page(
            "app_pages/advanced.py",
            title="Advanced comparison and batch scoring",
            icon=":material/science:",
        ),
        st.Page(
            "app_pages/fraud_prediction.py",
            title="Individual fraud prediction",
            icon=":material/shield_with_heart:",
        ),
    ],
    "Operations prototype": [
        st.Page(
            "app_pages/investigations.py",
            title="Fraud alerts and investigation",
            icon=":material/assignment_late:",
        ),
        st.Page(
            "app_pages/simulator.py",
            title="Live-style DEMO DATA simulator",
            icon=":material/stream:",
        ),
        st.Page(
            "app_pages/monitoring.py",
            title="Model performance monitoring",
            icon=":material/monitoring:",
        ),
        st.Page(
            "app_pages/architecture.py",
            title="System architecture",
            icon=":material/account_tree:",
        ),
        st.Page(
            "app_pages/conclusion.py",
            title="Conclusion and future scope",
            icon=":material/flag:",
        ),
    ],
}

st.sidebar.title("Risk intelligence")
st.sidebar.caption("AI-powered banking fraud risk · prototype")
st.sidebar.divider()
st.sidebar.info(
    "Real-time scoring prototype: submit transaction features for an immediate "
    "model response. Historical research data is used; there is no connected bank "
    "payment feed or production decision integration."
)

st.session_state.setdefault("investigation_status", {})
st.session_state.setdefault("demo_events", [])
st.session_state.setdefault("simulator_running", False)
st.session_state.setdefault("simulator_cursor", 0)

navigation = st.navigation(pages, position="sidebar")
navigation.run()
