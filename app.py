import streamlit as st
import pandas as pd
import joblib

# -------------------------------------------------
# Load Trained Pipeline
# -------------------------------------------------
pipeline = joblib.load("model.joblib")

cluster_labels = {
    0: "High-Value Customers",
    1: "Mid-Value Customers",
    2: "Low-Value / At-Risk Customers"
}

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Customer Smart Segmentation",
    layout="wide"
)

# -------------------------------------------------
# Professional Dark Theme Styling
# -------------------------------------------------
st.markdown("""
<style>

.stApp {
    background-color: #0f1117;
    color: #d6d6d6;
}

/* Main container */
.block-container {
    padding: 2.5rem 3rem;
}

/* Section cards */
.section-card {
    background-color: #161a23;
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 24px;
    border: 1px solid #262b3a;
}

/* Title */
.app-title {
    font-size: 34px;
    font-weight: 800;
    color: #e8eaf0;
}

.app-subtitle {
    font-size: 15px;
    color: #9aa4bf;
    margin-top: 6px;
}

/* Buttons */
.stButton > button {
    background-color: #2b5cff;
    color: #ffffff;
    border-radius: 10px;
    padding: 10px 18px;
    font-weight: 600;
    border: none;
}


/* Inputs */
.stNumberInput input {
    background-color: #1e2330 !important;
    color: #e8eaf0 !important;
    border-radius: 8px;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background-color: #161a23;
    border-radius: 12px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #121520;
    border-right: 1px solid #262b3a;
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: #d6d6d6;
}

/* Success / Error */
.stSuccess {
    background-color: #0f2d1f;
    color: #7ae0b2;
}

.stError {
    background-color: #2a1215;
    color: #ff8f8f;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown("""
<div class="section-card">
    <div class="app-title">Customer Smart Segmentation</div>
    <div class="app-subtitle">
        Machine Learning–based Customer Segmentation using K-Means Clustering
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.markdown("### Prediction Mode")
mode = st.sidebar.radio(
    "Choose how you want to provide data:",
    ["Manual Entry", "Upload CSV"]
)

# -------------------------------------------------
# Manual Input Mode
# -------------------------------------------------
if mode == "Manual Entry":

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Manual Customer Input")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 18, 100, 30)
        total_products = st.number_input("Total Products Purchased", 1, value=5)
        income = st.number_input("Annual Income", 1000, value=50000)

    with col2:
        total_purchases = st.number_input("Total Purchases", 1, value=10)
        recency = st.number_input("Recency (Days Since Last Purchase)", 0, value=30)

    input_df = pd.DataFrame([{
        "Age": age,
        "Total_Products": total_products,
        "Total_Purchases": total_purchases,
        "Income": income,
        "Recency": recency
    }])

    if st.button("Predict Customer Segment"):
        cluster = pipeline.predict(input_df)[0]
        segment = cluster_labels[cluster]

        st.success(f"Predicted Segment: **{segment}**")

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# CSV Upload Mode
# -------------------------------------------------
else:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Batch Prediction via CSV")

    st.markdown(
        "Required columns:\n"
        "`Age`, `Total_Products`, `Total_Purchases`, `Income`, `Recency`"
    )

    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file:
        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)

        required_cols = [
            "Age", "Total_Products", "Total_Purchases", "Income", "Recency"
        ]

        if not all(col in df.columns for col in required_cols):
            st.error("Uploaded file does not contain all required columns.")
        else:
            df_required = df[required_cols].copy()
            df_required = df_required.fillna(df_required.median())

            predictions = pipeline.predict(df_required)

            final_df = df_required.copy()
            final_df["Customer_Segment"] = [
                cluster_labels[p] for p in predictions
            ]

            st.success("Segmentation completed successfully.")
            st.dataframe(final_df)

            csv = final_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Segmented CSV",
                csv,
                "segmented_customers.csv",
                "text/csv"
            )

    st.markdown('</div>', unsafe_allow_html=True)
