import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import io

# =============== PAGE CONFIG ==================
st.set_page_config(page_title="KRA Progress Predictor", layout="wide")

# =============== NAVIGATION ===================
if "page" not in st.session_state:
    st.session_state.page = "Home"

menu = st.columns([1,1,1])
with menu[0]:
    if st.button("🏠 Home"):
        st.session_state.page = "Home"
with menu[1]:
    if st.button("📊 Forecasts"):
        st.session_state.page = "Forecasts"
with menu[2]:
    if st.button("📄 Reports"):
        st.session_state.page = "Reports"

st.markdown("---")

# =============== SIDEBAR ======================
st.sidebar.header("⚙️ Dashboard Settings")
report_title = st.sidebar.text_input("📑 Custom Report Title", value="Kenya Revenue Authority Forecast Report")
logo_file = st.sidebar.file_uploader("🏢 Upload KRA Logo (PNG/JPG)", type=["png", "jpg", "jpeg"])
years = st.sidebar.slider("⏳ Years to predict", 1, 10, 5)
model_choice = st.sidebar.radio("🔮 Model", ["Prophet", "ARIMA", "Holt-Winters"])
conf_interval = st.sidebar.slider("🎯 Confidence Interval (%)", 80, 99, 95)
uploaded_file = st.sidebar.file_uploader("📂 Upload Dataset (CSV)", type=["csv"])

# =============== DATASET ======================
METRIC_OPTIONS = {
    "Total Revenue (KES Billion)": "Total_Revenue_KES_Billion",
    "Tax Compliance Rate (%)": "Tax_Compliance_Rate_%",
    "Corporate Tax (KES Billion)": "Corporate_Tax_KES_Billion",
    "VAT Collection (KES Billion)": "VAT_Collection_KES_Billion",
    "Customs Duties (KES Billion)": "Customs_Duties_KES_Billion",
    "Taxpayer Growth (%)": "Taxpayer_Growth_%",
    "Digital Services Tax (KES Billion)": "Digital_Services_Tax_KES_Billion"
}
YEAR_COLUMN = "Year"

# =============== PAGE LOGIC ===================
if st.session_state.page == "Home":
    # Header
    col1, col2 = st.columns([1,6])
    with col1:
        if logo_file:
            logo_img = PILImage.open(logo_file)
            max_width = 100
            aspect_ratio = logo_img.height / logo_img.width
            new_height = int(max_width * aspect_ratio)
            st.image(logo_img.resize((max_width, new_height)))
    with col2:
        st.markdown(f"<h1 style='color:#b22222;'>📊 {report_title}</h1>", unsafe_allow_html=True)
        st.markdown("### Kenya Revenue Authority – Tax & Revenue Forecast Dashboard")

    st.markdown(
        """
        Welcome to the **KRA AI-Powered Forecasting Dashboard** 🇰🇪  

        🔹 Upload your official revenue dataset (CSV)  
        🔹 Choose forecasting models (Prophet, ARIMA, Holt-Winters)  
        🔹 Visualize predictions & accuracy metrics  
        🔹 Export custom reports in PDF format  
        """
    )

elif st.session_state.page == "Forecasts":
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.markdown("### 🔍 Preview of Uploaded Data")
        st.dataframe(data.head())

        category = st.multiselect(
            "📈 Choose metric(s) to forecast",
            options=list(METRIC_OPTIONS.keys()),
            default=["Total Revenue (KES Billion)"]
        )

        for metric_display in category:
            metric = METRIC_OPTIONS[metric_display]
            if YEAR_COLUMN not in data.columns or metric not in data.columns:
                st.error(f"Missing column(s): {YEAR_COLUMN} / {metric}")
                continue

            st.markdown(f"## 📉 Forecast for **{metric_display}**")

            # Dummy forecast plot (replace with real forecast function)
            fig, ax = plt.subplots(figsize=(8,4))
            ax.plot(data[YEAR_COLUMN], data[metric], label="History", marker="o")
            ax.plot(data[YEAR_COLUMN], data[metric]*1.05, label="Forecast", color="green")
            ax.legend()
            st.pyplot(fig)

            # Dummy metrics
            colA, colB = st.columns(2)
            colA.metric("📌 Mean Absolute Error (MAE)", f"{np.random.randint(10,50)}")
            colB.metric("📌 Root Mean Squared Error (RMSE)", f"{np.random.randint(20,70)}")
    else:
        st.warning("⬆️ Please upload a dataset first in the sidebar.")

elif st.session_state.page == "Reports":
    st.markdown("### 📄 Generate & Export Custom Reports")
    st.info("Once forecasts are generated, you will be able to **export PDF reports** with charts and metrics here.")
    if st.button("⬇️ Download Sample Report"):
        buf = io.BytesIO()
        buf.write(b"Sample Report for KRA")  # dummy content for now
        st.download_button("Download Report", data=buf, file_name="KRA_Report.pdf", mime="application/pdf")
