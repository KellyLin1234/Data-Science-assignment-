import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Gold Price Prediction", layout="wide")

st.title("💰 Gold Price Prediction System")
st.write("Compare Machine Learning Models for Gold Price Forecasting")

# ----------------------------
# Load models
# ----------------------------
@st.cache_resource
def load_models():
    rf = joblib.load("models/rf.pkl")
    xgb = joblib.load("models/xgb.pkl")
    gb = joblib.load("models/gb.pkl")
    hybrid = joblib.load("models/hybrid.pkl")
    return rf, xgb, gb, hybrid

rf_model, xgb_model, gb_model, hybrid_model = load_models()

# ----------------------------
# Sidebar
# ----------------------------
model_choice = st.sidebar.selectbox(
    "Select Model",
    ("Random Forest", "XGBoost", "Gradient Boosting", "Hybrid (RF + LSTM)")
)

st.sidebar.write("Enter input values:")

open_price = st.sidebar.number_input("Open Price", value=2000.0)
high_price = st.sidebar.number_input("High Price", value=2050.0)
low_price = st.sidebar.number_input("Low Price", value=1980.0)
volume = st.sidebar.number_input("Volume", value=1000000.0)
chg = st.sidebar.number_input("Change %", value=0.5)

# ----------------------------
# Prediction
# ----------------------------
if st.sidebar.button("Predict Price"):

    input_data = np.array([[open_price, high_price, low_price, volume, chg]])

    if model_choice == "Random Forest":
        prediction = rf_model.predict(input_data)[0]

    elif model_choice == "XGBoost":
        prediction = xgb_model.predict(input_data)[0]

    elif model_choice == "Gradient Boosting":
        prediction = gb_model.predict(input_data)[0]

    elif model_choice == "Hybrid (RF + LSTM)":
        prediction = hybrid_model.predict(input_data)[0]

    st.subheader("📊 Predicted Gold Price")
    st.success(f"${prediction:.2f}")

# ----------------------------
# Model Info Section
# ----------------------------
st.markdown("---")
st.subheader("📈 Model Information")

st.write("""
This system compares multiple machine learning models:

- Random Forest  
- XGBoost  
- Gradient Boosting  
- Hybrid Model (RF + LSTM feature-based)

Evaluation metrics used in report:
- RMSE
- MAE
- R²
- MAPE
""")

# ----------------------------
# Visualization Section
# ----------------------------
st.markdown("---")
st.subheader("📉 Sample Prediction Visualization")

if st.button("Show Sample Chart"):
    x = np.arange(0, 100)
    y_actual = np.sin(x / 10) * 100 + 2000
    y_pred = y_actual + np.random.normal(0, 20, 100)

    fig, ax = plt.subplots()
    ax.plot(y_actual, label="Actual Price")
    ax.plot(y_pred, label="Predicted Price")
    ax.legend()

    st.pyplot(fig)
