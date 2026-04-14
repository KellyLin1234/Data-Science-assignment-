import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

st.set_page_config(page_title="Gold Price Prediction", layout="wide")

st.title("💰 Gold Price Prediction System")
st.write("Compare Machine Learning & Deep Learning Models")

# ----------------------------
# Load models
# ----------------------------
@st.cache_resource
def load_models():
    rf = joblib.load("models/rf.pkl")
    xgb = joblib.load("models/xgb.pkl")
    gb = joblib.load("models/gb.pkl")
    hybrid = joblib.load("models/hybrid.pkl")
    lstm = load_model("models/lstm.h5")
    return rf, xgb, gb, hybrid, lstm

rf_model, xgb_model, gb_model, hybrid_model, lstm_model = load_models()

# ----------------------------
# Sidebar - model selection
# ----------------------------
model_choice = st.sidebar.selectbox(
    "Select Model",
    ("Random Forest", "XGBoost", "Gradient Boosting", "LSTM", "Hybrid (RF + LSTM)")
)

st.sidebar.write("Enter input values:")

# ----------------------------
# Input fields
# ----------------------------
open_price = st.sidebar.number_input("Open Price", value=2000.0)
high_price = st.sidebar.number_input("High Price", value=2050.0)
low_price = st.sidebar.number_input("Low Price", value=1980.0)
volume = st.sidebar.number_input("Volume", value=1000000.0)
chg = st.sidebar.number_input("Change %", value=0.5)

# For LSTM (sequence placeholder)
st.sidebar.write("⚠ LSTM uses last 60 days (demo uses static input)")

# ----------------------------
# Prediction button
# ----------------------------
if st.sidebar.button("Predict Price"):

    if model_choice == "Random Forest":
        input_data = np.array([[open_price, high_price, low_price, volume, chg]])
        prediction = rf_model.predict(input_data)[0]

    elif model_choice == "XGBoost":
        input_data = np.array([[open_price, high_price, low_price, volume, chg]])
        prediction = xgb_model.predict(input_data)[0]

    elif model_choice == "Gradient Boosting":
        input_data = np.array([[open_price, high_price, low_price, volume, chg]])
        prediction = gb_model.predict(input_data)[0]

    elif model_choice == "Hybrid (RF + LSTM)":
        input_data = np.array([[open_price, high_price, low_price, volume, chg]])
        prediction = hybrid_model.predict(input_data)[0]

    elif model_choice == "LSTM":
        st.warning("LSTM requires last 60-day sequence. Showing demo prediction.")

        dummy = np.random.random((1, 60, 1))
        prediction = lstm_model.predict(dummy)[0][0]

    st.subheader("📊 Predicted Gold Price")
    st.success(f"${prediction:.2f}")

# ----------------------------
# Model comparison section
# ----------------------------
st.markdown("---")
st.subheader("📈 Model Comparison Info")

st.write("""
This system compares:
- Random Forest (Machine Learning)
- XGBoost / Gradient Boosting
- LSTM (Deep Learning)
- Hybrid Model (RF + LSTM)

Evaluation metrics used:
- RMSE
- MAE
- R²
- MAPE
""")

# ----------------------------
# Simple visualization demo
# ----------------------------
st.markdown("---")
st.subheader("📉 Demo Visualization")

if st.button("Show Sample Chart"):
    x = np.arange(0, 100)
    y_actual = np.sin(x / 10) * 100 + 2000
    y_pred = y_actual + np.random.normal(0, 20, 100)

    fig, ax = plt.subplots()
    ax.plot(y_actual, label="Actual Price")
    ax.plot(y_pred, label="Predicted Price")
    ax.legend()

    st.pyplot(fig)
