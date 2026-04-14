import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Gold Price Prediction (RF Only)", layout="wide")

st.title("💰 Gold Price Prediction System")
st.write("Random Forest Model Only")

# ----------------------------
# Load model
# ----------------------------
@st.cache_resource
def load_model():
    rf = joblib.load("models/rf.pkl")
    return rf

rf_model = load_model()

# ----------------------------
# Sidebar inputs
# ----------------------------
st.sidebar.header("Input Features")

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
    prediction = rf_model.predict(input_data)[0]

    st.subheader("📊 Predicted Gold Price")
    st.success(f"${prediction:.2f}")

# ----------------------------
# Info section
# ----------------------------
st.markdown("---")
st.subheader("📌 Model Info")

st.write("""
This app uses a **Random Forest Regressor** trained on historical gold price data.

Input features:
- Open
- High
- Low
- Volume
- Change %

Output:
- Predicted Gold Price
""")
