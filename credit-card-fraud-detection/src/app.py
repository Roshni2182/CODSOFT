import os
import joblib
import streamlit as st
import numpy as np

# ===== PATH SETUP =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "fraud_model.pkl")

# Load model
model = joblib.load(MODEL_PATH)

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)

# ===== CUSTOM CSS =====
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #141E30, #243B55);
}

.card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
}

.title {
    text-align: center;
    font-size: 38px;
    font-weight: 700;
    color: white;
}

.subtitle {
    text-align: center;
    font-size: 15px;
    color: #d1d5db;
    margin-bottom: 25px;
}

.stButton > button {
    width: 100%;
    border-radius: 12px;
    padding: 0.6em;
    font-size: 18px;
    font-weight: 600;
    background: linear-gradient(90deg, #ff416c, #ff4b2b);
    color: white;
    border: none;
}

.result-fraud {
    margin-top: 20px;
    padding: 18px;
    border-radius: 15px;
    background: #ff4b2b;
    color: white;
    text-align: center;
    font-size: 22px;
    font-weight: 700;
}

.result-safe {
    margin-top: 20px;
    padding: 18px;
    border-radius: 15px;
    background: #22c55e;
    color: black;
    text-align: center;
    font-size: 22px;
    font-weight: 700;
}

.footer {
    text-align: center;
    color: #9ca3af;
    font-size: 13px;
    margin-top: 25px;
}
</style>
""", unsafe_allow_html=True)

# ===== UI =====
st.markdown('<div class="card">', unsafe_allow_html=True)

st.markdown('<div class="title">💳 Credit Card Fraud Detection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Machine Learning based Transaction Classification</div>',
    unsafe_allow_html=True
)

st.write("### 🔢 Enter Transaction Details")

# IMPORTANT: SAME ORDER AS TRAINING
time = st.number_input("Transaction Time", value=0.0)
v1 = st.number_input("V1", value=0.0)
v2 = st.number_input("V2", value=0.0)
v3 = st.number_input("V3", value=0.0)
v4 = st.number_input("V4", value=0.0)
amount = st.number_input("Transaction Amount", value=100.0)

if st.button("🔍 Check Transaction"):
    features = np.array([[time, v1, v2, v3, v4, amount]])

    prediction = model.predict(features)[0]

    if prediction == 1:
        st.markdown(
            '<div class="result-fraud">🚨 FRAUDULENT TRANSACTION</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="result-safe">✅ LEGITIMATE TRANSACTION</div>',
            unsafe_allow_html=True
        )

st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="footer">Built with ❤️ using Machine Learning & Streamlit</div>',
    unsafe_allow_html=True
)
