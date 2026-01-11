import os
import joblib
import streamlit as st
import numpy as np

# ===== PATH =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "churn_model.pkl")

model = joblib.load(MODEL_PATH)

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="centered"
)

# ===== CUSTOM CSS =====
st.markdown("""
<style>
body {
    background-color: #f5f7fa;
}
.card {
    background-color: white;
    padding: 30px;
    border-radius: 15px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.1);
}
.title {
    text-align: center;
    font-size: 34px;
    font-weight: 700;
    color: #1f2937;
}
.subtitle {
    text-align: center;
    color: #6b7280;
    margin-bottom: 25px;
}
.result-churn {
    background: #fee2e2;
    color: #991b1b;
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
}
.result-safe {
    background: #dcfce7;
    color: #065f46;
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
}
.footer {
    text-align: center;
    color: #6b7280;
    margin-top: 25px;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ===== UI START =====
st.markdown('<div class="card">', unsafe_allow_html=True)

st.markdown('<div class="title">📉 Customer Churn Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Predict whether a customer will leave the service</div>',
    unsafe_allow_html=True
)

st.write("### 🧾 Customer Details")

tenure = st.number_input("Tenure (in months)", min_value=0, value=12)
monthly = st.number_input("Monthly Charges (₹)", min_value=0.0, value=50.0)
total = st.number_input("Total Charges (₹)", min_value=0.0, value=600.0)

if st.button("🔍 Predict Churn"):
    features = np.array([[tenure, monthly, total]])
    churn_prob = model.predict_proba(features)[0][1]

    if churn_prob >= 0.5:
        st.markdown(
            f'<div class="result-churn">🚨 CUSTOMER WILL CHURN<br>Probability: {churn_prob:.2f}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="result-safe">✅ CUSTOMER WILL STAY<br>Probability: {churn_prob:.2f}</div>',
            unsafe_allow_html=True
        )

st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="footer">Built using Machine Learning & Streamlit</div>',
    unsafe_allow_html=True
)
