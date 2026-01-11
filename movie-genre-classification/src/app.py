import os
import joblib
import streamlit as st

# ===== PATH SETUP =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "genre_model.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "model", "tfidf.pkl")

# Load model
model = joblib.load(MODEL_PATH)
tfidf = joblib.load(TFIDF_PATH)

# ===== STREAMLIT UI =====
st.set_page_config(page_title="Movie Genre Classifier", layout="centered")

st.title("🎬 Movie Genre Classification")
st.write("Enter a movie plot summary and predict its genre")

plot_text = st.text_area(
    "✍️ Movie Plot Summary",
    height=150,
    placeholder="A hero fights criminals to save the city..."
)

if st.button("🎯 Predict Genre"):
    if plot_text.strip() == "":
        st.warning("Please enter a movie plot!")
    else:
        plot_tfidf = tfidf.transform([plot_text])
        prediction = model.predict(plot_tfidf)[0]
        st.success(f"🎉 Predicted Genre: **{prediction}**")
