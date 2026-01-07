import streamlit as st
import pickle
import re

# -------------------------
# Load trained artifacts
# -------------------------
@st.cache_resource
def load_artifacts():
    with open("tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("svm_model.pkl", "rb") as f:
        model = pickle.load(f)
    return tfidf, model

tfidf, model = load_artifacts()

# -------------------------
# Text preprocessing
# -------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -------------------------
# Streamlit UI
# -------------------------
st.title("Amazon Review Sentiment Analyzer")
st.write(
    "This application predicts **sentiment polarity** "
    "(Positive / Neutral / Negative) using a **Linear SVM model**."
)

review_text = st.text_area("Enter an Amazon product review:", height=150)

if st.button("Predict Sentiment"):
    if not review_text.strip():
        st.warning("Please enter a review.")
    else:
        cleaned = clean_text(review_text)
        X = tfidf.transform([cleaned])
        prediction = model.predict(X)[0]
        st.success(f"Predicted Sentiment: **{prediction.upper()}**")

