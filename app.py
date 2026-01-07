import streamlit as st
import pickle
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Page config (makes it look nicer)
# =========================
st.set_page_config(
    page_title="Amazon Sentiment Analyzer",
    page_icon="🛒",
    layout="wide"
)

# =========================
# Minimal styling (clean + modern)
# =========================
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.5rem; }
      .big-title { font-size: 2.1rem; font-weight: 700; }
      .subtle { color: #666; }
      .card { padding: 1rem; border-radius: 12px; border: 1px solid #eee; background: #fafafa; }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Load trained artifacts
# =========================
@st.cache_resource
def load_model_artifacts():
    with open("tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("svm_model.pkl", "rb") as f:
        model = pickle.load(f)
    return tfidf, model

tfidf, model = load_model_artifacts()

# =========================
# Load evaluation artifacts (optional but recommended)
# =========================
@st.cache_resource
def load_eval_artifacts():
    try:
        with open("eval_artifacts.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

eval_artifacts = load_eval_artifacts()

# =========================
# Text preprocessing (must match training)
# =========================
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# =========================
# Helper: plot confusion matrix
# =========================
def plot_confusion_matrix(cm: np.ndarray, labels: list[str]):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([l.title() for l in labels], rotation=30, ha="right")
    ax.set_yticklabels([l.title() for l in labels])

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Linear SVM)")
    fig.tight_layout()
    return fig

# =========================
# Helper: plot per-class F1
# =========================
def plot_f1_bars(report_dict: dict, labels: list[str]):
    f1_scores = []
    for lab in labels:
        f1_scores.append(report_dict.get(lab, {}).get("f1-score", 0.0) * 100)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([l.title() for l in labels], f1_scores)
    ax.set_ylim(0, 100)
    ax.set_ylabel("F1-score (%)")
    ax.set_title("Per-Class F1-score (Linear SVM)")
    fig.tight_layout()
    return fig

# =========================
# UI
# =========================
st.markdown('<div class="big-title">🛒 Amazon Review Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Linear SVM + TF-IDF | Predicts: Positive / Neutral / Negative</div>', unsafe_allow_html=True)
st.write("")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    show_cleaned = st.checkbox("Show cleaned text", value=False)
    st.caption("Tip: Keep your input as a normal review sentence.")

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Model Performance", "ℹ️ About"])

# -------------------------
# TAB 1: Prediction
# -------------------------
with tab1:
    colA, colB = st.columns([2, 1], gap="large")

    with colA:
        st.subheader("Enter a review")
        user_text = st.text_area("Paste an Amazon review text:", height=180)

        if st.button("Predict Sentiment ✅"):
            if not user_text.strip():
                st.warning("Please enter a review.")
            else:
                cleaned = clean_text(user_text)
                X = tfidf.transform([cleaned])
                pred = model.predict(X)[0]

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.success(f"Predicted Sentiment: **{pred.upper()}**")
                if show_cleaned:
                    st.write("Cleaned text:")
                    st.code(cleaned)
                st.markdown("</div>", unsafe_allow_html=True)

    with colB:
        st.subheader("Quick Examples")
        st.caption("Click to copy, then press Predict.")
        examples = [
            "This product is amazing and works perfectly. Highly recommended!",
            "It is okay. Not great, not bad. Just average.",
            "Very disappointed. Broke after two days and waste of money."
        ]
        for ex in examples:
            st.code(ex)

# -------------------------
# TAB 2: Performance Dashboard
# -------------------------
with tab2:
    st.subheader("Model Performance (Evaluation Results)")

    if eval_artifacts is None:
        st.warning(
            "eval_artifacts.pkl not found. "
            "To show accuracy + graphs, generate eval_artifacts.pkl in Colab and upload it with the app."
        )
    else:
        acc = eval_artifacts["accuracy_percent"]
        labels = eval_artifacts["labels_order"]
        report = eval_artifacts["classification_report"]
        cm = np.array(eval_artifacts["confusion_matrix"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Model", "Linear SVM")
        c2.metric("Accuracy", f"{acc:.2f}%")
        c3.metric("Classes", ", ".join([l.title() for l in labels]))

        st.write("")

        left, right = st.columns([1, 1], gap="large")

        with left:
            fig_cm = plot_confusion_matrix(cm, labels)
            st.pyplot(fig_cm)

        with right:
            fig_f1 = plot_f1_bars(report, labels)
            st.pyplot(fig_f1)

        st.write("")
        st.markdown("### Classification Report (Percentages)")

        # Convert report dict to a clean table
        rows = []
        for lab in labels:
            rows.append({
                "Class": lab.title(),
                "Precision (%)": round(report[lab]["precision"] * 100, 2),
                "Recall (%)": round(report[lab]["recall"] * 100, 2),
                "F1-score (%)": round(report[lab]["f1-score"] * 100, 2),
                "Support": int(report[lab]["support"])
            })
        report_df = pd.DataFrame(rows)
        st.dataframe(report_df, use_container_width=True)

# -------------------------
# TAB 3: About
# -------------------------
with tab3:
    st.subheader("About this Application")
    st.write(
        """
        **Objective:** Classify Amazon product reviews into **Positive**, **Neutral**, or **Negative** sentiment.

        **Pipeline:**
        1) Text cleaning  
        2) TF-IDF feature extraction  
        3) Linear SVM classification  

        **Deployment:** Streamlit web app using serialized model artifacts (`svm_model.pkl`) and vectorizer (`tfidf.pkl`).
        """
    )
