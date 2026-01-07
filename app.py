import streamlit as st
import pickle
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Amazon Review Sentiment Analyzer",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# STYLE (LinkedIn-friendly)
# -----------------------------
st.markdown("""
<style>
/* Global */
.block-container { padding-top: 1.2rem; max-width: 1200px; }
h1, h2, h3 { letter-spacing: -0.02em; }
.small-muted { color: rgba(255,255,255,0.65); font-size: 0.95rem; }
.hr { height: 1px; background: rgba(255,255,255,0.08); margin: 0.8rem 0 1.2rem 0; }

/* Cards */
.card {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px;
  padding: 16px 18px;
}
.card-title { font-weight: 650; font-size: 1.05rem; margin-bottom: 6px; }
.card-sub { color: rgba(255,255,255,0.7); font-size: 0.92rem; }

/* Pill badges */
.pill {
  display: inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  font-weight: 700;
  font-size: 0.88rem;
  letter-spacing: 0.02em;
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.06);
}
.pill-pos { background: rgba(34,197,94,0.16); border-color: rgba(34,197,94,0.35); }
.pill-neu { background: rgba(59,130,246,0.16); border-color: rgba(59,130,246,0.35); }
.pill-neg { background: rgba(239,68,68,0.16); border-color: rgba(239,68,68,0.35); }

/* Buttons */
.stButton > button {
  border-radius: 12px !important;
  padding: 0.55rem 0.9rem !important;
  font-weight: 650 !important;
}

/* Text area */
textarea {
  border-radius: 14px !important;
}

/* Hide Streamlit footer/menu for clean screenshots */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD ARTIFACTS
# -----------------------------
@st.cache_resource
def load_model_artifacts():
    with open("tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("svm_model.pkl", "rb") as f:
        model = pickle.load(f)
    return tfidf, model

tfidf, model = load_model_artifacts()

@st.cache_resource
def load_eval_artifacts():
    try:
        with open("eval_artifacts.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

eval_artifacts = load_eval_artifacts()

# -----------------------------
# PREPROCESSING (must match training)
# -----------------------------
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def sentiment_pill(label: str) -> str:
    label_l = label.lower()
    if label_l == "positive":
        return '<span class="pill pill-pos">POSITIVE</span>'
    if label_l == "neutral":
        return '<span class="pill pill-neu">NEUTRAL</span>'
    return '<span class="pill pill-neg">NEGATIVE</span>'

# -----------------------------
# PLOTS
# -----------------------------
def plot_confusion_matrix(cm: np.ndarray, labels: list[str]):
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.imshow(cm)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([l.title() for l in labels], rotation=25, ha="right")
    ax.set_yticklabels([l.title() for l in labels])

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Linear SVM)")
    fig.tight_layout()
    return fig

def plot_f1_bars(report_dict: dict, labels: list[str]):
    f1_scores = [(report_dict.get(l, {}).get("f1-score", 0.0) * 100) for l in labels]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([l.title() for l in labels], f1_scores)
    ax.set_ylim(0, 100)
    ax.set_ylabel("F1-score (%)")
    ax.set_title("Per-class F1-score (Linear SVM)")
    fig.tight_layout()
    return fig

# -----------------------------
# HERO HEADER
# -----------------------------
left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown("## 🛒 Amazon Review Sentiment Analyzer")
    st.markdown(
        "<div class='small-muted'>Linear SVM + TF-IDF • Predicts: Positive / Neutral / Negative</div>",
        unsafe_allow_html=True
    )
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>Model Snapshot</div>", unsafe_allow_html=True)

    if eval_artifacts is None:
        st.markdown("<div class='card-sub'>Upload <b>eval_artifacts.pkl</b> to show accuracy + charts.</div>",
                    unsafe_allow_html=True)
    else:
        acc = eval_artifacts["accuracy_percent"]
        st.metric("Accuracy", f"{acc:.2f}%")
        st.markdown("<div class='card-sub'>Best model selected for deployment: <b>Linear SVM</b></div>",
                    unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# TABS
# -----------------------------
tab_pred, tab_perf, tab_about = st.tabs(["🔮 Predict", "📊 Performance", "ℹ️ About"])

# =============================
# TAB 1: PREDICT
# =============================
with tab_pred:
    colA, colB = st.columns([2.2, 1], gap="large")

    examples = {
        "Positive example": "This product is amazing and works perfectly. Highly recommended!",
        "Neutral example": "It is okay. Not great, not bad. Just average.",
        "Negative example": "Very disappointed. Broke after two days and wasted my money."
    }

    # State init
    if "review_text" not in st.session_state:
        st.session_state.review_text = ""

    with colA:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Enter a review</div>", unsafe_allow_html=True)
        st.markdown("<div class='card-sub'>Paste an Amazon review and click predict.</div>", unsafe_allow_html=True)

        st.session_state.review_text = st.text_area(
            label="",
            value=st.session_state.review_text,
            height=170,
            placeholder="Example: Great quality, fast shipping, and worth the price..."
        )

        c1, c2, c3 = st.columns([1, 1, 1])
        show_cleaned = c1.checkbox("Show cleaned text", value=False)

        predict_clicked = c2.button("Predict ✅", use_container_width=True)
        clear_clicked = c3.button("Clear", use_container_width=True)

        if clear_clicked:
            st.session_state.review_text = ""
            st.rerun()

        if predict_clicked:
            if not st.session_state.review_text.strip():
                st.warning("Please enter a review.")
            else:
                cleaned = clean_text(st.session_state.review_text)
                X = tfidf.transform([cleaned])
                pred = model.predict(X)[0]

                st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
                st.markdown("<div class='card-title'>Prediction</div>", unsafe_allow_html=True)
                st.markdown(f"{sentiment_pill(pred)}", unsafe_allow_html=True)

                if show_cleaned:
                    st.markdown("<div class='card-sub' style='margin-top:10px;'>Cleaned text:</div>",
                                unsafe_allow_html=True)
                    st.code(cleaned)

        st.markdown("</div>", unsafe_allow_html=True)

    with colB:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Try an example</div>", unsafe_allow_html=True)
        st.markdown("<div class='card-sub'>Click a button to auto-fill the input box.</div>", unsafe_allow_html=True)

        for name, text in examples.items():
            if st.button(name, use_container_width=True):
                st.session_state.review_text = text
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

# =============================
# TAB 2: PERFORMANCE
# =============================
with tab_perf:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>Evaluation Results</div>", unsafe_allow_html=True)
    st.markdown("<div class='card-sub'>Accuracy, confusion matrix, and per-class F1-score.</div>",
                unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.write("")

    if eval_artifacts is None:
        st.info(
            "To show charts here, generate **eval_artifacts.pkl** in Colab and upload it to your repo "
            "along with the other files."
        )
    else:
        labels = eval_artifacts["labels_order"]
        report = eval_artifacts["classification_report"]
        cm = np.array(eval_artifacts["confusion_matrix"])
        acc = eval_artifacts["accuracy_percent"]

        m1, m2, m3 = st.columns(3)
        m1.metric("Model", "Linear SVM")
        m2.metric("Accuracy", f"{acc:.2f}%")
        m3.metric("Classes", " / ".join([l.title() for l in labels]))

        leftp, rightp = st.columns([1, 1], gap="large")
        with leftp:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.pyplot(plot_confusion_matrix(cm, labels))
            st.markdown("</div>", unsafe_allow_html=True)
        with rightp:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.pyplot(plot_f1_bars(report, labels))
            st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        st.markdown("### Classification Report (Percentages)")

        rows = []
        for lab in labels:
            rows.append({
                "Class": lab.title(),
                "Precision (%)": round(report[lab]["precision"] * 100, 2),
                "Recall (%)": round(report[lab]["recall"] * 100, 2),
                "F1-score (%)": round(report[lab]["f1-score"] * 100, 2),
                "Support": int(report[lab]["support"])
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

# =============================
# TAB 3: ABOUT
# =============================
with tab_about:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>About this Project</div>", unsafe_allow_html=True)
    st.write(
        """
        **Objective:** Classify Amazon product reviews into **Positive**, **Neutral**, or **Negative**.

        **Pipeline:**
        1) Text cleaning  
        2) TF-IDF feature extraction  
        3) Linear SVM classification  

        **Deployment:** Streamlit web application using serialized artifacts:
        `tfidf.pkl` and `svm_model.pkl` (and optional `eval_artifacts.pkl` for metrics & charts).
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)
