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
    page_title="Sentiment Analyzer",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# LOAD ARTIFACTS
# -----------------------------
@st.cache_resource
def load_artifacts():
    with open("tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("svm_model.pkl", "rb") as f:
        model = pickle.load(f)
    return tfidf, model

tfidf, model = load_artifacts()

@st.cache_resource
def load_eval_artifacts():
    try:
        with open("eval_artifacts.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

eval_artifacts = load_eval_artifacts()

# -----------------------------
# UTILS & PREPROCESSING
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def plot_confusion_matrix(cm: np.ndarray, labels: list[str]):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    
    # Transparent background for the figure
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    im = ax.imshow(cm, cmap='PuBuGn') 
    
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([l.title() for l in labels], rotation=25, ha="right", color="white")
    ax.set_yticklabels([l.title() for l in labels], color="white")
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white", fontweight='bold')

    ax.set_xlabel("Predicted", color="#b0b0b0")
    ax.set_ylabel("Actual", color="#b0b0b0")
    ax.set_title("Confusion Matrix", color="white", fontweight='bold', pad=20)
    
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    fig.tight_layout()
    return fig

def plot_f1_bars(report_dict: dict, labels: list[str]):
    plt.style.use('dark_background')
    f1_scores = [(report_dict.get(l, {}).get("f1-score", 0.0) * 100) for l in labels]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    
    bars = ax.bar([l.title() for l in labels], f1_scores, color=['#00f260', '#0575E6', '#e100ff'])
    
    ax.set_ylim(0, 110)
    ax.set_ylabel("F1-score (%)", color="#b0b0b0")
    ax.set_title("Per-class F1-score", color="white", fontweight='bold', pad=20)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{height:.1f}%',
                ha='center', va='bottom', color="white", fontweight="bold")
    
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    fig.tight_layout()
    return fig

# -----------------------------
# CSS STYLING
# -----------------------------
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    /* Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        background-attachment: fixed;
    }

    /* Input Fields (Text Area) - acting as "Cards" */
    .stTextArea textarea {
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    .stTextArea textarea:focus {
        border-color: #0575E6 !important;
        box-shadow: 0 0 15px rgba(5, 117, 230, 0.2);
        background-color: rgba(255, 255, 255, 0.08) !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(92deg, #00f260 0%, #0575E6 100%);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 242, 96, 0.2);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(5, 117, 230, 0.4);
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: rgba(255,255,255,0.03);
        border-radius: 8px;
        color: #b0b0b0;
        border: 1px solid rgba(255,255,255,0.05);
        padding: 0 20px;
        transition: all 0.2s;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255,255,255,0.1) !important;
        color: #fff !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
    }

    /* Custom Classes for pure HTML injections */
    .header-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        margin-bottom: 30px;
        backdrop-filter: blur(5px);
    }
    
    .header-title {
        background: linear-gradient(to right, #00f260, #0575E6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        padding-bottom: 10px;
    }
    
    .header-subtitle {
        color: #b0b0b0;
        font-size: 1.1rem;
        margin: 0;
    }

    .result-box {
        padding: 25px;
        border-radius: 16px;
        text-align: center;
        margin-top: 25px;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        animation: slideUp 0.5s cubic-bezier(0.16, 1, 0.3, 1);
    }
    
    .res-pos { background: linear-gradient(135deg, #11998e, #38ef7d); }
    .res-neg { background: linear-gradient(135deg, #cb2d3e, #ef473a); }
    .res-neu { background: linear-gradient(135deg, #2b5876, #4e4376); }

    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        color: #00f260;
    }
    div[data-testid="stMetricLabel"] {
        color: #b0b0b0;
    }

</style>
""", unsafe_allow_html=True)

# -----------------------------
# UI CONTROLLER
# -----------------------------

# --- Header Section (Pure HTML Card) ---
st.markdown("""
<div class="header-card">
    <h1 class="header-title">✨ Sentiment Analyzer</h1>
    <p class="header-subtitle">
        Powered by Linear SVM & TF-IDF <br>
        <span style="font-size: 0.9rem; opacity: 0.7;">Detects Positive, Neutral, and Negative sentiments instantly</span>
    </p>
</div>
""", unsafe_allow_html=True)

# Tab Navigation
tab_pred, tab_perf, tab_about = st.tabs(["🔮 Predict", "📊 Performance", "ℹ️ About"])

# --- TAB 1: PREDICT ---
with tab_pred:
    # No "main-card" wrapper here. Widgets float naturally.
    
    # 1. Quick Fill Buttons
    st.markdown("##### Try an example:")
    c1, c2, c3 = st.columns(3)
    if c1.button("Happy �"):
        st.session_state.review_text = "I absolutely love this product! It works wonders and arrived fast."
        st.rerun()
    if c2.button("Neutral 😐"):
        st.session_state.review_text = "The product is okay. It does what it says but quality could be better."
        st.rerun()
    if c3.button("Angry 😡"):
        st.session_state.review_text = "Worst purchase ever. Broken on arrival and terrible customer service."
        st.rerun()

    # 2. Input Area
    if "review_text" not in st.session_state:
        st.session_state.review_text = ""

    review_text = st.text_area(
        "Enter Review", 
        value=st.session_state.review_text,
        height=180, 
        placeholder="Type something here to analyze...",
        label_visibility="collapsed"
    )

    # 3. Analyze Button
    # Centered button
    col_l, col_btn, col_r = st.columns([1, 2, 1])
    with col_btn:
        analyze = st.button("🚀 Analyze Sentiment", use_container_width=True)

    # 4. Results
    if analyze:
        if not review_text.strip():
            st.warning("⚠️ Please enter some text first.")
        else:
            cleaned = clean_text(review_text)
            X = tfidf.transform([cleaned])
            prediction = model.predict(X)[0]
            
            # Display configuration
            if prediction.lower() == 'positive':
                cls, icon, title, desc = "res-pos", "🎉", "POSITIVE", "This review looks great!"
            elif prediction.lower() == 'negative':
                cls, icon, title, desc = "res-neg", "💀", "NEGATIVE", "This review seems critical."
            else:
                cls, icon, title, desc = "res-neu", "🤔", "NEUTRAL", "This review is balanced."

            st.markdown(f"""
            <div class="result-box {cls}">
                <div style="font-size: 3.5rem; margin-bottom: 10px;">{icon}</div>
                <h2 style="margin: 0; font-weight: 800; letter-spacing: 1px; color: white;">{title}</h2>
                <div style="margin-top: 5px; opacity: 0.9;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# --- TAB 2: PERFORMANCE ---
with tab_perf:
    # Use standard Streamlit columns, no wrapper div
    if eval_artifacts:
        acc = eval_artifacts["accuracy_percent"]
        labels = eval_artifacts["labels_order"]
        report = eval_artifacts["classification_report"]
        cm = np.array(eval_artifacts["confusion_matrix"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{acc:.2f}%")
        c2.metric("Precision (Weighted)", f"{report['weighted avg']['precision']*100:.1f}%")
        c3.metric("F1 Score (Weighted)", f"{report['weighted avg']['f1-score']*100:.1f}%")

        st.markdown("---")
        
        g1, g2 = st.columns(2)
        with g1:
            st.caption("Confusion Matrix")
            st.pyplot(plot_confusion_matrix(cm, labels))
        with g2:
            st.caption("Per-Class F1 Score")
            st.pyplot(plot_f1_bars(report, labels))
            
    else:
        st.info("ℹ️ Evaluation artifacts (`eval_artifacts.pkl`) were not found. Run the training notebook to generate them.")

# --- TAB 3: ABOUT ---
with tab_about:
    st.markdown("""
    <div style="background: rgba(255,255,255,0.05); padding: 25px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);">
        <h3 style="margin-top:0;">🤖 How it works</h3>
        <p style="color: #ccc;">
            This application uses a <b>Linear Support Vector Machine (SVM)</b> trained on Amazon Product Reviews.
        </p>
        <ul style="color: #ccc;">
            <li><b>Step 1:</b> Text is cleaned (urls removed, lowercase).</li>
            <li><b>Step 2:</b> Transformed into vectors using <b>TF-IDF</b>.</li>
            <li><b>Step 3:</b> The SVM model predicts the probability of sentiment.</li>
        </ul>
        <br>
        <small style="opacity: 0.5;">Built with Streamlit & Scikit-Learn</small>
    </div>
    """, unsafe_allow_html=True)
