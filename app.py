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
    # Dark theme compatible plot
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    
    # Custom color map
    im = ax.imshow(cm, cmap='PuBuGn') 
    
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([l.title() for l in labels], rotation=25, ha="right", color="white")
    ax.set_yticklabels([l.title() for l in labels], color="white")
    
    # Loop over data dimensions and create text annotations.
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white", fontweight='bold')

    ax.set_xlabel("Predicted", color="white")
    ax.set_ylabel("Actual", color="white")
    ax.set_title("Confusion Matrix", color="white", fontweight='bold')
    
    # Make spines invisible
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    fig.tight_layout()
    return fig

def plot_f1_bars(report_dict: dict, labels: list[str]):
    plt.style.use('dark_background')
    f1_scores = [(report_dict.get(l, {}).get("f1-score", 0.0) * 100) for l in labels]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([l.title() for l in labels], f1_scores, color=['#00f260', '#0575E6', '#e100ff'])
    
    ax.set_ylim(0, 100)
    ax.set_ylabel("F1-score (%)", color="white")
    ax.set_title("Per-class F1-score", color="white", fontweight='bold')
    
    # Value on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%',
                ha='center', va='bottom', color="white")
    
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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    /* Gradient Background for App */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        background-attachment: fixed;
    }

    /* Containers & Cards */
    .main-card {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 2rem;
        border-radius: 16px;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
    }

    /* Titles */
    h1 {
        background: linear-gradient(to right, #00f260, #0575E6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        text-align: center;
        padding-bottom: 10px;
    }
    
    h2, h3 {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(255,255,255,0.05);
        border-radius: 10px;
        color: white;
        font-weight: 600;
        padding: 0 20px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #00f260 0%, #0575E6 100%) !important;
        color: white !important;
        border: none !important;
    }

    /* Inputs */
    .stTextArea textarea {
        background-color: rgba(255, 255, 255, 0.07) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
    }
    .stTextArea textarea:focus {
        border-color: #0575E6 !important;
        box-shadow: 0 0 10px rgba(5, 117, 230, 0.3);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #00f260 0%, #0575E6 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 700;
        transition: transform 0.2s;
        width: 100%;
    }
    .stButton > button:hover {
        transform: scale(1.02);
    }

    /* Results */
    .result-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-top: 20px;
        animation: fadeIn 0.5s;
    }
    .res-pos { background: linear-gradient(135deg, rgba(17, 153, 142, 0.8), rgba(56, 239, 125, 0.8)); }
    .res-neg { background: linear-gradient(135deg, rgba(203, 45, 62, 0.8), rgba(239, 71, 58, 0.8)); }
    .res-neu { background: linear-gradient(135deg, rgba(43, 88, 118, 0.8), rgba(78, 67, 118, 0.8)); }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #00f260;
    }

</style>
""", unsafe_allow_html=True)

# -----------------------------
# UI LAYOUT
# -----------------------------

# Header
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.title("✨ Sentiment Analyzer")
st.markdown(
    """
    <p style='text-align: center; color: #b0b0b0;'>
    Advanced sentiment analysis powered by <b>Linear SVM</b> & <b>TF-IDF</b>.
    <br>Predictions are classified as Positive, Neutral, or Negative.
    </p>
    """, 
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)

# Tabs
tab_pred, tab_perf, tab_about = st.tabs(["🔮 Predict", "📊 Performance", "ℹ️ About"])

# --- TAB 1: PREDICT ---
with tab_pred:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown("### 📝 Analyze a Review")
    
    # Example buttons (small hack to put them above or near input)
    col_ex1, col_ex2, col_ex3 = st.columns(3)
    if col_ex1.button("Positive Example 😃"):
        st.session_state.review_text = "This product is amazing! Works perfectly and arrived early."
        st.rerun()
    if col_ex2.button("Neutral Example 😐"):
        st.session_state.review_text = "It's okay, does the job but nothing special."
        st.rerun()
    if col_ex3.button("Negative Example 😡"):
        st.session_state.review_text = "Terrible quality. Broke after one use. Do not buy!"
        st.rerun()

    # Input
    if "review_text" not in st.session_state:
        st.session_state.review_text = ""

    review_text = st.text_area(
        "", 
        value=st.session_state.review_text,
        height=150, 
        placeholder="Type or paste your Amazon review here...",
        label_visibility="collapsed"
    )

    if st.button("Analyze Sentiment 🚀"):
        if not review_text.strip():
            st.warning("⚠️ Please enter some text first.")
        else:
            cleaned = clean_text(review_text)
            X = tfidf.transform([cleaned])
            prediction = model.predict(X)[0]
            
            # Styles
            if prediction.lower() == 'positive':
                cls, icon, msg = "res-pos", "🎉", "Positive Sentiment"
            elif prediction.lower() == 'negative':
                cls, icon, msg = "res-neg", "💀", "Negative Sentiment"
            else:
                cls, icon, msg = "res-neu", "🤔", "Neutral Sentiment"

            st.markdown(f"""
            <div class="result-box {cls}">
                <div style="font-size: 3rem;">{icon}</div>
                <h2 style="margin: 10px 0 0 0;">{prediction.upper()}</h2>
                <div style="font-size: 0.9rem; opacity: 0.9;">{msg}</div>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 2: PERFORMANCE ---
with tab_perf:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Model Evaluation")
    
    if eval_artifacts:
        acc = eval_artifacts["accuracy_percent"]
        labels = eval_artifacts["labels_order"]
        report = eval_artifacts["classification_report"]
        cm = np.array(eval_artifacts["confusion_matrix"])

        c1, c2 = st.columns(2)
        c1.metric("Model Accuracy", f"{acc:.2f}%")
        c2.metric("Algorithm", "Linear SVM")
        
        st.markdown("---")
        
        c_chart1, c_chart2 = st.columns(2)
        with c_chart1:
            st.pyplot(plot_confusion_matrix(cm, labels))
        with c_chart2:
            st.pyplot(plot_f1_bars(report, labels))
            
    else:
        st.info("Performance artifacts not found. Please upload `eval_artifacts.pkl`.")
        
    st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 3: ABOUT ---
with tab_about:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown("### ℹ️ About This App")
    st.write(
        """
        This sentiment analysis tool uses a **Machine Learning** pipeline to classify text.
        
        **How it works:**
        1.  **Preprocessing**: Text is cleaned (lowercase, remove URLs, special chars).
        2.  **Vectorization**: Converted to numbers using **TF-IDF**.
        3.  **Classification**: A **Linear Support Vector Machine (SVM)** predicts the sentiment.
        
        **Tech Stack:**
        -   Python (`scikit-learn`, `pandas`)
        -   Streamlit (Frontend)
        -   Matplotlib (Visualization)
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div style="text-align: center; margin-top: 30px; opacity: 0.5; font-size: 0.8rem;">Designed with ❤️ for Data Science</div>', unsafe_allow_html=True)
