# Run with: streamlit run app.py
import streamlit as st
import pickle
import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide"
)

# ── Load Model & Vectorizer ──────────────────────────────────
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f: model = pickle.load(f)
    with open('tfidf.pkl', 'rb') as f: tfidf = pickle.load(f)
    return model, tfidf

model, tfidf = load_model()

# ── Text Cleaner (same as week 1) ────────────────────────────
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

# ── Header ───────────────────────────────────────────────────
st.title("🔍 Fake News Detector")
st.markdown("Paste any news article below and the model will predict if it's **Real** or **Fake**.")
st.divider()

# ── Tabs ─────────────────────────────────────────────────────
#tab1, tab2, tab3 = st.tabs(["🔎 Detect News", "📊 Model Performance", "📖 How It Works"])
tab1, tab2= st.tabs([" Detect News", "Model Performance"])

# ════════════════════════════════════════════════════════════
# TAB 1 — Detect News
# ════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Paste a news article")
    user_input = st.text_area(
        label="News Text",
        placeholder="Paste your news article here...",
        height=200,
        label_visibility="collapsed"
    )
    col1, col3 = st.columns([1, 3])
    with col1:
        predict_btn = st.button("🔍 Detect", use_container_width=True, type="primary")
    # with col2:
    #     clear_btn = st.button("🗑️ Clear", use_container_width=True)

    if predict_btn:
        if not user_input.strip():
            st.warning("⚠️ Please paste some news text first.")
        else:
            cleaned    = clean_text(user_input)
            vectorized = tfidf.transform([cleaned])
            prediction = model.predict(vectorized)[0]
            if hasattr(model, 'predict_proba'):
                proba      = model.predict_proba(vectorized)[0]
                confidence = proba[prediction] * 100
            else:
                confidence = None
            st.divider()
            if prediction == 1:
                st.success("## ✅ This news appears to be REAL")
            else:
                st.error("## ❌ This news appears to be FAKE")
            if confidence:
                st.markdown(f"**Confidence:** {confidence:.1f}%")
                st.progress(int(confidence))
            word_count = len(user_input.split())
            st.caption(f"Analyzed {word_count} words → {len(cleaned.split())} words after cleaning")

# ════════════════════════════════════════════════════════════
# TAB 2 — Model Performance
# ════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Model Performance")
    try:
        df = pd.read_csv('dataset/cleaned_news.csv').dropna()
        X, y = df['cleaned'], df['label']
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_test_tfidf = tfidf.transform(X_test)
        y_pred = model.predict(X_test_tfidf)
        acc = accuracy_score(y_test, y_pred)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy",  f"{acc*100:.1f}%")
        col2.metric("Real News", f"{sum(y==1):,}")
        col3.metric("Fake News", f"{sum(y==0):,}")
        col4.metric("Total",     f"{len(df):,}")
        st.divider()
        st.markdown("#### Confusion Matrix")
        st.caption("Shows correct vs incorrect predictions")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Fake', 'Real'],
                    yticklabels=['Fake', 'Real'], ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
        st.markdown("#### Dataset Balance")
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        counts = [sum(y==0), sum(y==1)]
        ax2.bar(['Fake', 'Real'], counts, color=['#ff4b4b', '#21c354'])
        ax2.set_ylabel('Number of Articles')
        ax2.set_title('Fake vs Real Articles')
        st.pyplot(fig2)
    except FileNotFoundError:
        st.info("Run week1 and week2 scripts first to see performance metrics.")

# ════════════════════════════════════════════════════════════
# TAB 3 — How It Works
# ════════════════════════════════════════════════════════════

# with tab3:
#     st.subheader("📖 How does this work?")
#     st.markdown("""
#     This app uses **Machine Learning** to detect fake news in 4 steps:

#     ---
#     **Step 1 — Clean the Text 🧹**
#     - Removes symbols, URLs, and common useless words
#     - Converts everything to lowercase

#     ---
#     **Step 2 — TF-IDF Vectorization 🔢**
#     - Computers only understand numbers, not words
#     - TF-IDF scores how important each word is
#     - 5000 most important words are kept as features

#     ---
#     **Step 3 — Model Prediction 🤖**
#     - Trained on 40,000+ articles
#     - Outputs: REAL (1) or FAKE (0)

#     ---
#     **Step 4 — Confidence Score 📈**
#     - Shows how sure the model is
#     - Higher % = more confident

#     ---
#     **Dataset:** ISOT Fake News Dataset (~44,000 articles)
#     """)



# st.divider()
# st.caption("Built with Python · Scikit-learn · Streamlit | Fake News Detector Project")
