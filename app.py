import streamlit as st
import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

st.set_page_config(page_title="Fake News Detector", page_icon="🔍", layout="wide")

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

@st.cache_resource
def train_model():
    true_df = pd.read_csv('dataset/True.csv')
    fake_df = pd.read_csv('dataset/Fake.csv')
    true_df['label'] = 1
    fake_df['label'] = 0
    df = pd.concat([true_df, fake_df], ignore_index=True)
    df['content'] = df['title'] + " " + df['text']
    df['cleaned'] = df['content'].apply(clean_text)
    df = df[['cleaned', 'label']].dropna()
    X, y = df['cleaned'], df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf  = tfidf.transform(X_test)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    return model, tfidf, X_test_tfidf, y_test, df

st.title("🔍 Fake News Detector")
st.markdown("Paste any news article below and the model will predict if it's **Real** or **Fake**.")
st.divider()

with st.spinner("🔄 Training model... please wait (~1 min on first load)"):
    model, tfidf, X_test_tfidf, y_test, df = train_model()

tab1, tab2, tab3 = st.tabs(["🔎 Detect News", "📊 Model Performance", "📖 How It Works"])

with tab1:
    st.subheader("Paste a news article")
    user_input = st.text_area(label="News Text", placeholder="Paste your news article here...", height=200, label_visibility="collapsed")
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        predict_btn = st.button("🔍 Detect", use_container_width=True, type="primary")
    if predict_btn:
        if not user_input.strip():
            st.warning("⚠️ Please paste some news text first.")
        else:
            cleaned    = clean_text(user_input)
            vectorized = tfidf.transform([cleaned])
            prediction = model.predict(vectorized)[0]
            proba      = model.predict_proba(vectorized)[0]
            confidence = proba[prediction] * 100
            st.divider()
            if prediction == 1:
                st.success("## ✅ This news appears to be REAL")
            else:
                st.error("## ❌ This news appears to be FAKE")
            st.markdown(f"**Confidence:** {confidence:.1f}%")
            st.progress(int(confidence))
            st.caption(f"Analyzed {len(user_input.split())} words → {len(cleaned.split())} words after cleaning")

with tab2:
    st.subheader("📊 How well does the model perform?")
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    y = df['label']
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎯 Accuracy",  f"{acc*100:.1f}%")
    col2.metric("📰 Real News", f"{sum(y==1):,}")
    col3.metric("📰 Fake News", f"{sum(y==0):,}")
    col4.metric("📦 Total",     f"{len(df):,}")
    st.divider()
    st.markdown("#### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)
    st.markdown("#### Dataset Balance")
    fig2, ax2 = plt.subplots(figsize=(4, 3))
    ax2.bar(['Fake', 'Real'], [sum(y==0), sum(y==1)], color=['#ff4b4b', '#21c354'])
    ax2.set_ylabel('Number of Articles')
    st.pyplot(fig2)

with tab3:
    st.subheader("📖 How does this work?")
    st.markdown("""
    **Step 1 — Clean the Text 🧹**
    Removes symbols, URLs, stopwords, converts to lowercase.

    ---
    **Step 2 — TF-IDF Vectorization 🔢**
    Converts words into numbers. Top 5000 important words kept.

    ---
    **Step 3 — Model Prediction 🤖**
    Logistic Regression trained on 40,000+ articles predicts REAL or FAKE.

    ---
    **Step 4 — Confidence Score 📈**
    Shows how sure the model is. Higher % = more confident.

    ---
    **Dataset:** ISOT Fake News Dataset (~44,000 articles)
    """)

st.divider()
st.caption("Built with Python · Scikit-learn · Streamlit | Fake News Detector Project")