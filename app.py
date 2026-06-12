import streamlit as st
import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from urllib.parse import urlparse
from sklearn.metrics import confusion_matrix, accuracy_score

from newsapi import NewsApiClient
newsapi = NewsApiClient(api_key='3553e6bea45e4976997d78058c8bcf41')

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

st.set_page_config(page_title="Fake News Detector", page_icon="🔍", layout="wide")

stop_words = set(stopwords.words('english'))

# ── Trusted & Untrusted Domains ───────────────────────────────
TRUSTED_DOMAINS = [
    "reuters",
    "bbc",
    "apnews",
    "guardian",
    "nytimes",
    "washingtonpost"
]
UNTRUSTED_DOMAINS = [
    "infowars.com", "breitbart.com", "naturalnews.com",
    "beforeitsnews.com", "yournewswire.com", "newspunch.com",
    "thegatewaypundit.com", "worldnewsdailyreport.com",
    "empirenews.net", "huzlers.com", "babylonbee.com",
    "dailybuzzlive.com", "newswatch33.com"
]

# ── Improved claim extractor ──────────────────────────────────
def extract_claim(text):
    # Remove markdown links like [text](url) → just keep the text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # Remove any leftover URLs
    text = re.sub(r'https?://\S+', '', text)
    
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.split()) > 5]

    keywords = [
        'says', 'claims', 'announces', 'confirms', 'states', 'warns',
        'deal', 'agreement', 'nuclear', 'war', 'peace', 'attack',
        'president', 'minister', 'government', 'official', 'manager',
        'election', 'vote', 'economy', 'policy', 'law', 'court',
        'killed', 'died', 'injured', 'arrested', 'signed', 'approved',
        'trophies', 'titles', 'champion', 'coach', 'contract', 'leave'
    ]

    def score(s):
        s_lower = s.lower()
        keyword_score = sum(1 for k in keywords if k in s_lower)
        length_score  = min(len(s.split()), 20)
        return keyword_score * 2 + length_score

    if not sentences:
        return text[:100]

    best = max(sentences, key=score)
    best = re.sub(r'[\"\'"\u201c\u201d]', '', best).strip()

    if len(best) > 120:
        best = best[:120].rsplit(' ', 1)[0]

    return best

# ── Search DuckDuckGo ─────────────────────────────────────────
def search_internet(query):
    try:
        response = newsapi.get_everything(
            q=query,
            language='en',
            sort_by='relevancy',
            page_size=6
        )
        articles = response.get('articles', [])
        # Convert to same format as before
        results = []
        for a in articles:
            results.append({
                'title': a.get('title', ''),
                'href':  a.get('url', ''),
                'body':  a.get('description', '')
            })
        return results
    except Exception as e:
        return []

# ── Check domain trust level ──────────────────────────────────
def check_domain(url):
    try:
        domain = urlparse(url).netloc.lower()

        # remove www.
        if domain.startswith("www."):
            domain = domain[4:]

        for trusted in TRUSTED_DOMAINS:
            if trusted in domain:
                return "trusted"

        for untrusted in UNTRUSTED_DOMAINS:
            if untrusted in domain:
                return "untrusted"

        return "unknown"

    except:
        return "unknown"

# ── Analyze results ───────────────────────────────────────────
def analyze_results(results):
    trusted_count   = 0
    untrusted_count = 0
    for r in results:
        status = check_domain(r.get('href', ''))
        if status == "trusted":
            trusted_count += 1
        elif status == "untrusted":
            untrusted_count += 1

    if trusted_count >= 2:
        return "corroborated", f"✅ {trusted_count} trusted sources cover this story"
    elif trusted_count == 1:
        return "partial",      f"🔎 1 trusted source found covering this story"
    elif untrusted_count >= 2:
        return "suspicious",   f"⚠️ Only untrusted sources cover this story"
    else:
        return "unverified",   "❓ No trusted sources found covering this story"

# ── Final Combined Verdict ────────────────────────────────────
def final_verdict(prediction, search_status):
    if   prediction == 1 and search_status == "corroborated":
        return "🟢 VERY LIKELY REAL",  "ML model says REAL and multiple trusted sources cover this story."
    elif prediction == 1 and search_status == "partial":
        return "🟢 LIKELY REAL",       "ML model says REAL and 1 trusted source found."
    elif prediction == 1 and search_status == "unverified":
        return "🟡 UNCERTAIN",         "ML model says REAL but no trusted sources found. Verify manually."
    elif prediction == 1 and search_status == "suspicious":
        return "🟡 UNCERTAIN",         "ML model says REAL but only untrusted sources cover this."
    elif prediction == 0 and search_status == "corroborated":
        return "🟡 UNCERTAIN",         "ML says FAKE but trusted sources cover this. May be a recent event."
    elif prediction == 0 and search_status == "partial":
        return "🟠 LIKELY REAL",       "ML says FAKE but 1 trusted source found. Possibly a recent article."
    elif prediction == 0 and search_status == "suspicious":
        return "🔴 VERY LIKELY FAKE",  "ML model says FAKE and only untrusted sources cover this."
    else:
        return "🔴 LIKELY FAKE",       "ML model says FAKE and no trusted sources found online."

# ── Text Cleaner ──────────────────────────────────────────────
def clean_text(text):
    t = text.lower()
    t = re.sub(r'\[.*?\]', '', t)
    t = re.sub(r'https?://\S+|www\.\S+', '', t)
    t = re.sub(r'[^a-zA-Z\s]', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    words = [w for w in t.split() if w not in stop_words]
    return " ".join(words)

# ── Train Model ───────────────────────────────────────────────
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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf  = tfidf.transform(X_test)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    return model, tfidf, X_test_tfidf, y_test, df

# ── Header ────────────────────────────────────────────────────
st.title("🔍 Fake News Detector")
st.markdown("Paste any news article — verified using **ML prediction + live internet search**.")
st.divider()

with st.spinner("🔄 Training model... please wait (~1 min on first load)"):
    model, tfidf, X_test_tfidf, y_test, df = train_model()

tab1, tab2, tab3 = st.tabs(["🔎 Detect News", "📊 Model Performance", "📖 How It Works"])

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

    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        predict_btn = st.button("🔍 Detect", use_container_width=True, type="primary")

    if predict_btn:
        if not user_input.strip():
            st.warning("⚠️ Please paste some news text first.")
        else:
            # ── ML Prediction ──────────────────────────────
            cleaned    = clean_text(user_input)
            vectorized = tfidf.transform([cleaned])
            prediction = model.predict(vectorized)[0]
            proba      = model.predict_proba(vectorized)[0]
            confidence = proba[prediction] * 100

            # ── Extract claim & search ─────────────────────
            claim = " ".join(cleaned.split()[:8])
            with st.spinner("🌐 Searching the internet to verify..."):
                results = search_internet(claim)

            # ── Analyze & verdict ──────────────────────────
            search_status, search_msg    = analyze_results(results)
            verdict, verdict_msg         = final_verdict(prediction, search_status)

            st.divider()

            # ── Final Verdict ──────────────────────────────
            st.markdown(f"## {verdict}")
            st.caption(verdict_msg)
            st.divider()

            # ── Two columns ────────────────────────────────
            col_ml, col_src = st.columns(2)

            with col_ml:
                st.markdown("#### 🤖 ML Model")
                if prediction == 1:
                    st.success(f"✅ REAL — {confidence:.1f}% confident")
                else:
                    st.error(f"❌ FAKE — {confidence:.1f}% confident")
                st.progress(int(confidence))

            with col_src:
                st.markdown("#### 🌐 Internet Verification")
                if search_status == "corroborated":
                    st.success(search_msg)
                elif search_status == "partial":
                    st.success(search_msg)
                elif search_status == "suspicious":
                    st.error(search_msg)
                else:
                    st.warning(search_msg)

            st.divider()

            # ── Search query used ──────────────────────────
            st.caption(f"🔎 Searched for: *\"{claim}\"*")

            # ── Search Results ─────────────────────────────
            if results:
                st.markdown("#### 🗞️ What the internet says")
                for r in results:
                    url    = r.get('href', '')
                    title  = r.get('title', 'No title')
                    body   = r.get('body', '')[:150] + "..."
                    status = check_domain(url)
                    icon   = "✅" if status == "trusted" else ("❌" if status == "untrusted" else "🔗")
                    with st.expander(f"{icon} {title}"):
                        st.markdown(f"**Source:** {url}")
                        st.markdown(f"**Preview:** {body}")
                        st.markdown(f"[Read full article →]({url})")
            else:
                st.warning("No search results found. Check your internet connection.")

            st.caption(f"Analyzed {len(user_input.split())} words → {len(cleaned.split())} after cleaning")

# ════════════════════════════════════════════════════════════
# TAB 2 — Model Performance
# ════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 How well does the model perform?")
    y_pred = model.predict(X_test_tfidf)
    acc    = accuracy_score(y_test, y_pred)
    y      = df['label']

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎯 Accuracy",  f"{acc*100:.1f}%")
    col2.metric("📰 Real News", f"{sum(y==1):,}")
    col3.metric("📰 Fake News", f"{sum(y==0):,}")
    col4.metric("📦 Total",     f"{len(df):,}")
    st.divider()

    st.markdown("#### Confusion Matrix")
    cm_mat = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fake', 'Real'],
                yticklabels=['Fake', 'Real'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

    st.markdown("#### Dataset Balance")
    fig2, ax2 = plt.subplots(figsize=(4, 3))
    ax2.bar(['Fake', 'Real'], [sum(y==0), sum(y==1)],
            color=['#ff4b4b', '#21c354'])
    ax2.set_ylabel('Number of Articles')
    st.pyplot(fig2)

# ════════════════════════════════════════════════════════════
# TAB 3 — How It Works
# ════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📖 How does this work?")
    st.markdown("""
    This app uses **two layers** of verification:

    ---
    **Layer 1 — ML Model 🤖**
    Cleans text → TF-IDF vectorization → Logistic Regression → REAL or FAKE + confidence %

    ---
    **Layer 2 — Live Internet Search 🌐**
    Extracts the most important claim → searches DuckDuckGo →
    checks if trusted sources cover the same story

    ---
    **Combined Verdict 🎯**

    | ML Model | Internet | Final Verdict |
    |---|---|---|
    | REAL | 2+ trusted sources | 🟢 Very Likely Real |
    | REAL | 1 trusted source | 🟢 Likely Real |
    | REAL | No sources found | 🟡 Uncertain |
    | FAKE | Trusted sources cover it | 🟡 Uncertain — may be recent news |
    | FAKE | Only untrusted sources | 🔴 Very Likely Fake |
    | FAKE | No sources found | 🔴 Likely Fake |

    ---
    **Why does the ML model sometimes say FAKE for real articles?**

    The model was trained on 2016–2017 news data. Articles about recent
    events (2024–2026) may be misclassified because the model has never
    seen that context before. The internet verification layer helps
    catch these cases.

    ---
    **Dataset:** ISOT Fake News Dataset (~44,000 articles)
    """)

    with st.expander("📋 Trusted domains list"):
        st.markdown(", ".join(TRUSTED_DOMAINS))

st.divider()
st.caption("Built with Python · Scikit-learn · Streamlit | Fake News Detector")