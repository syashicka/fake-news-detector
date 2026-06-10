import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import pickle

# ── STEP 1: Load cleaned data ────────────────────────────────
df = pd.read_csv('dataset/cleaned_news.csv')

# Drop any empty rows
df.dropna(inplace=True)

X = df['cleaned']   # input  → the news text
y = df['label']     # output → 1 (real) or 0 (fake)

print(f"✅ Data loaded: {len(df)} articles")
print()


# ── STEP 2: Split into Train & Test sets ─────────────────────
# 80% for training, 20% for testing
# random_state=42 just means results are reproducible
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"📊 Training set : {len(X_train)} articles")
print(f"📊 Testing set  : {len(X_test)} articles")
print()


# ── STEP 3: TF-IDF Vectorizer ────────────────────────────────
# Converts words → numbers the model can understand
# max_features=5000 means we only keep the 5000 most important words

print("🔢 Converting text to numbers using TF-IDF...")
tfidf = TfidfVectorizer(max_features=5000)

X_train_tfidf = tfidf.fit_transform(X_train)  # learn vocab + transform
X_test_tfidf  = tfidf.transform(X_test)       # only transform (don't relearn)

print("✅ TF-IDF done!")
print()


# ── STEP 4: Train Model 1 — Logistic Regression ──────────────
print("🤖 Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)
lr_pred = lr_model.predict(X_test_tfidf)
lr_acc  = accuracy_score(y_test, lr_pred)
print(f"   Accuracy: {lr_acc * 100:.2f}%")
print()


# ── STEP 5: Train Model 2 — Passive Aggressive Classifier ────
# This model is great for text classification — fast and accurate
print("🤖 Training Passive Aggressive Classifier...")
pa_model = PassiveAggressiveClassifier(max_iter=1000)
pa_model.fit(X_train_tfidf, y_train)
pa_pred = pa_model.predict(X_test_tfidf)
pa_acc  = accuracy_score(y_test, pa_pred)
print(f"   Accuracy: {pa_acc * 100:.2f}%")
print()


# ── STEP 6: Pick the best model & save ───────────────────────
best_model = lr_model if lr_acc >= pa_acc else pa_model
best_name  = "Logistic Regression" if lr_acc >= pa_acc else "Passive Aggressive"
print(f"🏆 Best model: {best_name} ({max(lr_acc, pa_acc)*100:.2f}% accuracy)")
print()

# Save both the model and the vectorizer using pickle
# (We need the vectorizer later to convert user input too)
with open('model.pkl',  'wb') as f: pickle.dump(best_model, f)
with open('tfidf.pkl',  'wb') as f: pickle.dump(tfidf, f)

print("💾 Model saved as model.pkl")
print("💾 Vectorizer saved as tfidf.pkl")
print()
print("🎉 Week 2 Done! Run week3_evaluate.py next.")
