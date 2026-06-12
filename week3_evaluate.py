import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    confusion_matrix, classification_report
)

#Load saved model & vectorizer
with open('model.pkl', 'rb') as f: model = pickle.load(f)
with open('tfidf.pkl', 'rb') as f: tfidf = pickle.load(f)

print("✅ Model and vectorizer loaded!")
print()

#Load data & recreate test set
df = pd.read_csv('dataset/cleaned_news.csv').dropna()
X = df['cleaned']
y = df['label']

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test_tfidf = tfidf.transform(X_test)
y_pred = model.predict(X_test_tfidf)


# Print evaluation metrics
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)

print("📊 Model Evaluation Results:")
print(f"   Accuracy  : {acc  * 100:.2f}%  ← overall correct predictions")
print(f"   Precision : {prec * 100:.2f}%  ← when it says REAL, how often it's right")
print(f"   Recall    : {rec  * 100:.2f}%  ← how many actual REAL news it caught")
print(f"   F1 Score  : {f1   * 100:.2f}%  ← balance of precision & recall")
print()
print("📋 Full Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Fake', 'Real'],
    yticklabels=['Fake', 'Real']
)
plt.title('Confusion Matrix', fontsize=14)
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()
print("💾 Confusion matrix saved as confusion_matrix.png")
print()


tn, fp, fn, tp = cm.ravel()
print("🔍 Breaking it down:")
print(f"   ✅ Correctly identified FAKE news  : {tn}")
print(f"   ✅ Correctly identified REAL news  : {tp}")
print(f"   ❌ Real news called FAKE (mistake) : {fn}")
print(f"   ❌ Fake news called REAL (mistake) : {fp}")
print()
print("🎉 Week 3 Done! Run: streamlit run app.py")
