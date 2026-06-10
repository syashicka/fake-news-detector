import pandas as pd
import re
import nltk

# Download stopwords (only needed once)
nltk.download('stopwords')
from nltk.corpus import stopwords

# ── STEP 1: Load the dataset ─────────────────────────────────
# Make sure your CSV files are inside the dataset/ folder
# ISOT dataset comes as two files: True.csv and Fake.csv

true_df  = pd.read_csv('dataset/True.csv')   # real news
fake_df  = pd.read_csv('dataset/Fake.csv')   # fake news

# Add a label column: 1 = Real, 0 = Fake
true_df['label'] = 1
fake_df['label'] = 0

# Combine both into one dataframe
df = pd.concat([true_df, fake_df], ignore_index=True)

print("✅ Dataset loaded!")
print(f"   Total articles : {len(df)}")
print(f"   Real news      : {len(true_df)}")
print(f"   Fake news      : {len(fake_df)}")
print()
print("📋 First 3 rows:")
print(df[['title', 'text', 'label']].head(3))
print()


# ── STEP 2: Combine title + text into one column ─────────────
# More text = better learning for the model
df['content'] = df['title'] + " " + df['text']


# ── STEP 3: Clean the text ───────────────────────────────────
# We remove anything that doesn't help the model learn

stop_words = set(stopwords.words('english'))  # common useless words

def clean_text(text):
    text = text.lower()                              # lowercase everything
    text = re.sub(r'\[.*?\]', '', text)              # remove [bracketed text]
    text = re.sub(r'https?://\S+|www\.\S+', '', text)# remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)          # remove numbers & symbols
    text = re.sub(r'\s+', ' ', text).strip()         # remove extra spaces
    words = text.split()
    words = [w for w in words if w not in stop_words]# remove stopwords
    return " ".join(words)

print("🧹 Cleaning text... (this may take a minute)")
df['cleaned'] = df['content'].apply(clean_text)

print("✅ Text cleaned!")
print()
print("📋 Example — Before cleaning:")
print(df['content'].iloc[0][:200])
print()
print("📋 Example — After cleaning:")
print(df['cleaned'].iloc[0][:200])
print()


# ── STEP 4: Save cleaned data ────────────────────────────────
df[['cleaned', 'label']].to_csv('dataset/cleaned_news.csv', index=False)
print("💾 Saved cleaned data to dataset/cleaned_news.csv")
print()
print("🎉 Week 1 Done! Run week2_model.py next.")
