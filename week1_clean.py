import pandas as pd
import re
import nltk

# Download stopwords 
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load the dataset

true_df  = pd.read_csv('dataset/True.csv')   # real news
fake_df  = pd.read_csv('dataset/Fake.csv')   # fake news

# label column: 1 = Real, 0 = Fake
true_df['label'] = 1
fake_df['label'] = 0

# Combine both 
df = pd.concat([true_df, fake_df], ignore_index=True)

print("✅ Dataset loaded!")
print(f"   Total articles : {len(df)}")
print(f"   Real news      : {len(true_df)}")
print(f"   Fake news      : {len(fake_df)}")
print()
print("📋 First 3 rows:")
print(df[['title', 'text', 'label']].head(3))
print()



df['content'] = df['title'] + " " + df['text']


# Clean the text

stop_words = set(stopwords.words('english'))  # common useless words

def clean_text(text):
    text = text.lower()                              # lowercase everything
    text = re.sub(r'\[.*?\]', '', text)              #  [bracketed text]
    text = re.sub(r'https?://\S+|www\.\S+', '', text)#  URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)          #  numbers & symbols
    text = re.sub(r'\s+', ' ', text).strip()         #  extra spaces
    words = text.split()
    words = [w for w in words if w not in stop_words]#  stopwords
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


#Save cleaned data
df[['cleaned', 'label']].to_csv('dataset/cleaned_news.csv', index=False)
print("💾 Saved cleaned data to dataset/cleaned_news.csv")
print()
print("🎉 Week 1 Done! Run week2_model.py next.")
