import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut

import matplotlib.pyplot as plt
import numpy as np
import re

file = r"C:\Users\jackl\OneDrive\Desktop\Project 5 - Anime Sentiments\mal_comments2.csv"
df = pd.read_csv(file)  # columns: episode, comment

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = str(text).lower()  # ensure text is string
    text = re.sub(r'[^a-z\s]', '', text)  # remove punctuation/numbers
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)  # reduce repeated letters (loooool -> lool)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Apply preprocessing
episode_texts = df.groupby('episode')['comment'].apply(lambda x: ' '.join(x)).reset_index()
episode_texts['ep_num'] = episode_texts['episode'].str.extract(r'Ep(\d+)').astype(int)   # extracts Ep10 --> 10
episode_texts = episode_texts.sort_values('ep_num').reset_index(drop=True)

episode_texts['clean_comments'] = [preprocess(c) for c in episode_texts['comment'] if str(c).strip() != '']

# TF-IDF at comment level
vectorizer = TfidfVectorizer(max_features=1000)
X_all = vectorizer.fit_transform(episode_texts['clean_comments'])  # shape: (#comments, #features)

feature_names = vectorizer.get_feature_names_out()

# Loop through each episode
frame = []
for ep_idx, ep in enumerate(episode_texts['episode']):
    # Get TF-IDF vector for this episode (row from X_all)
    tfidf_vector = X_all[ep_idx].toarray().flatten()
    
    # Sort indices by score (descending)
    top_indices = np.argsort(tfidf_vector) 
    
    # Get words and scores
    top_words = [(feature_names[i], tfidf_vector[i]) for i in top_indices if tfidf_vector[i] > 0]
    
    print(f"\nEpisode: {ep}")
    for word, score in top_words:
        row = {'Episode' : ep,
               'Word' : word,
               'Score' : score}
        frame.append(row)

#pd.DataFrame(frame).to_csv(r"C:\Users\jackl\OneDrive\Desktop\Project 5 - Anime Sentiments\TF-IDF_collection.csv", index=False)  --- TF-IDF collection of words

X = X_all.toarray()   # convert sparse matrix to numpy for easier slicing
y = np.array([4.0, 3.2, 3.5, 3.5, 3.9, 3.9, 4.1, 3.9, 4.0, 4.1, 4.2, 4.2, 4.4, 4.6])

# LOOCV setup 
loo = LeaveOneOut()
model = LinearRegression()

y_true, y_pred = [], []

for train_idx, test_idx in loo.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    
    y_true.append(float(y_test[0]))
    y_pred.append(pred[0])


# --- Final results ---
mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print ('mse:', mse,'\n---\n' ,'r2:', r2)

frame = {
    "Episode" : [f"Ep{n}" for n in range(1, 15)],
    "Actual ratings" : y_true,
    "Predicted ratings" : y_pred,
    "Difference" : np.round(np.array(y_pred) - np.array(y_true),2)
}

# pd.DataFrame(frame).to_csv(r'C:\Users\jackl\OneDrive\Desktop\Project 5 - Anime Sentiments\TF-IDF\Model_ratings.csv', index=False) --- Model ratings



# ????????????????Should I do this??????????????????
'''
# Get coefficients from the trained model
coefs = model.coef_

# Episode 3 vector
ep_idx = 2  # zero-based, so Ep3 is index 2
ep_vector = X[ep_idx]

# Contribution per word
contributions = ep_vector * coefs

# Sort by absolute contribution
top_factors = sorted(
    zip(feature_names, contributions), 
    key=lambda x: abs(x[1]), reverse=True
)

print("Top factors driving Episode 3 prediction:")
for word, contrib in top_factors[:10]:
    print(f"{word}: {contrib:.4f}")
'''
