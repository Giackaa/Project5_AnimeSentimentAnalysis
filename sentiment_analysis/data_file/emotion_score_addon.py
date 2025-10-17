import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline

emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True  # so you see probabilities for all emotions
)

df = pd.read_csv(r"C:\Users\jackl\OneDrive\Desktop\Project 5 - Anime Sentiments\mal_comments2.csv")

# VADER
sia = SentimentIntensityAnalyzer()
df["sentiment"] = df["comment"].apply(lambda x: sia.polarity_scores(str(x)))

df["neg"] = df["sentiment"].apply(lambda x: x["neg"])
df["neu"] = df["sentiment"].apply(lambda x: x["neu"])
df["pos"] = df["sentiment"].apply(lambda x: x["pos"])
df["compound"] = df["sentiment"].apply(lambda x: x["compound"])
df["overall"] = df["compound"].apply(
    lambda x: "positive" if x >= 0.1 else ("negative" if x <= -0.1 else "neutral")
)
df = df.drop(columns=["sentiment"])

ratings = [4.0, 3.2, 3.5, 3.5, 3.9, 3.9, 4.1, 3.9, 4.0, 4.1, 4.2, 4.2, 4.4, 4.6]

# Create mapping: Ep1 → 4.0, Ep2 → 3.2, ...
episode_to_rating = {f"Ep{num}": rating for num, rating in enumerate(ratings, start=1)}

# Add rating column
df["ep_rating"] = df["episode"].map(episode_to_rating)

# Face-Hugging Emotions Analysis
def get_emotions(text):
    result = emotion_classifier(
        str(text),
        truncation=True,   # cut off text if too long
        max_length=512     # DistilRoBERTa max length
    )[0]
    return {d["label"]: d["score"] for d in result}

emotion_scores = df["comment"].apply(get_emotions)

emotion_df = pd.DataFrame(list(emotion_scores))
df = pd.concat([df.reset_index(drop=True), emotion_df.reset_index(drop=True)], axis=1)

# Export to file
df.to_csv(r"C:\Users\jackl\OneDrive\Desktop\Project 5 - Anime Sentiments\TF-IDF\sentiments_repairedCSV2.csv", index=False)


