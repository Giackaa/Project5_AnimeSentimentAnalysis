import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load your dataset
df = pd.read_csv(r"C:\Users\jackl\OneDrive\Desktop\Project 5 - Anime Sentiments\TF-IDF\sentiments.csv")

# Define emotion columns
emotion_columns = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

def top_words_per_emotion(df, emotion_columns, top_n=10, percentile=0.9, output_path=None):
    results = []

    custom_skip_words = ENGLISH_STOP_WORDS.union({
    'episode', 'animation', 'character', 'series', 'show', 'movie', 'goal', 'season', 'anime', 'blue', 'lock', 'just', 'like', 'ep', 'match'
    })

    for emotion in emotion_columns:
        print(f"\nProcessing {emotion.upper()} ...")

        # Calculate 90th percentile threshold for this emotion
        threshold = df[emotion].quantile(percentile)

        # Filter comments above threshold
        high_emotion_df = df[df[emotion] >= threshold]
        comments = high_emotion_df['comment'].dropna().astype(str).tolist()


        # TF-IDF on selected comments
        vectorizer = TfidfVectorizer(stop_words= list(custom_skip_words), max_df=0.8, min_df=5, max_features=5000)
        X = vectorizer.fit_transform(comments)

        # Average TF-IDF score across all documents
        avg_scores = X.mean(axis=0).A1  # convert sparse matrix → array
        vocab = vectorizer.get_feature_names_out()

        # Rank top words
        top_indices = avg_scores.argsort()[::-1][:top_n]
        for i in top_indices:
            results.append({
                "Emotion": emotion,
                "Word": vocab[i],
                "TFIDF_Score": round(avg_scores[i], 5)
            })

        dfa = pd.DataFrame(results)
        if output_path:
            dfa.to_csv(output_path, index=False)

    return dfa


output_csv = r"C:\Users\jackl\OneDrive\Desktop\Project 5 - Anime Sentiments\TF-IDF\top_words_per_emotion.csv"
top_words_df = top_words_per_emotion(df, emotion_columns, top_n=10, percentile=0.9, output_path=output_csv)