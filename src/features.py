"""TF-IDF feature extraction for the classical baseline."""
from sklearn.feature_extraction.text import TfidfVectorizer

def build_tfidf_vectorizer(max_features=20000, ngram_range=(1, 2), min_df=3):
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=True,
    )
