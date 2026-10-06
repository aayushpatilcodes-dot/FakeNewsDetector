"""
Unified prediction interface used by the Streamlit app and tests.

Loads the best-performing model (decided after evaluate.py runs — see
results/comparison_table.csv) and exposes a single predict(text) -> (label,
confidence) function. Falls back gracefully between transformer and baseline
if one isn't available.
"""
import os

import joblib
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from preprocess import clean_text_basic, clean_text_for_baseline

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(ROOT, "models")
TRANSFORMER_DIR = os.path.join(MODELS_DIR, "distilbert_fake_news", "final")

LABEL_NAMES = ["FAKE", "REAL"]


class BaselinePredictor:
    name = "Logistic Regression (TF-IDF)"

    def __init__(self):
        self.vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
        self.model = joblib.load(os.path.join(MODELS_DIR, "logreg_baseline.joblib"))

    def predict(self, text: str):
        cleaned = clean_text_for_baseline(text)
        X = self.vectorizer.transform([cleaned])
        proba = self.model.predict_proba(X)[0]
        label_idx = int(proba.argmax())
        return LABEL_NAMES[label_idx], float(proba[label_idx])


class TransformerPredictor:
    name = "DistilBERT (fine-tuned)"

    def __init__(self, max_length=128):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(TRANSFORMER_DIR)
        self.model = DistilBertForSequenceClassification.from_pretrained(TRANSFORMER_DIR)
        self.model.eval()
        self.max_length = max_length

    def predict(self, text: str):
        cleaned = clean_text_basic(text)
        enc = self.tokenizer(
            cleaned, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt"
        )
        with torch.no_grad():
            logits = self.model(**enc).logits
            proba = torch.softmax(logits, dim=1)[0]
        label_idx = int(proba.argmax())
        return LABEL_NAMES[label_idx], float(proba[label_idx])


def load_best_predictor():
    """Prefer the transformer if it's been trained, else fall back to the baseline."""
    if os.path.exists(TRANSFORMER_DIR):
        try:
            return TransformerPredictor()
        except Exception as e:
            print(f"Could not load transformer predictor ({e}), falling back to baseline.")
    return BaselinePredictor()


if __name__ == "__main__":
    predictor = load_best_predictor()
    print(f"Loaded predictor: {predictor.name}")
    samples = [
        "Scientists confirm the earth revolves around the sun after decades of study.",
        "Aliens land in Ohio and endorse presidential candidate, eyewitnesses claim.",
    ]
    for s in samples:
        label, conf = predictor.predict(s)
        print(f"[{label} {conf:.2%}] {s}")
