"""
Train the classical baseline: TF-IDF features + Logistic Regression, with
Multinomial Naive Bayes trained alongside for comparison.

Usage:
    python src/train_baseline.py

Outputs:
    models/tfidf_vectorizer.joblib
    models/logreg_baseline.joblib
    models/nb_baseline.joblib
    data/processed/{train,validation,test}_clean_sample.csv  (500-row samples, committed)
    results/baseline_val_report.json  (validation-set metrics, used only for model selection)
"""
import json
import os
import random

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB

from features import build_tfidf_vectorizer
from preprocess import build_content_column, clean_text_for_baseline

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW_DIR = os.path.join(ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
MODELS_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_split(name):
    df = pd.read_csv(os.path.join(RAW_DIR, f"{name}.csv"))
    df["content"] = build_content_column(df)
    print(f"Cleaning {name} split ({len(df)} rows) for baseline...")
    df["content_clean"] = df["content"].apply(clean_text_for_baseline)
    return df


def main():
    train_df = load_split("train")
    val_df = load_split("validation")
    test_df = load_split("test")

    # Save small, committed samples of cleaned data for reproducibility/inspection.
    for name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        sample = df.sample(n=min(500, len(df)), random_state=SEED)
        sample[["title", "text", "content_clean", "label"]].to_csv(
            os.path.join(PROCESSED_DIR, f"{name}_clean_sample.csv"), index=False
        )

    vectorizer = build_tfidf_vectorizer()
    X_train = vectorizer.fit_transform(train_df["content_clean"])
    X_val = vectorizer.transform(val_df["content_clean"])
    y_train = train_df["label"].values
    y_val = val_df["label"].values

    print("Training Logistic Regression...")
    logreg = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)
    logreg.fit(X_train, y_train)
    val_pred_lr = logreg.predict(X_val)
    lr_acc = accuracy_score(y_val, val_pred_lr)
    lr_f1 = f1_score(y_val, val_pred_lr)
    print(f"LogReg  val_accuracy={lr_acc:.4f}  val_f1={lr_f1:.4f}")

    print("Training Multinomial Naive Bayes...")
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    val_pred_nb = nb.predict(X_val)
    nb_acc = accuracy_score(y_val, val_pred_nb)
    nb_f1 = f1_score(y_val, val_pred_nb)
    print(f"NB      val_accuracy={nb_acc:.4f}  val_f1={nb_f1:.4f}")

    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    joblib.dump(logreg, os.path.join(MODELS_DIR, "logreg_baseline.joblib"))
    joblib.dump(nb, os.path.join(MODELS_DIR, "nb_baseline.joblib"))

    report = {
        "seed": SEED,
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "logreg": {"val_accuracy": lr_acc, "val_f1": lr_f1},
        "naive_bayes": {"val_accuracy": nb_acc, "val_f1": nb_f1},
        "tfidf_params": {"max_features": 20000, "ngram_range": [1, 2], "min_df": 3},
    }
    with open(os.path.join(RESULTS_DIR, "baseline_val_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("Saved models to", MODELS_DIR)
    print("Saved validation report to results/baseline_val_report.json")


if __name__ == "__main__":
    main()
