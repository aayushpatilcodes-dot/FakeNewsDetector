"""
Evaluate the baseline (TF-IDF + LogReg, TF-IDF + NB) and the fine-tuned
DistilBERT model on the SAME held-out test set (data/raw/test.csv, 8,117 rows).

Outputs:
    results/metrics.json                  -- full metrics for all models
    results/plots/confusion_matrix_<model>.png
    results/plots/roc_curve_comparison.png
    results/comparison_table.csv
    results/error_analysis.json           -- misclassified examples per model

Usage:
    python src/evaluate.py
"""
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from preprocess import build_content_column, clean_text_basic, clean_text_for_baseline

SEED = 42
ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW_DIR = os.path.join(ROOT, "data", "raw")
MODELS_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

LABEL_NAMES = ["FAKE", "REAL"]


def load_test_df():
    df = pd.read_csv(os.path.join(RAW_DIR, "test.csv"))
    df["content"] = build_content_column(df)
    return df


def eval_baseline(model_name, model_path, test_df, vectorizer):
    model = joblib.load(model_path)
    content_clean = test_df["content"].apply(clean_text_for_baseline)
    X_test = vectorizer.transform(content_clean)
    y_true = test_df["label"].values
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return y_true, y_pred, y_proba


def eval_transformer(model_dir, test_df, max_length=128, batch_size=32):
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    texts = test_df["content"].apply(clean_text_basic).tolist()
    y_true = test_df["label"].values
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
            )
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=1)[:, 1].numpy()
            all_probs.append(probs)
            if (i // batch_size) % 20 == 0:
                print(f"  transformer inference {i}/{len(texts)}")

    y_proba = np.concatenate(all_probs)
    y_pred = (y_proba >= 0.5).astype(int)
    return y_true, y_pred, y_proba


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(LABEL_NAMES)
    ax.set_yticklabels(LABEL_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    safe_name = model_name.lower().replace(" ", "_").replace("+", "")
    fig.savefig(os.path.join(PLOTS_DIR, f"confusion_matrix_{safe_name}.png"), dpi=150)
    plt.close(fig)
    return cm.tolist()


def plot_roc_comparison(results):
    fig, ax = plt.subplots(figsize=(6, 5))
    for model_name, (y_true, y_pred, y_proba) in results.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "roc_curve_comparison.png"), dpi=150)
    plt.close(fig)


def error_analysis(test_df, y_true, y_pred, y_proba, model_name, n=10):
    mis_idx = np.where(y_true != y_pred)[0]
    rng = np.random.RandomState(SEED)
    chosen = rng.choice(mis_idx, size=min(n, len(mis_idx)), replace=False) if len(mis_idx) > 0 else []
    examples = []
    for idx in chosen:
        row = test_df.iloc[idx]
        examples.append({
            "title": str(row.get("title", ""))[:200],
            "text_snippet": str(row.get("text", ""))[:300],
            "true_label": LABEL_NAMES[y_true[idx]],
            "predicted_label": LABEL_NAMES[y_pred[idx]],
            "predicted_confidence": float(y_proba[idx]),
        })
    return {"model": model_name, "num_misclassified": int(len(mis_idx)),
            "misclassification_rate": float(len(mis_idx) / len(y_true)),
            "examples": examples}


def main():
    test_df = load_test_df()
    print(f"Loaded test set: {len(test_df)} rows")

    vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))

    all_metrics = {}
    all_results = {}
    all_errors = []

    print("Evaluating Logistic Regression...")
    y_true, y_pred, y_proba = eval_baseline(
        "Logistic Regression", os.path.join(MODELS_DIR, "logreg_baseline.joblib"), test_df, vectorizer
    )
    all_metrics["Logistic Regression (TF-IDF)"] = compute_metrics(y_true, y_pred, y_proba)
    all_metrics["Logistic Regression (TF-IDF)"]["confusion_matrix"] = plot_confusion_matrix(
        y_true, y_pred, "Logistic Regression"
    )
    all_results["Logistic Regression"] = (y_true, y_pred, y_proba)
    all_errors.append(error_analysis(test_df, y_true, y_pred, y_proba, "Logistic Regression (TF-IDF)"))

    print("Evaluating Naive Bayes...")
    y_true, y_pred, y_proba = eval_baseline(
        "Naive Bayes", os.path.join(MODELS_DIR, "nb_baseline.joblib"), test_df, vectorizer
    )
    all_metrics["Naive Bayes (TF-IDF)"] = compute_metrics(y_true, y_pred, y_proba)
    all_metrics["Naive Bayes (TF-IDF)"]["confusion_matrix"] = plot_confusion_matrix(
        y_true, y_pred, "Naive Bayes"
    )
    all_results["Naive Bayes"] = (y_true, y_pred, y_proba)

    transformer_dir = os.path.join(MODELS_DIR, "distilbert_fake_news", "final")
    if os.path.exists(transformer_dir):
        print("Evaluating DistilBERT...")
        y_true, y_pred, y_proba = eval_transformer(transformer_dir, test_df)
        all_metrics["DistilBERT (fine-tuned)"] = compute_metrics(y_true, y_pred, y_proba)
        all_metrics["DistilBERT (fine-tuned)"]["confusion_matrix"] = plot_confusion_matrix(
            y_true, y_pred, "DistilBERT"
        )
        all_results["DistilBERT"] = (y_true, y_pred, y_proba)
        all_errors.append(error_analysis(test_df, y_true, y_pred, y_proba, "DistilBERT (fine-tuned)"))
    else:
        print(f"WARNING: transformer model not found at {transformer_dir}, skipping.")

    plot_roc_comparison(all_results)

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump({"test_set_size": len(test_df), "models": all_metrics}, f, indent=2)

    comparison_rows = []
    for model_name, m in all_metrics.items():
        comparison_rows.append({
            "model": model_name,
            "accuracy": round(m["accuracy"], 4),
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "f1": round(m["f1"], 4),
            "roc_auc": round(m["roc_auc"], 4),
        })
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(os.path.join(RESULTS_DIR, "comparison_table.csv"), index=False)
    print("\n" + comparison_df.to_string(index=False))

    with open(os.path.join(RESULTS_DIR, "error_analysis.json"), "w") as f:
        json.dump(all_errors, f, indent=2)

    print("\nSaved: results/metrics.json, results/comparison_table.csv, results/error_analysis.json")
    print(f"Saved plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
