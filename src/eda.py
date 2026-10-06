"""
Exploratory data analysis for the fake/real news dataset.

Produces, under results/plots/:
  - class_distribution.png
  - text_length_distribution.png
  - top_words_real.png / top_words_fake.png
And a results/eda_findings.txt summary (class balance, length stats, and
label-leakage tokens found, which drives the cleaning rules in preprocess.py).
"""
import os
import re
from collections import Counter

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

LABEL_NAMES = {0: "FAKE", 1: "REAL"}
COLORS = {"FAKE": "#d64550", "REAL": "#3a7ca5"}


def load_all():
    frames = []
    for split in ["train", "validation", "test"]:
        path = os.path.join(RAW_DIR, f"{split}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["split"] = split
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def plot_class_distribution(df, findings):
    counts = df["label"].map(LABEL_NAMES).value_counts()
    findings.append(f"Overall class counts: {counts.to_dict()} (total={len(df)})")
    for split in ["train", "validation", "test"]:
        sub = df[df["split"] == split]
        c = sub["label"].map(LABEL_NAMES).value_counts().to_dict()
        findings.append(f"  split={split}: {c} (n={len(sub)})")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(counts.index, counts.values, color=[COLORS[k] for k in counts.index])
    ax.set_title("Class Distribution (REAL vs FAKE)")
    ax.set_ylabel("Number of articles")
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(counts.values) * 0.01, str(v), ha="center")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "class_distribution.png"), dpi=150)
    plt.close(fig)


def plot_text_length_distribution(df, findings):
    df = df.copy()
    df["text"] = df["text"].fillna("")
    df["word_count"] = df["text"].str.split().apply(len)
    stats = df.groupby(df["label"].map(LABEL_NAMES))["word_count"].describe()
    findings.append("Word count stats by class:\n" + stats.to_string())

    fig, ax = plt.subplots(figsize=(7, 4))
    for label_name, color in COLORS.items():
        sub = df[df["label"].map(LABEL_NAMES) == label_name]["word_count"]
        sub = sub[sub < sub.quantile(0.99)]
        ax.hist(sub, bins=50, alpha=0.6, label=label_name, color=color)
    ax.set_xlabel("Word count per article")
    ax.set_ylabel("Frequency")
    ax.set_title("Article Length Distribution by Class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "text_length_distribution.png"), dpi=150)
    plt.close(fig)


def check_leakage(df, findings):
    """Look for tokens that trivially separate the two classes."""
    df = df.copy()
    df["text"] = df["text"].fillna("")
    suspects = ["Reuters", "(Reuters)", "Getty Images", "Featured image",
                "21st Century Wire", "Image via"]
    findings.append("\nLabel-leakage scan (occurrence rate per class):")
    for token in suspects:
        mask = df["text"].str.contains(re.escape(token), case=False, na=False)
        rate_fake = mask[df["label"] == 0].mean()
        rate_real = mask[df["label"] == 1].mean()
        findings.append(
            f"  '{token}': FAKE={rate_fake:.3%}  REAL={rate_real:.3%}"
        )


SIMPLE_STOPWORDS = set(stopwords.words("english")) | {
    "said", "says", "say", "also", "one", "two", "told", "would", "could",
    "like", "just", "will", "may", "new", "york", "mr", "u", "via",
}


def most_frequent_words(df, findings):
    df = df.copy()
    df["text"] = df["text"].fillna("")

    def top_words(texts, n=20):
        counter = Counter()
        for t in texts:
            words = re.findall(r"[a-zA-Z]{3,}", t.lower())
            words = [w for w in words if w not in SIMPLE_STOPWORDS]
            counter.update(words)
        return counter.most_common(n)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, label_val, label_name in [(axes[0], 1, "REAL"), (axes[1], 0, "FAKE")]:
        texts = df[df["label"] == label_val]["text"]
        top = top_words(texts)
        words, counts = zip(*top)
        ax.barh(list(words)[::-1], list(counts)[::-1], color=COLORS[label_name])
        ax.set_title(f"Top words — {label_name}")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "top_words_by_class.png"), dpi=150)
    plt.close(fig)
    findings.append("\nSaved top_words_by_class.png")


def main():
    df = load_all()
    findings = [f"EDA on GonzaloA/fake_news — {len(df)} total rows\n"]
    plot_class_distribution(df, findings)
    plot_text_length_distribution(df, findings)
    check_leakage(df, findings)
    most_frequent_words(df, findings)

    out_path = os.path.join(RESULTS_DIR, "eda_findings.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(findings))
    print("\n".join(findings))
    print(f"\nSaved findings to {out_path} and plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
