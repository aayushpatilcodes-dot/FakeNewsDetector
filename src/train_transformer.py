"""
Fine-tune DistilBERT (distilbert-base-uncased) for REAL/FAKE news classification.

CPU-only training constraint: the full training set (24,353 rows) is too slow
to fine-tune a transformer on a laptop CPU in a reasonable time, so we
subsample. Subsample sizes and max sequence length are deliberately small and
documented here and in PROJECT_DOCUMENTATION.md:

    TRAIN_SUBSAMPLE = 6000   (class-balanced)
    VAL_SUBSAMPLE   = 1200   (class-balanced)
    MAX_LENGTH      = 128 tokens
    EPOCHS          = 2
    BATCH_SIZE      = 16

These were chosen after a timing smoke test on this machine (16-core CPU):
at max_length=128, batch_size=16, throughput was ~6 samples/sec, giving an
estimated full-run time of ~35 minutes for 6000 train rows x 2 epochs.

The full test set (8,117 rows) is used at evaluation time (inference only,
run separately in evaluate.py) so the model comparison is on the same,
complete held-out set as the baseline.

Usage:
    python src/train_transformer.py
"""
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)

from preprocess import build_content_column, clean_text_basic

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
set_seed(SEED)

MODEL_NAME = "distilbert-base-uncased"
TRAIN_SUBSAMPLE = 6000
VAL_SUBSAMPLE = 1200
MAX_LENGTH = 128
EPOCHS = 2
BATCH_SIZE = 16

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW_DIR = os.path.join(ROOT, "data", "raw")
MODELS_DIR = os.path.join(ROOT, "models", "distilbert_fake_news")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_balanced_subsample(name, n):
    df = pd.read_csv(os.path.join(RAW_DIR, f"{name}.csv"))
    df["content"] = build_content_column(df)
    df["content"] = df["content"].apply(clean_text_basic)
    n_per_class = n // 2
    parts = []
    for label in [0, 1]:
        sub = df[df["label"] == label].sample(
            n=min(n_per_class, (df["label"] == label).sum()), random_state=SEED
        )
        parts.append(sub)
    out = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return out


class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]))
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
    }


def main():
    print(f"Loading class-balanced subsamples: train={TRAIN_SUBSAMPLE}, val={VAL_SUBSAMPLE}")
    train_df = load_balanced_subsample("train", TRAIN_SUBSAMPLE)
    val_df = load_balanced_subsample("validation", VAL_SUBSAMPLE)
    print(f"Actual train rows={len(train_df)}  val rows={len(val_df)}")
    print("Train label balance:", train_df["label"].value_counts().to_dict())

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    def tokenize(df):
        return tokenizer(
            list(df["content"]),
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

    print("Tokenizing...")
    train_enc = tokenize(train_df)
    val_enc = tokenize(val_df)
    train_ds = NewsDataset(train_enc, train_df["label"].values)
    val_ds = NewsDataset(val_enc, val_df["label"].values)

    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir=os.path.join(MODELS_DIR, "checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=20,
        seed=SEED,
        report_to=[],
        no_cuda=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    print(f"Training took {elapsed:.1f} seconds ({elapsed/60:.1f} min)")

    val_metrics = trainer.evaluate()
    print("Final validation metrics:", val_metrics)

    final_dir = os.path.join(MODELS_DIR, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    report = {
        "seed": SEED,
        "model_name": MODEL_NAME,
        "train_subsample": len(train_df),
        "val_subsample": len(val_df),
        "max_length": MAX_LENGTH,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "training_time_seconds": elapsed,
        "validation_metrics": val_metrics,
    }
    with open(os.path.join(RESULTS_DIR, "transformer_val_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("Saved model to", final_dir)
    print("Saved validation report to results/transformer_val_report.json")


if __name__ == "__main__":
    main()
