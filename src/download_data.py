"""
Download the fake/real news dataset from the Hugging Face Hub and save a local,
version-controlled copy under data/raw/.

Dataset: GonzaloA/fake_news
  - A Hugging Face mirror of the ISOT Fake News Dataset (University of Victoria,
    ISOT research lab): https://huggingface.co/datasets/GonzaloA/fake_news
  - Original source: True.csv (real articles scraped from Reuters.com) and
    Fake.csv (fake articles from websites flagged unreliable by PolitiFact).
  - No login/authentication required to download.

Running this script writes:
  data/raw/train.csv, data/raw/validation.csv, data/raw/test.csv  (full splits)
  data/raw/dataset_info.txt  (row counts, column names, class balance)
"""
import os
from datasets import load_dataset
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

def main():
    print("Downloading GonzaloA/fake_news from the Hugging Face Hub...")
    ds = load_dataset("GonzaloA/fake_news")
    print(ds)

    info_lines = [f"Dataset: GonzaloA/fake_news (HF mirror of ISOT Fake News Dataset)", ""]

    for split_name, split_ds in ds.items():
        df = split_ds.to_pandas()
        out_path = os.path.join(RAW_DIR, f"{split_name}.csv")
        df.to_csv(out_path, index=False)
        label_col = "label" if "label" in df.columns else df.columns[-1]
        counts = df[label_col].value_counts().to_dict()
        info_lines.append(f"split={split_name}  rows={len(df)}  columns={list(df.columns)}  label_counts={counts}")
        print(f"Saved {out_path}  rows={len(df)}  label_counts={counts}")

    with open(os.path.join(RAW_DIR, "dataset_info.txt"), "w") as f:
        f.write("\n".join(info_lines))

if __name__ == "__main__":
    main()
