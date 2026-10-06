# Fake News Detection using NLP

A college NLP project that classifies news articles as **REAL** or **FAKE**, comparing a classical TF-IDF + Logistic Regression baseline against a fine-tuned DistilBERT transformer, wrapped in a Streamlit app.

Full methodology, dataset details, hyperparameters, results, error analysis, and decision log are in **[PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)**. This README only covers setup and how to run things.

## Project structure

```
fake-news-detector/
  data/
    raw/            full dataset CSVs (gitignored — regenerate with download_data.py)
    processed/      small committed samples of cleaned data (500 rows/split)
  src/
    download_data.py    downloads the dataset from Hugging Face
    eda.py               exploratory analysis -> results/plots/
    preprocess.py        text cleaning (leakage removal, stopwords, lemmatization)
    features.py           TF-IDF vectorizer
    train_baseline.py    TF-IDF + Logistic Regression / Naive Bayes
    train_transformer.py fine-tunes DistilBERT
    evaluate.py           scores all models on the same test set
    predict.py             unified predict(text) -> (label, confidence)
  app/
    app.py                 Streamlit app
  results/                 metrics.json, comparison_table.csv, plots/, error_analysis.json
  models/                  trained models (gitignored — regenerate via train scripts)
  tests/                   pytest unit tests
  docs/                    app screenshots
  PROJECT_DOCUMENTATION.md
  requirements.txt
```

## Setup

Requires Python 3.10+ (tested on 3.12.10). CPU only, no GPU required.

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('omw-1.4')"
```

## Running the full pipeline

All commands below are run from the project root, with the venv activated.

```bash
# 1. Download the dataset (GonzaloA/fake_news on Hugging Face, ~40k rows, no login needed)
python src/download_data.py

# 2. Exploratory data analysis -> results/plots/, results/eda_findings.txt
python src/eda.py

# 3. Train the classical baseline (TF-IDF + Logistic Regression + Naive Bayes)
python src/train_baseline.py

# 4. Fine-tune DistilBERT (CPU, ~35-45 min on a 16-core laptop; see
#    PROJECT_DOCUMENTATION.md Section 5.5 for subsample/hyperparameter rationale)
python src/train_transformer.py

# 5. Evaluate all models on the same held-out test set
python src/evaluate.py

# 6. Run the web app (loads the best available model automatically)
streamlit run app/app.py
```

## Running tests

```bash
pytest tests/ -v
```

## Results summary

See [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) Section 7 for the full results table, confusion matrices, ROC curves, and error analysis with real numbers from the trained models in this repository.
