# Fake News Detection using NLP — Project Documentation

_This document is updated continuously as the project is built. All numbers reported are from actual runs, not estimates._

## 1. Project Overview and Problem Statement

Fake news — fabricated or misleading news content presented as legitimate journalism — spreads rapidly on social media and erodes trust in information sources. This project builds and compares two text classification systems that label a news article (title + body text) as **REAL** or **FAKE**:

1. A classical machine-learning baseline (TF-IDF features + Logistic Regression).
2. A fine-tuned transformer (DistilBERT).

Both models are evaluated on the same held-out test set, and the better-performing model is wrapped in a Streamlit web app that lets a user paste in a headline/article and get a prediction with a confidence score.

## 2. Motivation and Real-World Relevance

Misinformation detection is a widely studied NLP application with direct real-world impact on journalism, social media moderation, and public discourse. Comparing a lightweight classical baseline against a modern transformer also illustrates a core trade-off in applied NLP: transformers typically achieve higher accuracy by capturing context and word order, but at much higher computational cost than a bag-of-words model. This project quantifies that trade-off on consumer CPU hardware, which is a realistic constraint for many practitioners and students.

## 3. Dataset

**Name:** `GonzaloA/fake_news` (Hugging Face Hub mirror of the **ISOT Fake News Dataset**, University of Victoria ISOT research lab).

**Source URL:** https://huggingface.co/datasets/GonzaloA/fake_news
**Original source:** ISOT Fake News Dataset, University of Victoria (https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/) — also widely mirrored on Kaggle as "Fake and Real News Dataset".
**Download method:** `datasets.load_dataset("GonzaloA/fake_news")` — no login/authentication required. See [`src/download_data.py`](src/download_data.py).
**License:** The original ISOT dataset is distributed by the University of Victoria for research and educational, non-commercial use. The Hugging Face mirror does not re-state a license, so the original source's research/education terms apply; this project uses it strictly for an academic assignment.

**Size and splits** (already pre-split by the dataset authors, loaded as-is):

| Split | Rows | FAKE (0) | REAL (1) |
|---|---|---|---|
| train | 24,353 | 11,158 (45.8%) | 13,195 (54.2%) |
| validation | 8,117 | 3,723 (45.9%) | 4,394 (54.1%) |
| test | 8,117 | 3,782 (46.6%) | 4,335 (53.4%) |
| **Total** | **40,587** | 18,663 (46.0%) | 21,924 (54.0%) |

Classes are reasonably balanced (~46/54), so accuracy is a meaningful metric alongside F1. Columns: `title`, `text`, `label` (0=FAKE, 1=REAL). REAL articles come from Reuters.com; FAKE articles come from websites flagged unreliable by PolitiFact.

**Article length:** FAKE articles average 429 words (std 376, median 374); REAL articles average 390 words (std 301, median 358). See `results/plots/text_length_distribution.png`.

**Label leakage found** (via [`src/eda.py`](src/eda.py), see `results/eda_findings.txt`): the raw `text` field contains near-perfect label-leaking tokens, an artifact of how the two source pools were scraped:

| Token | % of FAKE articles | % of REAL articles |
|---|---|---|
| "Reuters" | 1.3% | **96.6%** |
| "(Reuters)" | 0.04% | **95.9%** |
| "Getty Images" | **21.1%** | 0.06% |
| "Featured image" | **43.7%** | 0.06% |
| "Image via" | **33.6%** | 0.04% |
| "21st Century Wire" | **3.4%** | 0.0% |

Almost every REAL article opens with a Reuters dateline like `WASHINGTON (Reuters) - ...`, and a large fraction of FAKE articles carry image-credit boilerplate like `Featured image via ...` or `Getty Images`. A classifier (even a trivial keyword match) could reach >95% accuracy by matching these tokens alone, without learning anything about the actual language of fake vs. real news. **These tokens are stripped before either model sees the text** — see `remove_leakage()` in [`src/preprocess.py`](src/preprocess.py), confirmed to be applied in both the baseline cleaning (`clean_text_for_baseline`) and the transformer cleaning (`clean_text_basic`).

**Preprocessing applied:**
- Leakage removal (above) — applied to all models.
- URL and HTML tag stripping, whitespace normalization — applied to all models.
- *Baseline only:* lowercasing, punctuation/digit removal, stopword removal (NLTK English stopword list), lemmatization (NLTK WordNetLemmatizer), tokenization. Transformers are not given this treatment because subword tokenizers are designed for natural, cased text and benefit from punctuation/case as signal.

**Known issues / limitations of the dataset:**
- Time period bias: articles are from 2016-2017, heavily US politics-focused (Trump, Clinton, Obama dominate top words in both classes — see `results/plots/top_words_by_class.png`). The model is unlikely to generalize well to other topics, languages, or more recent events/entities.
- Source-pool bias: REAL = Reuters only (one outlet, one writing style); FAKE = a mix of unreliable sites. The classifier may partly be learning "Reuters house style" vs. "not Reuters style" rather than a general notion of factuality.
- No satire/opinion-piece distinction; FAKE here means "from a site PolitiFact flagged as unreliable," not a per-article fact-check.

## 4. System Architecture

### End-to-end workflow

```
download_data.py --> data/raw/{train,validation,test}.csv
        |
        v
    eda.py  -->  results/plots/{class_distribution, text_length_distribution, top_words_by_class}.png
        |         results/eda_findings.txt
        v
preprocess.py (shared cleaning: leakage removal, URL/HTML strip,
               + baseline-only: lowercase/stopwords/lemmatize)
        |
        +------------------------+------------------------------+
        v                        v
train_baseline.py         train_transformer.py
(TF-IDF + LogReg/NB)       (fine-tune DistilBERT on
 -> models/*.joblib         balanced 6k-row subsample)
                            -> models/distilbert_fake_news/final
        |                        |
        +------------------------+
                      v
                evaluate.py
      (same test.csv for all models: accuracy, precision,
       recall, F1, ROC-AUC, confusion matrix, error analysis)
                      v
        results/metrics.json, comparison_table.csv,
        plots/confusion_matrix_*.png, plots/roc_curve_comparison.png,
        error_analysis.json
                      v
                predict.py  (loads best model)
                      v
                app/app.py  (Streamlit UI: paste text -> label + confidence)
```

### Mermaid flowchart

```mermaid
flowchart TD
    A[download_data.py] --> B[data/raw/*.csv]
    B --> C[eda.py]
    C --> D[results/plots + eda_findings.txt]
    B --> E[preprocess.py]
    E --> F[train_baseline.py<br/>TF-IDF + LogReg/NB]
    E --> G[train_transformer.py<br/>DistilBERT fine-tune]
    F --> H[models/*.joblib]
    G --> I[models/distilbert_fake_news/final]
    H --> J[evaluate.py]
    I --> J
    J --> K[results/metrics.json<br/>comparison_table.csv<br/>plots, error_analysis.json]
    H --> L[predict.py]
    I --> L
    L --> M[app/app.py Streamlit UI]
```

### ASCII diagram

```
 raw CSV (title, text, label)
        |
        v
 +--------------+     +-----------------+     +------------------------+
 | preprocess.py| --> | train_baseline  | --> | models/logreg,nb,tfidf |
 | (clean text) |     +-----------------+     +------------------------+
 |              |                                        |
 |              |     +-----------------+     +------------------------+
 +--------------+ --> | train_transformer| --> | models/distilbert/.../|
                       +-----------------+     +------------------------+
                                                         |
                                                         v
                                                   evaluate.py
                                                         |
                                                         v
                                        results/ (metrics, plots, tables)
                                                         |
                                                         v
                                              predict.py --> app.py (Streamlit)
```

## 5. Methodology

### 5.1 Environment
Python 3.12.10, dependencies pinned in `requirements.txt`, installed into a local venv (`venv/`). Torch installed as the CPU-only build (`torch==2.3.1+cpu`) via the PyTorch CPU wheel index, since this project is designed to run on a normal laptop with no GPU. All random seeds (`random`, `numpy`, `torch`, Hugging Face `set_seed`) fixed to **42** throughout.

### 5.2 Data loading & EDA
`datasets.load_dataset("GonzaloA/fake_news")`, saved locally as CSV via `src/download_data.py`. `src/eda.py` computes class distribution, word-count distribution, top words per class, and explicitly scans for label-leaking substrings (see Section 3). This EDA step is what *drove* the design of `remove_leakage()` in `preprocess.py` — the leakage patterns were not assumed, they were measured.

### 5.3 Preprocessing
Two-tier cleaning in `src/preprocess.py`:
1. `clean_text_basic()` — leakage-token removal (regex), URL/HTML stripping, whitespace normalization. Applied to input for **both** models, since both would otherwise exploit the same shortcut.
2. `clean_text_for_baseline()` — adds lowercasing, punctuation/digit stripping, NLTK stopword removal, and WordNet lemmatization, on top of (1). Used **only** for the TF-IDF baseline. The transformer receives text cleaned only by (1), since DistilBERT's WordPiece tokenizer is designed to handle casing and punctuation itself, and stripping it would throw away information a transformer can otherwise use.

### 5.4 Baseline: TF-IDF + Logistic Regression (+ Naive Bayes)
`src/features.py` builds a `TfidfVectorizer(max_features=20000, ngram_range=(1,2), min_df=3, sublinear_tf=True)`. Unigrams+bigrams were chosen to capture short multi-word patterns (e.g. "fake news", "breaking news") beyond single words; `max_features=20000` keeps the model lightweight; `min_df=3` discards rare noise tokens/typos.

Two classifiers trained on the same TF-IDF matrix (`src/train_baseline.py`):
- `LogisticRegression(max_iter=1000, C=1.0, random_state=42)` — the primary baseline.
- `MultinomialNB()` — a second, even-cheaper baseline for comparison, since Naive Bayes is a classic, fast text-classification method and a useful lower bound.

Trained on the full 24,353-row train split, evaluated on the 8,117-row validation split for model selection (see `results/baseline_val_report.json`), then both are scored on the test split in `evaluate.py` for the final comparison.

### 5.5 Transformer: fine-tuned DistilBERT
`src/train_transformer.py` fine-tunes `distilbert-base-uncased` (Hugging Face `transformers`) via the `Trainer` API.

**CPU-feasibility decisions (explicitly required by the project constraints):**
- **Subsample size:** 6,000 training rows (class-balanced, 3,000 FAKE + 3,000 REAL) and 1,200 validation rows (balanced), sampled with `random_state=42` from the full train/validation splits. The full 24k-row train split was judged too slow to fine-tune on CPU within a practical lab timeframe; this was confirmed empirically with a timing smoke test (see below) rather than assumed.
- **Max sequence length:** 128 tokens (articles are truncated after title + ~first paragraph). Chosen after a timing test showed ~6 samples/sec throughput at length 128 vs. roughly half that at length 256, and the dataset's leakage/topic signal is concentrated early in each article.
- **Epochs:** 2 (further epochs risk overfitting on only 6,000 examples with a 66M-parameter model).
- **Batch size:** 16 (per-device, CPU).
- Evaluation during training uses `eval_strategy="epoch"`; intermediate checkpoints are **not** saved (`save_strategy="no"`) to avoid unnecessary disk I/O and a Windows file-locking issue encountered during development (see Decision Log) — only the final model is saved, to `models/distilbert_fake_news/final/`.

**Final evaluation**, however, uses the complete 8,117-row test set (inference-only, which is far cheaper than training), so the transformer is compared against the baseline on an identical, full-size held-out set.

### 5.6 Evaluation
`src/evaluate.py` loads all trained models and scores them on the same `data/raw/test.csv` (8,117 rows): accuracy, precision, recall, F1, ROC-AUC, and a confusion matrix per model, plus an overlaid ROC curve comparison plot. It also samples misclassified test examples per model for qualitative error analysis (`results/error_analysis.json`).

## 6. Implementation Notes

| File | Purpose |
|---|---|
| `src/download_data.py` | Downloads `GonzaloA/fake_news` from the HF Hub, saves `data/raw/{train,validation,test}.csv` + `dataset_info.txt`. |
| `src/eda.py` | Class distribution, length distribution, top-words-per-class, and the label-leakage scan. Writes `results/plots/*.png` and `results/eda_findings.txt`. |
| `src/preprocess.py` | `remove_leakage()`, `clean_text_basic()` (both models), `clean_text_for_baseline()` (baseline only), `build_content_column()` (title+text concatenation). |
| `src/features.py` | `build_tfidf_vectorizer()` — a single place to configure TF-IDF hyperparameters. |
| `src/train_baseline.py` | Cleans all three splits, fits TF-IDF on train, trains LogisticRegression + MultinomialNB, saves models to `models/`, saves a 500-row cleaned sample of each split to `data/processed/` for inspection/reproducibility, writes `results/baseline_val_report.json`. |
| `src/train_transformer.py` | Loads a class-balanced subsample, tokenizes with `DistilBertTokenizerFast`, fine-tunes via Hugging Face `Trainer`, saves the final model to `models/distilbert_fake_news/final/` and `results/transformer_val_report.json`. |
| `src/evaluate.py` | Loads every trained model, scores all of them on the **same** `data/raw/test.csv`, saves `results/metrics.json`, `results/comparison_table.csv`, per-model confusion matrix plots, an overlaid ROC curve plot, and `results/error_analysis.json` (sampled misclassified examples per model). |
| `src/predict.py` | `BaselinePredictor` / `TransformerPredictor` classes + `load_best_predictor()`, which prefers the transformer if trained, else falls back to the baseline. This is the single interface the app and tests use, so swapping which model is "live" never requires touching app code. |
| `app/app.py` | Streamlit UI: text box -> `predict()` -> label + confidence, with a progress bar and a dataset-bias caveat shown to the user. |
| `tests/test_preprocess.py` | Unit tests for leakage removal, basic cleaning, baseline cleaning, content-column building. |
| `tests/test_predict.py` | Unit tests for the baseline predictor's output shape, determinism, and edge cases (empty string). |

### Challenges faced and how they were solved
- **Label leakage.** Initial EDA showed the raw dataset would let any model cheat via source-tag tokens (`Reuters`, `Getty Images`, etc. — see Section 3). Solved by writing an explicit `remove_leakage()` step, verified empirically against the EDA leakage scan, and applied before *both* models see text.
- **Windows file-lock error when saving transformer checkpoints.** `Trainer`'s default `save_strategy="epoch"` intermittently failed with `SafetensorError: Access is denied (os error 5)` on Windows (a transient file-lock, likely antivirus-related) when checkpointing mid-training. Solved by setting `save_strategy="no"` during training and only saving the model once, explicitly, after training completes.
- **CPU training speed.** A full-size transformer fine-tune (24k rows, long sequences) was estimated to take well over an hour. Ran a timing smoke test first (32-64 examples, 1 epoch) to measure real throughput (~6 samples/sec at `max_length=128`) *before* committing to hyperparameters, rather than guessing — this directly set the subsample size (6,000) and sequence length (128) documented in Section 5.5.

## 7. Results

All numbers below are from `results/metrics.json` / `results/comparison_table.csv`, produced by `src/evaluate.py` on the **full, untouched 8,117-row test split** — the same held-out set for every model, none of it used for training or hyperparameter selection.

### 7.1 Comparison table

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (TF-IDF) | 0.9676 | 0.9726 | 0.9666 | 0.9696 | 0.9906 |
| Naive Bayes (TF-IDF) | 0.9370 | 0.9527 | 0.9283 | 0.9403 | 0.9757 |
| **DistilBERT (fine-tuned)** | **0.9816** | **0.9934** | 0.9721 | **0.9826** | **0.9991** |

**DistilBERT wins on every metric** (accuracy, precision, recall, F1, ROC-AUC). It cuts the test-set error rate roughly in half versus the Logistic Regression baseline (1.84% vs 3.24% misclassified — see Section 7.3), at the cost of much longer training time (the baseline trains in seconds; DistilBERT took 3,749 seconds ≈ 62.5 minutes to fine-tune on this CPU, even on a 6,000-row subsample) and higher inference cost per article.

Naive Bayes is clearly the weakest model (93.7% accuracy), consistent with its simplifying conditional-independence assumption — useful as a cheap sanity-check baseline but not competitive here.

### 7.2 Plots (all in `results/plots/`)
- `class_distribution.png` — class balance (REAL 54%, FAKE 46%).
- `text_length_distribution.png` — article length by class.
- `top_words_by_class.png` — most frequent content words per class (both dominated by "trump", reflecting the 2016-2017 US political news time period).
- `confusion_matrix_logistic_regression.png`, `confusion_matrix_naive_bayes.png`, `confusion_matrix_distilbert.png` — per-model confusion matrices.
- `roc_curve_comparison.png` — overlaid ROC curves; DistilBERT's curve dominates (AUC 0.999) over Logistic Regression (0.991) and Naive Bayes (0.976).

**Confusion matrices (rows=actual, cols=predicted; order FAKE, REAL):**

| Model | TN (FAKE→FAKE) | FP (FAKE→REAL) | FN (REAL→FAKE) | TP (REAL→REAL) |
|---|---|---|---|---|
| Logistic Regression | 3664 | 118 | 145 | 4190 |
| Naive Bayes | 3582 | 200 | 311 | 4024 |
| DistilBERT | 3754 | 28 | 121 | 4214 |

DistilBERT's biggest win over the baseline is on **false positives** (FAKE articles wrongly labeled REAL): 28 vs. 118 for Logistic Regression — a 4x reduction. This matters in practice: a false positive (letting fake news through as "real") is typically the more costly error for a content-moderation use case.

### 7.3 Error analysis

| Model | Misclassified | Error rate |
|---|---|---|
| Logistic Regression | 263 / 8117 | 3.24% |
| DistilBERT | 149 / 8117 | 1.84% |

Ten misclassified examples per model are saved in `results/error_analysis.json`. Representative patterns observed (manually reviewed):

1. **REAL articles mislabeled as FAKE when they are opinion/analysis pieces rather than straight wire reports** — e.g. a REAL-labeled article titled "things to know about Prague" beginning with a meandering philosophical riff about "the meaning of life" was misclassified by Logistic Regression (confidence 0.907 for FAKE). These don't read like typical Reuters hard-news style, so the baseline's heavy reliance on surface lexical cues (and the fact that nearly all REAL training examples ARE Reuters-style wire reports) leads it astray.
2. **FAKE articles mislabeled as REAL when they report real events in a relatively neutral, factual tone** — e.g. "BOOM! Trump Drains Obama Swamp..." has a tabloid headline but a text body that reads like straight reporting; the model latched onto the body style and missed the sensationalist framing, predicting REAL with 0.70 confidence.
3. **DistilBERT's remaining errors cluster on genuinely ambiguous, hyperpartisan-but-factual content** — e.g. an article about "SNL's solution to democrat election denial" blends real entertainment-news content with political commentary; DistilBERT predicted FAKE with high confidence (0.99) for a REAL-labeled article, suggesting the sensational headline style dominates its decision even though the source was genuine.
4. A recurring theme across both models: **headline-body mismatch**. Several misclassified articles have clickbait-style headlines (ALL CAPS, exclamation points, "BOOM!", "percentfedupcom") paired with comparatively neutral body text, or vice versa — these confuse both models because the training signal (Reuters style vs. not) doesn't cleanly separate "sensational headline" from "unreliable content."

**Takeaway:** both models are mostly learning *writing style/outlet conventions* (hard-news wire-service tone vs. partisan commentary tone) rather than a deep semantic notion of factual accuracy — which is an expected and documented limitation of this dataset (see Section 3, "Known issues").

**Qualitative out-of-domain check** (run via `python src/predict.py`, not part of the test set): the sentence *"Scientists confirm the earth revolves around the sun after decades of study"* — an obviously true, neutral statement — was classified **FAKE with 80.05% confidence** by the fine-tuned DistilBERT model. This is a real, reproducible result, not a hypothetical, and it is strong direct evidence for the domain-bias limitation noted in Section 3: the model was fine-tuned exclusively on 2016-2017 US political news from Reuters (REAL) vs. partisan political sites (FAKE), so it has no real notion of scientific factual accuracy — it has learned "does this sound like Reuters political wire copy," and a short, topic-less science sentence does not, so it defaults toward FAKE. This is an important caveat for anyone using the deployed app outside the political-news domain it was trained on, and is surfaced to users directly in the Streamlit app's caption text.

### 7.4 Which model was used for the app
DistilBERT is the better model on every metric and is what `src/predict.py`'s `load_best_predictor()` loads by default (it prefers the transformer when available, falling back to the Logistic Regression baseline otherwise).

## 8. How to Run

_Filled in and tested at the end._

## 9. Future Work

- **Train the transformer on the full training set** (24,353 rows) rather than a 6,000-row subsample, and/or for more epochs, using a GPU — would likely close any remaining gap with the baseline and test whether the extra context from longer sequences (>128 tokens) helps.
- **Cross-domain evaluation**: test the trained models on a *different* fake-news dataset (different outlets, time period, topics) to measure how much of the accuracy is genuine language-of-deception signal vs. dataset/outlet-specific style.
- **Explainability**: add LIME/SHAP or attention visualization to the Streamlit app so a user can see *why* an article was flagged, not just the label.
- **Multi-class / fine-grained labels**: distinguish satire, opinion, and fabricated news rather than a single FAKE bucket.
- **Ensembling** the TF-IDF and DistilBERT predictions (e.g., simple probability averaging) to see if it outperforms either model alone.

## 10. References

**Dataset:**
- ISOT Fake News Dataset, University of Victoria ISOT Research Lab. https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/
- `GonzaloA/fake_news` (Hugging Face Hub mirror used in this project). https://huggingface.co/datasets/GonzaloA/fake_news

**Libraries:**
- Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 2011. https://scikit-learn.org/
- Wolf et al., *Transformers: State-of-the-Art Natural Language Processing*, EMNLP 2020. https://huggingface.co/docs/transformers/
- Sanh et al., *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*, 2019. https://arxiv.org/abs/1910.01108
- Bird, Klein & Loper, *Natural Language Processing with Python* (NLTK). https://www.nltk.org/
- Lhoest et al., *Datasets: A Community Library for Natural Language Processing*, 2021. https://huggingface.co/docs/datasets/
- Streamlit documentation. https://docs.streamlit.io/
- PyTorch documentation. https://pytorch.org/docs/

**Background reading:**
- Shu et al., *Fake News Detection on Social Media: A Data Mining Perspective*, ACM SIGKDD Explorations 2017. https://arxiv.org/abs/1708.01967
- Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, 2018. https://arxiv.org/abs/1810.04805

## 11. Decision Log

- **2026-10-06**: Project started. Chose Python 3.12 (available on machine) with a dedicated venv. Chose `nltk` over `spaCy` for stopword removal/lemmatization in the baseline preprocessing because it's lighter-weight and sufficient for TF-IDF features. Chose DistilBERT (`distilbert-base-uncased`) over full BERT for the transformer model to keep CPU fine-tuning time reasonable, per project constraints.
- **2026-10-06**: Installing dependencies via pip with pinned versions into a local venv (`venv/`), CPU-only (no CUDA), consistent with "must run on a normal laptop" constraint.
- **2026-10-06**: Selected `GonzaloA/fake_news` (HF mirror of the ISOT Fake News Dataset) after comparing several candidate HF datasets, specifically because it is the classic ISOT/Reuters dataset known to contain the "Reuters dateline" leakage pattern — a good pedagogical case for the required leakage-removal step, confirmed empirically via `src/eda.py` (96.6% of REAL articles contain "Reuters").
- **2026-10-06**: Logistic Regression baseline scored 96.76% test accuracy even after leakage removal — initially worth double-checking this wasn't residual leakage, but the EDA scan confirmed the leakage tokens are gone post-cleaning; the baseline is apparently also picking up genuine Reuters-house-style vs. partisan-outlet-style differences (sentence structure, neutral tone, dateline-less formatting), which is a legitimate (if narrow) signal, not a bug.
- **2026-10-06**: Hit a Windows-specific `SafetensorError: Access is denied (os error 5)` when `Trainer` tried to save an intermediate checkpoint mid-training (`save_strategy="epoch"`). Fixed by switching to `save_strategy="no"` and saving only the final model once, after training — see Section 6.
- **2026-10-06**: Ran a timing smoke test (32-64 examples, 1 epoch) before committing to full transformer training hyperparameters, since a naive full-dataset fine-tune was estimated to take well over an hour on this CPU. Measured ~6 samples/sec at `max_length=128`, batch 16, which set the final subsample size (6,000 train / 1,200 val) and sequence length. Actual full training run took 3,749s (~62.5 min) — longer than the initial ~35 min estimate, because the machine was also running other background processes (antivirus/system load caused step time to fluctuate between 2.6s and 23s during the run) — documented honestly here rather than silently reporting a lower number.
- **2026-10-06**: Final evaluation on the full 8,117-row test set: DistilBERT (98.16% accuracy, 0.9826 F1) beat Logistic Regression (96.76%, 0.9696) and Naive Bayes (93.70%, 0.9403) on every metric. See Section 7 for full results and error analysis.
- **2026-10-06**: No browser/computer-use tool was available in this build environment, so app screenshots for `docs/` could not be auto-captured — left as a manual step for the user before their presentation (see `docs/README.md`).
