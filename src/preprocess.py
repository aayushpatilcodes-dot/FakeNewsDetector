"""
Text cleaning and preprocessing utilities shared across the baseline and
transformer pipelines.

Two levels of cleaning are provided:
  - clean_text_basic(): leakage removal + whitespace/HTML normalization.
    Used for BOTH the TF-IDF baseline and the transformer, since leaking
    tokens (e.g. "(Reuters)") must be removed before either model sees the
    text, otherwise the task becomes trivial string matching instead of
    genuine fake-vs-real language modelling.
  - clean_text_for_baseline(): basic cleaning + lowercasing + punctuation/
    digit stripping + stopword removal + lemmatization. Used ONLY for the
    TF-IDF + Logistic Regression baseline, since bag-of-words models benefit
    from vocabulary reduction, whereas transformers expect natural text and
    have their own subword tokenizer.
"""
import re
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

_STOPWORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()

# Patterns that leak the label directly and were found during EDA (see
# PROJECT_DOCUMENTATION.md, Section 3 "Leakage found"):
#   - Real (Reuters) articles almost always open with a dateline like
#     "WASHINGTON (Reuters) - " or "(Reuters) - ".
#   - Some fake articles carry boilerplate like "Featured image via ..." or
#     social-media call-to-action text that never appears in real articles.
_LEAKAGE_PATTERNS = [
    r"^[A-Z][A-Za-z.,\s]{0,40}\(Reuters\)\s*-\s*",  # "WASHINGTON (Reuters) - "
    r"\(Reuters\)",
    r"Reuters",
    r"Featured image via [^\n.]*\.?",
    r"Image via [^\n.]*\.?",
    r"Getty Images",
    r"21st Century Wire",
]
_LEAKAGE_RE = re.compile("|".join(_LEAKAGE_PATTERNS))

_URL_RE = re.compile(r"http\S+|www\.\S+")
_HTML_RE = re.compile(r"<.*?>")
_WHITESPACE_RE = re.compile(r"\s+")


def remove_leakage(text: str) -> str:
    """Strip known label-leaking tokens (source tags, dateline boilerplate)."""
    return _LEAKAGE_RE.sub(" ", text)


def clean_text_basic(text: str) -> str:
    """Light cleaning applied before BOTH models see the text."""
    if not isinstance(text, str):
        return ""
    text = remove_leakage(text)
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def clean_text_for_baseline(text: str) -> str:
    """Full cleaning pipeline for the TF-IDF baseline: lowercase, strip
    punctuation/digits, tokenize, remove stopwords, lemmatize."""
    text = clean_text_basic(text)
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()

    tokens = word_tokenize(text)
    tokens = [
        _LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok not in _STOPWORDS and len(tok) > 1
    ]
    return " ".join(tokens)


def build_content_column(df, title_col="title", text_col="text"):
    """Combine title + body into a single content string."""
    title = df[title_col].fillna("") if title_col in df.columns else ""
    body = df[text_col].fillna("") if text_col in df.columns else df.iloc[:, 0].fillna("")
    return (title + ". " + body).str.strip() if title_col in df.columns else body
