import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocess import (  # noqa: E402
    build_content_column,
    clean_text_basic,
    clean_text_for_baseline,
    remove_leakage,
)


def test_remove_leakage_strips_reuters_dateline():
    text = "WASHINGTON (Reuters) - The president spoke today about the economy."
    cleaned = remove_leakage(text)
    assert "Reuters" not in cleaned
    assert "WASHINGTON" not in cleaned


def test_remove_leakage_strips_fake_news_boilerplate():
    text = "Shocking news. Featured image via Getty Images of the event."
    cleaned = remove_leakage(text)
    assert "Featured image" not in cleaned
    assert "Getty Images" not in cleaned


def test_clean_text_basic_removes_urls_and_html():
    text = "Check this out <b>now</b> at http://example.com/fake-news"
    cleaned = clean_text_basic(text)
    assert "http" not in cleaned
    assert "<b>" not in cleaned


def test_clean_text_basic_handles_non_string():
    assert clean_text_basic(None) == ""
    assert clean_text_basic(float("nan")) == ""


def test_clean_text_for_baseline_lowercases_and_removes_stopwords():
    text = "The Quick Brown Fox jumps over the lazy dog."
    cleaned = clean_text_for_baseline(text)
    assert cleaned == cleaned.lower()
    assert "the" not in cleaned.split()


def test_clean_text_for_baseline_removes_punctuation_and_digits():
    text = "Breaking: 100 people saw this, incredible!!!"
    cleaned = clean_text_for_baseline(text)
    assert "!" not in cleaned
    assert "100" not in cleaned
    assert "," not in cleaned


def test_build_content_column_combines_title_and_text():
    df = pd.DataFrame({"title": ["Headline A"], "text": ["Body text here."]})
    content = build_content_column(df)
    assert "Headline A" in content.iloc[0]
    assert "Body text here." in content.iloc[0]


def test_build_content_column_handles_missing_title():
    df = pd.DataFrame({"text": ["Just body text."]})
    content = build_content_column(df)
    assert "Just body text." in content.iloc[0]
