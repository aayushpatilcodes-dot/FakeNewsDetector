"""
Streamlit web app for the Fake News Detector.

Run with:
    streamlit run app/app.py
"""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from predict import load_best_predictor  # noqa: E402

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")


@st.cache_resource
def get_predictor():
    return load_best_predictor()


st.title("📰 Fake News Detector")
st.caption(
    "Paste a news headline or article below. The model predicts whether it reads "
    "as REAL or FAKE news, with a confidence score."
)

predictor = get_predictor()
st.sidebar.markdown("### Model in use")
st.sidebar.write(predictor.name)
st.sidebar.markdown(
    "---\nBuilt for a college NLP project comparing a TF-IDF + Logistic "
    "Regression baseline against a fine-tuned DistilBERT model. "
    "See `PROJECT_DOCUMENTATION.md` for methodology and results."
)

text = st.text_area(
    "News text",
    height=220,
    placeholder="Paste a headline, lead paragraph, or full article text here...",
)

if st.button("Analyze", type="primary"):
    if not text.strip():
        st.warning("Please paste some text first.")
    else:
        with st.spinner("Analyzing..."):
            label, confidence = predictor.predict(text)

        if label == "REAL":
            st.success(f"✅ Predicted: **REAL** ({confidence:.1%} confidence)")
        else:
            st.error(f"🚩 Predicted: **FAKE** ({confidence:.1%} confidence)")

        st.progress(confidence)
        st.caption(
            "Note: this model was trained on a specific dataset (ISOT fake/real "
            "news, 2016-2017 era US political/world news) and may not generalize "
            "well to other topics, time periods, or satire."
        )
