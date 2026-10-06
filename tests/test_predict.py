import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from predict import BaselinePredictor, TransformerPredictor, load_best_predictor  # noqa: E402

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
TRANSFORMER_DIR = os.path.join(MODELS_DIR, "distilbert_fake_news", "final")


@pytest.fixture(scope="module")
def baseline_predictor():
    if not os.path.exists(os.path.join(MODELS_DIR, "logreg_baseline.joblib")):
        pytest.skip("Baseline model not trained yet (run src/train_baseline.py first)")
    return BaselinePredictor()


def test_predict_returns_valid_label_and_confidence(baseline_predictor):
    label, confidence = baseline_predictor.predict("The government announced a new policy today.")
    assert label in ("REAL", "FAKE")
    assert 0.0 <= confidence <= 1.0


def test_predict_confidence_is_at_least_half(baseline_predictor):
    # Since confidence is defined as the probability of the PREDICTED class,
    # it must be >= 0.5 for a binary classifier.
    _, confidence = baseline_predictor.predict("Some fairly neutral sounding news text.")
    assert confidence >= 0.5


def test_predict_handles_empty_string(baseline_predictor):
    label, confidence = baseline_predictor.predict("")
    assert label in ("REAL", "FAKE")


def test_predict_is_deterministic(baseline_predictor):
    text = "Local officials confirmed the results of the annual budget review."
    r1 = baseline_predictor.predict(text)
    r2 = baseline_predictor.predict(text)
    assert r1 == r2


@pytest.fixture(scope="module")
def transformer_predictor():
    if not os.path.exists(TRANSFORMER_DIR):
        pytest.skip("Transformer model not trained yet (run src/train_transformer.py first)")
    return TransformerPredictor()


def test_transformer_predict_returns_valid_label_and_confidence(transformer_predictor):
    label, confidence = transformer_predictor.predict(
        "The government announced a new economic policy on Tuesday."
    )
    assert label in ("REAL", "FAKE")
    assert 0.0 <= confidence <= 1.0


def test_load_best_predictor_prefers_transformer_when_available():
    predictor = load_best_predictor()
    if os.path.exists(TRANSFORMER_DIR):
        assert isinstance(predictor, TransformerPredictor)
    else:
        assert isinstance(predictor, BaselinePredictor)
