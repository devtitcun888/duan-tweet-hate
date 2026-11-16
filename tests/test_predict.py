"""Inference and saved model reproducibility tests (P6 & P10)."""

from pathlib import Path
import pytest

from src.models import build_pipeline
from src.predict import Predictor, load_model
from src.utils import get_project_root

MODELS_DIR = get_project_root() / "models"


def test_predictor_wrapper_handles_normal_and_empty_inputs():
    """Test Predictor wrapper on a small synthetic fit so it always runs even before P6."""
    texts = [
        "happy sunny day with friends love peace",
        "wonderful family dinner grateful",
        "enjoying music and art festival",
        "racist bigot hate speech against women",
        "sexist misogyny hate slur attack",
        "racist hate crime discrimination",
    ]
    labels = [0, 0, 0, 1, 1, 1]
    pipe = build_pipeline("lr", vectorizer_kind="word_tfidf", balance="class_weight")
    pipe.named_steps["tfidf"].set_params(min_df=1)
    pipe.fit(texts, labels)

    pred = Predictor(pipe, threshold=0.5)

    res_empty = pred.predict_one("")
    assert res_empty["label"] == 0
    assert res_empty["score"] == 0.0

    res_none = pred.predict_one(None)
    assert res_none["label"] == 0

    res_pos = pred.predict_one("racist sexist hate attack")
    assert res_pos["label"] in (0, 1)
    assert 0.0 <= res_pos["score"] <= 1.0
    assert isinstance(res_pos["top_terms"], list)

    batch_df = pred.predict_batch(["happy day", "racist sexist hate", ""])
    assert len(batch_df) == 3
    assert list(batch_df.columns) == [
        "text",
        "text_clean",
        "score",
        "threshold",
        "label",
        "label_name",
    ]


@pytest.mark.skipif(
    not (MODELS_DIR / "best_model.joblib").exists(),
    reason="best_model.joblib not yet generated (run P6 first).",
)
def test_saved_best_model_loads_and_predicts_deterministically():
    predictor = load_model()
    samples = [
        "Looking forward to a wonderful weekend with my family!",
        "These racist bigots keep spreading hatred and sexism against women",
        "",
    ]
    out1 = [predictor.predict_one(s) for s in samples]
    out2 = [predictor.predict_one(s) for s in samples]
    for a, b in zip(out1, out2):
        assert a["label"] == b["label"]
        assert abs(a["score"] - b["score"]) < 1e-9
    assert out1[0]["label"] == 0
    assert out1[2]["label"] == 0
