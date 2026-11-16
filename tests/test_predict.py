"""Inference and Multilingual (EN/VI) routing tests (P6, P10 & V7)."""

import time
import pytest

from src.models import build_pipeline
from src.predict import MultiPredictor, Predictor, load_model
from src.utils import get_project_root

MODELS_DIR = get_project_root() / "models"


def test_predictor_wrapper_handles_normal_and_empty_inputs():
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
    reason="best_model.joblib not yet generated.",
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


@pytest.mark.skipif(
    not (MODELS_DIR / "vi" / "best_vi_bin_a.joblib").exists(),
    reason="Vietnamese model not yet generated (run V3 first).",
)
def test_multilingual_predictor_language_detection_and_override():
    mp = MultiPredictor()

    # 1. Accented Vietnamese -> 'vi'
    r_vi = mp.predict("Đội tuyển bóng đá Việt Nam thi đấu rất nỗ lực và đoàn kết", lang="auto")
    assert r_vi["lang"] == "vi"
    assert r_vi["status"] == "ok"
    assert "Tiếng Việt" in r_vi["model_used"]

    # 2. English sentence -> 'en'
    r_en = mp.predict("Just finished reading an amazing book on machine learning with my friends", lang="auto")
    assert r_en["lang"] == "en"
    assert r_en["status"] == "ok"
    assert "English" in r_en["model_used"]

    # 3. Empty string -> no crash
    r_empty = mp.predict("", lang="auto")
    assert r_empty["status"] == "empty"
    assert r_empty["label"] is None

    # 4. Too short / ambiguous string -> 'unknown' warning when auto
    r_short = mp.predict("ok bro", lang="auto")
    assert r_short["lang"] == "unknown"
    assert r_short["label"] is None
    assert r_short["warning"] is not None

    # 5. Manual language choice overrides automatic detection
    r_override_vi = mp.predict("thang oc cho nay ngu dot va mat day kinh khung", lang="Tiếng Việt")
    assert r_override_vi["lang"] == "vi"
    assert r_override_vi["status"] == "ok"
    assert r_override_vi["label"] in (0, 1)

    r_override_en = mp.predict("ok bro", lang="English")
    assert r_override_en["lang"] == "en"
    assert r_override_en["status"] == "ok"

    # 6. Batch 1,000 mixed rows completes in < 15 seconds
    mixed_1k = (
        ["Just finished reading an amazing book on machine learning!"] * 400
        + ["Đội tuyển bóng đá Việt Nam chiến thắng tuyệt vời!"] * 400
        + ["Thằng óc chó này ngu si đần độn mất dạy"] * 150
        + ["ok"] * 50
    )
    t0 = time.perf_counter()
    df_out = mp.predict_batch(mixed_1k, lang="auto")
    elapsed = time.perf_counter() - t0
    assert len(df_out) == 1000
    assert "lang" in df_out.columns
    assert "model_used" in df_out.columns
    assert (df_out["label"] == "NA").sum() == 50
    assert elapsed < 15.0, f"Batch 1000 mixed rows took {elapsed:.2f}s >= 15s"
