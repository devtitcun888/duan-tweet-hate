"""Model loading, single-tweet and batch inference, and feature attribution (P10.1)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.clean import clean_text
from src.models import get_model_scores
from src.utils import get_project_root, mask_profanity

PROJECT_ROOT = get_project_root()


def _extract_linear_coef(clf) -> np.ndarray | None:
    """Extract 1D linear weight vector from classifier if available."""
    if hasattr(clf, "coef_"):
        coef = np.asarray(clf.coef_)
        return coef.ravel() if coef.ndim > 1 else coef
    if hasattr(clf, "calibrated_classifiers_") and clf.calibrated_classifiers_:
        coefs = []
        for cc in clf.calibrated_classifiers_:
            est = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
            if est is not None and hasattr(est, "coef_"):
                c = np.asarray(est.coef_)
                coefs.append(c.ravel() if c.ndim > 1 else c)
        if coefs:
            return np.mean(coefs, axis=0)
    if hasattr(clf, "feature_log_prob_"):
        flp = np.asarray(clf.feature_log_prob_)
        if flp.shape[0] >= 2:
            return flp[1] - flp[0]
    return None


class Predictor:
    """Wrapper around a trained sklearn/imblearn Pipeline and tuned threshold."""

    def __init__(self, pipeline: Any, threshold: float = 0.5):
        self.pipeline = pipeline
        self.threshold = float(threshold)
        self.vectorizer = pipeline.named_steps.get("tfidf")
        self.clf = pipeline.named_steps.get("clf")
        self._coef = _extract_linear_coef(self.clf)
        self._feature_names: np.ndarray | None = None
        if self.vectorizer is not None and hasattr(self.vectorizer, "get_feature_names_out"):
            try:
                self._feature_names = self.vectorizer.get_feature_names_out()
            except Exception:
                self._feature_names = None

    def _get_top_terms(self, cleaned: str, top_k: int = 8) -> list[tuple[str, float]]:
        """Compute top-k contributing features (tfidf_value * linear_weight) for a single cleaned text."""
        if not cleaned or self.vectorizer is None or self._coef is None or self._feature_names is None:
            return []
        try:
            vec = self.vectorizer.transform([cleaned])
            nonzero_idx = vec.nonzero()[1]
            if len(nonzero_idx) == 0:
                return []
            contribs = []
            for idx in nonzero_idx:
                if idx < len(self._coef) and idx < len(self._feature_names):
                    val = float(vec[0, idx]) * float(self._coef[idx])
                    term = str(self._feature_names[idx])
                    # Skip char n-grams when explaining word contributions if word+char union is used
                    if term.startswith("char__"):
                        continue
                    if term.startswith("word__"):
                        term = term[len("word__") :]
                    contribs.append((mask_profanity(term), round(val, 4)))
            contribs.sort(key=lambda x: abs(x[1]), reverse=True)
            return contribs[:top_k]
        except Exception:
            return []

    def predict_one(self, text: str | None) -> dict[str, Any]:
        """Predict label, probability score, and top contributing terms for one tweet."""
        raw_truncated = str(text or "")[:1000]
        cleaned = clean_text(raw_truncated)
        if not cleaned:
            return {
                "label": 0,
                "score": 0.0,
                "threshold": self.threshold,
                "clean_text": "",
                "top_terms": [],
            }

        score = float(get_model_scores(self.pipeline, [cleaned])[0])
        label = int(score >= self.threshold)
        top_terms = self._get_top_terms(cleaned, top_k=8)
        return {
            "label": label,
            "score": round(score, 4),
            "threshold": round(self.threshold, 4),
            "clean_text": cleaned,
            "top_terms": top_terms,
        }

    def predict_batch(self, texts: list[str] | pd.Series) -> pd.DataFrame:
        """Predict labels and scores for a batch of tweets (up to 20,000 rows)."""
        raw_list = [str(t if t is not None else "")[:1000] for t in texts]
        cleaned_list = [clean_text(t) for t in raw_list]

        scores = np.zeros(len(cleaned_list), dtype=float)
        non_empty_indices = [i for i, c in enumerate(cleaned_list) if c]
        if non_empty_indices:
            non_empty_texts = [cleaned_list[i] for i in non_empty_indices]
            pred_scores = get_model_scores(self.pipeline, non_empty_texts)
            for idx, sc in zip(non_empty_indices, pred_scores):
                scores[idx] = float(sc)

        labels = (scores >= self.threshold).astype(int)
        return pd.DataFrame(
            {
                "text": raw_list,
                "text_clean": cleaned_list,
                "score": np.round(scores, 4),
                "threshold": round(self.threshold, 4),
                "label": labels,
                "label_name": ["Phản cảm (Racist/Sexist)" if l == 1 else "Bình thường" for l in labels],
            }
        )


def load_model(
    path: str | Path = "models/best_model.joblib",
    thr_path: str | Path = "models/threshold.json",
) -> Predictor:
    """Load the saved best pipeline and threshold JSON into a Predictor instance."""
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    t = Path(thr_path)
    if not t.is_absolute():
        t = PROJECT_ROOT / t

    pipeline = joblib.load(p)
    threshold = 0.5
    if t.exists():
        with open(t, "r", encoding="utf-8") as f:
            thr_data = json.load(f)
            threshold = float(thr_data.get("best_threshold", 0.5))

    return Predictor(pipeline=pipeline, threshold=threshold)


def predict_text(
    text: str,
    path: str | Path = "models/best_model.joblib",
    thr_path: str | Path = "models/threshold.json",
) -> dict[str, Any]:
    """Convenience function to load model and predict a single text."""
    predictor = load_model(path=path, thr_path=thr_path)
    return predictor.predict_one(text)
