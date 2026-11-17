"""Model loading, single-tweet and batch inference, feature attribution, and Multilingual (EN/VI) routing (P10 & V7)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.clean import clean_text
from src.models import get_model_scores
from src.utils import get_project_root, mask_profanity
from src.vi_clean import vi_clean, vi_mask_profanity
from src.vi_data_io import has_vietnamese_accents

PROJECT_ROOT = get_project_root()

try:
    from langdetect import DetectorFactory, detect_langs

    DetectorFactory.seed = 0
except ImportError:
    detect_langs = None


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
    """Wrapper around a trained sklearn/imblearn Pipeline and tuned threshold (supports EN or VI)."""

    def __init__(self, pipeline: Any, threshold: float = 0.5, lang: str = "en"):
        self.pipeline = pipeline
        self.threshold = float(threshold)
        self.lang = lang
        self.vectorizer = pipeline.named_steps.get("tfidf")
        self.clf = pipeline.named_steps.get("clf")
        self._coef = _extract_linear_coef(self.clf)
        self._feature_names: np.ndarray | None = None
        if self.vectorizer is not None and hasattr(self.vectorizer, "get_feature_names_out"):
            try:
                self._feature_names = self.vectorizer.get_feature_names_out()
            except Exception:
                self._feature_names = None

    def _clean(self, raw_text: str) -> str:
        if self.lang == "vi":
            return vi_clean(raw_text, segment=True, expand_teencode=True)
        return clean_text(raw_text)

    def _mask(self, text: str) -> str:
        if self.lang == "vi":
            return vi_mask_profanity(text)
        return mask_profanity(text)

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
                    if term.startswith("char__"):
                        continue
                    if term.startswith("word__"):
                        term = term[len("word__") :]
                    contribs.append((self._mask(term), round(val, 4)))
            contribs.sort(key=lambda x: abs(x[1]), reverse=True)
            return contribs[:top_k]
        except Exception:
            return []

    def predict_one(self, text: str | None) -> dict[str, Any]:
        """Predict label, probability score, and top contributing terms for one text."""
        raw_truncated = str(text or "")[:1000]
        cleaned = self._clean(raw_truncated)
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
        """Predict labels and scores for a batch of texts (up to 20,000 rows)."""
        raw_list = [str(t if t is not None else "")[:1000] for t in texts]
        cleaned_list = [self._clean(t) for t in raw_list]

        scores = np.zeros(len(cleaned_list), dtype=float)
        non_empty_indices = [i for i, c in enumerate(cleaned_list) if c]
        if non_empty_indices:
            non_empty_texts = [cleaned_list[i] for i in non_empty_indices]
            pred_scores = get_model_scores(self.pipeline, non_empty_texts)
            for idx, sc in zip(non_empty_indices, pred_scores):
                scores[idx] = float(sc)

        labels = (scores >= self.threshold).astype(int)
        pos_name = (
            "Thù ghét / Xúc phạm (ViHSD Hate)"
            if self.lang == "vi"
            else "Phản cảm (Racist/Sexist)"
        )
        return pd.DataFrame(
            {
                "text": raw_list,
                "text_clean": cleaned_list,
                "score": np.round(scores, 4),
                "threshold": round(self.threshold, 4),
                "label": labels,
                "label_name": [pos_name if l == 1 else "Bình thường" for l in labels],
            }
        )


def _rebuild_model_fallback(lang: str, target_path: Path):
    """Self-healing fallback if joblib binary was pickled under a different Python/sklearn version on cloud deployment."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.svm import LinearSVC

    if lang == "vi":
        tr = pd.read_csv(PROJECT_ROOT / "data" / "splits_vi" / "train.csv")
        dv = pd.read_csv(PROJECT_ROOT / "data" / "splits_vi" / "dev.csv")
        df = pd.concat([tr, dv], ignore_index=True)
        sub = df[df["label_3"].isin(["CLEAN", "HATE"])].copy()
        X = sub["text_clean_seg"].fillna("").astype(str).to_numpy()
        y = (sub["label_3"] == "HATE").astype(int).to_numpy()
        vec = FeatureUnion(
            [
                ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, max_features=40000, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_df=0.95, max_features=50000, sublinear_tf=True)),
            ]
        )
        clf = CalibratedClassifierCV(estimator=LinearSVC(C=0.4, class_weight="balanced", random_state=42, max_iter=3000), cv=3)
        pipe = Pipeline([("tfidf", vec), ("clf", clf)])
        pipe.fit(X, y)
    else:
        tr = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "train.csv")
        va = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "val.csv")
        df = pd.concat([tr, va], ignore_index=True)
        X = df["text_clean"].fillna("").astype(str).to_numpy()
        y = df["label"].astype(int).to_numpy()
        vec = TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_df=0.9, max_features=50000, sublinear_tf=True)
        clf = CalibratedClassifierCV(
            estimator=LinearSVC(C=0.50191, loss="squared_hinge", class_weight="balanced", random_state=42, max_iter=3000),
            cv=3,
        )
        pipe = Pipeline([("tfidf", vec), ("clf", clf)])
        pipe.fit(X, y)

    try:
        joblib.dump(pipe, target_path, protocol=4)
    except Exception:
        pass
    return pipe


def load_model(
    path: str | Path = "models/best_model.joblib",
    thr_path: str | Path = "models/threshold.json",
    lang: str = "en",
) -> Predictor:
    """Load the saved best pipeline and threshold JSON into a Predictor instance."""
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    t = Path(thr_path)
    if not t.is_absolute():
        t = PROJECT_ROOT / t

    try:
        pipeline = joblib.load(p)
    except Exception:
        pipeline = _rebuild_model_fallback(lang=lang, target_path=p)

    threshold = 0.5
    if t.exists():
        with open(t, "r", encoding="utf-8") as f:
            thr_data = json.load(f)
            threshold = float(thr_data.get("best_threshold", 0.5))

    return Predictor(pipeline=pipeline, threshold=threshold, lang=lang)


def load_vi_model(
    path: str | Path = "models/vi/best_vi_bin_a.joblib",
    thr_path: str | Path = "models/vi/threshold_vi_bin_a.json",
) -> Predictor:
    """Load the trained Vietnamese best pipeline and threshold."""
    return load_model(path=path, thr_path=thr_path, lang="vi")


def predict_text(
    text: str,
    path: str | Path = "models/best_model.joblib",
    thr_path: str | Path = "models/threshold.json",
) -> dict[str, Any]:
    """Convenience function to load English model and predict a single text."""
    predictor = load_model(path=path, thr_path=thr_path, lang="en")
    return predictor.predict_one(text)


class MultiPredictor:
    """Multilingual (English & Vietnamese) hate-speech predictor with automatic language detection (V7)."""

    def __init__(
        self,
        en_path: str | Path = "models/best_model.joblib",
        en_thr_path: str | Path = "models/threshold.json",
        vi_path: str | Path = "models/vi/best_vi_bin_a.joblib",
        vi_thr_path: str | Path = "models/vi/threshold_vi_bin_a.json",
    ):
        self.en = load_model(path=en_path, thr_path=en_thr_path, lang="en")
        vi_full = PROJECT_ROOT / vi_path if not Path(vi_path).is_absolute() else Path(vi_path)
        if vi_full.exists():
            self.vi: Predictor | None = load_vi_model(path=vi_path, thr_path=vi_thr_path)
        else:
            self.vi = None

    def detect_lang(self, text: str | None) -> str:
        """Detect language according to V7 rules:
        1) Contains accented Vietnamese characters (ă â ê ô ơ ư đ + tone marks) -> 'vi'
        2) Otherwise, if >=3 words and >=10 alphabetic chars, run langdetect with threshold >= 0.80 ('vi' -> 'vi', 'en' -> 'en')
        3) Otherwise -> 'unknown' (short or unaccented texts are prone to misdetection)
        """
        if text is None:
            return "unknown"
        s = str(text).strip()
        if not s:
            return "unknown"

        # Rule 1: Accented Vietnamese characters -> 'vi'
        if has_vietnamese_accents(s):
            return "vi"

        # Strip URLs, mentions, numbers, punctuation
        s_clean = re.sub(r"https?://\S+|www\.\S+|@\w+|[^a-zA-Z\s]", " ", s)
        words = [w for w in s_clean.split() if len(w) >= 2]
        alpha_chars = sum(len(w) for w in words)

        # Too short without Vietnamese accents -> 'unknown'
        if len(words) < 3 or alpha_chars < 10:
            return "unknown"

        if detect_langs is None:
            return "unknown"

        try:
            langs = detect_langs(" ".join(words))
            if not langs:
                return "unknown"
            top = langs[0]
            if top.prob >= 0.80:
                if top.lang == "vi":
                    return "vi"
                if top.lang == "en":
                    return "en"
        except Exception:
            return "unknown"

        return "unknown"

    @staticmethod
    def _normalize_lang_choice(lang: str) -> str:
        l_low = str(lang or "auto").strip().lower()
        if l_low in ("en", "english", "tiếng anh"):
            return "en"
        if l_low in ("vi", "vietnamese", "tiếng việt", "tieng viet"):
            return "vi"
        return "auto"

    def predict(self, text: str | None, lang: str = "auto") -> dict[str, Any]:
        """Predict hate/offensive speech with manual language override or automatic detection."""
        s = str(text or "").strip()
        if not s:
            return {
                "status": "empty",
                "lang": "unknown",
                "model_used": "NA",
                "definition": "NA",
                "label": None,
                "score": 0.0,
                "threshold": None,
                "clean_text": "",
                "top_terms": [],
                "warning": "Văn bản rỗng. Vui lòng nhập nội dung cần phân tích.",
            }

        choice = self._normalize_lang_choice(lang)
        resolved_lang = self.detect_lang(s) if choice == "auto" else choice

        if resolved_lang == "unknown":
            return {
                "status": "unknown_lang",
                "lang": "unknown",
                "model_used": "NA",
                "definition": "NA",
                "label": None,
                "score": None,
                "threshold": None,
                "clean_text": "",
                "top_terms": [],
                "warning": (
                    "Không nhận diện đủ độ tin cậy ngôn ngữ (≥ 0.80) do văn bản quá ngắn hoặc là tiếng Việt không dấu. "
                    "Vui lòng chọn ngôn ngữ thủ công ('English' hoặc 'Tiếng Việt') để dự đoán chính xác."
                ),
            }

        if resolved_lang == "vi":
            if self.vi is None:
                raise RuntimeError("Vietnamese model (models/vi/best_vi_bin_a.joblib) is not loaded.")
            res = self.vi.predict_one(s)
            return {
                "status": "ok",
                "lang": "vi",
                "model_used": "Tiếng Việt — Linear SVM (ViHSD bin_A)",
                "definition": "Tiếng Việt: Thù ghét / Xúc phạm cá nhân hoặc nhóm (ViHSD Hate Speech)",
                **res,
                "warning": None,
            }

        res = self.en.predict_one(s)
        return {
            "status": "ok",
            "lang": "en",
            "model_used": "English — Linear SVM (Calibrated)",
            "definition": "English: Racist / Sexist Hate Speech (Phân biệt chủng tộc / giới tính)",
            **res,
            "warning": None,
        }

    def predict_batch(self, texts: list[str] | pd.Series, lang: str = "auto") -> pd.DataFrame:
        """Fast batch inference across English, Vietnamese, and Unknown rows."""
        raw_list = [str(t if t is not None else "")[:1000] for t in texts]
        choice = self._normalize_lang_choice(lang)
        langs = [self.detect_lang(t) if choice == "auto" else choice for t in raw_list]

        n = len(raw_list)
        clean_col = [""] * n
        score_col: list[float | None] = [None] * n
        thr_col: list[float | None] = [None] * n
        label_col: list[Any] = ["NA"] * n
        label_name_col = ["NA (Không rõ ngôn ngữ)"] * n
        model_used_col = ["NA"] * n

        en_idx = [i for i, l in enumerate(langs) if l == "en"]
        vi_idx = [i for i, l in enumerate(langs) if l == "vi"]

        if en_idx:
            en_df = self.en.predict_batch([raw_list[i] for i in en_idx])
            for pos, r_idx in enumerate(en_idx):
                clean_col[r_idx] = str(en_df["text_clean"].iloc[pos])
                score_col[r_idx] = float(en_df["score"].iloc[pos])
                thr_col[r_idx] = float(en_df["threshold"].iloc[pos])
                lbl_val = int(en_df["label"].iloc[pos])
                label_col[r_idx] = lbl_val
                label_name_col[r_idx] = (
                    "1: Racist/Sexist (EN)" if lbl_val == 1 else "0: Normal (EN)"
                )
                model_used_col[r_idx] = "EN: Linear SVM"

        if vi_idx and self.vi is not None:
            vi_df = self.vi.predict_batch([raw_list[i] for i in vi_idx])
            for pos, r_idx in enumerate(vi_idx):
                clean_col[r_idx] = str(vi_df["text_clean"].iloc[pos])
                score_col[r_idx] = float(vi_df["score"].iloc[pos])
                thr_col[r_idx] = float(vi_df["threshold"].iloc[pos])
                lbl_val = int(vi_df["label"].iloc[pos])
                label_col[r_idx] = lbl_val
                label_name_col[r_idx] = (
                    "1: Thù ghét/Xúc phạm (VI)" if lbl_val == 1 else "0: Bình thường (VI)"
                )
                model_used_col[r_idx] = "VI: Linear SVM (ViHSD)"

        return pd.DataFrame(
            {
                "text": raw_list,
                "lang": langs,
                "model_used": model_used_col,
                "text_clean": clean_col,
                "score": score_col,
                "threshold": thr_col,
                "label": label_col,
                "label_name": label_name_col,
            }
        )
