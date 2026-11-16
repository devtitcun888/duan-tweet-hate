"""Model zoo and leak-free Pipeline builders for hate-speech classification (P5.2)."""

from __future__ import annotations

from typing import Any

import numpy as np
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

from src.features import build_vectorizer
from src.utils import load_config

MODEL_ZOO: dict[str, str] = {
    "lr": "Logistic Regression",
    "nb": "Complement Naive Bayes",
    "svm_linear": "Linear SVM (Calibrated)",
    "svm_rbf": "RBF-Kernel SVM",
    "dt": "Decision Tree",
    "rf": "Random Forest",
    "knn": "K-Nearest Neighbors (Cosine)",
    "xgb": "XGBoost",
    "lgbm": "LightGBM",
}


def build_estimator(
    model_key: str,
    balance: str = "class_weight",
    pos_weight_ratio: float = 13.3,
    seed: int = 42,
):
    """Instantiate the classifier for `model_key` with appropriate class-weighting."""
    use_cw = balance == "class_weight"
    cw = "balanced" if use_cw else None

    if model_key == "lr":
        return LogisticRegression(
            C=1.0,
            max_iter=2000,
            solver="liblinear",
            class_weight=cw,
            random_state=seed,
        )
    elif model_key == "nb":
        return ComplementNB(alpha=0.5)
    elif model_key == "svm_linear":
        base_svm = LinearSVC(C=1.0, class_weight=cw, random_state=seed, max_iter=3000)
        return CalibratedClassifierCV(estimator=base_svm, cv=3)
    elif model_key == "svm_rbf":
        return SVC(
            kernel="rbf",
            C=1.0,
            class_weight=cw,
            probability=False,
            random_state=seed,
        )
    elif model_key == "dt":
        return DecisionTreeClassifier(
            max_depth=30,
            class_weight=cw,
            random_state=seed,
        )
    elif model_key == "rf":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=40,
            class_weight=cw,
            n_jobs=-1,
            random_state=seed,
        )
    elif model_key == "knn":
        return KNeighborsClassifier(
            n_neighbors=5,
            metric="cosine",
            weights="distance" if use_cw else "uniform",
            n_jobs=-1,
        )
    elif model_key == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=(pos_weight_ratio if use_cw else 1.0),
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=seed,
        )
    elif model_key == "lgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            num_leaves=31,
            learning_rate=0.1,
            class_weight=cw,
            n_jobs=-1,
            random_state=seed,
            verbose=-1,
        )
    else:
        raise ValueError(f"Unknown model_key: {model_key!r}. Valid keys: {list(MODEL_ZOO)}")


def build_sampler(balance: str, seed: int = 42):
    """Build an imbalanced-learn sampler when resampling is selected."""
    if balance in ("none", "class_weight", None):
        return None
    elif balance == "ros":
        return RandomOverSampler(random_state=seed)
    elif balance == "smote":
        return SMOTE(k_neighbors=5, random_state=seed)
    elif balance == "rus":
        return RandomUnderSampler(random_state=seed)
    elif balance == "smote_tomek":
        return SMOTETomek(random_state=seed)
    else:
        raise ValueError(f"Unknown balance strategy: {balance!r}")


def build_pipeline(
    model_key: str,
    vectorizer_kind: str = "word_tfidf",
    balance: str = "class_weight",
    pos_weight_ratio: float = 13.3,
    cfg: dict[str, Any] | None = None,
    seed: int = 42,
):
    """Construct a complete leak-free Pipeline(vectorizer -> [sampler] -> clf)."""
    if cfg is None:
        cfg = load_config()
    vec = build_vectorizer(vectorizer_kind, cfg=cfg)
    clf = build_estimator(
        model_key=model_key,
        balance=balance,
        pos_weight_ratio=pos_weight_ratio,
        seed=seed,
    )
    sampler = build_sampler(balance, seed=seed)

    if sampler is not None:
        return ImbPipeline(
            [
                ("tfidf", vec),
                ("sampler", sampler),
                ("clf", clf),
            ]
        )
    return SkPipeline(
        [
            ("tfidf", vec),
            ("clf", clf),
        ]
    )


def get_model_scores(pipeline, X) -> np.ndarray:
    """Return continuous positive-class scores in [0, 1] for any fitted pipeline."""
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(X)
        return np.asarray(probs[:, 1], dtype=float)
    elif hasattr(pipeline, "decision_function"):
        dec = np.asarray(pipeline.decision_function(X), dtype=float)
        # Sigmoid mapping so 0.0 -> 0.5 and values lie in (0, 1)
        return 1.0 / (1.0 + np.exp(-np.clip(dec, -30.0, 30.0)))
    else:
        preds = pipeline.predict(X)
        return np.asarray(preds, dtype=float)
