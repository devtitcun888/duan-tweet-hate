"""V3 Vietnamese Classical Modeling: bin_A, bin_B, multi3, Balancing Comparison, 8 Models x Vectorizers, Ablation Study, and Tuning (V3)."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src.evaluate import compute_metrics
from src.models import get_model_scores
from src.threshold import pick_threshold
from src.utils import get_project_root, set_seed
from src.vi_clean import restore_unaccented_vietnamese, strip_accents, vi_clean

PROJECT_ROOT = get_project_root()
VI_SPLITS_DIR = PROJECT_ROOT / "data" / "splits_vi"
VI_MODELS_DIR = PROJECT_ROOT / "models" / "vi"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
TAB_DIR = REPORTS_DIR / "tables"

VI_MODEL_ZOO = {
    "lr": "Logistic Regression",
    "svm_linear": "Linear SVM (Calibrated)",
    "nb": "Complement Naive Bayes",
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "rf": "Random Forest",
    "knn": "Cosine KNN",
    "dt": "Decision Tree",
}


def log_vi_experiment(
    task: str,
    model: str,
    vectorizer: str,
    balance: str,
    params: dict[str, Any] | str,
    dev_f1_primary: float,
    dev_pr_auc_or_acc: float,
    fit_seconds: float,
) -> None:
    """Append one Vietnamese experiment record to reports/vi_experiments.csv."""
    exp_path = REPORTS_DIR / "vi_experiments.csv"
    exp_path.parent.mkdir(parents=True, exist_ok=True)
    exists = exp_path.exists() and exp_path.stat().st_size > 0
    fieldnames = [
        "timestamp",
        "task",
        "model",
        "vectorizer",
        "balance",
        "params_json",
        "dev_primary_f1",
        "dev_pr_auc_or_acc",
        "fit_seconds",
    ]
    with open(exp_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "task": task,
                "model": model,
                "vectorizer": vectorizer,
                "balance": balance,
                "params_json": json.dumps(params, ensure_ascii=False) if isinstance(params, dict) else str(params),
                "dev_primary_f1": round(float(dev_f1_primary), 4),
                "dev_pr_auc_or_acc": round(float(dev_pr_auc_or_acc), 4),
                "fit_seconds": round(float(fit_seconds), 3),
            }
        )


def build_vi_vectorizer(kind: str = "word_tfidf"):
    """Build a Vietnamese TF-IDF vectorizer preserving '_' in compound words."""
    if kind == "word_tfidf":
        return TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9,
            max_features=45000,
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b",
        )
    elif kind == "char_tfidf":
        return TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=3,
            max_df=0.95,
            max_features=60000,
            sublinear_tf=True,
        )
    elif kind == "word+char":
        return FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=3,
                        max_df=0.9,
                        max_features=40000,
                        sublinear_tf=True,
                        token_pattern=r"(?u)\b\w+\b",
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(2, 5),
                        min_df=3,
                        max_df=0.95,
                        max_features=50000,
                        sublinear_tf=True,
                    ),
                ),
            ]
        )
    else:
        raise ValueError(f"Unknown vectorizer kind: {kind}")


def build_vi_estimator(model_key: str, balance: str = "class_weight", pos_weight: float = 7.8, seed: int = 42, is_multiclass: bool = False):
    use_cw = balance == "class_weight"
    cw = "balanced" if use_cw else None

    if model_key == "lr":
        return LogisticRegression(
            C=1.5,
            max_iter=2000,
            solver="lbfgs" if is_multiclass else "liblinear",
            class_weight=cw,
            random_state=seed,
        )
    elif model_key == "svm_linear":
        base_svm = LinearSVC(C=0.6, class_weight=cw, random_state=seed, max_iter=3000)
        return CalibratedClassifierCV(estimator=base_svm, cv=3)
    elif model_key == "nb":
        return ComplementNB(alpha=0.3)
    elif model_key == "dt":
        return DecisionTreeClassifier(max_depth=30, class_weight=cw, random_state=seed)
    elif model_key == "rf":
        return RandomForestClassifier(n_estimators=200, max_depth=40, class_weight=cw, n_jobs=-1, random_state=seed)
    elif model_key == "knn":
        return KNeighborsClassifier(n_neighbors=5, metric="cosine", weights="distance" if use_cw else "uniform", n_jobs=-1)
    elif model_key == "xgb":
        from xgboost import XGBClassifier

        if is_multiclass:
            return XGBClassifier(
                n_estimators=250,
                max_depth=6,
                learning_rate=0.1,
                tree_method="hist",
                n_jobs=-1,
                random_state=seed,
            )
        return XGBClassifier(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=(pos_weight if use_cw else 1.0),
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        )
    elif model_key == "lgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=250,
            max_depth=8,
            num_leaves=31,
            learning_rate=0.1,
            class_weight=cw,
            n_jobs=-1,
            random_state=seed,
            verbose=-1,
        )
    else:
        raise ValueError(f"Unknown model_key: {model_key}")


def prepare_task_split(df: pd.DataFrame, task: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_seg, X_raw, y) for task in {'bin_A', 'bin_B', 'multi3'}."""
    if task == "bin_A":
        # HATE (1) vs CLEAN (0), drop OFFENSIVE
        sub = df[df["label_3"].isin(["CLEAN", "HATE"])].copy()
        y = (sub["label_3"] == "HATE").astype(int).to_numpy()
    elif task == "bin_B":
        # OFFENSIVE + HATE (1) vs CLEAN (0)
        sub = df.copy()
        y = (sub["label_3"].isin(["OFFENSIVE", "HATE"])).astype(int).to_numpy()
    elif task == "multi3":
        # 0=CLEAN, 1=OFFENSIVE, 2=HATE
        sub = df.copy()
        y = sub["label_id"].astype(int).to_numpy()
    else:
        raise ValueError(f"Unknown task: {task}")

    X_seg = sub["text_clean_seg"].fillna("").astype(str).to_numpy()
    X_raw = sub["text_clean_raw"].fillna("").astype(str).to_numpy()
    return X_seg, X_raw, y


def run_v3_classical() -> dict[str, Any]:
    """Execute full V3 Vietnamese classical modeling, balancing comparison, ablation study, and tuning."""
    set_seed(42)
    VI_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(VI_SPLITS_DIR / "train.csv")
    dev_df = pd.read_csv(VI_SPLITS_DIR / "dev.csv")

    X_tr_seg_a, X_tr_raw_a, y_tr_a = prepare_task_split(train_df, "bin_A")
    X_dv_seg_a, X_dv_raw_a, y_dv_a = prepare_task_split(dev_df, "bin_A")
    pos_weight_a = round(float((y_tr_a == 0).sum() / max((y_tr_a == 1).sum(), 1)), 2)

    # =========================================================================
    # 1. Balancing comparison on bin_A (train -> dev)
    # =========================================================================
    bal_strategies = [
        ("none", "Không cân bằng (None)"),
        ("class_weight", "Trọng số lớp (class_weight='balanced')"),
        ("ros", "RandomOverSampler (ROS)"),
        ("smote", "SMOTE (k=5)"),
        ("rus", "RandomUnderSampler (RUS)"),
    ]
    bal_rows = []
    for b_key, b_desc in bal_strategies:
        vec = build_vi_vectorizer("word_tfidf")
        clf = LogisticRegression(
            C=1.5,
            max_iter=2000,
            solver="liblinear",
            class_weight="balanced" if b_key == "class_weight" else None,
            random_state=42,
        )
        if b_key == "ros":
            pipe = ImbPipeline([("tfidf", vec), ("sampler", RandomOverSampler(random_state=42)), ("clf", clf)])
        elif b_key == "smote":
            pipe = ImbPipeline([("tfidf", vec), ("sampler", SMOTE(k_neighbors=5, random_state=42)), ("clf", clf)])
        elif b_key == "rus":
            pipe = ImbPipeline([("tfidf", vec), ("sampler", RandomUnderSampler(random_state=42)), ("clf", clf)])
        else:
            pipe = Pipeline([("tfidf", vec), ("clf", clf)])

        t0 = time.perf_counter()
        pipe.fit(X_tr_seg_a, y_tr_a)
        fit_sec = time.perf_counter() - t0
        scores = get_model_scores(pipe, X_dv_seg_a)
        m = compute_metrics(y_dv_a, scores, threshold=0.5)
        bal_rows.append(
            {
                "strategy": b_key,
                "description": b_desc,
                "precision_pos": m["precision_pos"],
                "recall_pos": m["recall_pos"],
                "f1_pos": m["f1_pos"],
                "pr_auc": m["pr_auc"],
                "roc_auc": m["roc_auc"],
                "fit_seconds": round(fit_sec, 3),
            }
        )
        log_vi_experiment("bin_A_balance", "lr", "word_tfidf", b_key, {}, m["f1_pos"], m["pr_auc"], fit_sec)

    bal_df = pd.DataFrame(bal_rows).sort_values("f1_pos", ascending=False).reset_index(drop=True)
    bal_df.to_csv(TAB_DIR / "V3_balance_comparison.csv", index=False, encoding="utf-8")

    # =========================================================================
    # 2. Full Model Comparison: 8 Models x {word_tfidf, word+char} on bin_A + Top Models on bin_B & multi3
    # =========================================================================
    comp_rows = []

    # Task bin_A: 8 models x {word_tfidf, word+char} + char_tfidf for LR & SVM
    bin_a_specs = []
    for mk in VI_MODEL_ZOO:
        bin_a_specs.append((mk, "word_tfidf"))
        if mk in ("lr", "svm_linear", "nb", "lgbm", "xgb", "rf", "dt", "knn"):
            bin_a_specs.append((mk, "word+char"))
    bin_a_specs.append(("lr", "char_tfidf"))
    bin_a_specs.append(("svm_linear", "char_tfidf"))

    for mk, vk in bin_a_specs:
        X_tr_in = X_tr_raw_a if vk == "char_tfidf" else X_tr_seg_a
        X_dv_in = X_dv_raw_a if vk == "char_tfidf" else X_dv_seg_a
        pipe = Pipeline(
            [
                ("tfidf", build_vi_vectorizer(vk)),
                ("clf", build_vi_estimator(mk, balance="class_weight", pos_weight=pos_weight_a, seed=42)),
            ]
        )
        t0 = time.perf_counter()
        pipe.fit(X_tr_in, y_tr_a)
        fit_sec = time.perf_counter() - t0
        sc = get_model_scores(pipe, X_dv_in)
        m = compute_metrics(y_dv_a, sc, threshold=0.5)
        row = {
            "task": "bin_A",
            "model_key": mk,
            "model_name": VI_MODEL_ZOO[mk],
            "vectorizer": vk,
            "balance": "class_weight",
            "primary_metric_name": "f1_pos",
            "primary_metric": m["f1_pos"],
            "precision": m["precision_pos"],
            "recall": m["recall_pos"],
            "f1_pos": m["f1_pos"],
            "f1_macro": m["f1_macro"],
            "pr_auc": m["pr_auc"],
            "roc_auc": m["roc_auc"],
            "accuracy": m["accuracy"],
            "fit_seconds": round(fit_sec, 3),
        }
        comp_rows.append(row)
        log_vi_experiment("bin_A", mk, vk, "class_weight", {}, m["f1_pos"], m["pr_auc"], fit_sec)

    # Task bin_B: OFFENSIVE + HATE vs CLEAN
    X_tr_seg_b, _, y_tr_b = prepare_task_split(train_df, "bin_B")
    X_dv_seg_b, _, y_dv_b = prepare_task_split(dev_df, "bin_B")
    pos_weight_b = round(float((y_tr_b == 0).sum() / max((y_tr_b == 1).sum(), 1)), 2)

    for mk in ["lr", "svm_linear", "nb", "lgbm", "rf"]:
        for vk in ["word_tfidf", "word+char"]:
            pipe = Pipeline(
                [
                    ("tfidf", build_vi_vectorizer(vk)),
                    ("clf", build_vi_estimator(mk, balance="class_weight", pos_weight=pos_weight_b, seed=42)),
                ]
            )
            t0 = time.perf_counter()
            pipe.fit(X_tr_seg_b, y_tr_b)
            fit_sec = time.perf_counter() - t0
            sc = get_model_scores(pipe, X_dv_seg_b)
            m = compute_metrics(y_dv_b, sc, threshold=0.5)
            comp_rows.append(
                {
                    "task": "bin_B",
                    "model_key": mk,
                    "model_name": VI_MODEL_ZOO[mk],
                    "vectorizer": vk,
                    "balance": "class_weight",
                    "primary_metric_name": "f1_pos",
                    "primary_metric": m["f1_pos"],
                    "precision": m["precision_pos"],
                    "recall": m["recall_pos"],
                    "f1_pos": m["f1_pos"],
                    "f1_macro": m["f1_macro"],
                    "pr_auc": m["pr_auc"],
                    "roc_auc": m["roc_auc"],
                    "accuracy": m["accuracy"],
                    "fit_seconds": round(fit_sec, 3),
                }
            )
            log_vi_experiment("bin_B", mk, vk, "class_weight", {}, m["f1_pos"], m["pr_auc"], fit_sec)

    # Task multi3: 3-class (0=CLEAN, 1=OFFENSIVE, 2=HATE) evaluated by f1_macro
    X_tr_seg_m, _, y_tr_m = prepare_task_split(train_df, "multi3")
    X_dv_seg_m, _, y_dv_m = prepare_task_split(dev_df, "multi3")

    best_multi3_f1 = -1.0
    best_multi3_pipe = None
    best_multi3_info = {}

    for mk in ["lr", "svm_linear", "nb", "lgbm", "rf"]:
        for vk in ["word_tfidf", "word+char"]:
            pipe = Pipeline(
                [
                    ("tfidf", build_vi_vectorizer(vk)),
                    ("clf", build_vi_estimator(mk, balance="class_weight", seed=42, is_multiclass=True)),
                ]
            )
            t0 = time.perf_counter()
            pipe.fit(X_tr_seg_m, y_tr_m)
            fit_sec = time.perf_counter() - t0
            y_pred_m = pipe.predict(X_dv_seg_m)
            f1_mac = round(float(f1_score(y_dv_m, y_pred_m, average="macro", zero_division=0)), 4)
            prec_mac = round(float(precision_score(y_dv_m, y_pred_m, average="macro", zero_division=0)), 4)
            rec_mac = round(float(recall_score(y_dv_m, y_pred_m, average="macro", zero_division=0)), 4)
            acc_m = round(float(accuracy_score(y_dv_m, y_pred_m)), 4)
            per_cls_f1 = f1_score(y_dv_m, y_pred_m, average=None, labels=[0, 1, 2], zero_division=0)

            comp_rows.append(
                {
                    "task": "multi3",
                    "model_key": mk,
                    "model_name": VI_MODEL_ZOO[mk],
                    "vectorizer": vk,
                    "balance": "class_weight",
                    "primary_metric_name": "f1_macro",
                    "primary_metric": f1_mac,
                    "precision": prec_mac,
                    "recall": rec_mac,
                    "f1_pos": round(float(per_cls_f1[2]), 4),
                    "f1_macro": f1_mac,
                    "pr_auc": 0.0,
                    "roc_auc": 0.0,
                    "accuracy": acc_m,
                    "fit_seconds": round(fit_sec, 3),
                }
            )
            log_vi_experiment("multi3", mk, vk, "class_weight", {}, f1_mac, acc_m, fit_sec)

            if f1_mac > best_multi3_f1:
                best_multi3_f1 = f1_mac
                best_multi3_pipe = pipe
                best_multi3_info = {
                    "model_key": mk,
                    "model_name": VI_MODEL_ZOO[mk],
                    "vectorizer": vk,
                    "dev_f1_macro": f1_mac,
                    "dev_accuracy": acc_m,
                    "dev_f1_clean": round(float(per_cls_f1[0]), 4),
                    "dev_f1_offensive": round(float(per_cls_f1[1]), 4),
                    "dev_f1_hate": round(float(per_cls_f1[2]), 4),
                }

    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(REPORTS_DIR / "vi_model_comparison.csv", index=False, encoding="utf-8")

    # =========================================================================
    # 3. Ablation Study on bin_A (Segmentation, Teencode, Stopwords & No-Accent Augmentation)
    # =========================================================================
    sub_tr_a = train_df[train_df["label_3"].isin(["CLEAN", "HATE"])].copy()
    sub_dv_a = dev_df[dev_df["label_3"].isin(["CLEAN", "HATE"])].copy()

    X_tr_noacc_a = np.array([restore_unaccented_vietnamese(strip_accents(t)) for t in X_tr_raw_a])
    X_dv_noacc_a = np.array([restore_unaccented_vietnamese(strip_accents(t)) for t in X_dv_raw_a])

    ablation_configs = [
        ("Full Pipeline + Tăng cường Không dấu (Accent-Drop Augmentation)", True, True, False, True),
        ("Full Pipeline (Có tách từ + Chuẩn hóa Teencode + Giữ Stopword)", True, True, False, False),
        ("Không tách từ (segment=False)", False, True, False, False),
        ("Không chuẩn hóa Teencode (expand_teencode=False)", True, False, False, False),
        ("Bỏ Stopwords (remove_stopwords=True)", True, True, True, False),
    ]
    ablation_rows = []
    for ab_name, seg_flag, tc_flag, sw_flag, aug_noacc_flag in ablation_configs:
        if seg_flag and tc_flag and not sw_flag:
            X_tr_ab = sub_tr_a["text_clean_seg"].fillna("").to_numpy()
            X_dv_ab = sub_dv_a["text_clean_seg"].fillna("").to_numpy()
        elif not seg_flag and tc_flag and not sw_flag:
            X_tr_ab = sub_tr_a["text_clean_raw"].fillna("").to_numpy()
            X_dv_ab = sub_dv_a["text_clean_raw"].fillna("").to_numpy()
        else:
            X_tr_ab = np.array([vi_clean(t, segment=seg_flag, expand_teencode=tc_flag, remove_stopwords=sw_flag) for t in sub_tr_a["text"]])
            X_dv_ab = np.array([vi_clean(t, segment=seg_flag, expand_teencode=tc_flag, remove_stopwords=sw_flag) for t in sub_dv_a["text"]])

        if aug_noacc_flag:
            X_tr_fit = np.concatenate([X_tr_ab, X_tr_noacc_a])
            y_tr_fit = np.concatenate([y_tr_a, y_tr_a])
        else:
            X_tr_fit = X_tr_ab
            y_tr_fit = y_tr_a

        pipe_ab = Pipeline(
            [
                ("tfidf", build_vi_vectorizer("word_tfidf")),
                ("clf", build_vi_estimator("svm_linear", balance="class_weight", seed=42)),
            ]
        )
        pipe_ab.fit(X_tr_fit, y_tr_fit)
        sc_ab = get_model_scores(pipe_ab, X_dv_ab)
        m_ab = compute_metrics(y_dv_a, sc_ab, threshold=0.5)
        sc_ab_noacc = get_model_scores(pipe_ab, X_dv_noacc_a)
        m_ab_noacc = compute_metrics(y_dv_a, sc_ab_noacc, threshold=0.5)
        ablation_rows.append(
            {
                "config": ab_name,
                "segment": seg_flag,
                "expand_teencode": tc_flag,
                "remove_stopwords": sw_flag,
                "noaccent_augmentation": aug_noacc_flag,
                "precision_pos": m_ab["precision_pos"],
                "recall_pos": m_ab["recall_pos"],
                "f1_pos": m_ab["f1_pos"],
                "f1_no_accents": m_ab_noacc["f1_pos"],
                "pr_auc": m_ab["pr_auc"],
                "roc_auc": m_ab["roc_auc"],
            }
        )

    ab_df = pd.DataFrame(ablation_rows)
    ab_df.to_csv(TAB_DIR / "vi_ablation.csv", index=False, encoding="utf-8")

    # =========================================================================
    # 4. Tuning & Threshold Optimization on Dev for bin_A and bin_B (with Accent-Drop Augmentation)
    # =========================================================================
    X_tr_aug_a = np.concatenate([X_tr_seg_a, X_tr_noacc_a])
    y_tr_aug_a = np.concatenate([y_tr_a, y_tr_a])

    best_a_candidates = []
    for mk in ["svm_linear", "lr"]:
        for vk in ["word+char", "word_tfidf"]:
            for c_val in [0.4, 0.8, 1.5, 3.0]:
                vec = build_vi_vectorizer(vk)
                if mk == "svm_linear":
                    clf = CalibratedClassifierCV(
                        estimator=LinearSVC(C=c_val, class_weight="balanced", random_state=42, max_iter=3000),
                        cv=3,
                    )
                else:
                    clf = LogisticRegression(C=c_val, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=42)
                pipe = Pipeline([("tfidf", vec), ("clf", clf)])
                pipe.fit(X_tr_aug_a, y_tr_aug_a)
                sc_dv = get_model_scores(pipe, X_dv_seg_a)
                thr_res = pick_threshold(y_dv_a, sc_dv, strategy="f1")
                m_tuned = compute_metrics(y_dv_a, sc_dv, threshold=thr_res["best_threshold"])
                best_a_candidates.append((m_tuned["f1_pos"], mk, vk, c_val, pipe, thr_res, m_tuned))

    best_a_candidates.sort(key=lambda x: x[0], reverse=True)
    _, best_mk_a, best_vk_a, best_c_a, best_pipe_a, best_thr_a, best_m_a = best_a_candidates[0]

    # Refit best bin_A model on Train + Dev (including unaccented augmented copies of Train + Dev) and save
    X_tr_dv_a = np.concatenate([X_tr_seg_a, X_dv_seg_a, X_tr_noacc_a, X_dv_noacc_a])
    y_tr_dv_a = np.concatenate([y_tr_a, y_dv_a, y_tr_a, y_dv_a])
    best_pipe_a.fit(X_tr_dv_a, y_tr_dv_a)
    joblib.dump(best_pipe_a, VI_MODELS_DIR / "best_vi_bin_a.joblib")
    with open(VI_MODELS_DIR / "threshold_vi_bin_a.json", "w", encoding="utf-8") as f:
        json.dump(best_thr_a, f, indent=2, ensure_ascii=False)

    # Also tune & save best bin_B model (OFFENSIVE+HATE vs CLEAN) with unaccented augmentation
    _, X_tr_raw_b, _ = prepare_task_split(train_df, "bin_B")
    _, X_dv_raw_b, _ = prepare_task_split(dev_df, "bin_B")
    X_tr_noacc_b = np.array([restore_unaccented_vietnamese(strip_accents(t)) for t in X_tr_raw_b])
    X_dv_noacc_b = np.array([restore_unaccented_vietnamese(strip_accents(t)) for t in X_dv_raw_b])

    best_pipe_b = Pipeline(
        [
            ("tfidf", build_vi_vectorizer(best_vk_a)),
            (
                "clf",
                CalibratedClassifierCV(
                    estimator=LinearSVC(C=best_c_a, class_weight="balanced", random_state=42, max_iter=3000),
                    cv=3,
                )
                if best_mk_a == "svm_linear"
                else LogisticRegression(C=best_c_a, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=42),
            ),
        ]
    )
    best_pipe_b.fit(np.concatenate([X_tr_seg_b, X_tr_noacc_b]), np.concatenate([y_tr_b, y_tr_b]))
    sc_dv_b = get_model_scores(best_pipe_b, X_dv_seg_b)
    best_thr_b = pick_threshold(y_dv_b, sc_dv_b, strategy="f1")
    best_m_b = compute_metrics(y_dv_b, sc_dv_b, threshold=best_thr_b["best_threshold"])
    best_pipe_b.fit(
        np.concatenate([X_tr_seg_b, X_dv_seg_b, X_tr_noacc_b, X_dv_noacc_b]),
        np.concatenate([y_tr_b, y_dv_b, y_tr_b, y_dv_b]),
    )
    joblib.dump(best_pipe_b, VI_MODELS_DIR / "best_vi_bin_b.joblib")
    with open(VI_MODELS_DIR / "threshold_vi_bin_b.json", "w", encoding="utf-8") as f:
        json.dump(best_thr_b, f, indent=2, ensure_ascii=False)

    # Refit best multi3 model on Train + Dev (plus unaccented augmented copies) and save
    if best_multi3_pipe is not None:
        best_multi3_pipe.fit(
            np.concatenate([X_tr_seg_m, X_dv_seg_m, X_tr_noacc_b, X_dv_noacc_b]),
            np.concatenate([y_tr_m, y_dv_m, y_tr_m, y_dv_m]),
        )
        joblib.dump(best_multi3_pipe, VI_MODELS_DIR / "best_vi_multi3.joblib")

    # =========================================================================
    # 5. Plot V3 Figures
    # =========================================================================
    bin_a_df = comp_df[comp_df["task"] == "bin_A"].sort_values("f1_pos", ascending=False).reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=150)
    lbls_a = [f"{r.model_key} ({r.vectorizer})" for r in bin_a_df.itertuples()]
    x = np.arange(len(lbls_a))
    w = 0.35
    axes[0].bar(x - w / 2, bin_a_df["f1_pos"], w, label="Dev F1-Score (HATE)", color="#e53935", edgecolor="black")
    axes[0].bar(x + w / 2, bin_a_df["pr_auc"], w, label="Dev PR-AUC", color="#1e88e5", edgecolor="black")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(lbls_a, rotation=35, ha="right", fontsize=8.5)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("Điểm số (Dev)")
    axes[0].set_title("So sánh 8 Mô hình × Vectorizer trên bài toán bin_A (HATE vs CLEAN)", fontweight="bold")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    # Multi3 comparison bar chart vs published baseline (0.60 - 0.675)
    m3_df = comp_df[comp_df["task"] == "multi3"].sort_values("f1_macro", ascending=False).reset_index(drop=True)
    lbls_m3 = [f"{r.model_key} ({r.vectorizer})" for r in m3_df.itertuples()]
    xm = np.arange(len(lbls_m3))
    axes[1].bar(xm, m3_df["f1_macro"], color="#7e57c2", edgecolor="black", width=0.5, label="Dev Macro-F1 (3 lớp)")
    axes[1].axhline(0.675, color="#d32f2f", linestyle="--", lw=1.8, label="Mốc công bố PhoBERT-CNN (0.675)")
    axes[1].axhline(0.620, color="#fb8c00", linestyle=":", lw=1.6, label="Mốc Baseline công bố (~0.620)")
    axes[1].set_xticks(xm)
    axes[1].set_xticklabels(lbls_m3, rotation=30, ha="right", fontsize=9)
    axes[1].set_ylim(0, 0.85)
    axes[1].set_ylabel("Macro-F1 (3 lớp CLEAN/OFFENSIVE/HATE)")
    axes[1].set_title("Kết quả bài toán 3 lớp (multi3) so với mốc công bố trên ViHSD", fontweight="bold")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "V3_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot Ablation & Balancing comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), dpi=150)
    y_ab = np.arange(len(ab_df))
    h_ab = 0.36
    ab_rev = ab_df.iloc[::-1].reset_index(drop=True)
    axes[0].barh(y_ab + h_ab / 2, ab_rev["f1_pos"], h_ab, label="Dev F1 (Có dấu)", color="#1e88e5", edgecolor="black")
    axes[0].barh(y_ab - h_ab / 2, ab_rev["f1_no_accents"], h_ab, label="Dev F1 (Không dấu)", color="#43a047", edgecolor="black")
    axes[0].set_yticks(y_ab)
    axes[0].set_yticklabels(ab_rev["config"], fontsize=8.5)
    axes[0].set_xlim(0, max(ab_df["f1_pos"]) * 1.22)
    axes[0].set_title("Ablation: Tác động Tách từ, Teencode & Tăng cường Không dấu", fontweight="bold")
    axes[0].set_xlabel("Dev F1-Score (bin_A)")
    axes[0].legend(loc="lower right", fontsize=8.5)
    axes[0].grid(axis="x", alpha=0.3)

    axes[1].bar(bal_df["strategy"], bal_df["f1_pos"], color="#00897b", edgecolor="black", width=0.5)
    for idx, val in enumerate(bal_df["f1_pos"]):
        axes[1].text(idx, val + 0.01, f"{val:.4f}", ha="center", fontweight="bold", fontsize=9)
    axes[1].set_ylim(0, max(bal_df["f1_pos"]) * 1.18)
    axes[1].set_title("So sánh 5 chiến lược cân bằng dữ liệu trên bin_A (Dev F1)", fontweight="bold")
    axes[1].set_ylabel("Dev F1-Score")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "V3_ablation_and_balance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    vi_model_card = {
        "best_bin_A": {
            "task": "bin_A (HATE vs CLEAN, bỏ OFFENSIVE)",
            "model_key": best_mk_a,
            "model_name": VI_MODEL_ZOO[best_mk_a],
            "vectorizer": best_vk_a,
            "C": best_c_a,
            "best_threshold": best_thr_a["best_threshold"],
            "dev_metrics_tuned": best_m_a,
            "dev_metrics_default_05": best_thr_a["default_05"],
        },
        "best_bin_B": {
            "task": "bin_B (OFFENSIVE+HATE vs CLEAN)",
            "model_key": best_mk_a,
            "model_name": VI_MODEL_ZOO[best_mk_a],
            "vectorizer": best_vk_a,
            "C": best_c_a,
            "best_threshold": best_thr_b["best_threshold"],
            "dev_metrics_tuned": best_m_b,
        },
        "best_multi3": best_multi3_info,
        "ablation": ablation_rows,
        "balance_comparison": bal_rows,
    }
    with open(VI_MODELS_DIR / "vi_model_card.json", "w", encoding="utf-8") as f:
        json.dump(vi_model_card, f, indent=2, ensure_ascii=False)
    with open(REPORTS_DIR / "vi_v3_summary.json", "w", encoding="utf-8") as f:
        json.dump(vi_model_card, f, indent=2, ensure_ascii=False)

    return vi_model_card


if __name__ == "__main__":
    res = run_v3_classical()
    print("V3 Classical finished. Best bin_A Dev F1:", res["best_bin_A"]["dev_metrics_tuned"]["f1_pos"])
    print("Best multi3 Dev Macro-F1:", res["best_multi3"]["dev_f1_macro"])
