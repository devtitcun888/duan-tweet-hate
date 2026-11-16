"""P6 Hyperparameter Tuning (RandomizedSearchCV) & Decision Threshold Optimization (P6)."""

from __future__ import annotations

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
from scipy.stats import loguniform
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.evaluate import compute_metrics
from src.models import get_model_scores
from src.threshold import pick_threshold
from src.utils import get_project_root, load_config, log_experiment, set_seed

PROJECT_ROOT = get_project_root()
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
TAB_DIR = REPORTS_DIR / "tables"


def run_p6_tuning() -> dict[str, Any]:
    """Run RandomizedSearchCV (n_iter=40, cv=5) on top models, optimize threshold on val, and save best_model."""
    set_seed(42)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "val.csv")

    X_train = train_df["text_clean"].fillna("").astype(str).to_numpy()
    y_train = train_df["label"].astype(int).to_numpy()
    X_val = val_df["text_clean"].fillna("").astype(str).to_numpy()
    y_val = val_df["label"].astype(int).to_numpy()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Load P5 baseline metrics for before/after comparison
    comp_df = pd.read_csv(REPORTS_DIR / "model_comparison.csv")
    baseline_lookup = {}
    for row in comp_df[comp_df["vectorizer"] == "word_tfidf"].itertuples():
        baseline_lookup[row.model_key] = {
            "val_f1_before": float(row.f1_pos),
            "val_precision_before": float(row.precision_pos),
            "val_recall_before": float(row.recall_pos),
            "val_pr_auc_before": float(row.pr_auc),
            "cv_f1_before": float(row.cv_f1_mean),
        }

    # Define candidate pipelines & hyperparameter spaces (n_iter=40 as specified in P6)
    candidates = [
        (
            "lr",
            "Logistic Regression",
            Pipeline(
                [
                    ("tfidf", TfidfVectorizer(sublinear_tf=True, max_df=0.9)),
                    ("clf", LogisticRegression(max_iter=2000, solver="liblinear", random_state=42)),
                ]
            ),
            {
                "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
                "tfidf__min_df": [2, 3, 5],
                "tfidf__max_features": [20000, 50000, 100000],
                "clf__C": loguniform(1e-2, 1e2),
                "clf__penalty": ["l1", "l2"],
                "clf__class_weight": [None, "balanced"],
            },
        ),
        (
            "svm_linear",
            "Linear SVM (Calibrated)",
            Pipeline(
                [
                    ("tfidf", TfidfVectorizer(sublinear_tf=True, max_df=0.9)),
                    ("clf", LinearSVC(random_state=42, max_iter=3000, dual="auto")),
                ]
            ),
            {
                "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
                "tfidf__min_df": [2, 3, 5],
                "tfidf__max_features": [20000, 50000, 100000],
                "clf__C": loguniform(1e-2, 1e2),
                "clf__loss": ["hinge", "squared_hinge"],
                "clf__class_weight": [None, "balanced"],
            },
        ),
        (
            "nb",
            "Complement Naive Bayes",
            Pipeline(
                [
                    ("tfidf", TfidfVectorizer(sublinear_tf=True, max_df=0.9)),
                    ("clf", ComplementNB()),
                ]
            ),
            {
                "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
                "tfidf__min_df": [2, 3, 5],
                "tfidf__max_features": [20000, 50000, 100000],
                "clf__alpha": loguniform(1e-2, 2.0),
                "clf__norm": [True, False],
            },
        ),
    ]

    before_after_rows = []
    tuned_estimators = {}
    tuned_val_scores = {}
    tuned_best_params = {}
    tuned_thr_results = {}

    for model_key, model_name, base_pipe, param_dist in candidates:
        print(f"[P6] RandomizedSearchCV (n_iter=40, cv=5) for {model_name}...")
        t0 = time.perf_counter()
        search = RandomizedSearchCV(
            estimator=base_pipe,
            param_distributions=param_dist,
            n_iter=40,
            scoring="f1",
            cv=skf,
            random_state=42,
            n_jobs=1,
            refit=True,
        )
        search.fit(X_train, y_train)
        fit_sec = round(time.perf_counter() - t0, 2)

        best_pipe = search.best_estimator_
        best_params = {
            k: (round(float(v), 5) if isinstance(v, (float, np.floating)) else v)
            for k, v in search.best_params_.items()
        }

        # For svm_linear, wrap the tuned LinearSVC in CalibratedClassifierCV(cv=3) on train so probabilities are calibrated
        if model_key == "svm_linear":
            tuned_vec = best_pipe.named_steps["tfidf"]
            tuned_svm = best_pipe.named_steps["clf"]
            cal_pipe = Pipeline(
                [
                    ("tfidf", tuned_vec),
                    ("clf", CalibratedClassifierCV(estimator=tuned_svm, cv=3)),
                ]
            )
            cal_pipe.fit(X_train, y_train)
            best_pipe = cal_pipe

        val_scores = get_model_scores(best_pipe, X_val)
        m_after_05 = compute_metrics(y_val, val_scores, threshold=0.5)
        thr_info = pick_threshold(y_val, val_scores, strategy="f1")
        m_after_thr = compute_metrics(y_val, val_scores, threshold=thr_info["best_threshold"])

        log_experiment(
            model=f"{model_key}_tuned",
            vectorizer="word_tfidf_tuned",
            balance=str(best_params.get("clf__class_weight", "balanced")),
            params=best_params,
            cv_f1_mean=float(search.best_score_),
            cv_f1_std=float(search.cv_results_["std_test_score"][search.best_index_]),
            val_f1=m_after_thr["f1_pos"],
            val_pr_auc=m_after_thr["pr_auc"],
            fit_seconds=fit_sec,
        )

        base_info = baseline_lookup.get(model_key, {})
        row = {
            "model_key": model_key,
            "model_name": model_name,
            "cv_f1_before": base_info.get("cv_f1_before", 0.0),
            "cv_f1_after": round(float(search.best_score_), 4),
            "val_f1_before": base_info.get("val_f1_before", 0.0),
            "val_f1_after_thr05": m_after_05["f1_pos"],
            "best_threshold": thr_info["best_threshold"],
            "val_f1_after_tuned_thr": m_after_thr["f1_pos"],
            "val_precision_after": m_after_thr["precision_pos"],
            "val_recall_after": m_after_thr["recall_pos"],
            "val_pr_auc_after": m_after_thr["pr_auc"],
            "val_roc_auc_after": m_after_thr["roc_auc"],
            "f1_gain_vs_baseline": round(
                float(m_after_thr["f1_pos"]) - float(base_info.get("val_f1_before", 0.0)), 4
            ),
            "search_seconds": fit_sec,
            "best_params_json": json.dumps(best_params, ensure_ascii=False),
        }
        before_after_rows.append(row)
        tuned_estimators[model_key] = best_pipe
        tuned_val_scores[model_key] = val_scores
        tuned_best_params[model_key] = best_params
        tuned_thr_results[model_key] = thr_info

    ba_df = pd.DataFrame(before_after_rows).sort_values("val_f1_after_tuned_thr", ascending=False).reset_index(drop=True)
    ba_df.to_csv(TAB_DIR / "P6_before_after.csv", index=False, encoding="utf-8")

    # Select final best model (if LR is within 0.005 F1 of the top model, prefer LR for direct coefficient explainability & speed; otherwise pick top model)
    top_row = ba_df.iloc[0]
    lr_row = ba_df[ba_df["model_key"] == "lr"].iloc[0]
    if (
        top_row["model_key"] != "lr"
        and float(top_row["val_f1_after_tuned_thr"]) - float(lr_row["val_f1_after_tuned_thr"]) < 0.005
    ):
        chosen_key = "lr"
        selection_reason = (
            f"Logistic Regression đạt Val F1 = {lr_row['val_f1_after_tuned_thr']:.4f} "
            f"(chênh lệch < 0.5 điểm % so với {top_row['model_name']} đạt {top_row['val_f1_after_tuned_thr']:.4f}), "
            "được chọn làm best_model theo quy tắc P6 vì tốc độ suy luận cực nhanh, xuất trực tiếp xác suất chuẩn "
            "và hỗ trợ giải thích đóng góp từng từ (TF-IDF × hệ số) minh bạch trong ứng dụng Streamlit."
        )
    else:
        chosen_key = str(top_row["model_key"])
        selection_reason = (
            f"{top_row['model_name']} đạt Val F1 sau tinh chỉnh ngưỡng cao nhất "
            f"({top_row['val_f1_after_tuned_thr']:.4f}, PR-AUC = {top_row['val_pr_auc_after']:.4f}) "
            f"với ngưỡng quyết định tối ưu threshold = {top_row['best_threshold']:.4f}."
        )

    chosen_row = ba_df[ba_df["model_key"] == chosen_key].iloc[0].to_dict()
    chosen_thr_info = tuned_thr_results[chosen_key]
    chosen_val_scores = tuned_val_scores[chosen_key]

    # Save models/threshold.json for the chosen best model
    with open(MODELS_DIR / "threshold.json", "w", encoding="utf-8") as f:
        json.dump(chosen_thr_info, f, indent=2, ensure_ascii=False)

    # Plot P6 Figure 1: Precision-Recall curve & Threshold scan for chosen model
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5), dpi=150)
    curve_df = pd.DataFrame(chosen_thr_info["curve"])
    best_thr = chosen_thr_info["best_threshold"]
    best_f1 = chosen_thr_info["best_val_f1"]

    axes[0].plot(curve_df["threshold"], curve_df["precision"], label="Precision (Lớp 1)", color="#1e88e5", lw=2)
    axes[0].plot(curve_df["threshold"], curve_df["recall"], label="Recall (Lớp 1)", color="#fb8c00", lw=2)
    axes[0].plot(curve_df["threshold"], curve_df["f1"], label="F1-Score (Lớp 1)", color="#e53935", lw=2.5)
    axes[0].axvline(best_thr, color="black", linestyle="--", lw=1.5, label=f"Ngưỡng tối ưu = {best_thr:.3f} (F1={best_f1:.4f})")
    axes[0].axvline(0.5, color="gray", linestyle=":", lw=1.2, label=f"Ngưỡng mặc định = 0.500 (F1={chosen_thr_info['default_05']['f1']:.4f})")
    axes[0].set_title(f"Quét ngưỡng quyết định trên Validation ({chosen_row['model_name']})", fontweight="bold")
    axes[0].set_xlabel("Ngưỡng xác suất (Decision Threshold)")
    axes[0].set_ylabel("Điểm số (Score)")
    axes[0].legend(fontsize=8.5)
    axes[0].grid(alpha=0.3)

    prec_c, rec_c, _ = precision_recall_curve(y_val, chosen_val_scores)
    axes[1].plot(rec_c, prec_c, color="#8e24aa", lw=2.2, label=f"PR Curve (PR-AUC={chosen_row['val_pr_auc_after']:.4f})")
    axes[1].scatter(
        [chosen_thr_info["best_val_recall"]],
        [chosen_thr_info["best_val_precision"]],
        color="#e53935",
        s=90,
        zorder=5,
        label=f"Điểm tối ưu (Thr={best_thr:.3f}, P={chosen_thr_info['best_val_precision']:.3f}, R={chosen_thr_info['best_val_recall']:.3f})",
    )
    axes[1].scatter(
        [chosen_thr_info["default_05"]["recall"]],
        [chosen_thr_info["default_05"]["precision"]],
        color="#1e88e5",
        s=75,
        marker="s",
        zorder=5,
        label=f"Ngưỡng 0.50 (P={chosen_thr_info['default_05']['precision']:.3f}, R={chosen_thr_info['default_05']['recall']:.3f})",
    )
    axes[1].set_title("Đường cong Precision-Recall & Điểm vận hành tối ưu", fontweight="bold")
    axes[1].set_xlabel("Recall (Độ phủ lớp 1)")
    axes[1].set_ylabel("Precision (Độ chính xác lớp 1)")
    axes[1].legend(fontsize=8.5, loc="lower left")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "P6_threshold_tuning.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot P6 Figure 2: Before vs After Tuning comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    x = np.arange(len(ba_df))
    w = 0.25
    ax.bar(x - w, ba_df["val_f1_before"], w, label="Trước Tuning (Thr=0.5)", color="#90a4ae", edgecolor="black")
    ax.bar(x, ba_df["val_f1_after_thr05"], w, label="Sau Tuning Tham số (Thr=0.5)", color="#42a5f5", edgecolor="black")
    ax.bar(x + w, ba_df["val_f1_after_tuned_thr"], w, label="Sau Tuning Tham số + Ngưỡng", color="#e53935", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(ba_df["model_name"], fontsize=10.5)
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("Validation F1-Score (Lớp 1)")
    ax.set_title("So sánh F1-Score trước và sau khi tinh chỉnh tham số & ngưỡng (P6)", fontweight="bold")
    for idx, r in ba_df.iterrows():
        ax.text(idx - w, r["val_f1_before"] + 0.01, f"{r['val_f1_before']:.4f}", ha="center", fontsize=8.5)
        ax.text(idx, r["val_f1_after_thr05"] + 0.01, f"{r['val_f1_after_thr05']:.4f}", ha="center", fontsize=8.5)
        ax.text(idx + w, r["val_f1_after_tuned_thr"] + 0.01, f"{r['val_f1_after_tuned_thr']:.4f}", ha="center", fontsize=8.5, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "P6_before_after.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Final step of P6: Refit chosen best_model on train + val (keeping threshold from val) and save best_model.joblib
    X_train_val = np.concatenate([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])
    final_best_pipe = tuned_estimators[chosen_key]
    final_best_pipe.fit(X_train_val, y_train_val)
    joblib.dump(final_best_pipe, MODELS_DIR / "best_model.joblib")

    model_card = {
        "model_key": chosen_key,
        "model_name": chosen_row["model_name"],
        "selection_reason": selection_reason,
        "best_params": tuned_best_params[chosen_key],
        "best_threshold": chosen_thr_info["best_threshold"],
        "val_metrics_before_tuning": baseline_lookup.get(chosen_key, {}),
        "val_metrics_after_tuning_thr05": compute_metrics(y_val, chosen_val_scores, threshold=0.5),
        "val_metrics_after_tuning_best_thr": compute_metrics(
            y_val, chosen_val_scores, threshold=chosen_thr_info["best_threshold"]
        ),
        "refit_on_train_plus_val_rows": int(len(X_train_val)),
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "sklearn_version": sklearn.__version__,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(MODELS_DIR / "model_card.json", "w", encoding="utf-8") as f:
        json.dump(model_card, f, indent=2, ensure_ascii=False)

    summary = {
        "chosen_model_key": chosen_key,
        "chosen_model_name": chosen_row["model_name"],
        "selection_reason": selection_reason,
        "best_params": tuned_best_params[chosen_key],
        "best_threshold": chosen_thr_info["best_threshold"],
        "before_after_table": before_after_rows,
        "model_card": model_card,
    }
    with open(REPORTS_DIR / "p6_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


if __name__ == "__main__":
    res = run_p6_tuning()
    print("P6 finished. Chosen model:", res["chosen_model_name"], "Threshold:", res["best_threshold"])
