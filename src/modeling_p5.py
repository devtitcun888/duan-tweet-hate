"""P5 Model Comparison across 9 Classifiers and Vectorizer Representations (P5)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.evaluate import compute_metrics, plot_roc_pr
from src.models import MODEL_ZOO, build_pipeline, get_model_scores
from src.utils import get_project_root, load_config, log_experiment, set_seed

PROJECT_ROOT = get_project_root()
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
TAB_DIR = REPORTS_DIR / "tables"


def run_p5_model_comparison() -> dict[str, Any]:
    """Train and evaluate all 9 models in MODEL_ZOO with 5-fold CV on train and evaluation on val."""
    set_seed(42)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    balance_strategy = cfg.get("balance", "class_weight")

    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "val.csv")

    X_train = train_df["text_clean"].fillna("").astype(str).to_numpy()
    y_train = train_df["label"].astype(int).to_numpy()
    X_val = val_df["text_clean"].fillna("").astype(str).to_numpy()
    y_val = val_df["label"].astype(int).to_numpy()

    n_pos = max(int((y_train == 1).sum()), 1)
    n_neg = int((y_train == 0).sum())
    pos_weight_ratio = round(n_neg / n_pos, 2)

    # Stratified 8,000-row subset of train ONLY for slow kernel/instance models (svm_rbf) as specified in P5.2
    X_train_sub, _, y_train_sub, _ = train_test_split(
        X_train, y_train, train_size=min(8000, len(X_train)), stratify=y_train, random_state=42
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Primary comparison: all 9 models with word_tfidf + top candidates with word+char
    experiment_specs = [
        ("lr", "word_tfidf"),
        ("lr", "word+char"),
        ("svm_linear", "word_tfidf"),
        ("svm_linear", "word+char"),
        ("nb", "word_tfidf"),
        ("nb", "word+char"),
        ("xgb", "word_tfidf"),
        ("lgbm", "word_tfidf"),
        ("rf", "word_tfidf"),
        ("dt", "word_tfidf"),
        ("knn", "word_tfidf"),
        ("svm_rbf", "word_tfidf"),
    ]

    comparison_rows = []
    val_scores_by_model: dict[str, np.ndarray] = {}
    cv_fold_f1s: dict[str, list[float]] = {}

    for model_key, vec_kind in experiment_specs:
        run_id = f"{model_key}__{vec_kind}"
        display_name = f"{MODEL_ZOO[model_key]} ({vec_kind})"
        print(f"[P5] Running {display_name}...")

        use_sub = model_key == "svm_rbf"
        X_tr_curr = X_train_sub if use_sub else X_train
        y_tr_curr = y_train_sub if use_sub else y_train

        fold_f1s = []
        t_cv0 = time.perf_counter()
        for tr_idx, va_idx in skf.split(X_tr_curr, y_tr_curr):
            pipe_cv = build_pipeline(
                model_key=model_key,
                vectorizer_kind=vec_kind,
                balance=balance_strategy,
                pos_weight_ratio=pos_weight_ratio,
                cfg=cfg,
                seed=42,
            )
            pipe_cv.fit(X_tr_curr[tr_idx], y_tr_curr[tr_idx])
            sc_va = get_model_scores(pipe_cv, X_tr_curr[va_idx])
            m_cv = compute_metrics(y_tr_curr[va_idx], sc_va, threshold=0.5)
            fold_f1s.append(float(m_cv["f1_pos"]))

        cv_f1_mean = round(float(np.mean(fold_f1s)), 4)
        cv_f1_std = round(float(np.std(fold_f1s, ddof=1)), 4)
        cv_fold_f1s[run_id] = fold_f1s

        # Fit on full train (or 8k subset for svm_rbf) and evaluate on val
        t_fit0 = time.perf_counter()
        pipe_full = build_pipeline(
            model_key=model_key,
            vectorizer_kind=vec_kind,
            balance=balance_strategy,
            pos_weight_ratio=pos_weight_ratio,
            cfg=cfg,
            seed=42,
        )
        pipe_full.fit(X_tr_curr, y_tr_curr)
        fit_seconds = round(time.perf_counter() - t_fit0, 3)

        t_pred0 = time.perf_counter()
        val_scores = get_model_scores(pipe_full, X_val)
        pred_elapsed = time.perf_counter() - t_pred0
        predict_ms_per_1k = round((pred_elapsed / max(len(X_val), 1)) * 1000.0 * 1000.0, 3)

        val_metrics = compute_metrics(y_val, val_scores, threshold=0.5)

        log_experiment(
            model=model_key,
            vectorizer=vec_kind,
            balance=balance_strategy,
            params={"subset_8k": use_sub},
            cv_f1_mean=cv_f1_mean,
            cv_f1_std=cv_f1_std,
            val_f1=val_metrics["f1_pos"],
            val_pr_auc=val_metrics["pr_auc"],
            fit_seconds=fit_seconds,
        )

        row = {
            "run_id": run_id,
            "model_key": model_key,
            "model_name": MODEL_ZOO[model_key],
            "vectorizer": vec_kind,
            "balance": balance_strategy,
            "cv_f1_mean": cv_f1_mean,
            "cv_f1_std": cv_f1_std,
            **val_metrics,
            "fit_seconds": fit_seconds,
            "predict_ms_per_1k": predict_ms_per_1k,
            "note": "Train trên mẫu con 8,000 dòng do độ phức tạp bậc 2" if use_sub else "Train toàn bộ tập train",
        }
        comparison_rows.append(row)

        if vec_kind == "word_tfidf":
            val_scores_by_model[MODEL_ZOO[model_key]] = val_scores

    comp_df = pd.DataFrame(comparison_rows).sort_values("f1_pos", ascending=False).reset_index(drop=True)
    comp_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False, encoding="utf-8")

    # Paired t-test across top-3 configurations on 5-fold CV F1
    top3_runs = comp_df.head(3)["run_id"].tolist()
    paired_tests = []
    for i in range(len(top3_runs)):
        for j in range(i + 1, len(top3_runs)):
            r1, r2 = top3_runs[i], top3_runs[j]
            s1, s2 = cv_fold_f1s[r1], cv_fold_f1s[r2]
            t_stat, p_val = ttest_rel(s1, s2)
            paired_tests.append(
                {
                    "run_a": r1,
                    "run_b": r2,
                    "mean_diff_f1": round(float(np.mean(s1) - np.mean(s2)), 4),
                    "t_statistic": round(float(t_stat), 4),
                    "p_value": round(float(p_val), 4),
                    "significant_at_005": bool(p_val < 0.05),
                }
            )
    pd.DataFrame(paired_tests).to_csv(TAB_DIR / "P5_paired_ttest.csv", index=False, encoding="utf-8")

    # Figure 1: Bar chart of F1 and PR-AUC across all runs
    fig, ax = plt.subplots(figsize=(13.5, 6), dpi=150)
    labels = [f"{r.model_key} ({r.vectorizer})" for r in comp_df.itertuples()]
    x = np.arange(len(labels))
    w = 0.28
    ax.bar(x - w, comp_df["cv_f1_mean"], w, yerr=comp_df["cv_f1_std"], capsize=3, label="5-Fold CV F1 (Train)", color="#42a5f5", edgecolor="black")
    ax.bar(x, comp_df["f1_pos"], w, label="Val F1-Score (Lớp 1)", color="#e53935", edgecolor="black")
    ax.bar(x + w, comp_df["pr_auc"], w, label="Val PR-AUC", color="#7e57c2", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=9.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Điểm số (Score)")
    ax.set_title("So sánh hiệu năng 9 mô hình phân loại nhị phân trên tập Validation (P5)", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "P5_f1_prauc_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: Overlaid ROC & PR curves for the 9 word_tfidf models
    fig_roc_pr = plot_roc_pr(val_scores_by_model, y_val, save_path=FIG_DIR / "P5_roc_pr_curves.png")
    plt.close(fig_roc_pr)

    # Figure 3: 3x3 Confusion Matrix Grid for all 9 word_tfidf models
    fig, axes = plt.subplots(3, 3, figsize=(14, 12), dpi=150)
    axes_flat = axes.ravel()
    word_tfidf_df = comp_df[comp_df["vectorizer"] == "word_tfidf"].reset_index(drop=True)
    for idx, row in word_tfidf_df.iterrows():
        if idx >= 9:
            break
        ax_curr = axes_flat[idx]
        m_name = row["model_name"]
        sc = val_scores_by_model[m_name]
        y_pred = (sc >= 0.5).astype(int)
        cm = confusion_matrix(y_val, y_pred, labels=[0, 1])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["0 (Thường)", "1 (Phản cảm)"],
            yticklabels=["0 (Thường)", "1 (Phản cảm)"],
            cbar=False,
            ax=ax_curr,
        )
        ax_curr.set_title(
            f"{m_name}\nF1={row['f1_pos']:.3f} | P={row['precision_pos']:.3f} | R={row['recall_pos']:.3f}",
            fontsize=10,
            fontweight="bold",
        )
        ax_curr.set_xlabel("Dự đoán")
        ax_curr.set_ylabel("Thực tế")

    plt.suptitle("Lưới 3×3 Ma trận nhầm lẫn của 9 mô hình trên tập Validation (word_tfidf)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "P5_confusion_grid_3x3.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Unique top-3 model keys to advance to P6
    top3_models = []
    for mk in comp_df["model_key"]:
        if mk not in top3_models:
            top3_models.append(mk)
        if len(top3_models) == 3:
            break

    summary = {
        "top_3_model_keys": top3_models,
        "best_untuned_run": comp_df.iloc[0].to_dict(),
        "paired_ttests": paired_tests,
        "all_runs": comparison_rows,
    }
    with open(REPORTS_DIR / "p5_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


if __name__ == "__main__":
    res = run_p5_model_comparison()
    print("P5 finished. Best untuned:", res["best_untuned_run"]["run_id"], "F1=", res["best_untuned_run"]["f1_pos"])
