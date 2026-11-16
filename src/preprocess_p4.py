"""P4 Preprocessing, Near-Duplicate & Conflict Removal, Stratified Splitting, and Class Balancing (P4)."""

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
from sklearn.model_selection import StratifiedKFold, train_test_split
import yaml

from src.clean import clean_text
from src.evaluate import compute_metrics
from src.models import build_pipeline, get_model_scores
from src.utils import get_project_root, load_config, log_experiment, mask_profanity, set_seed

PROJECT_ROOT = get_project_root()
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
PROC_DIR = PROJECT_ROOT / "data" / "processed"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
TAB_DIR = PROJECT_ROOT / "reports" / "tables"


def create_clean_splits(force: bool = False) -> dict[str, Any]:
    """Clean hate_train.csv, remove conflicting/duplicate text_clean, and create 70/15/15 splits."""
    set_seed(42)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "hate_train.csv")
    n_initial = int(len(raw_df))

    raw_df["text"] = raw_df["text"].fillna("").astype(str)
    raw_df["text_clean"] = raw_df["text"].map(lambda s: clean_text(s, remove_stopwords=False, lemmatize=True))

    # 1. Drop empty text_clean rows
    empty_mask = raw_df["text_clean"].str.strip() == ""
    n_empty_dropped = int(empty_mask.sum())
    df_nonempty = raw_df[~empty_mask].copy().reset_index(drop=True)

    # 2. Identify conflicting labels for identical text_clean
    label_nunique = df_nonempty.groupby("text_clean")["label"].nunique()
    conflicting_texts = set(label_nunique[label_nunique > 1].index)
    conflict_rows = df_nonempty[df_nonempty["text_clean"].isin(conflicting_texts)].copy()
    conflict_rows_masked = conflict_rows.copy()
    conflict_rows_masked["text"] = conflict_rows_masked["text"].map(mask_profanity)
    conflict_rows_masked["text_clean"] = conflict_rows_masked["text_clean"].map(mask_profanity)
    conflict_rows_masked.sort_values("text_clean").to_csv(
        TAB_DIR / "P4_conflicting_labels.csv", index=False, encoding="utf-8"
    )
    n_conflict_groups = int(len(conflicting_texts))
    n_conflict_rows_dropped = int(len(conflict_rows))

    # Remove all rows belonging to conflicting groups
    df_no_conflict = df_nonempty[~df_nonempty["text_clean"].isin(conflicting_texts)].copy()

    # 3. Drop near-duplicates (identical text_clean with consistent label)
    n_before_dedup = int(len(df_no_conflict))
    df_dedup = df_no_conflict.drop_duplicates(subset=["text_clean"], keep="first").copy().reset_index(drop=True)
    n_dup_dropped = int(n_before_dedup - len(df_dedup))
    n_final_clean = int(len(df_dedup))

    # Save full processed dataset
    df_dedup[["id", "text", "text_clean", "label"]].to_csv(
        PROC_DIR / "clean.csv", index=False, encoding="utf-8"
    )

    # Save 12 before/after cleaning examples (profanity masked) for notebook & report
    sample_examples = pd.concat(
        [
            df_dedup[df_dedup["label"] == 0].head(6),
            df_dedup[df_dedup["label"] == 1].head(6),
        ],
        ignore_index=True,
    )
    examples_out = pd.DataFrame(
        {
            "id": sample_examples["id"],
            "label": sample_examples["label"],
            "text_raw_masked": sample_examples["text"].map(mask_profanity),
            "text_clean_masked": sample_examples["text_clean"].map(mask_profanity),
        }
    )
    examples_out.to_csv(TAB_DIR / "P4_clean_examples.csv", index=False, encoding="utf-8")

    # 4. Stratified 70 / 15 / 15 split (seed=42)
    train_val_df, test_df = train_test_split(
        df_dedup[["id", "text", "text_clean", "label"]],
        test_size=0.15,
        stratify=df_dedup["label"],
        random_state=42,
    )
    val_rel_size = 0.15 / 0.85
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_rel_size,
        stratify=train_val_df["label"],
        random_state=42,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df.to_csv(SPLITS_DIR / "train.csv", index=False, encoding="utf-8")
    val_df.to_csv(SPLITS_DIR / "val.csv", index=False, encoding="utf-8")
    test_df.to_csv(SPLITS_DIR / "test.csv", index=False, encoding="utf-8")

    # Plot word length before vs after cleaning
    raw_words = df_dedup["text"].str.split().str.len()
    clean_words = df_dedup["text_clean"].str.split().str.len()
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    ax.hist(raw_words, bins=35, alpha=0.55, label=f"Trước làm sạch (Mean={raw_words.mean():.1f} từ)", color="#1976d2")
    ax.hist(clean_words, bins=35, alpha=0.55, label=f"Sau làm sạch (Mean={clean_words.mean():.1f} từ)", color="#43a047")
    ax.set_title("Phân bố độ dài số từ trước và sau khi làm sạch (P4)", fontweight="bold")
    ax.set_xlabel("Số từ / tweet")
    ax.set_ylabel("Số lượng tweet")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "P4_length_before_after.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    split_summary = {
        "n_initial_raw": n_initial,
        "n_empty_dropped": n_empty_dropped,
        "n_conflict_groups": n_conflict_groups,
        "n_conflict_rows_dropped": n_conflict_rows_dropped,
        "n_near_duplicate_rows_dropped": n_dup_dropped,
        "n_total_dropped": int(n_initial - n_final_clean),
        "n_final_unique_clean": n_final_clean,
        "final_label_0": int((df_dedup["label"] == 0).sum()),
        "final_label_1": int((df_dedup["label"] == 1).sum()),
        "final_pos_rate_pct": round(100.0 * float((df_dedup["label"] == 1).mean()), 4),
        "splits": {
            "train": {
                "n_rows": int(len(train_df)),
                "label_0": int((train_df["label"] == 0).sum()),
                "label_1": int((train_df["label"] == 1).sum()),
                "pos_rate_pct": round(100.0 * float((train_df["label"] == 1).mean()), 4),
            },
            "val": {
                "n_rows": int(len(val_df)),
                "label_0": int((val_df["label"] == 0).sum()),
                "label_1": int((val_df["label"] == 1).sum()),
                "pos_rate_pct": round(100.0 * float((val_df["label"] == 1).mean()), 4),
            },
            "test": {
                "n_rows": int(len(test_df)),
                "label_0": int((test_df["label"] == 0).sum()),
                "label_1": int((test_df["label"] == 1).sum()),
                "pos_rate_pct": round(100.0 * float((test_df["label"] == 1).mean()), 4),
            },
        },
        "word_len_before_mean": round(float(raw_words.mean()), 2),
        "word_len_after_mean": round(float(clean_words.mean()), 2),
    }
    return split_summary


def compare_balancing_strategies() -> dict[str, Any]:
    """Compare 6 class-balancing strategies using 5-fold StratifiedKFold CV on train.csv only (P4.4)."""
    set_seed(42)
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    X_train = train_df["text_clean"].fillna("").astype(str).to_numpy()
    y_train = train_df["label"].astype(int).to_numpy()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    strategies = [
        ("none", "Không cân bằng (None)"),
        ("class_weight", "Trọng số lớp (class_weight='balanced')"),
        ("ros", "RandomOverSampler (ROS)"),
        ("smote", "SMOTE (k=5)"),
        ("rus", "RandomUnderSampler (RUS)"),
        ("smote_tomek", "SMOTETomek"),
    ]

    rows = []
    for strat_key, strat_desc in strategies:
        fold_metrics = []
        t0 = time.perf_counter()
        for tr_idx, va_idx in skf.split(X_train, y_train):
            X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
            X_va, y_va = X_train[va_idx], y_train[va_idx]

            pipe = build_pipeline("lr", vectorizer_kind="word_tfidf", balance=strat_key, seed=42)
            pipe.fit(X_tr, y_tr)
            scores = get_model_scores(pipe, X_va)
            m = compute_metrics(y_va, scores, threshold=0.5)
            fold_metrics.append(m)

        elapsed = time.perf_counter() - t0
        df_folds = pd.DataFrame(fold_metrics)

        row = {
            "strategy": strat_key,
            "description": strat_desc,
            "precision_mean": round(float(df_folds["precision_pos"].mean()), 4),
            "precision_std": round(float(df_folds["precision_pos"].std()), 4),
            "recall_mean": round(float(df_folds["recall_pos"].mean()), 4),
            "recall_std": round(float(df_folds["recall_pos"].std()), 4),
            "f1_pos_mean": round(float(df_folds["f1_pos"].mean()), 4),
            "f1_pos_std": round(float(df_folds["f1_pos"].std()), 4),
            "pr_auc_mean": round(float(df_folds["pr_auc"].mean()), 4),
            "pr_auc_std": round(float(df_folds["pr_auc"].std()), 4),
            "roc_auc_mean": round(float(df_folds["roc_auc"].mean()), 4),
            "fit_seconds": round(elapsed, 2),
        }
        rows.append(row)

        log_experiment(
            model="lr_baseline_balance_check",
            vectorizer="word_tfidf",
            balance=strat_key,
            params={"solver": "liblinear", "max_iter": 2000},
            cv_f1_mean=row["f1_pos_mean"],
            cv_f1_std=row["f1_pos_std"],
            val_f1=None,
            val_pr_auc=None,
            fit_seconds=elapsed,
        )

    bal_df = pd.DataFrame(rows).sort_values("f1_pos_mean", ascending=False).reset_index(drop=True)
    bal_df.to_csv(TAB_DIR / "P4_balance_comparison.csv", index=False, encoding="utf-8")

    # Select best strategy by CV f1_pos_mean (if within 0.005 of class_weight, prefer class_weight for simplicity)
    best_row = bal_df.iloc[0]
    cw_row = bal_df[bal_df["strategy"] == "class_weight"].iloc[0]
    if float(best_row["f1_pos_mean"]) - float(cw_row["f1_pos_mean"]) < 0.005:
        chosen_strategy = "class_weight"
        chosen_reason = (
            f"Chiến lược '{best_row['strategy']}' (F1={best_row['f1_pos_mean']:.4f}) và "
            f"'class_weight' (F1={cw_row['f1_pos_mean']:.4f}) chênh lệch < 0.5 điểm %, "
            "nên ưu tiên chọn 'class_weight' vì nhanh hơn, không sinh mẫu nhân tạo và tương thích mọi pipeline."
        )
    else:
        chosen_strategy = str(best_row["strategy"])
        chosen_reason = (
            f"Chiến lược '{chosen_strategy}' đạt CV F1 lớp dương cao nhất "
            f"({best_row['f1_pos_mean']:.4f} ± {best_row['f1_pos_std']:.4f}) và PR-AUC = {best_row['pr_auc_mean']:.4f} "
            f"trên 5-fold StratifiedKFold tập train."
        )

    # Update config.yaml with chosen balance strategy
    cfg = load_config()
    cfg["balance"] = chosen_strategy
    with open(PROJECT_ROOT / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    # Plot comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5), dpi=150)
    x = np.arange(len(bal_df))
    w = 0.26
    axes[0].bar(x - w, bal_df["precision_mean"], w, label="Precision (Lớp 1)", color="#1e88e5", edgecolor="black")
    axes[0].bar(x, bal_df["recall_mean"], w, label="Recall (Lớp 1)", color="#fb8c00", edgecolor="black")
    axes[0].bar(
        x + w,
        bal_df["f1_pos_mean"],
        w,
        yerr=bal_df["f1_pos_std"],
        capsize=4,
        label="F1-Score (Lớp 1)",
        color="#e53935",
        edgecolor="black",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(bal_df["strategy"], rotation=20, ha="right")
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("Điểm số (5-fold CV trên Train)")
    axes[0].set_title("So sánh 6 kỹ thuật cân bằng dữ liệu (LR + Word TF-IDF)", fontweight="bold")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(
        bal_df["strategy"],
        bal_df["fit_seconds"],
        color="#5e35b1",
        edgecolor="black",
        width=0.5,
    )
    axes[1].set_title("Thời gian huấn luyện 5-fold CV theo kỹ thuật cân bằng (giây)", fontweight="bold")
    axes[1].set_ylabel("Thời gian (giây)")
    axes[1].set_xticks(range(len(bal_df)))
    axes[1].set_xticklabels(bal_df["strategy"], rotation=20, ha="right")
    for idx, val in enumerate(bal_df["fit_seconds"]):
        axes[1].text(idx, val + 0.15, f"{val:.1f}s", ha="center", fontsize=9, fontweight="bold")
    axes[1].set_ylim(0, max(bal_df["fit_seconds"]) * 1.2)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "P4_balance_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "chosen_strategy": chosen_strategy,
        "chosen_reason": chosen_reason,
        "table": rows,
    }


def run_p4() -> dict[str, Any]:
    split_info = create_clean_splits()
    bal_info = compare_balancing_strategies()
    out = {"split_summary": split_info, "balance_comparison": bal_info}
    with open(PROJECT_ROOT / "reports" / "p4_summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out


if __name__ == "__main__":
    res = run_p4()
    print("P4 finished. Chosen balance:", res["balance_comparison"]["chosen_strategy"])
