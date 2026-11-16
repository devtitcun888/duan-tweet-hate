"""Evaluation metrics and visualization helpers for binary hate-speech classification (P5.1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(
    y_true: np.ndarray | list[int],
    y_score: np.ndarray | list[float],
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Compute full suite of binary classification metrics for imbalanced hate-speech detection.

    Returns dict with keys:
      precision_pos, recall_pos, f1_pos, f1_macro, accuracy, balanced_acc,
      roc_auc, pr_auc, mcc, tn, fp, fn, tp
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)
    y_pred = (y_score_arr >= threshold).astype(int)

    precision_pos = float(precision_score(y_true_arr, y_pred, pos_label=1, zero_division=0))
    recall_pos = float(recall_score(y_true_arr, y_pred, pos_label=1, zero_division=0))
    f1_pos = float(f1_score(y_true_arr, y_pred, pos_label=1, zero_division=0))
    f1_macro = float(f1_score(y_true_arr, y_pred, average="macro", zero_division=0))
    acc = float(accuracy_score(y_true_arr, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true_arr, y_pred))
    mcc = float(matthews_corrcoef(y_true_arr, y_pred))

    if len(np.unique(y_true_arr)) > 1:
        roc_auc = float(roc_auc_score(y_true_arr, y_score_arr))
        pr_auc = float(average_precision_score(y_true_arr, y_score_arr))
    else:
        roc_auc = 0.0
        pr_auc = 0.0

    cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in cm.ravel())

    return {
        "precision_pos": round(precision_pos, 4),
        "recall_pos": round(recall_pos, 4),
        "f1_pos": round(f1_pos, 4),
        "f1_macro": round(f1_macro, 4),
        "accuracy": round(acc, 4),
        "balanced_acc": round(bal_acc, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "mcc": round(mcc, 4),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def plot_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Ma trận nhầm lẫn",
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """Plot a labeled confusion matrix on the given axes."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["0 (Bình thường)", "1 (Phản cảm)"],
        yticklabels=["0 (Bình thường)", "1 (Phản cảm)"],
        ax=ax,
        cbar=False,
    )
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Dự đoán (Predicted)")
    ax.set_ylabel("Thực tế (Actual)")
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax


def plot_roc_pr(
    models_scores: dict[str, np.ndarray],
    y_true: np.ndarray,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot overlaid ROC and Precision-Recall curves for multiple models."""
    y_true_arr = np.asarray(y_true, dtype=int)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    # Panel 1: ROC curves
    ax_roc = axes[0]
    for name, scores in models_scores.items():
        fpr, tpr, _ = roc_curve(y_true_arr, scores)
        auc_val = roc_auc_score(y_true_arr, scores)
        ax_roc.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc_val:.3f})")
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Ngẫu nhiên (0.500)")
    ax_roc.set_title("Đường cong ROC (ROC Curves)", fontsize=12, fontweight="bold")
    ax_roc.set_xlabel("False Positive Rate (FPR)")
    ax_roc.set_ylabel("True Positive Rate (Recall)")
    ax_roc.legend(loc="lower right", fontsize=8.5)
    ax_roc.grid(alpha=0.3)

    # Panel 2: Precision-Recall curves
    ax_pr = axes[1]
    pos_rate = float(np.mean(y_true_arr))
    for name, scores in models_scores.items():
        prec, rec, _ = precision_recall_curve(y_true_arr, scores)
        ap_val = average_precision_score(y_true_arr, scores)
        ax_pr.plot(rec, prec, lw=2, label=f"{name} (PR-AUC={ap_val:.3f})")
    ax_pr.axhline(pos_rate, color="k", linestyle="--", lw=1, alpha=0.6, label=f"Baseline ({pos_rate:.3f})")
    ax_pr.set_title("Đường cong Precision-Recall (PR Curves)", fontsize=12, fontweight="bold")
    ax_pr.set_xlabel("Recall (Độ phủ lớp 1)")
    ax_pr.set_ylabel("Precision (Độ chính xác lớp 1)")
    ax_pr.legend(loc="upper right", fontsize=8.5)
    ax_pr.grid(alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
