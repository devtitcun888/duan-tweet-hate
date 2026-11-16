"""Decision threshold optimization on validation probabilities (P6)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score


def pick_threshold(
    y_val: np.ndarray | list[int],
    score_val: np.ndarray | list[float],
    strategy: str = "f1",
    save_path: str | Path | None = None,
) -> dict[str, Any]:
    """Scan thresholds in [0.05, 0.95] on validation scores to maximize the chosen metric.

    Args:
        y_val: Validation ground-truth labels (0/1). NEVER pass test labels here.
        score_val: Validation predicted probabilities for class 1.
        strategy: 'f1' (default) or 'f2' (beta=2, prioritizing recall).
        save_path: Optional path to save JSON result (e.g., models/threshold.json).
    """
    y_true = np.asarray(y_val, dtype=int)
    scores = np.asarray(score_val, dtype=float)

    thresholds = np.round(np.arange(0.05, 0.955, 0.005), 4)
    curve: list[dict[str, float]] = []

    best_thr = 0.5
    best_obj = -1.0
    best_metrics = {"f1": 0.0, "f2": 0.0, "precision": 0.0, "recall": 0.0}

    for thr in thresholds:
        preds = (scores >= thr).astype(int)
        prec = float(precision_score(y_true, preds, pos_label=1, zero_division=0))
        rec = float(recall_score(y_true, preds, pos_label=1, zero_division=0))
        f1 = float(f1_score(y_true, preds, pos_label=1, zero_division=0))
        f2 = float(fbeta_score(y_true, preds, beta=2.0, pos_label=1, zero_division=0))

        obj = f2 if strategy == "f2" else f1
        if obj > best_obj:
            best_obj = obj
            best_thr = float(thr)
            best_metrics = {
                "f1": round(f1, 4),
                "f2": round(f2, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
            }

        curve.append(
            {
                "threshold": float(thr),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "f2": round(f2, 4),
            }
        )

    preds_05 = (scores >= 0.5).astype(int)
    default_05 = {
        "threshold": 0.5,
        "precision": round(float(precision_score(y_true, preds_05, pos_label=1, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, preds_05, pos_label=1, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, preds_05, pos_label=1, zero_division=0)), 4),
    }

    result = {
        "strategy": strategy,
        "best_threshold": round(best_thr, 4),
        "best_val_f1": best_metrics["f1"],
        "best_val_f2": best_metrics["f2"],
        "best_val_precision": best_metrics["precision"],
        "best_val_recall": best_metrics["recall"],
        "default_05": default_05,
        "curve": curve,
    }

    if save_path is not None:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    return result
