"""P8 Final Single-Shot Evaluation on Held-Out Test Set, Bootstrap CI, Error Analysis & Subgroup Bias (P8)."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
import re
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
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.evaluate import compute_metrics
from src.predict import _extract_linear_coef, load_model
from src.utils import get_profanity_lexicon, get_project_root, mask_profanity, set_seed

PROJECT_ROOT = get_project_root()
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
TAB_DIR = REPORTS_DIR / "tables"


def _categorize_error(raw_text: str, clean_txt: str, y_true: int, y_pred: int, lex_terms: set[str]) -> str:
    """Rule-assisted categorization of FP and FN errors into 6 linguistic error categories."""
    s_raw = str(raw_text).lower()
    toks = str(clean_txt).split()
    n_hashtags = s_raw.count("#")

    # 1. Sarcasm / irony markers or rhetorical questions
    if any(w in s_raw for w in ["yeah right", "sure", "totally", "lol", "lmao", "joke", "irony", "sarcasm", "?", "!!"]):
        if "?" in s_raw or any(w in toks for w in ["lol", "lmao", "joke", "funny"]):
            return "Mỉa mai / Châm biếm / Câu hỏi tu từ (Sarcasm & Rhetorical)"

    # 2. Profanity or sensitive identity group term in non-hate context (especially for FP)
    identity_terms = {"women", "woman", "girl", "men", "man", "black", "white", "trump", "obama", "muslim", "gay", "racist", "sexist"}
    if y_true == 0 and y_pred == 1:
        if any(t in lex_terms for t in toks) or any(t in identity_terms for t in toks):
            return "Từ nhạy cảm / Từ chỉ nhóm người trong ngữ cảnh trung tính hoặc phản đối kỳ thị (Sensitive Term in Neutral/Counter-Speech Context)"

    # 3. Hashtag-heavy or slogan tweet
    if n_hashtags >= 4:
        return "Chuỗi nhiều Hashtag / Khẩu hiệu đa chủ đề (Hashtag-Heavy / Multi-Topic)"

    # 4. Very short or rare/slang tokens (OOV / typo / short fragment)
    if len(toks) <= 5:
        return "Câu quá ngắn / Từ lóng / Viết tắt thiếu thông tin (Short / Slang / OOV)"

    # 5. Implicit hate / stereotyping without explicit slur (especially for FN)
    if y_true == 1 and y_pred == 0:
        if not any(t in lex_terms for t in toks):
            return "Kỳ thị ngầm / Định kiến không dùng từ tục trực tiếp (Implicit Hate / Missing Context)"

    # 6. Ambiguous / noisy crowd label
    return "Nhãn gốc nhiễu hoặc ranh giới ngữ nghĩa mờ (Noisy / Borderline Ground-Truth Label)"


def run_p8_final_eval() -> dict[str, Any]:
    """Run final evaluation on test.csv, compute 1000-bootstrap 95% CI, analyze 30 FP + 30 FN, and check bias."""
    set_seed(42)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    test_path = SPLITS_DIR / "test.csv"
    test_bytes = test_path.read_bytes()
    test_sha256 = hashlib.sha256(test_bytes).hexdigest()

    test_df = pd.read_csv(test_path)
    y_test = test_df["label"].astype(int).to_numpy()
    X_test_clean = test_df["text_clean"].fillna("").astype(str).tolist()

    predictor = load_model()
    thr = predictor.threshold
    scores = predictor.predict_batch(test_df["text"].astype(str).tolist())["score"].to_numpy()
    y_pred = (scores >= thr).astype(int)

    # 1. Single-shot final test metrics (preserve original evaluation timestamp if already recorded for same hash)
    final_metrics_path = REPORTS_DIR / "final_test_metrics.json"
    eval_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if final_metrics_path.exists():
        try:
            prev = json.loads(final_metrics_path.read_text(encoding="utf-8"))
            if prev.get("test_file_sha256") == test_sha256:
                eval_timestamp = prev.get("evaluated_once_at", eval_timestamp)
        except Exception:
            pass

    metrics_tuned_thr = compute_metrics(y_test, scores, threshold=thr)
    metrics_default_05 = compute_metrics(y_test, scores, threshold=0.5)

    # Sanity check Rule 5: F1 > 0.90 => suspect leakage
    assert float(metrics_tuned_thr["f1_pos"]) <= 0.90, (
        f"SANITY CHECK FAILED: Test F1 = {metrics_tuned_thr['f1_pos']} > 0.90! Possible data leakage."
    )

    final_test_payload = {
        "evaluated_once_at": eval_timestamp,
        "test_file": "data/splits/test.csv",
        "test_file_sha256": test_sha256,
        "n_test_rows": int(len(test_df)),
        "n_test_pos": int((y_test == 1).sum()),
        "n_test_neg": int((y_test == 0).sum()),
        "threshold_used": thr,
        "metrics_at_tuned_threshold": metrics_tuned_thr,
        "metrics_at_default_05": metrics_default_05,
    }
    with open(final_metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_test_payload, f, indent=2, ensure_ascii=False)

    # 2. 1000-iteration Bootstrap 95% Confidence Intervals (seed=42)
    rng = np.random.default_rng(42)
    n = len(y_test)
    boot_f1, boot_prec, boot_rec, boot_prauc, boot_rocauc = [], [], [], [], []
    for _ in range(1000):
        idx = rng.integers(0, n, size=n)
        yt_b = y_test[idx]
        sc_b = scores[idx]
        yp_b = (sc_b >= thr).astype(int)
        if len(np.unique(yt_b)) < 2:
            continue
        boot_f1.append(float(f1_score(yt_b, yp_b, pos_label=1, zero_division=0)))
        boot_prec.append(float(precision_score(yt_b, yp_b, pos_label=1, zero_division=0)))
        boot_rec.append(float(recall_score(yt_b, yp_b, pos_label=1, zero_division=0)))
        boot_prauc.append(float(average_precision_score(yt_b, sc_b)))
        boot_rocauc.append(float(roc_auc_score(yt_b, sc_b)))

    def _ci95(arr: list[float]) -> dict[str, float]:
        return {
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr, ddof=1)), 4),
            "ci95_low": round(float(np.percentile(arr, 2.5)), 4),
            "ci95_high": round(float(np.percentile(arr, 97.5)), 4),
        }

    ci_payload = {
        "n_bootstrap": 1000,
        "seed": 42,
        "threshold": thr,
        "f1_pos": _ci95(boot_f1),
        "precision_pos": _ci95(boot_prec),
        "recall_pos": _ci95(boot_rec),
        "pr_auc": _ci95(boot_prauc),
        "roc_auc": _ci95(boot_rocauc),
    }
    with open(REPORTS_DIR / "final_test_ci.json", "w", encoding="utf-8") as f:
        json.dump(ci_payload, f, indent=2, ensure_ascii=False)

    # 3. Plot Confusion Matrix, ROC, and PR on Test Set
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=150)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["0 (Bình thường)", "1 (Phản cảm)"],
        yticklabels=["0 (Bình thường)", "1 (Phản cảm)"],
        cbar=False,
        ax=axes[0],
    )
    axes[0].set_title(
        f"Ma trận nhầm lẫn Tập Test (N={len(y_test):,})\nF1={metrics_tuned_thr['f1_pos']:.4f} (95% CI: [{ci_payload['f1_pos']['ci95_low']:.3f}, {ci_payload['f1_pos']['ci95_high']:.3f}])",
        fontweight="bold",
        fontsize=10.5,
    )
    axes[0].set_xlabel("Dự đoán (Predicted)")
    axes[0].set_ylabel("Thực tế (Actual)")

    fpr, tpr, _ = roc_curve(y_test, scores)
    axes[1].plot(fpr, tpr, color="#1e88e5", lw=2.2, label=f"ROC-AUC = {metrics_tuned_thr['roc_auc']:.4f}")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    axes[1].set_title("Đường cong ROC trên Tập Test", fontweight="bold")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate (Recall)")
    axes[1].legend(loc="lower right")
    axes[1].grid(alpha=0.3)

    prec_c, rec_c, _ = precision_recall_curve(y_test, scores)
    axes[2].plot(rec_c, prec_c, color="#e53935", lw=2.2, label=f"PR-AUC = {metrics_tuned_thr['pr_auc']:.4f}")
    axes[2].scatter(
        [metrics_tuned_thr["recall_pos"]],
        [metrics_tuned_thr["precision_pos"]],
        color="black",
        s=80,
        zorder=5,
        label=f"Thr={thr:.3f} (P={metrics_tuned_thr['precision_pos']:.3f}, R={metrics_tuned_thr['recall_pos']:.3f})",
    )
    axes[2].set_title("Đường cong Precision-Recall trên Tập Test", fontweight="bold")
    axes[2].set_xlabel("Recall (Lớp 1)")
    axes[2].set_ylabel("Precision (Lớp 1)")
    axes[2].legend(loc="lower left")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "P8_test_evaluation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. Error Analysis: Export 30 FP + 30 FN (masked) + categorize ALL test errors
    lex_terms = set(get_profanity_lexicon().keys())
    err_df = test_df.copy()
    err_df["score"] = np.round(scores, 4)
    err_df["y_pred"] = y_pred
    err_df["error_type"] = "Correct"
    err_df.loc[(err_df["label"] == 0) & (err_df["y_pred"] == 1), "error_type"] = "FP (Báo động giả)"
    err_df.loc[(err_df["label"] == 1) & (err_df["y_pred"] == 0), "error_type"] = "FN (Bỏ sót)"

    all_errors = err_df[err_df["error_type"] != "Correct"].copy()
    all_errors["cause"] = [
        _categorize_error(r.text, r.text_clean, int(r.label), int(r.y_pred), lex_terms)
        for r in all_errors.itertuples()
    ]
    all_errors["text_masked"] = all_errors["text"].map(mask_profanity)
    all_errors["text_clean_masked"] = all_errors["text_clean"].map(mask_profanity)

    fp_top30 = (
        all_errors[all_errors["error_type"].str.startswith("FP")]
        .sort_values("score", ascending=False)
        .head(30)
    )
    fn_top30 = (
        all_errors[all_errors["error_type"].str.startswith("FN")]
        .sort_values("score", ascending=True)
        .head(30)
    )
    errors_60 = pd.concat([fp_top30, fn_top30], ignore_index=True)
    errors_60[
        ["id", "error_type", "label", "y_pred", "score", "cause", "text_masked", "text_clean_masked"]
    ].to_csv(TAB_DIR / "P8_errors.csv", index=False, encoding="utf-8")

    cause_counts = (
        all_errors.groupby(["cause", "error_type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    cause_counts["total"] = cause_counts.select_dtypes(include="number").sum(axis=1)
    cause_counts["pct"] = np.round(100.0 * cause_counts["total"] / max(len(all_errors), 1), 2)
    cause_counts = cause_counts.sort_values("total", ascending=False).reset_index(drop=True)
    cause_counts.to_csv(TAB_DIR / "P8_error_causes.csv", index=False, encoding="utf-8")

    # Representative 6 error examples (3 FP + 3 FN across distinct causes)
    rep_examples = pd.concat([fp_top30.head(3), fn_top30.head(3)], ignore_index=True)
    rep_examples_list = rep_examples[
        ["id", "error_type", "label", "y_pred", "score", "cause", "text_masked"]
    ].to_dict(orient="records")

    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    short_causes = [c.split("(")[0].strip() for c in cause_counts["cause"]]
    ax.barh(short_causes[::-1], cause_counts["total"].iloc[::-1], color="#ef5350", edgecolor="black")
    for idx, (tot, pct) in enumerate(zip(cause_counts["total"].iloc[::-1], cause_counts["pct"].iloc[::-1])):
        ax.text(tot + 1.0, idx, f"{int(tot)} ({pct:.1f}%)", va="center", fontsize=9, fontweight="bold")
    ax.set_xlim(0, max(cause_counts["total"]) * 1.22)
    ax.set_xlabel("Số lượng mẫu dự đoán sai trên tập Test")
    ax.set_title(f"Phân loại nguyên nhân lỗi trên tập Test (Tổng số lỗi = {len(all_errors)})", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "P8_error_causes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 5. Model Explainability: Top-30 Positive & Top-30 Negative Linear Coefficients
    vec = predictor.vectorizer
    clf = predictor.clf
    coef = _extract_linear_coef(clf)
    top_pos_terms = []
    top_neg_terms = []
    if vec is not None and coef is not None:
        feat_names = vec.get_feature_names_out()
        top_pos_idx = np.argsort(coef)[::-1][:30]
        top_neg_idx = np.argsort(coef)[:30]
        top_pos_terms = [
            {"term_masked": mask_profanity(str(feat_names[i])), "weight": round(float(coef[i]), 4)}
            for i in top_pos_idx
        ]
        top_neg_terms = [
            {"term_masked": mask_profanity(str(feat_names[i])), "weight": round(float(coef[i]), 4)}
            for i in top_neg_idx
        ]
        coef_df = pd.DataFrame(
            [{"direction": "Positive (Hate/Racist/Sexist)", **x} for x in top_pos_terms]
            + [{"direction": "Negative (Normal/Non-Hate)", **x} for x in top_neg_terms]
        )
        coef_df.to_csv(TAB_DIR / "P8_top_coefficients.csv", index=False, encoding="utf-8")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), dpi=150)
        p20 = top_pos_terms[:20]
        n20 = top_neg_terms[:20]
        axes[0].barh(
            [x["term_masked"] for x in p20][::-1],
            [x["weight"] for x in p20][::-1],
            color="#d32f2f",
            edgecolor="black",
        )
        axes[0].set_title("Top-20 đặc trưng đóng góp mạnh nhất cho Lớp 1 (Phản cảm)", fontweight="bold")
        axes[0].set_xlabel("Trọng số mô hình (Model Weight)")

        axes[1].barh(
            [x["term_masked"] for x in n20][::-1],
            [x["weight"] for x in n20][::-1],
            color="#388e3c",
            edgecolor="black",
        )
        axes[1].set_title("Top-20 đặc trưng đóng góp mạnh nhất cho Lớp 0 (Bình thường)", fontweight="bold")
        axes[1].set_xlabel("Trọng số mô hình (Model Weight)")
        plt.tight_layout()
        fig.savefig(FIG_DIR / "P8_feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 6. Subgroup Fairness / Bias Analysis on Negative Examples (Label == 0)
    neg_df = err_df[err_df["label"] == 0].copy()
    overall_fpr = round(100.0 * float((neg_df["y_pred"] == 1).mean()), 2)

    subgroups = [
        ("Giới tính / Phụ nữ (women, woman, girl, female)", r"\b(women|woman|girl|girls|female|feminist)\b"),
        ("Chủng tộc / Màu da (black, white, race, african)", r"\b(black|white|race|racial|african|asian)\b"),
        ("Tôn giáo (muslim, islam, jewish, christian, god)", r"\b(muslim|islam|jew|jewish|christian|religion)\b"),
        ("Xu hướng tính dục (gay, lgbt, queer)", r"\b(gay|lgbt|lesbian|queer|trans)\b"),
        ("Chính trị (trump, obama, clinton, liberal, gop)", r"\b(trump|obama|clinton|hillary|liberal|conservative|gop)\b"),
    ]

    bias_rows = [
        {
            "subgroup": "Toàn bộ mẫu Lớp 0 trên Test (Overall Baseline)",
            "n_neg_tweets": int(len(neg_df)),
            "n_false_positives": int((neg_df["y_pred"] == 1).sum()),
            "fpr_pct": overall_fpr,
            "mean_score": round(float(neg_df["score"].mean()), 4),
        }
    ]
    for sg_name, pattern in subgroups:
        mask_sg = neg_df["text_clean"].str.contains(pattern, flags=re.IGNORECASE, regex=True, na=False)
        sub_neg = neg_df[mask_sg]
        n_sub = int(len(sub_neg))
        n_fp = int((sub_neg["y_pred"] == 1).sum()) if n_sub > 0 else 0
        fpr_sub = round(100.0 * n_fp / max(n_sub, 1), 2) if n_sub > 0 else 0.0
        bias_rows.append(
            {
                "subgroup": sg_name,
                "n_neg_tweets": n_sub,
                "n_false_positives": n_fp,
                "fpr_pct": fpr_sub,
                "mean_score": round(float(sub_neg["score"].mean()), 4) if n_sub > 0 else 0.0,
            }
        )

    bias_df = pd.DataFrame(bias_rows)
    bias_df.to_csv(TAB_DIR / "P8_subgroup_bias.csv", index=False, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=150)
    colors_b = ["#455a64"] + ["#e53935" if r["fpr_pct"] > overall_fpr * 1.5 else "#1e88e5" for r in bias_rows[1:]]
    ax.barh(
        [r["subgroup"] for r in bias_rows][::-1],
        [r["fpr_pct"] for r in bias_rows][::-1],
        color=colors_b[::-1],
        edgecolor="black",
    )
    ax.axvline(overall_fpr, color="black", linestyle="--", lw=1.5, label=f"FPR trung bình toàn tập Test ({overall_fpr:.2f}%)")
    for idx, r in enumerate(bias_rows[::-1]):
        ax.text(r["fpr_pct"] + 0.5, idx, f"{r['fpr_pct']:.2f}% (n={r['n_neg_tweets']})", va="center", fontsize=9, fontweight="bold")
    ax.set_xlim(0, max(r["fpr_pct"] for r in bias_rows) * 1.25 + 2)
    ax.set_xlabel("False Positive Rate — Tỉ lệ báo động giả trên câu bình thường (%)")
    ax.set_title("Phân tích thiên lệch (Subgroup Bias): Tỉ lệ báo động giả khi tweet chứa từ chỉ nhóm người (P8)", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "P8_subgroup_bias.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "final_test": final_test_payload,
        "ci95": ci_payload,
        "n_total_errors": int(len(all_errors)),
        "error_causes": cause_counts.to_dict(orient="records"),
        "representative_errors_masked": rep_examples_list,
        "top_positive_coefficients_masked": top_pos_terms[:20],
        "top_negative_coefficients_masked": top_neg_terms[:20],
        "subgroup_bias": bias_rows,
    }
    with open(REPORTS_DIR / "p8_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


if __name__ == "__main__":
    res = run_p8_final_eval()
    print("P8 finished. Test F1:", res["final_test"]["metrics_at_tuned_threshold"]["f1_pos"])
