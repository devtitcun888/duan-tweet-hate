"""V5 Vietnamese Final Held-Out Test Evaluation, Bootstrap 95% CI, 3-Version Robustness Test (Original / No Accents / Teencode), and Error Analysis (V5)."""

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

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.evaluate import compute_metrics
from src.models import get_model_scores
from src.predict import _extract_linear_coef
from src.utils import get_project_root, set_seed
from src.vi_clean import load_vi_profanity_lexicon, strip_accents, vi_clean, vi_mask_profanity
from src.vi_data_io import has_vietnamese_accents
from src.vi_modeling_v3 import prepare_task_split

PROJECT_ROOT = get_project_root()
VI_SPLITS_DIR = PROJECT_ROOT / "data" / "splits_vi"
VI_MODELS_DIR = PROJECT_ROOT / "models" / "vi"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
TAB_DIR = REPORTS_DIR / "tables"

REVERSE_TEENCODE = {
    "không": "ko",
    "được": "đc",
    "với": "vs",
    "gì": "j",
    "như thế nào": "ntn",
    "bình thường": "bt",
    "điện thoại": "đt",
    "mình": "mk",
    "người": "ng",
    "bây giờ": "bh",
    "trước": "trc",
    "cũng": "cx",
    "thích": "thik",
    "yêu": "iu",
    "mọi người": "mn",
    "anh em": "ae",
    "việt nam": "vn",
    "facebook": "fb",
    "youtube": "ytb",
    "bình luận": "cmt",
    "rồi": "r",
    "quá": "wa",
    "luôn": "lun",
    "biết": "bik",
    "chưa": "chx",
    "vậy": "z",
    "nhiều": "nhìu",
}


def simulate_teencode_and_repeat(text: str, rng: np.random.Generator) -> str:
    """Deterministically inject teencode abbreviations and character lengthening into a test comment."""
    s = str(text or "")
    words = s.split()
    out_words = []
    for w in words:
        w_low = w.lower()
        if w_low in REVERSE_TEENCODE:
            out_words.append(REVERSE_TEENCODE[w_low])
        elif len(w) >= 3 and w[-1].isalpha() and rng.random() < 0.35:
            # Lengthen last character 4 times (e.g. 'ngu' -> 'nguuuu')
            out_words.append(w + w[-1] * 3)
        else:
            out_words.append(w)
    return " ".join(out_words)


def _categorize_vi_error(raw_text: str, seg_text: str, y_true: int, y_pred: int, lex_set: set[str]) -> str:
    s_raw = str(raw_text).lower()
    toks = set(str(seg_text).split())

    if not has_vietnamese_accents(raw_text):
        return "Bình luận không dấu / Teencode biến thể nặng (Unaccented / Heavy Teencode)"

    if any(m in s_raw for m in ["haha", "hihi", ":)))", "=)))", "đùa", "troll", "mỉa", "chắc", "?"]):
        if "?" in s_raw or ":)" in s_raw or "=)" in s_raw:
            return "Mỉa mai / Châm biếm / Câu hỏi tu từ (Sarcasm & Rhetorical)"

    if y_true == 0 and y_pred == 1:
        if len(toks & lex_set) > 0:
            return "Chửi thề 'thân mật' / Cảm thán hoặc Phản đối kỳ thị chứa từ nhạy cảm (Friendly Profanity / Counter-Speech)"

    if len(toks) <= 4:
        return "Bình luận quá ngắn / Thiếu ngữ cảnh bài đăng gốc (Short / Missing Context)"

    if y_true == 1 and y_pred == 0 and len(toks & lex_set) == 0:
        return "Kỳ thị ngầm / Từ lóng vùng miền hoặc ẩn dụ không dùng từ tục (Implicit Hate / Regional Slang)"

    return "Nhãn gốc nhiễu hoặc ranh giới mờ giữa CLEAN – OFFENSIVE – HATE (Borderline / Noisy Label)"


def run_v5_final_eval() -> dict[str, Any]:
    """Run single-shot final evaluation on ViHSD test.csv for bin_A, bin_B, multi3, robustness test, and error analysis."""
    set_seed(42)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    test_path = VI_SPLITS_DIR / "test.csv"
    test_sha256 = hashlib.sha256(test_path.read_bytes()).hexdigest()
    test_df = pd.read_csv(test_path)

    pipe_a = joblib.load(VI_MODELS_DIR / "best_vi_bin_a.joblib")
    thr_a = float(json.loads((VI_MODELS_DIR / "threshold_vi_bin_a.json").read_text(encoding="utf-8"))["best_threshold"])

    pipe_b = joblib.load(VI_MODELS_DIR / "best_vi_bin_b.joblib")
    thr_b = float(json.loads((VI_MODELS_DIR / "threshold_vi_bin_b.json").read_text(encoding="utf-8"))["best_threshold"])

    pipe_m3 = joblib.load(VI_MODELS_DIR / "best_vi_multi3.joblib")

    # 1. Evaluate bin_A on test split
    X_te_seg_a, _, y_te_a = prepare_task_split(test_df, "bin_A")
    scores_a = get_model_scores(pipe_a, X_te_seg_a)
    metrics_a = compute_metrics(y_te_a, scores_a, threshold=thr_a)
    metrics_a_05 = compute_metrics(y_te_a, scores_a, threshold=0.5)

    # Bootstrap 95% CI (1,000 iterations) for bin_A on test
    rng = np.random.default_rng(42)
    n_a = len(y_te_a)
    b_f1, b_prec, b_rec, b_pr, b_roc = [], [], [], [], []
    for _ in range(1000):
        idx = rng.integers(0, n_a, size=n_a)
        yt_b = y_te_a[idx]
        sc_b = scores_a[idx]
        yp_b = (sc_b >= thr_a).astype(int)
        if len(np.unique(yt_b)) < 2:
            continue
        b_f1.append(float(f1_score(yt_b, yp_b, pos_label=1, zero_division=0)))
        b_prec.append(float(precision_score(yt_b, yp_b, pos_label=1, zero_division=0)))
        b_rec.append(float(recall_score(yt_b, yp_b, pos_label=1, zero_division=0)))
        b_pr.append(float(average_precision_score(yt_b, sc_b)))
        b_roc.append(float(roc_auc_score(yt_b, sc_b)))

    def _ci(arr):
        return {
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr, ddof=1)), 4),
            "ci95_low": round(float(np.percentile(arr, 2.5)), 4),
            "ci95_high": round(float(np.percentile(arr, 97.5)), 4),
        }

    ci95_a = {
        "f1_pos": _ci(b_f1),
        "precision_pos": _ci(b_prec),
        "recall_pos": _ci(b_rec),
        "pr_auc": _ci(b_pr),
        "roc_auc": _ci(b_roc),
    }

    # 2. Evaluate bin_B on test split
    X_te_seg_b, _, y_te_b = prepare_task_split(test_df, "bin_B")
    scores_b = get_model_scores(pipe_b, X_te_seg_b)
    metrics_b = compute_metrics(y_te_b, scores_b, threshold=thr_b)

    # 3. Evaluate multi3 on test split
    X_te_seg_m, _, y_te_m = prepare_task_split(test_df, "multi3")
    y_pred_m = pipe_m3.predict(X_te_seg_m)
    f1_mac_m = round(float(f1_score(y_te_m, y_pred_m, average="macro", zero_division=0)), 4)
    prec_mac_m = round(float(precision_score(y_te_m, y_pred_m, average="macro", zero_division=0)), 4)
    rec_mac_m = round(float(recall_score(y_te_m, y_pred_m, average="macro", zero_division=0)), 4)
    acc_m = round(float(accuracy_score(y_te_m, y_pred_m)), 4)
    per_cls_p = precision_score(y_te_m, y_pred_m, average=None, labels=[0, 1, 2], zero_division=0)
    per_cls_r = recall_score(y_te_m, y_pred_m, average=None, labels=[0, 1, 2], zero_division=0)
    per_cls_f1 = f1_score(y_te_m, y_pred_m, average=None, labels=[0, 1, 2], zero_division=0)
    cm_m3 = confusion_matrix(y_te_m, y_pred_m, labels=[0, 1, 2]).tolist()

    # Sanity check from Section 0: macro-F1 3 classes > 0.80 => suspect leakage
    assert f1_mac_m <= 0.80, f"Sanity check failed: multi3 macro-F1 = {f1_mac_m} > 0.80!"

    multi3_metrics = {
        "f1_macro": f1_mac_m,
        "precision_macro": prec_mac_m,
        "recall_macro": rec_mac_m,
        "accuracy": acc_m,
        "per_class": {
            "CLEAN": {"precision": round(float(per_cls_p[0]), 4), "recall": round(float(per_cls_r[0]), 4), "f1": round(float(per_cls_f1[0]), 4)},
            "OFFENSIVE": {"precision": round(float(per_cls_p[1]), 4), "recall": round(float(per_cls_r[1]), 4), "f1": round(float(per_cls_f1[1]), 4)},
            "HATE": {"precision": round(float(per_cls_p[2]), 4), "recall": round(float(per_cls_r[2]), 4), "f1": round(float(per_cls_f1[2]), 4)},
        },
        "confusion_matrix": cm_m3,
    }

    # =========================================================================
    # 4. V5 Robustness Test: Original vs No-Accents (strip_accents) vs Teencode/Lengthened
    # =========================================================================
    sub_te_a = test_df[test_df["label_3"].isin(["CLEAN", "HATE"])].copy().reset_index(drop=True)
    raw_texts_a = sub_te_a["text"].fillna("").astype(str).tolist()

    # Version 1: Original
    m_orig = metrics_a

    # Version 2: Accents stripped (strip_accents -> vi_clean)
    no_accent_texts = [vi_clean(strip_accents(t), segment=True, expand_teencode=True) for t in raw_texts_a]
    scores_no_acc = get_model_scores(pipe_a, np.array(no_accent_texts))
    m_no_acc = compute_metrics(y_te_a, scores_no_acc, threshold=thr_a)

    # Version 3: Teencode + character lengthening injected (then processed by our pipeline)
    rng_tc = np.random.default_rng(42)
    tc_raw_texts = [simulate_teencode_and_repeat(t, rng_tc) for t in raw_texts_a]
    tc_clean_with_norm = [vi_clean(t, segment=True, expand_teencode=True) for t in tc_raw_texts]
    tc_clean_without_norm = [vi_clean(t, segment=True, expand_teencode=False) for t in tc_raw_texts]

    scores_tc_norm = get_model_scores(pipe_a, np.array(tc_clean_with_norm))
    m_tc_norm = compute_metrics(y_te_a, scores_tc_norm, threshold=thr_a)

    scores_tc_nonorm = get_model_scores(pipe_a, np.array(tc_clean_without_norm))
    m_tc_nonorm = compute_metrics(y_te_a, scores_tc_nonorm, threshold=thr_a)

    robustness = {
        "orig": {
            "f1_pos": m_orig["f1_pos"],
            "precision_pos": m_orig["precision_pos"],
            "recall_pos": m_orig["recall_pos"],
            "pr_auc": m_orig["pr_auc"],
            "f1_drop": 0.0,
        },
        "no_accents": {
            "f1_pos": m_no_acc["f1_pos"],
            "precision_pos": m_no_acc["precision_pos"],
            "recall_pos": m_no_acc["recall_pos"],
            "pr_auc": m_no_acc["pr_auc"],
            "f1_drop": round(float(m_orig["f1_pos"] - m_no_acc["f1_pos"]), 4),
        },
        "teencode_with_normalizer": {
            "f1_pos": m_tc_norm["f1_pos"],
            "precision_pos": m_tc_norm["precision_pos"],
            "recall_pos": m_tc_norm["recall_pos"],
            "pr_auc": m_tc_norm["pr_auc"],
            "f1_drop": round(float(m_orig["f1_pos"] - m_tc_norm["f1_pos"]), 4),
        },
        "teencode_without_normalizer": {
            "f1_pos": m_tc_nonorm["f1_pos"],
            "precision_pos": m_tc_nonorm["precision_pos"],
            "recall_pos": m_tc_nonorm["recall_pos"],
            "pr_auc": m_tc_nonorm["pr_auc"],
            "f1_drop": round(float(m_orig["f1_pos"] - m_tc_nonorm["f1_pos"]), 4),
        },
    }
    pd.DataFrame(robustness).T.reset_index(names="variant").to_csv(
        TAB_DIR / "V5_robustness_test.csv", index=False, encoding="utf-8"
    )

    # =========================================================================
    # 5. Error Analysis (30 FP + 30 FN on bin_A Test Set, masked)
    # =========================================================================
    vi_lex = load_vi_profanity_lexicon()
    sub_te_a["y_true"] = y_te_a
    sub_te_a["score"] = np.round(scores_a, 4)
    sub_te_a["y_pred"] = (scores_a >= thr_a).astype(int)
    sub_te_a["error_type"] = "Correct"
    sub_te_a.loc[(sub_te_a["y_true"] == 0) & (sub_te_a["y_pred"] == 1), "error_type"] = "FP (Báo động giả)"
    sub_te_a.loc[(sub_te_a["y_true"] == 1) & (sub_te_a["y_pred"] == 0), "error_type"] = "FN (Bỏ sót)"

    errs = sub_te_a[sub_te_a["error_type"] != "Correct"].copy()
    errs["cause"] = [
        _categorize_vi_error(r.text, r.text_clean_seg, int(r.y_true), int(r.y_pred), vi_lex)
        for r in errs.itertuples()
    ]
    errs["text_masked"] = errs["text"].map(vi_mask_profanity)
    errs["text_clean_masked"] = errs["text_clean_seg"].map(vi_mask_profanity)

    fp_30 = errs[errs["error_type"].str.startswith("FP")].sort_values("score", ascending=False).head(30)
    fn_30 = errs[errs["error_type"].str.startswith("FN")].sort_values("score", ascending=True).head(30)
    err_60 = pd.concat([fp_30, fn_30], ignore_index=True)
    err_60[["id", "error_type", "y_true", "y_pred", "score", "cause", "text_masked", "text_clean_masked"]].to_csv(
        TAB_DIR / "V5_vi_errors_60.csv", index=False, encoding="utf-8"
    )

    cause_df = (
        errs.groupby(["cause", "error_type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    cause_df["total"] = cause_df.select_dtypes(include="number").sum(axis=1)
    cause_df["pct"] = np.round(100.0 * cause_df["total"] / max(len(errs), 1), 2)
    cause_df = cause_df.sort_values("total", ascending=False).reset_index(drop=True)
    cause_df.to_csv(TAB_DIR / "V5_vi_error_causes.csv", index=False, encoding="utf-8")

    # =========================================================================
    # 6. Feature Attribution (Top-25 Positive & Negative Word Weights)
    # =========================================================================
    vec_a = pipe_a.named_steps["tfidf"]
    clf_a = pipe_a.named_steps["clf"]
    coef_a = _extract_linear_coef(clf_a)
    top_pos_vi, top_neg_vi = [], []
    if coef_a is not None and hasattr(vec_a, "get_feature_names_out"):
        fnames = vec_a.get_feature_names_out()
        # Filter to word features so explanations are human-readable Vietnamese words
        word_indices = [i for i, fn in enumerate(fnames) if not str(fn).startswith("char__")]
        word_coefs = np.array([coef_a[i] for i in word_indices])
        word_names = [str(fnames[i]).replace("word__", "") for i in word_indices]

        pos_idx = np.argsort(word_coefs)[::-1][:25]
        neg_idx = np.argsort(word_coefs)[:25]
        top_pos_vi = [{"term_masked": vi_mask_profanity(word_names[i]), "weight": round(float(word_coefs[i]), 4)} for i in pos_idx]
        top_neg_vi = [{"term_masked": vi_mask_profanity(word_names[i]), "weight": round(float(word_coefs[i]), 4)} for i in neg_idx]

    # =========================================================================
    # 7. Plot V5 Figures (Test Evaluation + Robustness + Error Causes)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5), dpi=150)

    # Panel 1: Confusion matrix for bin_A on test
    cm_a = confusion_matrix(y_te_a, (scores_a >= thr_a).astype(int), labels=[0, 1])
    sns.heatmap(
        cm_a,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["CLEAN (0)", "HATE (1)"],
        yticklabels=["CLEAN (0)", "HATE (1)"],
        cbar=False,
        ax=axes[0],
    )
    axes[0].set_title(
        f"Test bin_A (HATE vs CLEAN)\nF1={metrics_a['f1_pos']:.4f} (95% CI: [{ci95_a['f1_pos']['ci95_low']:.3f}, {ci95_a['f1_pos']['ci95_high']:.3f}])",
        fontweight="bold",
        fontsize=10.5,
    )
    axes[0].set_xlabel("Dự đoán")
    axes[0].set_ylabel("Thực tế")

    # Panel 2: Confusion matrix for multi3 (3 classes) on test
    sns.heatmap(
        np.array(cm_m3),
        annot=True,
        fmt="d",
        cmap="Purples",
        xticklabels=["CLEAN", "OFFENSIVE", "HATE"],
        yticklabels=["CLEAN", "OFFENSIVE", "HATE"],
        cbar=False,
        ax=axes[1],
    )
    axes[1].set_title(
        f"Test multi3 (3 lớp ViHSD)\nMacro-F1={f1_mac_m:.4f} | Acc={acc_m*100:.2f}%",
        fontweight="bold",
        fontsize=10.5,
    )
    axes[1].set_xlabel("Dự đoán")
    axes[1].set_ylabel("Thực tế")

    # Panel 3: Robustness test (Original vs Teencode vs No-Accents)
    rob_labels = [
        "1. Bản gốc\n(Original)",
        "2. Thêm Teencode\n(Có chuẩn hóa)",
        "3. Thêm Teencode\n(Không chuẩn hóa)",
        "4. Bỏ dấu hoàn toàn\n(strip_accents)",
    ]
    rob_f1s = [
        robustness["orig"]["f1_pos"],
        robustness["teencode_with_normalizer"]["f1_pos"],
        robustness["teencode_without_normalizer"]["f1_pos"],
        robustness["no_accents"]["f1_pos"],
    ]
    b_rob = axes[2].bar(
        rob_labels,
        rob_f1s,
        color=["#43a047", "#1e88e5", "#fb8c00", "#e53935"],
        edgecolor="black",
        width=0.52,
    )
    axes[2].set_ylim(0, max(rob_f1s) * 1.22)
    axes[2].set_ylabel("Test F1-Score (bin_A)")
    axes[2].set_title("Kiểm tra Độ bền (Robustness): Gốc vs Teencode vs Không dấu", fontweight="bold", fontsize=10.5)
    for b in b_rob:
        axes[2].text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.012,
            f"{b.get_height():.4f}",
            ha="center",
            fontweight="bold",
            fontsize=9,
        )
    axes[2].tick_params(axis="x", rotation=15)
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "V5_test_and_robustness.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save single-shot final test metrics JSON
    out_path = REPORTS_DIR / "vi_final_test_metrics.json"
    eval_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            if prev.get("test_file_sha256") == test_sha256:
                eval_ts = prev.get("evaluated_once_at", eval_ts)
        except Exception:
            pass

    payload = {
        "evaluated_once_at": eval_ts,
        "test_file": "data/splits_vi/test.csv",
        "test_file_sha256": test_sha256,
        "bin_A": {
            "threshold": thr_a,
            "metrics_tuned": metrics_a,
            "metrics_default_05": metrics_a_05,
            "ci95": ci95_a,
        },
        "bin_B": {
            "threshold": thr_b,
            "metrics_tuned": metrics_b,
        },
        "multi3": multi3_metrics,
        "robustness": robustness,
        "error_causes": cause_df.to_dict(orient="records"),
        "top_positive_words_masked": top_pos_vi[:15],
        "top_negative_words_masked": top_neg_vi[:15],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return payload


if __name__ == "__main__":
    res = run_v5_final_eval()
    print("V5 finished. Test bin_A F1:", res["bin_A"]["metrics_tuned"]["f1_pos"], "multi3 Macro-F1:", res["multi3"]["f1_macro"])
