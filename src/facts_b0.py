"""B0 Facts Aggregator: Collects all verified numbers from P1-P9 artifacts into reports/facts.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import get_project_root, mask_profanity

PROJECT_ROOT = get_project_root()
REPORTS_DIR = PROJECT_ROOT / "reports"
TAB_DIR = REPORTS_DIR / "tables"


def build_facts_json() -> dict[str, Any]:
    """Read all generated artifacts and write reports/facts.json with zero missing numbers."""
    data_report = json.loads((REPORTS_DIR / "data_report.json").read_text(encoding="utf-8"))
    p2 = json.loads((REPORTS_DIR / "p2_findings.json").read_text(encoding="utf-8"))
    p3 = json.loads((REPORTS_DIR / "p3_sql_summary.json").read_text(encoding="utf-8"))
    p4 = json.loads((REPORTS_DIR / "p4_summary.json").read_text(encoding="utf-8"))
    p5 = json.loads((REPORTS_DIR / "p5_summary.json").read_text(encoding="utf-8"))
    p6 = json.loads((REPORTS_DIR / "p6_summary.json").read_text(encoding="utf-8"))
    p7 = json.loads((REPORTS_DIR / "p7_summary.json").read_text(encoding="utf-8"))
    p8 = json.loads((REPORTS_DIR / "p8_summary.json").read_text(encoding="utf-8"))
    p9 = json.loads((REPORTS_DIR / "realworld_metrics.json").read_text(encoding="utf-8"))

    comp_df = pd.read_csv(REPORTS_DIR / "model_comparison.csv")
    errors_df = pd.read_csv(TAB_DIR / "P8_errors.csv")

    # Select 20 representative misclassified examples (10 FP + 10 FN) from test set
    fp_10 = errors_df[errors_df["error_type"].str.startswith("FP")].head(10)
    fn_10 = errors_df[errors_df["error_type"].str.startswith("FN")].head(10)
    errors_20 = pd.concat([fp_10, fn_10], ignore_index=True).to_dict(orient="records")

    ft_m = p8["final_test"]["metrics_at_tuned_threshold"]
    ft_05 = p8["final_test"]["metrics_at_default_05"]

    facts: dict[str, Any] = {
        "project_title": "Gắn nhãn các đoạn tweet có nội dung phân biệt chủng tộc, phân biệt giới tính (Twitter Sentiment Analysis)",
        "seed": 42,
        "datasets": {
            "assigned": data_report["assigned_twitter"],
            "assigned_reddit": data_report["assigned_reddit"],
            "hate": data_report["hate_train"],
            "hate_unlabeled_test": data_report.get("hate_unlabeled_test", {}),
            "davidson_cross_domain": data_report.get("davidson_cross_domain", {}),
            "domain_jaccard_top5000": p2["domain_jaccard_top5000"],
            "domain_shared_words_top5000": p2["domain_shared_words_top5000"],
        },
        "eda_highlights": {
            "length_stats": p2["length_stats"],
            "surface_tests": p2["surface_tests"],
            "top_unigrams_label_1_masked": p2["top_unigrams_label_1_masked"],
            "top_bigrams_label_1_masked": p2["top_bigrams_label_1_masked"],
            "top_hashtags_label_1_masked": p2["top_hashtags_label_1_masked"],
        },
        "sql_analysis": {
            "n_train_tweets_in_db": p3["n_train_tweets_in_db"],
            "n_tokens_in_db": p3["n_tokens_in_db"],
            "n_bigrams_in_db": p3["n_bigrams_in_db"],
            "n_lexicon_terms": p3["n_lexicon_terms"],
            "lexicon_coverage": p3["lexicon_coverage"],
            "lexicon_by_category": p3["lexicon_by_category"],
            "top_20_log_odds_masked": p3["top_20_log_odds_masked"],
            "top_15_cooccur_masked": p3["top_15_cooccur_masked"],
        },
        "preprocessing_and_splits": p4["split_summary"],
        "balance_choice": p4["balance_comparison"],
        "models": comp_df.to_dict(orient="records"),
        "paired_ttests": p5["paired_ttests"],
        "tuning_before_after": p6["before_after_table"],
        "best_model": {
            "name": p6["chosen_model_name"],
            "key": p6["chosen_model_key"],
            "selection_reason": p6["selection_reason"],
            "params": p6["best_params"],
            "threshold": p6["best_threshold"],
            "val_metrics_before_tuning": p6["model_card"]["val_metrics_before_tuning"],
            "val_metrics_after_tuning_thr05": p6["model_card"]["val_metrics_after_tuning_thr05"],
            "val_metrics_after_tuning_best_thr": p6["model_card"]["val_metrics_after_tuning_best_thr"],
        },
        "bonus_models": {
            "has_cuda": p7["has_cuda"],
            "bilstm": p7["bilstm"]["metrics"],
            "distilbert": p7["distilbert"]["metrics"],
            "tradeoff_conclusion": p7["tradeoff_conclusion"],
        },
        "final_test": {
            "evaluated_once_at": p8["final_test"]["evaluated_once_at"],
            "test_file_sha256": p8["final_test"]["test_file_sha256"],
            "n_test_rows": p8["final_test"]["n_test_rows"],
            "n_test_pos": p8["final_test"]["n_test_pos"],
            "n_test_neg": p8["final_test"]["n_test_neg"],
            "threshold": p8["final_test"]["threshold_used"],
            "f1_pos": ft_m["f1_pos"],
            "precision_pos": ft_m["precision_pos"],
            "recall_pos": ft_m["recall_pos"],
            "f1_macro": ft_m["f1_macro"],
            "accuracy": ft_m["accuracy"],
            "balanced_acc": ft_m["balanced_acc"],
            "roc_auc": ft_m["roc_auc"],
            "pr_auc": ft_m["pr_auc"],
            "mcc": ft_m["mcc"],
            "tn": ft_m["tn"],
            "fp": ft_m["fp"],
            "fn": ft_m["fn"],
            "tp": ft_m["tp"],
            "default_05_f1_pos": ft_05["f1_pos"],
            "default_05_precision_pos": ft_05["precision_pos"],
            "default_05_recall_pos": ft_05["recall_pos"],
            "ci95": p8["ci95"],
        },
        "realworld": {
            "n": p9["n_total"],
            "n_pos": p9["n_pos"],
            "n_neg": p9["n_neg"],
            "kappa": p9["cohen_kappa"],
            "agreement_pct": p9["annotator_agreement_pct"],
            "n_disagreements": p9["n_disagreements"],
            "overall_metrics": p9["overall_metrics"],
            "f1_by_source": p9["metrics_by_source"],
            "davidson_cross_domain": p9["davidson_cross_domain_metrics"],
            "assigned_twitter_by_sentiment": p9["assigned_twitter_by_sentiment"],
            "sample_errors_masked": p9["sample_errors_masked"],
            "sample_correct_masked": p9["sample_correct_masked"],
        },
        "top_discriminative_terms_masked": {
            "log_odds_top20": p3["top_20_log_odds_masked"],
            "model_positive_top20": p8["top_positive_coefficients_masked"],
            "model_negative_top20": p8["top_negative_coefficients_masked"],
        },
        "error_causes": p8["error_causes"],
        "errors_20_masked": errors_20,
        "subgroup_bias": p8["subgroup_bias"],
    }

    out_path = REPORTS_DIR / "facts.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(facts, f, indent=2, ensure_ascii=False)
    return facts


if __name__ == "__main__":
    f = build_facts_json()
    print("facts.json generated successfully. Final Test F1:", f["final_test"]["f1_pos"])
