"""End-to-end deterministic pipeline runner (P12: `python run_all.py --cpu-only`)."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import pandas as pd

from src.data_io import generate_data_report
from src.eda_p2 import run_p2_eda
from src.final_eval_p8 import run_p8_final_eval
from src.modeling_p5 import run_p5_model_comparison
from src.preprocess_p4 import run_p4
from src.realworld_p9 import run_p9_realworld
from src.sql_p3 import run_p3_sql
from src.tuning_p6 import run_p6_tuning
from src.utils import get_project_root, set_seed

PROJECT_ROOT = get_project_root()


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce Tweet Hate Speech Classification Pipeline.")
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Run P1->P6, P8, P9 deterministically on CPU (skipping GPU-heavy P7 if already present).",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Re-run P8 evaluation and verify exact match (<1e-6) against existing final_test_metrics.json.",
    )
    args = parser.parse_args()

    set_seed(42)
    t_start = time.perf_counter()

    ft_path = PROJECT_ROOT / "reports" / "final_test_metrics.json"
    prev_test_f1 = None
    if ft_path.exists():
        prev_data = json.loads(ft_path.read_text(encoding="utf-8"))
        prev_test_f1 = float(prev_data["metrics_at_tuned_threshold"]["f1_pos"])

    if not args.verify_only:
        print("=== [1/7] P1: Data Collection & Verification ===")
        generate_data_report()

        print("=== [2/7] P2: Exploratory Data Analysis & Descriptive Statistics ===")
        run_p2_eda()

        print("=== [3/7] P4: Text Cleaning, Deduplication, Stratified Split & Class Balancing ===")
        run_p4()

        print("=== [4/7] P3: SQLite Database & SQL Query Analysis ===")
        run_p3_sql()

        print("=== [5/7] P5: Model Comparison (9 Classifiers) ===")
        run_p5_model_comparison()

        print("=== [6/7] P6: Hyperparameter Tuning & Threshold Optimization ===")
        run_p6_tuning()

        if not args.cpu_only:
            from src.bonus_p7 import run_p7_bonus

            print("=== [Bonus] P7: Deep Learning & Transformer Bonus ===")
            run_p7_bonus()
        elif (PROJECT_ROOT / "reports" / "p7_summary.json").exists():
            # Preserve existing P7 rows in model_comparison.csv if P7 was already run
            p7_data = json.loads((PROJECT_ROOT / "reports" / "p7_summary.json").read_text(encoding="utf-8"))
            comp_path = PROJECT_ROOT / "reports" / "model_comparison.csv"
            comp_df = pd.read_csv(comp_path)
            comp_df = comp_df[~comp_df["model_key"].isin(["bilstm_deep", "distilbert"])].copy()
            bonus_rows = []
            for mk, mname, vec_name, bal_name, sub_key in [
                ("bilstm_deep", "BiLSTM (PyTorch Embedding)", "embedding_128", "pos_weight", "bilstm"),
                ("distilbert", "DistilBERT (Fine-tuned)", "subword_wp_80", "class_weight", "distilbert"),
            ]:
                if sub_key in p7_data:
                    m = p7_data[sub_key]["metrics"]
                    bonus_rows.append(
                        {
                            "run_id": f"{mk}__{vec_name}",
                            "model_key": mk,
                            "model_name": mname,
                            "vectorizer": vec_name,
                            "balance": bal_name,
                            "cv_f1_mean": m["f1_pos"],
                            "cv_f1_std": 0.0,
                            **{
                                k: m[k]
                                for k in [
                                    "precision_pos",
                                    "recall_pos",
                                    "f1_pos",
                                    "f1_macro",
                                    "accuracy",
                                    "balanced_acc",
                                    "roc_auc",
                                    "pr_auc",
                                    "mcc",
                                    "tn",
                                    "fp",
                                    "fn",
                                    "tp",
                                    "fit_seconds",
                                    "predict_ms_per_1k",
                                ]
                            },
                            "note": f"Bonus Deep Learning ({m.get('device', 'cpu')})",
                        }
                    )
            if bonus_rows:
                comp_df = pd.concat([comp_df, pd.DataFrame(bonus_rows)], ignore_index=True)
                comp_df = comp_df.sort_values("f1_pos", ascending=False).reset_index(drop=True)
                comp_df.to_csv(comp_path, index=False, encoding="utf-8")

    print("=== [7/7] P8 & P9: Final Held-out Test Evaluation & Real-World Evaluation ===")
    p8_res = run_p8_final_eval()
    p9_res = run_p9_realworld()

    new_test_f1 = float(p8_res["final_test"]["metrics_at_tuned_threshold"]["f1_pos"])
    if prev_test_f1 is not None:
        diff = abs(new_test_f1 - prev_test_f1)
        print(f"[REPRODUCIBILITY CHECK] Previous Test F1 = {prev_test_f1:.6f}, Current Test F1 = {new_test_f1:.6f}, Diff = {diff:.8f}")
        assert diff < 1e-6, f"Reproducibility diff {diff} >= 1e-6!"

    total_sec = round(time.perf_counter() - t_start, 2)
    print(f"=== Pipeline completed successfully in {total_sec}s (Real-world F1={p9_res['overall_metrics']['f1_pos']:.4f}) ===")


if __name__ == "__main__":
    main()
