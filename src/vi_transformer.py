"""V4 Vietnamese Transformer Models: PhoBERT (vinai/phobert-base-v2, segmented input) & XLM-RoBERTa (xlm-roberta-base, unsegmented input) across 3 seeds (42, 43, 44)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ["HF_HOME"] = str(PROJECT_ROOT / ".cache" / "hf")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

from src.evaluate import compute_metrics
from src.utils import set_seed
from src.vi_modeling_v3 import log_vi_experiment, prepare_task_split

VI_SPLITS_DIR = PROJECT_ROOT / "data" / "splits_vi"
VI_MODELS_DIR = PROJECT_ROOT / "models" / "vi"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"


def _extract_transformer_features(
    model_name: str,
    texts: list[str],
    use_fast_tokenizer: bool,
    device: torch.device,
    max_length: int = 48,
    batch_size: int = 64,
) -> np.ndarray:
    """Extract mean+max pooled hidden representations from pre-trained Transformer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast_tokenizer)
    backbone = AutoModel.from_pretrained(model_name).to(device)
    backbone.eval()

    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    feats = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            b_ids = input_ids[i : i + batch_size].to(device)
            b_mask = attention_mask[i : i + batch_size].to(device)
            out = backbone(input_ids=b_ids, attention_mask=b_mask)
            hidden = out.last_hidden_state  # (B, L, D)
            mask_exp = b_mask.unsqueeze(-1).float()
            mean_pool = (hidden * mask_exp).sum(dim=1) / torch.clamp(mask_exp.sum(dim=1), min=1e-6)
            cls_tok = hidden[:, 0, :]
            combined = torch.cat([cls_tok, mean_pool], dim=-1).cpu().numpy()
            feats.append(combined)

    return np.vstack(feats)


def _train_head_for_seed(
    H_tr: np.ndarray,
    y_tr: np.ndarray,
    H_dv: np.ndarray,
    y_dv: np.ndarray,
    seed: int,
    device: torch.device,
    base_extract_sec: float,
    epochs: int = 12,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> dict[str, Any]:
    set_seed(seed)
    in_dim = H_tr.shape[1]

    class ClassificationHead(nn.Module):
        def __init__(self, d_in: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(d_in),
                nn.Dropout(0.3),
                nn.Linear(d_in, 256),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(256, 2),
            )

        def forward(self, x):
            return self.net(x)

    head = ClassificationHead(in_dim).to(device)
    n_pos = max(int((y_tr == 1).sum()), 1)
    n_neg = int((y_tr == 0).sum())
    weights = torch.tensor([1.0, min(n_neg / n_pos, 8.0)], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-3)

    tr_ds = TensorDataset(torch.tensor(H_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long))
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    X_dv_t = torch.tensor(H_dv, dtype=torch.float32, device=device)

    t0 = time.perf_counter()
    best_f1 = -1.0
    best_metrics = {}
    for _ in range(epochs):
        head.train()
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = head(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

        head.eval()
        with torch.no_grad():
            val_logits = head(X_dv_t)
            val_probs = torch.softmax(val_logits, dim=-1)[:, 1].cpu().numpy()
        m = compute_metrics(y_dv, val_probs, threshold=0.5)
        if m["f1_pos"] > best_f1:
            best_f1 = float(m["f1_pos"])
            best_metrics = m

    head_sec = time.perf_counter() - t0
    best_metrics["fit_seconds"] = round(base_extract_sec + head_sec, 2)
    best_metrics["seed"] = seed
    best_metrics["n_train_used"] = int(len(y_tr))
    best_metrics["device"] = str(device)
    return best_metrics


def run_v4_transformers() -> dict[str, Any]:
    """Run PhoBERT (segmented input) and XLM-R (unsegmented input) across 3 seeds (42, 43, 44) on bin_A."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    VI_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(6, os.cpu_count() or 4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_csv(VI_SPLITS_DIR / "train.csv")
    dev_df = pd.read_csv(VI_SPLITS_DIR / "dev.csv")

    X_tr_seg, X_tr_raw, y_tr = prepare_task_split(train_df, "bin_A")
    X_dv_seg, X_dv_raw, y_dv = prepare_task_split(dev_df, "bin_A")

    # On CPU, use a stratified 2,000-sample training subset so feature extraction completes in ~20s per model
    if device.type == "cpu" and len(y_tr) > 2000:
        idx_tr, _ = train_test_split(
            np.arange(len(y_tr)),
            train_size=2000,
            stratify=y_tr,
            random_state=42,
        )
        X_tr_seg_use = X_tr_seg[idx_tr]
        X_tr_raw_use = X_tr_raw[idx_tr]
        y_tr_use = y_tr[idx_tr]
    else:
        X_tr_seg_use, X_tr_raw_use, y_tr_use = X_tr_seg, X_tr_raw, y_tr

    seeds = [42, 43, 44]
    results = {}

    for short_key, hf_name, use_seg, use_fast in [
        ("phobert", "vinai/phobert-base-v2", True, False),
        ("xlmr", "xlm-roberta-base", False, True),
    ]:
        print(f"[V4] Extracting features with {hf_name} (segmented={use_seg}, device={device})...")
        t_ext0 = time.perf_counter()
        tr_texts = (X_tr_seg_use if use_seg else X_tr_raw_use).tolist()
        dv_texts = (X_dv_seg if use_seg else X_dv_raw).tolist()
        H_tr = _extract_transformer_features(hf_name, tr_texts, use_fast, device, max_length=48, batch_size=64)
        H_dv = _extract_transformer_features(hf_name, dv_texts, use_fast, device, max_length=48, batch_size=64)
        ext_sec = time.perf_counter() - t_ext0

        seed_runs = []
        for s in seeds:
            m = _train_head_for_seed(H_tr, y_tr_use, H_dv, y_dv, seed=s, device=device, base_extract_sec=ext_sec / 3.0)
            seed_runs.append(m)
            log_vi_experiment(
                task="bin_A",
                model=short_key,
                vectorizer="segmented_bpe" if use_seg else "unsegmented_spm",
                balance="class_weight",
                params={"model_name": hf_name, "seed": s, "n_train": m["n_train_used"], "device": m["device"]},
                dev_f1_primary=m["f1_pos"],
                dev_pr_auc_or_acc=m["pr_auc"],
                fit_seconds=m["fit_seconds"],
            )

        f1s = [r["f1_pos"] for r in seed_runs]
        praucs = [r["pr_auc"] for r in seed_runs]
        rocaucs = [r["roc_auc"] for r in seed_runs]
        precs = [r["precision_pos"] for r in seed_runs]
        recs = [r["recall_pos"] for r in seed_runs]
        secs = [r["fit_seconds"] for r in seed_runs]

        results[short_key] = {
            "model_name": hf_name,
            "input_type": "Văn bản ĐÃ tách từ (ViTokenizer)" if use_seg else "Văn bản CHƯA tách từ (Raw SentencePiece)",
            "seeds": seeds,
            "f1_pos_mean": round(float(np.mean(f1s)), 4),
            "f1_pos_std": round(float(np.std(f1s, ddof=1)), 4),
            "pr_auc_mean": round(float(np.mean(praucs)), 4),
            "pr_auc_std": round(float(np.std(praucs, ddof=1)), 4),
            "roc_auc_mean": round(float(np.mean(rocaucs)), 4),
            "precision_mean": round(float(np.mean(precs)), 4),
            "recall_mean": round(float(np.mean(recs)), 4),
            "fit_seconds_mean": round(float(np.mean(secs)), 2),
            "device": str(device),
            "n_train_used": int(len(y_tr_use)),
            "seed_runs": seed_runs,
        }

    # Append PhoBERT and XLM-R mean rows to reports/vi_model_comparison.csv
    comp_path = REPORTS_DIR / "vi_model_comparison.csv"
    if comp_path.exists():
        comp_df = pd.read_csv(comp_path)
        comp_df = comp_df[~comp_df["model_key"].isin(["phobert", "xlmr"])].copy()
        extra_rows = []
        for sk, info in results.items():
            extra_rows.append(
                {
                    "task": "bin_A",
                    "model_key": sk,
                    "model_name": info["model_name"],
                    "vectorizer": "segmented_bpe" if sk == "phobert" else "unsegmented_spm",
                    "balance": "class_weight",
                    "primary_metric_name": "f1_pos",
                    "primary_metric": info["f1_pos_mean"],
                    "precision": info["precision_mean"],
                    "recall": info["recall_mean"],
                    "f1_pos": info["f1_pos_mean"],
                    "f1_macro": 0.0,
                    "pr_auc": info["pr_auc_mean"],
                    "roc_auc": info["roc_auc_mean"],
                    "accuracy": 0.0,
                    "fit_seconds": info["fit_seconds_mean"],
                }
            )
        comp_df = pd.concat([comp_df, pd.DataFrame(extra_rows)], ignore_index=True)
        comp_df.to_csv(comp_path, index=False, encoding="utf-8")

    # Plot V4 Transformer vs Classical comparison
    v3_data = json.loads((REPORTS_DIR / "vi_v3_summary.json").read_text(encoding="utf-8"))
    cls_best = v3_data["best_bin_A"]

    fig, ax = plt.subplots(figsize=(10.5, 5), dpi=150)
    names = [
        f"{cls_best['model_name']}\n({cls_best['vectorizer']} — Tuned)",
        f"PhoBERT-base-v2\n(3 seeds, {results['phobert']['device']})",
        f"XLM-RoBERTa-base\n(3 seeds, {results['xlmr']['device']})",
    ]
    f1_means = [
        cls_best["dev_metrics_tuned"]["f1_pos"],
        results["phobert"]["f1_pos_mean"],
        results["xlmr"]["f1_pos_mean"],
    ]
    f1_stds = [0.0, results["phobert"]["f1_pos_std"], results["xlmr"]["f1_pos_std"]]
    pr_means = [
        cls_best["dev_metrics_tuned"]["pr_auc"],
        results["phobert"]["pr_auc_mean"],
        results["xlmr"]["pr_auc_mean"],
    ]

    x = np.arange(len(names))
    w = 0.34
    ax.bar(x - w / 2, f1_means, w, yerr=f1_stds, capsize=5, label="Dev F1-Score (Mean ± Std)", color="#e53935", edgecolor="black")
    ax.bar(x + w / 2, pr_means, w, label="Dev PR-AUC", color="#1e88e5", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Điểm số (Dev — bin_A)")
    ax.set_title("Đối sánh Mô hình Cổ điển Tối ưu vs PhoBERT & XLM-RoBERTa (3 seeds: 42, 43, 44)", fontweight="bold")
    for i in range(len(x)):
        ax.text(x[i] - w / 2, f1_means[i] + 0.025, f"{f1_means[i]:.4f}", ha="center", fontweight="bold", fontsize=9)
        ax.text(x[i] + w / 2, pr_means[i] + 0.025, f"{pr_means[i]:.4f}", ha="center", fontsize=9)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "V4_transformers_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(REPORTS_DIR / "vi_v4_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results


if __name__ == "__main__":
    res = run_v4_transformers()
    print("V4 finished. PhoBERT F1:", res["phobert"]["f1_pos_mean"], "XLM-R F1:", res["xlmr"]["f1_pos_mean"])
