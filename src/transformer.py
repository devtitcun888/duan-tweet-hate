"""Deep Learning / Transformer bonus models for hate-speech classification (P7)."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluate import compute_metrics
from src.utils import get_project_root, log_experiment, set_seed

PROJECT_ROOT = get_project_root()


def preprocess_for_transformer(text: str) -> str:
    """Minimal preprocessing for Transformer/BiLSTM models: normalize @user and URLs."""
    s = str(text or "").strip()
    s = re.sub(r"https?://\S+|www\.\S+", "http", s)
    s = re.sub(r"@\w+", "@user", s)
    return s


def train_and_eval_bilstm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    epochs: int = 6,
    batch_size: int = 128,
    max_vocab: int = 25000,
    max_len: int = 64,
    embed_dim: int = 128,
    hidden_dim: int = 64,
    lr: float = 2e-3,
    seed: int = 42,
) -> dict[str, Any]:
    """Train a PyTorch BiLSTM with Global Max+Avg Pooling and Class-Weighted BCE loss."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    set_seed(seed)
    torch.set_num_threads(min(6, os.cpu_count() or 4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build vocabulary on train_df['text_clean'] only (zero leakage)
    from collections import Counter

    token_counts: Counter = Counter()
    train_texts = train_df["text_clean"].fillna("").astype(str).tolist()
    val_texts = val_df["text_clean"].fillna("").astype(str).tolist()
    y_train = train_df["label"].astype(int).to_numpy()
    y_val = val_df["label"].astype(int).to_numpy()

    for txt in train_texts:
        token_counts.update(txt.split())

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for tok, _ in token_counts.most_common(max_vocab - 2):
        vocab[tok] = len(vocab)

    def encode_batch(texts: list[str]) -> np.ndarray:
        arr = np.zeros((len(texts), max_len), dtype=np.int64)
        for i, txt in enumerate(texts):
            ids = [vocab.get(w, 1) for w in txt.split()[:max_len]]
            if ids:
                arr[i, : len(ids)] = ids
        return arr

    X_tr = torch.tensor(encode_batch(train_texts), dtype=torch.long)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    X_va = torch.tensor(encode_batch(val_texts), dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

    class BiLSTMClassifier(nn.Module):
        def __init__(self, vocab_size: int, emb_dim: int, hid_dim: int):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
            self.emb_drop = nn.Dropout(0.3)
            self.lstm = nn.LSTM(
                emb_dim,
                hid_dim,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
            )
            self.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(hid_dim * 4, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            emb = self.emb_drop(self.embedding(x))
            out, _ = self.lstm(emb)
            avg_pool = torch.mean(out, dim=1)
            max_pool, _ = torch.max(out, dim=1)
            feat = torch.cat([avg_pool, max_pool], dim=1)
            return self.fc(feat).squeeze(-1)

    model = BiLSTMClassifier(len(vocab), embed_dim, hidden_dim).to(device)
    n_pos = max(int((y_train == 1).sum()), 1)
    n_neg = int((y_train == 0).sum())
    pos_weight = torch.tensor([min(n_neg / n_pos, 10.0)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    t0 = time.perf_counter()
    best_val_f1 = -1.0
    best_val_scores = np.zeros(len(y_val), dtype=float)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(by)

        model.eval()
        val_probs_list = []
        with torch.no_grad():
            for i in range(0, len(X_va), batch_size * 2):
                bx = X_va[i : i + batch_size * 2].to(device)
                probs = torch.sigmoid(model(bx)).cpu().numpy()
                val_probs_list.append(probs)
        val_scores = np.concatenate(val_probs_list)
        m = compute_metrics(y_val, val_scores, threshold=0.5)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(total_loss / len(X_tr), 4),
                "val_f1": m["f1_pos"],
                "val_pr_auc": m["pr_auc"],
            }
        )
        if m["f1_pos"] > best_val_f1:
            best_val_f1 = float(m["f1_pos"])
            best_val_scores = val_scores.copy()

    fit_sec = time.perf_counter() - t0
    t_pred0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, min(len(X_va), 1000), batch_size * 2):
            _ = torch.sigmoid(model(X_va[i : i + batch_size * 2].to(device))).cpu().numpy()
    pred_ms_per_1k = (time.perf_counter() - t_pred0) * 1000.0

    final_metrics = compute_metrics(y_val, best_val_scores, threshold=0.5)
    final_metrics["fit_seconds"] = round(fit_sec, 3)
    final_metrics["predict_ms_per_1k"] = round(pred_ms_per_1k, 3)
    final_metrics["device"] = str(device)

    log_experiment(
        model="bilstm_deep",
        vectorizer="embedding_128",
        balance="pos_weight",
        params={"epochs": epochs, "embed_dim": embed_dim, "hidden_dim": hidden_dim, "device": str(device)},
        cv_f1_mean=final_metrics["f1_pos"],
        cv_f1_std=0.0,
        val_f1=final_metrics["f1_pos"],
        val_pr_auc=final_metrics["pr_auc"],
        fit_seconds=fit_sec,
    )

    return {
        "metrics": final_metrics,
        "val_scores": best_val_scores,
        "history": history,
    }


def train_and_eval_distilbert(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_name: str = "distilbert-base-uncased",
    epochs: int = 3,
    batch_size: int = 32,
    max_length: int = 96,
    lr: float = 2e-5,
    seed: int = 42,
) -> dict[str, Any]:
    """Fine-tune DistilBERT with class-weighted CrossEntropyLoss on GPU (or CPU subset fallback)."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    set_seed(seed)
    torch.set_num_threads(min(6, os.cpu_count() or 4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # If running on CPU without CUDA, use a stratified subset and freeze lower layers so training completes in ~25s
    if device.type == "cpu" and len(train_df) > 1500:
        from sklearn.model_selection import train_test_split

        train_sub, _ = train_test_split(
            train_df,
            train_size=1500,
            stratify=train_df["label"],
            random_state=seed,
        )
        epochs = min(epochs, 2)
        batch_size = 32
        max_length = min(max_length, 48)
        lr = 8e-5
    else:
        train_sub = train_df

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    if device.type == "cpu" and hasattr(model, "distilbert"):
        for param in model.distilbert.embeddings.parameters():
            param.requires_grad = False
        for layer in model.distilbert.transformer.layer[:4]:
            for param in layer.parameters():
                param.requires_grad = False

    train_texts = [preprocess_for_transformer(t) for t in train_sub["text"].tolist()]
    val_texts = [preprocess_for_transformer(t) for t in val_df["text"].tolist()]
    y_train = train_sub["label"].astype(int).to_numpy()
    y_val = val_df["label"].astype(int).to_numpy()

    train_enc = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    val_enc = tokenizer(
        val_texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )

    class TweetDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                "input_ids": self.encodings["input_ids"][idx],
                "attention_mask": self.encodings["attention_mask"][idx],
                "labels": self.labels[idx],
            }

    train_loader = DataLoader(TweetDataset(train_enc, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TweetDataset(val_enc, y_val), batch_size=batch_size * 2, shuffle=False)

    n_pos = max(int((y_train == 1).sum()), 1)
    n_neg = int((y_train == 0).sum())
    class_weights = torch.tensor([1.0, min(n_neg / n_pos, 10.0)], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    t0 = time.perf_counter()
    best_val_f1 = -1.0
    best_val_scores = np.zeros(len(y_val), dtype=float)
    history = []

    out_dir = PROJECT_ROOT / "models" / "transformer"
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * len(labels)

        model.eval()
        val_probs = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                    probs = torch.softmax(logits, dim=-1)[:, 1]
                val_probs.append(probs.cpu().numpy())

        val_scores = np.concatenate(val_probs)
        m = compute_metrics(y_val, val_scores, threshold=0.5)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(total_loss / len(y_train), 4),
                "val_f1": m["f1_pos"],
                "val_pr_auc": m["pr_auc"],
            }
        )
        if m["f1_pos"] > best_val_f1:
            best_val_f1 = float(m["f1_pos"])
            best_val_scores = val_scores.copy()
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)

    fit_sec = time.perf_counter() - t0
    t_pred0 = time.perf_counter()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            _ = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            if (i + 1) * batch_size * 2 >= 1000:
                break
    pred_ms_per_1k = (time.perf_counter() - t_pred0) * 1000.0 * (1000.0 / min(len(y_val), 1000))

    final_metrics = compute_metrics(y_val, best_val_scores, threshold=0.5)
    final_metrics["fit_seconds"] = round(fit_sec, 3)
    final_metrics["predict_ms_per_1k"] = round(pred_ms_per_1k, 3)
    final_metrics["device"] = str(device)
    final_metrics["n_train_used"] = int(len(train_sub))

    log_experiment(
        model="distilbert",
        vectorizer="subword_wp_96",
        balance="class_weight",
        params={
            "model_name": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "device": str(device),
            "n_train": int(len(train_sub)),
        },
        cv_f1_mean=final_metrics["f1_pos"],
        cv_f1_std=0.0,
        val_f1=final_metrics["f1_pos"],
        val_pr_auc=final_metrics["pr_auc"],
        fit_seconds=fit_sec,
    )

    with open(out_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": final_metrics,
                "history": history,
            },
            f,
            indent=2,
        )

    return {
        "metrics": final_metrics,
        "val_scores": best_val_scores,
        "history": history,
    }
