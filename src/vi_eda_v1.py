"""V1 Vietnamese EDA & SQL Analysis on ViHSD (8+ figures, SQLite queries 50 & 60, masked profanity)."""

from __future__ import annotations

from collections import Counter
import json
import math
import os
import re
import sqlite3
import unicodedata
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
from sklearn.feature_extraction.text import CountVectorizer

from src.utils import get_project_root, set_seed
from src.vi_clean import (
    TEENCODE,
    VI_STOPWORDS_LIGHT,
    load_vi_profanity_lexicon,
    vi_mask_profanity,
)
from src.vi_data_io import has_vietnamese_accents

PROJECT_ROOT = get_project_root()
VI_SPLITS_DIR = PROJECT_ROOT / "data" / "splits_vi"
DB_PATH = PROJECT_ROOT / "db" / "tweets.db"
SQL_DIR = PROJECT_ROOT / "sql"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
TAB_DIR = REPORTS_DIR / "tables"

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
plt.rcParams["axes.unicode_minus"] = False

try:
    import emoji as _emoji_pkg
except ImportError:
    _emoji_pkg = None


def run_v1_eda_and_sql() -> dict[str, Any]:
    """Run all V1 Vietnamese EDA analyses and SQL queries, saving >=8 figures and summary JSON."""
    set_seed(42)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(VI_SPLITS_DIR / "train.csv")
    dev_df = pd.read_csv(VI_SPLITS_DIR / "dev.csv")
    test_df = pd.read_csv(VI_SPLITS_DIR / "test.csv")
    full_df = pd.concat([train_df, dev_df, test_df], ignore_index=True)

    for d in (train_df, dev_df, test_df, full_df):
        d["text"] = d["text"].fillna("").astype(str)
        d["text_clean_seg"] = d["text_clean_seg"].fillna("").astype(str)
        d["text_clean_raw"] = d["text_clean_raw"].fillna("").astype(str)

    # =========================================================================
    # Figure V1_01: Label distribution overall & across splits (train/dev/test)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5), dpi=150)
    lbl_order = ["CLEAN", "OFFENSIVE", "HATE"]
    lbl_colors = ["#43a047", "#fb8c00", "#e53935"]

    vc_all = full_df["label_3"].value_counts()
    vals_all = [int(vc_all.get(k, 0)) for k in lbl_order]
    bars1 = axes[0].bar(lbl_order, vals_all, color=lbl_colors, edgecolor="black", width=0.55)
    axes[0].set_title(f"Phân bố 3 nhãn trên toàn bộ ViHSD (N = {len(full_df):,})", fontweight="bold")
    axes[0].set_ylabel("Số lượng bình luận")
    for b in bars1:
        axes[0].text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 400,
            f"{int(b.get_height()):,}\n({100*b.get_height()/len(full_df):.1f}%)",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    axes[0].set_ylim(0, max(vals_all) * 1.2)

    split_names = ["train", "dev", "test"]
    x = np.arange(len(split_names))
    w = 0.25
    for idx, (lbl, col) in enumerate(zip(lbl_order, lbl_colors)):
        pcts = [
            100.0 * float((full_df[full_df["split"] == sp]["label_3"] == lbl).mean())
            for sp in split_names
        ]
        axes[1].bar(x + (idx - 1) * w, pcts, w, label=lbl, color=col, edgecolor="black")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["Train (24,048)", "Dev (2,672)", "Test (6,680)"])
    axes[1].set_ylabel("Tỉ lệ phần trăm (%)")
    axes[1].set_title("Tỉ lệ 3 nhãn đồng nhất giữa Train / Dev / Test chuẩn", fontweight="bold")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "V1_01_split_label_dist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # Figure V1_02: Length distribution (characters & words) by label on Train
    # =========================================================================
    train_df["n_chars"] = train_df["text"].str.len()
    train_df["n_words"] = train_df["text"].str.split().str.len()

    len_rows = []
    for lbl in lbl_order:
        sub = train_df[train_df["label_3"] == lbl]
        for col in ["n_chars", "n_words"]:
            s = sub[col]
            len_rows.append(
                {
                    "label_3": lbl,
                    "metric": col,
                    "mean": round(float(s.mean()), 2),
                    "median": round(float(s.median()), 2),
                    "std": round(float(s.std()), 2),
                    "p95": round(float(np.percentile(s, 95)), 2),
                }
            )
    pd.DataFrame(len_rows).to_csv(TAB_DIR / "V1_length_stats.csv", index=False, encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), dpi=150)
    sns.boxplot(
        data=train_df[train_df["n_chars"] <= 400],
        x="label_3",
        y="n_chars",
        order=lbl_order,
        hue="label_3",
        palette=dict(zip(lbl_order, lbl_colors)),
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Phân bố độ dài ký tự (n_chars ≤ 400) theo nhãn (Train)", fontweight="bold")
    axes[0].set_xlabel("Nhãn (ViHSD)")
    axes[0].set_ylabel("Số ký tự")

    sns.boxplot(
        data=train_df[train_df["n_words"] <= 80],
        x="label_3",
        y="n_words",
        order=lbl_order,
        hue="label_3",
        palette=dict(zip(lbl_order, lbl_colors)),
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Phân bố số từ (n_words ≤ 80) theo nhãn (Train)", fontweight="bold")
    axes[1].set_xlabel("Nhãn (ViHSD)")
    axes[1].set_ylabel("Số từ")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "V1_02_length_by_label.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # Figure V1_03 & V1_04: Top Unigrams & Bigrams by label (segmented, masked)
    # =========================================================================
    def _top_vi_ngrams(series: pd.Series, ngram_range=(1, 1), top_k=18) -> list[tuple[str, int]]:
        vec = CountVectorizer(
            ngram_range=ngram_range,
            min_df=5,
            token_pattern=r"(?u)\b\w[\w_]+\b",
            stop_words=list(VI_STOPWORDS_LIGHT),
        )
        X = vec.fit_transform(series.dropna())
        cnts = np.asarray(X.sum(axis=0)).ravel()
        vocab = vec.get_feature_names_out()
        idx = np.argsort(cnts)[::-1][:top_k]
        return [(vi_mask_profanity(str(vocab[i])), int(cnts[i])) for i in idx]

    uni_clean = _top_vi_ngrams(train_df[train_df["label_3"] == "CLEAN"]["text_clean_seg"], (1, 1), 18)
    uni_off = _top_vi_ngrams(train_df[train_df["label_3"] == "OFFENSIVE"]["text_clean_seg"], (1, 1), 18)
    uni_hate = _top_vi_ngrams(train_df[train_df["label_3"] == "HATE"]["text_clean_seg"], (1, 1), 18)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6), dpi=150)
    for ax_c, data_u, title_u, col_u in [
        (axes[0], uni_clean, "Top-18 Từ (CLEAN)", "#43a047"),
        (axes[1], uni_off, "Top-18 Từ (OFFENSIVE — Đã che)", "#fb8c00"),
        (axes[2], uni_hate, "Top-18 Từ (HATE — Đã che)", "#e53935"),
    ]:
        ax_c.barh([w for w, _ in data_u][::-1], [c for _, c in data_u][::-1], color=col_u, edgecolor="black")
        ax_c.set_title(title_u, fontweight="bold")
        ax_c.set_xlabel("Tần suất (Train)")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "V1_03_top_unigrams_by_label.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    bi_off = _top_vi_ngrams(train_df[train_df["label_3"] == "OFFENSIVE"]["text_clean_seg"], (2, 2), 18)
    bi_hate = _top_vi_ngrams(train_df[train_df["label_3"] == "HATE"]["text_clean_seg"], (2, 2), 18)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6), dpi=150)
    axes[0].barh([w for w, _ in bi_off][::-1], [c for _, c in bi_off][::-1], color="#fb8c00", edgecolor="black")
    axes[0].set_title("Top-18 Bigram — Lớp OFFENSIVE (Đã tách từ & che từ tục)", fontweight="bold")
    axes[0].set_xlabel("Tần suất (Train)")

    axes[1].barh([w for w, _ in bi_hate][::-1], [c for _, c in bi_hate][::-1], color="#e53935", edgecolor="black")
    axes[1].set_title("Top-18 Bigram — Lớp HATE (Đã tách từ & che từ tục)", fontweight="bold")
    axes[1].set_xlabel("Tần suất (Train)")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "V1_04_top_bigrams_by_label.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # Figure V1_05: Surface features by label (caps, '!', '?', emoji, repeated chars, unaccented)
    # =========================================================================
    repeat_re = re.compile(r"([^\W\d_])\1{2,}", flags=re.UNICODE)
    train_df["has_exclaim"] = train_df["text"].str.contains("!", regex=False).astype(int)
    train_df["has_question"] = train_df["text"].str.contains("?", regex=False).astype(int)
    train_df["has_repeat"] = train_df["text"].map(lambda s: 1 if repeat_re.search(s) else 0)
    train_df["is_unaccented"] = train_df["text"].map(lambda s: 0 if has_vietnamese_accents(s) else 1)
    if _emoji_pkg is not None:
        train_df["has_emoji"] = train_df["text"].map(lambda s: 1 if _emoji_pkg.emoji_count(s) > 0 else 0)
    else:
        train_df["has_emoji"] = 0

    surf_rows = []
    for lbl in lbl_order:
        sub = train_df[train_df["label_3"] == lbl]
        surf_rows.append(
            {
                "label_3": lbl,
                "pct_exclaim": round(100.0 * float(sub["has_exclaim"].mean()), 2),
                "pct_question": round(100.0 * float(sub["has_question"].mean()), 2),
                "pct_emoji": round(100.0 * float(sub["has_emoji"].mean()), 2),
                "pct_repeat_chars": round(100.0 * float(sub["has_repeat"].mean()), 2),
                "pct_unaccented": round(100.0 * float(sub["is_unaccented"].mean()), 2),
            }
        )
    surf_df = pd.DataFrame(surf_rows)
    surf_df.to_csv(TAB_DIR / "V1_surface_by_label.csv", index=False, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(11.5, 5), dpi=150)
    feat_cols = ["pct_exclaim", "pct_question", "pct_emoji", "pct_repeat_chars", "pct_unaccented"]
    feat_names = ["Có dấu '!'", "Có dấu '?'", "Có Emoji", "Ký tự kéo dài (≥3)", "Không dấu"]
    x = np.arange(len(feat_cols))
    w = 0.25
    for idx, (lbl, col) in enumerate(zip(lbl_order, lbl_colors)):
        vals = [float(surf_df[surf_df["label_3"] == lbl][fc].iloc[0]) for fc in feat_cols]
        ax.bar(x + (idx - 1) * w, vals, w, label=lbl, color=col, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(feat_names)
    ax.set_ylabel("Tỉ lệ bình luận (%)")
    ax.set_title("Đặc trưng bề mặt theo 3 nhãn trên tập Train ViHSD (V1)", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "V1_05_surface_features.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # SQL Analysis in db/tweets.db: vi_comments, vi_tokens, 50_vi_log_odds.sql, 60_vi_cooccur.sql
    # =========================================================================
    conn = sqlite3.connect(str(DB_PATH))
    conn.create_function("LN", 1, lambda v: math.log(float(v)) if v and float(v) > 0 else 0.0)
    conn.executescript(
        """
        DROP TABLE IF EXISTS vi_comments;
        DROP TABLE IF EXISTS vi_tokens;
        DROP TABLE IF EXISTS vi_top_hate_tokens;
        CREATE TABLE vi_comments(id INTEGER PRIMARY KEY, text TEXT, label_3 TEXT, split TEXT);
        CREATE TABLE vi_tokens(comment_id INT, label_3 TEXT, token TEXT);
        CREATE TABLE vi_top_hate_tokens(token TEXT PRIMARY KEY, c_hate INT, c_off INT, c_clean INT, log_odds_hate_vs_clean REAL);
        CREATE INDEX ix_vi_tokens_tok ON vi_tokens(token, label_3);
        CREATE INDEX ix_vi_tokens_cid ON vi_tokens(comment_id);
        """
    )

    train_df[["id", "text", "label_3", "split"]].to_sql("vi_comments", conn, if_exists="append", index=False, chunksize=5000)

    vi_token_rows = []
    for r in train_df.itertuples(index=False):
        cid = int(r.id)
        l3 = str(r.label_3)
        for tok in str(r.text_clean_seg).split():
            if len(tok) > 1 and tok not in VI_STOPWORDS_LIGHT and not tok.isdigit():
                vi_token_rows.append((cid, l3, tok))

    pd.DataFrame(vi_token_rows, columns=["comment_id", "label_3", "token"]).to_sql(
        "vi_tokens", conn, if_exists="append", index=False, chunksize=5000
    )
    conn.commit()

    q50_sql = (SQL_DIR / "50_vi_log_odds.sql").read_text(encoding="utf-8")
    df_q50 = pd.read_sql_query(q50_sql, conn)
    df_q50[["token", "c_hate", "c_off", "c_clean", "log_odds_hate_vs_clean"]].to_sql(
        "vi_top_hate_tokens", conn, if_exists="replace", index=False
    )
    conn.commit()

    q60_sql = (SQL_DIR / "60_vi_cooccur.sql").read_text(encoding="utf-8")
    df_q60 = pd.read_sql_query(q60_sql, conn)
    conn.close()

    df_q50_masked = df_q50.copy()
    df_q50_masked["token_masked"] = df_q50_masked["token"].map(vi_mask_profanity)
    df_q50_masked.drop(columns=["token"]).to_csv(TAB_DIR / "V1_sql_log_odds.csv", index=False, encoding="utf-8")

    df_q60_masked = df_q60.copy()
    df_q60_masked["neighbor_masked"] = df_q60_masked["neighbor"].map(vi_mask_profanity)
    df_q60_masked.drop(columns=["neighbor"]).to_csv(TAB_DIR / "V1_sql_cooccur.csv", index=False, encoding="utf-8")

    # =========================================================================
    # Figure V1_06: Smoothed Log-Odds Ratio for HATE vs CLEAN & OFFENSIVE vs CLEAN
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2), dpi=150)
    top20_hate = df_q50_masked.sort_values("log_odds_hate_vs_clean", ascending=False).head(20)
    top20_off = df_q50_masked.sort_values("log_odds_off_vs_clean", ascending=False).head(20)

    axes[0].barh(
        top20_hate["token_masked"].iloc[::-1],
        top20_hate["log_odds_hate_vs_clean"].iloc[::-1],
        color="#d32f2f",
        edgecolor="black",
    )
    axes[0].set_title("SQL 50: Top-20 từ đặc trưng HATE vs CLEAN (Log-Odds, đã che)", fontweight="bold")
    axes[0].set_xlabel("Smoothed Log-Odds Ratio")

    axes[1].barh(
        top20_off["token_masked"].iloc[::-1],
        top20_off["log_odds_off_vs_clean"].iloc[::-1],
        color="#ef6c00",
        edgecolor="black",
    )
    axes[1].set_title("SQL 50: Top-20 từ đặc trưng OFFENSIVE vs CLEAN (Log-Odds, đã che)", fontweight="bold")
    axes[1].set_xlabel("Smoothed Log-Odds Ratio")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "V1_06_log_odds_hate_offensive.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # Figure V1_07: Teencode distribution & Top-15 Teencode words
    # =========================================================================
    vi_data_rep = json.loads((REPORTS_DIR / "vi_data_report.json").read_text(encoding="utf-8"))
    top_tc = vi_data_rep["linguistic_characteristics"]["top_teencode_terms"]

    def _has_tc(s: str) -> int:
        toks = re.sub(r"[^\w\s]", " ", str(s or "").lower()).split()
        return 1 if any(t in TEENCODE for t in toks) else 0

    train_df["has_teencode"] = train_df["text"].map(_has_tc)
    tc_by_lbl = [
        100.0 * float(train_df[train_df["label_3"] == lbl]["has_teencode"].mean())
        for lbl in lbl_order
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5), dpi=150)
    axes[0].barh(
        [ f"{w} → {TEENCODE.get(w, '')}" for w, _ in top_tc ][::-1],
        [ c for _, c in top_tc ][::-1],
        color="#5e35b1",
        edgecolor="black",
    )
    axes[0].set_title("Top-15 từ viết tắt / Teencode phổ biến nhất trong ViHSD", fontweight="bold")
    axes[0].set_xlabel("Số lần xuất hiện")

    b_tc = axes[1].bar(lbl_order, tc_by_lbl, color=lbl_colors, edgecolor="black", width=0.5)
    axes[1].set_title("Tỉ lệ bình luận chứa Teencode theo từng nhãn (%)", fontweight="bold")
    axes[1].set_ylabel("Tỉ lệ (%)")
    for b in b_tc:
        axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{b.get_height():.1f}%", ha="center", fontweight="bold")
    axes[1].set_ylim(0, max(tc_by_lbl) * 1.25)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "V1_07_teencode_and_accents.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # Figure V1_08: SQL Co-occurrence & Profanity Lexicon coverage by label
    # =========================================================================
    vi_lex = load_vi_profanity_lexicon()

    def _has_vi_lex(seg_txt: str) -> int:
        toks = set(str(seg_txt or "").split())
        return 1 if len(toks & vi_lex) > 0 else 0

    train_df["has_lexicon"] = train_df["text_clean_seg"].map(_has_vi_lex)
    lex_by_lbl = []
    for lbl in lbl_order:
        sub = train_df[train_df["label_3"] == lbl]
        lex_by_lbl.append(
            {
                "label_3": lbl,
                "n_comments": int(len(sub)),
                "n_with_lexicon": int(sub["has_lexicon"].sum()),
                "pct_with_lexicon": round(100.0 * float(sub["has_lexicon"].mean()), 2),
            }
        )
    pd.DataFrame(lex_by_lbl).to_csv(TAB_DIR / "V1_lexicon_coverage.csv", index=False, encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5), dpi=150)
    top15_co = df_q60_masked.head(15)
    axes[0].barh(
        top15_co["neighbor_masked"].iloc[::-1],
        top15_co["cnt"].iloc[::-1],
        color="#c62828",
        edgecolor="black",
    )
    axes[0].set_title("SQL 60: Top-15 từ đồng xuất hiện trong bình luận HATE (Đã che)", fontweight="bold")
    axes[0].set_xlabel("Số lần đồng xuất hiện")

    pct_lex_vals = [r["pct_with_lexicon"] for r in lex_by_lbl]
    b_lx = axes[1].bar(lbl_order, pct_lex_vals, color=lbl_colors, edgecolor="black", width=0.5)
    axes[1].set_title("Tỉ lệ bình luận chứa từ trong vi_profanity.txt theo nhãn (%)", fontweight="bold")
    axes[1].set_ylabel("Tỉ lệ (%)")
    for b in b_lx:
        axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, f"{b.get_height():.2f}%", ha="center", fontweight="bold")
    axes[1].set_ylim(0, max(pct_lex_vals) * 1.25)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "V1_08_sql_cooccur_lexicon.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "n_train_tokens_in_db": len(vi_token_rows),
        "length_stats": len_rows,
        "surface_by_label": surf_rows,
        "lexicon_coverage": lex_by_lbl,
        "top_15_log_odds_hate_masked": top20_hate.head(15).to_dict(orient="records"),
        "top_15_log_odds_off_masked": top20_off.head(15).to_dict(orient="records"),
        "top_15_cooccur_masked": top15_co.to_dict(orient="records"),
    }
    with open(REPORTS_DIR / "vi_eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


if __name__ == "__main__":
    res = run_v1_eda_and_sql()
    print("V1 EDA & SQL finished. Lexicon coverage:", res["lexicon_coverage"])
