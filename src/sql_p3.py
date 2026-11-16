"""P3 SQLite Database Construction & SQL Query Analysis for Common Patterns in Hate Tweets (P3)."""

from __future__ import annotations

import json
import math
import os
import sqlite3
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

from src.clean import STOPWORDS_SET, clean_text
from src.utils import get_profanity_lexicon, get_project_root, mask_profanity, set_seed

PROJECT_ROOT = get_project_root()
DB_PATH = PROJECT_ROOT / "db" / "tweets.db"
SQL_DIR = PROJECT_ROOT / "sql"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
TAB_DIR = PROJECT_ROOT / "reports" / "tables"


def run_p3_sql() -> dict[str, Any]:
    """Build db/tweets.db from train split and assigned dataset, run all 4 SQL queries, save tables & figures."""
    set_seed(42)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    # Use train split if already created by P4, otherwise create splits first
    train_split_path = PROJECT_ROOT / "data" / "splits" / "train.csv"
    if not train_split_path.exists():
        from src.preprocess_p4 import create_clean_splits

        create_clean_splits()

    train_df = pd.read_csv(train_split_path)
    raw_twitter = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "Twitter_Data.csv")
    tw_valid = raw_twitter.dropna(subset=["clean_text", "category"]).copy().reset_index(drop=True)

    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(str(DB_PATH))
    conn.create_function("LN", 1, lambda x: math.log(float(x)) if x and float(x) > 0 else 0.0)

    # 1. Execute 00_schema.sql
    schema_sql = (SQL_DIR / "00_schema.sql").read_text(encoding="utf-8")
    conn.executescript(schema_sql)

    # 2. Populate tweets table (from train_df only to prevent test leakage)
    tweets_table_df = pd.DataFrame(
        {
            "id": train_df["id"].astype(int),
            "text": train_df["text"].astype(str),
            "label": train_df["label"].astype(int),
            "n_chars": train_df["text"].astype(str).str.len(),
            "n_words": train_df["text"].astype(str).str.split().str.len(),
        }
    )
    tweets_table_df.to_sql("tweets", conn, if_exists="append", index=False, chunksize=5000)

    # 3. Tokenize train tweets (filtering stopwords for semantic SQL analysis) and populate tokens & bigrams
    token_rows = []
    bigram_rows = []
    for row in train_df.itertuples(index=False):
        tid = int(row.id)
        lbl = int(row.label)
        cleaned = str(row.text_clean if pd.notna(row.text_clean) else "")
        toks = [t for t in cleaned.split() if t not in STOPWORDS_SET and len(t) > 1]
        for tok in toks:
            token_rows.append((tid, lbl, tok))
        for i in range(len(toks) - 1):
            bigram_rows.append((tid, lbl, f"{toks[i]} {toks[i + 1]}"))

    tokens_df = pd.DataFrame(token_rows, columns=["tweet_id", "label", "token"])
    tokens_df.to_sql("tokens", conn, if_exists="append", index=False, chunksize=5000)

    bigrams_df = pd.DataFrame(bigram_rows, columns=["tweet_id", "label", "bigram"])
    bigrams_df.to_sql("bigrams", conn, if_exists="append", index=False, chunksize=5000)

    # 4. Populate assigned table
    assigned_df = pd.DataFrame(
        {
            "id": range(1, len(tw_valid) + 1),
            "text": tw_valid["clean_text"].astype(str),
            "sentiment": tw_valid["category"].astype(float).astype(int),
        }
    )
    assigned_df.to_sql("assigned", conn, if_exists="append", index=False, chunksize=5000)

    # 5. Populate lexicon table (including lemmatized forms so matches on cleaned tokens are accurate)
    lex_dict = get_profanity_lexicon()
    lex_entries = {}
    for term, cat in lex_dict.items():
        lex_entries[term.lower()] = cat
        cleaned_term = clean_text(term, remove_stopwords=False, lemmatize=True)
        if cleaned_term and " " not in cleaned_term:
            lex_entries.setdefault(cleaned_term, cat)

    lex_df = pd.DataFrame(
        [{"term": k, "category": v} for k, v in sorted(lex_entries.items())]
    )
    lex_df.to_sql("lexicon", conn, if_exists="append", index=False, chunksize=5000)
    conn.commit()

    # =========================================================================
    # Execute mandatory SQL queries
    # =========================================================================
    # Query 1: 10_token_freq.sql
    q1_sql = (SQL_DIR / "10_token_freq.sql").read_text(encoding="utf-8")
    df_q1 = pd.read_sql_query(q1_sql, conn)
    df_q1_masked = df_q1.copy()
    df_q1_masked["token_masked"] = df_q1_masked["token"].map(mask_profanity)
    df_q1_masked.drop(columns=["token"]).head(200).to_csv(
        TAB_DIR / "P3_token_freq_top200.csv", index=False, encoding="utf-8"
    )

    # Query 2: 20_log_odds.sql
    q2_sql = (SQL_DIR / "20_log_odds.sql").read_text(encoding="utf-8")
    df_q2 = pd.read_sql_query(q2_sql, conn)
    # Populate top_hate_tokens table for Query 3
    df_q2[["token", "c1", "c0", "log_odds"]].to_sql(
        "top_hate_tokens", conn, if_exists="replace", index=False
    )
    conn.commit()

    df_q2_masked = df_q2.copy()
    df_q2_masked["token"] = df_q2_masked["token"].map(mask_profanity)
    df_q2_masked.to_csv(TAB_DIR / "P3_top_hate_tokens.csv", index=False, encoding="utf-8")

    # Query 3: 30_cooccur.sql
    q3_sql = (SQL_DIR / "30_cooccur.sql").read_text(encoding="utf-8")
    df_q3 = pd.read_sql_query(q3_sql, conn)
    df_q3_masked = df_q3.copy()
    df_q3_masked["neighbor"] = df_q3_masked["neighbor"].map(mask_profanity)
    df_q3_masked.to_csv(TAB_DIR / "P3_cooccurrence.csv", index=False, encoding="utf-8")

    # Query 4: 40_lexicon.sql
    q4_sql = (SQL_DIR / "40_lexicon.sql").read_text(encoding="utf-8")
    df_q4 = pd.read_sql_query(q4_sql, conn)
    df_q4.to_csv(TAB_DIR / "P3_lexicon_coverage.csv", index=False, encoding="utf-8")

    # Bonus SQL Query 5: Breakdown of lexicon hits by category in label=1 vs label=0
    q5_sql = """
    SELECT
        k.category,
        SUM(CASE WHEN t.label = 1 THEN 1 ELSE 0 END) AS hits_label_1,
        SUM(CASE WHEN t.label = 0 THEN 1 ELSE 0 END) AS hits_label_0,
        COUNT(DISTINCT CASE WHEN t.label = 1 THEN t.id END) AS tweets_label_1,
        COUNT(DISTINCT CASE WHEN t.label = 0 THEN t.id END) AS tweets_label_0
    FROM tokens tk
    JOIN tweets t ON t.id = tk.tweet_id
    JOIN lexicon k ON k.term = tk.token
    GROUP BY k.category
    ORDER BY tweets_label_1 DESC;
    """
    df_q5 = pd.read_sql_query(q5_sql, conn)
    df_q5.to_csv(TAB_DIR / "P3_lexicon_by_category.csv", index=False, encoding="utf-8")

    # Bonus SQL Query 6: Top bigrams in label=1 via SQL
    q6_sql = """
    SELECT bigram, COUNT(*) AS cnt
    FROM bigrams
    WHERE label = 1
    GROUP BY bigram
    ORDER BY cnt DESC
    LIMIT 30;
    """
    df_q6 = pd.read_sql_query(q6_sql, conn)
    df_q6["bigram"] = df_q6["bigram"].map(mask_profanity)
    df_q6.to_csv(TAB_DIR / "P3_top_bigrams_sql.csv", index=False, encoding="utf-8")

    conn.close()

    # Plot P3 SQL summary figure (3 panels)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.5), dpi=150)

    top15_lo = df_q2_masked.head(15)
    axes[0].barh(
        top15_lo["token"].iloc[::-1],
        top15_lo["log_odds"].iloc[::-1],
        color="#c62828",
        edgecolor="black",
    )
    axes[0].set_title("SQL Query 20: Top-15 từ đặc trưng Lớp 1 (Log-Odds)", fontweight="bold")
    axes[0].set_xlabel("Smoothed Log-Odds Ratio")

    top15_co = df_q3_masked.head(15)
    axes[1].barh(
        top15_co["neighbor"].iloc[::-1],
        top15_co["cnt"].iloc[::-1],
        color="#ef6c00",
        edgecolor="black",
    )
    axes[1].set_title("SQL Query 30: Top-15 từ đồng xuất hiện (Co-occurrence)", fontweight="bold")
    axes[1].set_xlabel("Số lần đi cùng top-20 từ phản cảm")

    lbl_names = ["Lớp 0 (Bình thường)", "Lớp 1 (Phản cảm)"]
    pct_lex = df_q4["pct_with_lexicon"].tolist()
    b_lex = axes[2].bar(lbl_names, pct_lex, color=["#43a047", "#d32f2f"], edgecolor="black", width=0.5)
    axes[2].set_title("SQL Query 40: Tỉ lệ tweet chứa từ trong Lexicon (%)", fontweight="bold")
    axes[2].set_ylabel("Tỉ lệ (%)")
    for b in b_lex:
        axes[2].text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 1.0,
            f"{b.get_height():.2f}%",
            ha="center",
            fontweight="bold",
        )
    axes[2].set_ylim(0, max(pct_lex) * 1.25)
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "P3_sql_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "db_path": str(DB_PATH.relative_to(PROJECT_ROOT)),
        "n_train_tweets_in_db": int(len(tweets_table_df)),
        "n_tokens_in_db": int(len(tokens_df)),
        "n_bigrams_in_db": int(len(bigrams_df)),
        "n_assigned_in_db": int(len(assigned_df)),
        "n_lexicon_terms": int(len(lex_df)),
        "top_20_log_odds_masked": df_q2_masked.head(20).to_dict(orient="records"),
        "top_15_cooccur_masked": df_q3_masked.head(15).to_dict(orient="records"),
        "lexicon_coverage": df_q4.to_dict(orient="records"),
        "lexicon_by_category": df_q5.to_dict(orient="records"),
        "top_15_bigrams_sql_masked": df_q6.head(15).to_dict(orient="records"),
    }
    with open(PROJECT_ROOT / "reports" / "p3_sql_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


if __name__ == "__main__":
    res = run_p3_sql()
    print("P3 SQL finished. Lexicon coverage:", res["lexicon_coverage"])
