"""P2 Exploratory Data Analysis & Descriptive Statistics generator (10 mandatory analyses & figures)."""

from __future__ import annotations

from collections import Counter
import json
import re
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from wordcloud import WordCloud

from src.clean import STOPWORDS_SET, clean_text, extract_meta_features
from src.utils import get_project_root, mask_profanity, set_seed

PROJECT_ROOT = get_project_root()
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
TAB_DIR = PROJECT_ROOT / "reports" / "tables"

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
plt.rcParams["axes.unicode_minus"] = False


def run_p2_eda() -> dict[str, Any]:
    """Execute all 10 mandatory P2 EDA analyses, save >=10 figures and summary tables."""
    set_seed(42)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    raw_dir = PROJECT_ROOT / "data" / "raw"
    hate_df = pd.read_csv(raw_dir / "hate_train.csv")
    raw_twitter = pd.read_csv(raw_dir / "Twitter_Data.csv")
    raw_reddit = pd.read_csv(raw_dir / "Reddit_Data.csv")

    # Clean text for hate_df and assigned twitter
    hate_df["text"] = hate_df["text"].fillna("").astype(str)
    hate_df["text_clean"] = hate_df["text"].map(lambda s: clean_text(s, remove_stopwords=False))
    hate_df["text_nostop"] = hate_df["text"].map(lambda s: clean_text(s, remove_stopwords=True))

    # Extract surface features
    meta_records = [extract_meta_features(t) for t in hate_df["text"]]
    meta_df = pd.DataFrame(meta_records)
    for col in meta_df.columns:
        hate_df[col] = meta_df[col]

    # Assigned Twitter cleaned subset
    tw_valid = raw_twitter.dropna(subset=["clean_text", "category"]).copy()
    tw_valid["text"] = tw_valid["clean_text"].astype(str)
    tw_valid["sentiment"] = tw_valid["category"].astype(float).astype(int)
    tw_valid["n_chars"] = tw_valid["text"].str.len()
    tw_valid["n_words"] = tw_valid["text"].str.split().str.len()

    # =========================================================================
    # 1. Quy mô & chất lượng dữ liệu (Scale & Data Quality)
    # =========================================================================
    n_hate_total = len(hate_df)
    n_hate_exact_dup = int(hate_df["text"].duplicated().sum())
    n_hate_clean_dup = int(hate_df["text_clean"].duplicated().sum())
    n_tw_total = len(raw_twitter)
    n_tw_null = int(raw_twitter.isna().any(axis=1).sum())
    n_tw_dup = int(raw_twitter["clean_text"].dropna().duplicated().sum())
    n_rd_total = len(raw_reddit)
    n_rd_null = int(raw_reddit.isna().any(axis=1).sum())

    summary_rows = [
        {
            "dataset": "Hate Speech Train (hate_train.csv)",
            "n_rows": n_hate_total,
            "n_cols": hate_df.shape[1],
            "n_null_rows": int(hate_df["text"].isna().sum()),
            "n_exact_duplicates": n_hate_exact_dup,
            "n_clean_duplicates": n_hate_clean_dup,
            "label_col": "label (0=Normal, 1=Racist/Sexist)",
        },
        {
            "dataset": "Assigned Twitter (Twitter_Data.csv)",
            "n_rows": n_tw_total,
            "n_cols": raw_twitter.shape[1],
            "n_null_rows": n_tw_null,
            "n_exact_duplicates": n_tw_dup,
            "n_clean_duplicates": n_tw_dup,
            "label_col": "category (-1=Neg, 0=Neu, 1=Pos)",
        },
        {
            "dataset": "Assigned Reddit (Reddit_Data.csv)",
            "n_rows": n_rd_total,
            "n_cols": raw_reddit.shape[1],
            "n_null_rows": n_rd_null,
            "n_exact_duplicates": int(raw_reddit["clean_comment"].dropna().duplicated().sum()),
            "n_clean_duplicates": int(raw_reddit["clean_comment"].dropna().duplicated().sum()),
            "label_col": "category (-1=Neg, 0=Neu, 1=Pos)",
        },
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(TAB_DIR / "P2_summary_stats.csv", index=False, encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=150)
    ds_names = ["Hate Train\n(Racist/Sexist)", "Assigned Twitter\n(Sentiment)", "Assigned Reddit\n(Sentiment)"]
    ds_counts = [n_hate_total, n_tw_total, n_rd_total]
    bars = axes[0].bar(ds_names, ds_counts, color=["#2b5c8f", "#388e3c", "#f57c00"], edgecolor="black")
    axes[0].set_title("Quy mô số dòng của các bộ dữ liệu", fontweight="bold")
    axes[0].set_ylabel("Số dòng (Rows)")
    for b in bars:
        axes[0].text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 2000,
            f"{int(b.get_height()):,}",
            ha="center",
            fontsize=9.5,
            fontweight="bold",
        )
    axes[0].set_ylim(0, max(ds_counts) * 1.15)

    qual_labels = ["Trùng nguyên bản\n(Exact Dup)", "Trùng sau làm sạch\n(Near Dup / Retweet)", "Dòng rỗng sau\nlàm sạch"]
    n_empty_after_clean = int((hate_df["text_clean"] == "").sum())
    qual_vals = [n_hate_exact_dup, n_hate_clean_dup, n_empty_after_clean]
    bars2 = axes[1].bar(qual_labels, qual_vals, color=["#fbc02d", "#e64a19", "#7b1fa2"], edgecolor="black")
    axes[1].set_title("Kiểm tra trùng lặp & nhiễu trên bộ Hate Speech (N=31,962)", fontweight="bold")
    axes[1].set_ylabel("Số lượng tweet")
    for b in bars2:
        axes[1].text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 60,
            f"{int(b.get_height()):,} ({100*b.get_height()/n_hate_total:.1f}%)",
            ha="center",
            fontsize=9.5,
            fontweight="bold",
        )
    axes[1].set_ylim(0, max(qual_vals) * 1.2)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "P2_01_dataset_quality.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # 2. Phân bố nhãn (Label Distribution & Class Imbalance)
    # =========================================================================
    vc_hate = hate_df["label"].value_counts().sort_index()
    n_label0 = int(vc_hate.get(0, 0))
    n_label1 = int(vc_hate.get(1, 0))
    imbalance_ratio = round(n_label0 / max(n_label1, 1), 2)
    pos_pct = round(100.0 * n_label1 / len(hate_df), 2)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=150)
    b_lbl = axes[0].bar(
        ["Lớp 0: Bình thường\n(Non-hate)", "Lớp 1: Phản cảm\n(Racist/Sexist)"],
        [n_label0, n_label1],
        color=["#4caf50", "#e53935"],
        edgecolor="black",
        width=0.55,
    )
    axes[0].set_title(
        f"Phân bố nhãn bộ Hate Speech (Tỉ lệ lệch = {imbalance_ratio}:1)",
        fontweight="bold",
    )
    axes[0].set_ylabel("Số lượng tweet")
    for b in b_lbl:
        axes[0].text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 600,
            f"{int(b.get_height()):,}\n({100*b.get_height()/len(hate_df):.2f}%)",
            ha="center",
            fontsize=9.5,
            fontweight="bold",
        )
    axes[0].set_ylim(0, n_label0 * 1.18)

    axes[1].pie(
        [n_label0, n_label1],
        labels=[f"Lớp 0: Bình thường ({100-pos_pct:.2f}%)", f"Lớp 1: Phản cảm ({pos_pct:.2f}%)"],
        autopct="%1.2f%%",
        startangle=140,
        colors=["#66bb6a", "#ef5350"],
        explode=(0.0, 0.08),
        shadow=True,
    )
    axes[1].set_title("Tỉ lệ phần trăm hai lớp (Mất cân bằng nghiêm trọng)", fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "P2_02_label_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # 3. Độ dài tweet (ký tự & từ) theo nhãn + Bảng thống kê
    # =========================================================================
    len_stats_rows = []
    for lbl_val, lbl_name in [(0, "Label 0 (Normal)"), (1, "Label 1 (Racist/Sexist)")]:
        sub = hate_df[hate_df["label"] == lbl_val]
        for metric_col in ["n_chars", "n_words"]:
            s = sub[metric_col]
            len_stats_rows.append(
                {
                    "dataset": "hate_train",
                    "group": lbl_name,
                    "metric": metric_col,
                    "mean": round(float(s.mean()), 2),
                    "median": round(float(s.median()), 2),
                    "std": round(float(s.std()), 2),
                    "p95": round(float(np.percentile(s, 95)), 2),
                    "min": int(s.min()),
                    "max": int(s.max()),
                }
            )
    len_stats_df = pd.DataFrame(len_stats_rows)
    len_stats_df.to_csv(TAB_DIR / "P2_length_stats.csv", index=False, encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), dpi=150)
    sns.histplot(
        data=hate_df,
        x="n_chars",
        hue="label",
        stat="density",
        common_norm=False,
        bins=40,
        palette={0: "#2e7d32", 1: "#c62828"},
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("Phân bố độ dài ký tự (n_chars) theo nhãn", fontweight="bold")
    axes[0, 0].set_xlabel("Số ký tự / tweet")

    sns.boxplot(
        data=hate_df,
        x="label",
        y="n_chars",
        hue="label",
        palette={0: "#66bb6a", 1: "#ef5350"},
        legend=False,
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("Boxplot số ký tự theo nhãn (0 vs 1)", fontweight="bold")
    axes[0, 1].set_xlabel("Nhãn (0 = Bình thường, 1 = Phản cảm)")
    axes[0, 1].set_ylabel("Số ký tự")

    sns.histplot(
        data=hate_df,
        x="n_words",
        hue="label",
        stat="density",
        common_norm=False,
        bins=35,
        palette={0: "#2e7d32", 1: "#c62828"},
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("Phân bố số từ (n_words) theo nhãn", fontweight="bold")
    axes[1, 0].set_xlabel("Số từ / tweet")

    sns.boxplot(
        data=hate_df,
        x="label",
        y="n_words",
        hue="label",
        palette={0: "#66bb6a", 1: "#ef5350"},
        legend=False,
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Boxplot số từ theo nhãn (0 vs 1)", fontweight="bold")
    axes[1, 1].set_xlabel("Nhãn (0 = Bình thường, 1 = Phản cảm)")
    axes[1, 1].set_ylabel("Số từ")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "P2_03_length_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # 4. Đặc trưng bề mặt theo nhãn + Kiểm định thống kê (Mann-Whitney U & Chi2)
    # =========================================================================
    hate_df["has_url"] = (hate_df["n_urls"] > 0).astype(int)
    hate_df["has_mention"] = (hate_df["n_mentions"] > 0).astype(int)
    hate_df["has_hashtag"] = (hate_df["n_hashtags"] > 0).astype(int)
    hate_df["has_emoji"] = (hate_df["n_emojis"] > 0).astype(int)
    hate_df["has_exclaim"] = (hate_df["n_exclaim"] > 0).astype(int)
    hate_df["has_question"] = (hate_df["n_question"] > 0).astype(int)

    g0 = hate_df[hate_df["label"] == 0]
    g1 = hate_df[hate_df["label"] == 1]

    surface_tests = []
    binary_cols = [
        ("has_mention", "% có @user"),
        ("has_hashtag", "% có #hashtag"),
        ("has_exclaim", "% có dấu '!'"),
        ("has_question", "% có dấu '?'"),
        ("has_emoji", "% có Emoji"),
        ("has_url", "% có URL"),
    ]
    for col, desc in binary_cols:
        p0 = 100.0 * float(g0[col].mean())
        p1 = 100.0 * float(g1[col].mean())
        ct = pd.crosstab(hate_df["label"], hate_df[col])
        if ct.shape == (2, 2):
            chi2_stat, p_val, _, _ = chi2_contingency(ct)
        else:
            chi2_stat, p_val = 0.0, 1.0
        surface_tests.append(
            {
                "feature": col,
                "description": desc,
                "label_0_mean": round(p0, 2),
                "label_1_mean": round(p1, 2),
                "test_type": "Chi-square",
                "statistic": round(float(chi2_stat), 4),
                "p_value": float(f"{p_val:.4e}"),
                "significant_005": bool(p_val < 0.05),
            }
        )

    cont_cols = [
        ("n_chars", "Số ký tự (n_chars)"),
        ("n_words", "Số từ (n_words)"),
        ("caps_ratio", "Tỉ lệ chữ HOA (caps_ratio)"),
        ("n_hashtags", "Số hashtag / tweet"),
        ("n_mentions", "Số @mention / tweet"),
    ]
    for col, desc in cont_cols:
        m0 = float(g0[col].mean())
        m1 = float(g1[col].mean())
        u_stat, p_val = mannwhitneyu(g0[col], g1[col], alternative="two-sided")
        surface_tests.append(
            {
                "feature": col,
                "description": desc,
                "label_0_mean": round(m0, 4),
                "label_1_mean": round(m1, 4),
                "test_type": "Mann-Whitney U",
                "statistic": round(float(u_stat), 2),
                "p_value": float(f"{p_val:.4e}"),
                "significant_005": bool(p_val < 0.05),
            }
        )

    surface_df = pd.DataFrame(surface_tests)
    surface_df.to_csv(TAB_DIR / "P2_surface_tests.csv", index=False, encoding="utf-8")

    bin_sub = surface_df[surface_df["test_type"] == "Chi-square"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=150)
    x = np.arange(len(bin_sub))
    w = 0.36
    axes[0].bar(x - w / 2, bin_sub["label_0_mean"], w, label="Lớp 0 (Bình thường)", color="#4caf50", edgecolor="black")
    axes[0].bar(x + w / 2, bin_sub["label_1_mean"], w, label="Lớp 1 (Phản cảm)", color="#e53935", edgecolor="black")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(bin_sub["description"], rotation=20, ha="right")
    axes[0].set_ylabel("Tỉ lệ tweet (%)")
    axes[0].set_title("Tỉ lệ xuất hiện đặc trưng bề mặt theo nhãn (Kiểm định Chi²)", fontweight="bold")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    cont_sub = surface_df[surface_df["feature"].isin(["caps_ratio", "n_hashtags", "n_mentions"])].copy()
    x2 = np.arange(len(cont_sub))
    axes[1].bar(x2 - w / 2, cont_sub["label_0_mean"], w, label="Lớp 0 (Bình thường)", color="#4caf50", edgecolor="black")
    axes[1].bar(x2 + w / 2, cont_sub["label_1_mean"], w, label="Lớp 1 (Phản cảm)", color="#e53935", edgecolor="black")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(cont_sub["description"], rotation=15, ha="right")
    axes[1].set_ylabel("Giá trị trung bình")
    axes[1].set_title("Trung bình tỉ lệ chữ HOA, số Hashtag & @user (Mann-Whitney U)", fontweight="bold")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "P2_04_surface_features.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # 5. Top-20 Unigram & Bigram theo nhãn (Đã bỏ stopword & che từ tục)
    # =========================================================================
    def _get_top_ngrams(texts: pd.Series, ngram_range=(1, 1), top_k=20) -> list[tuple[str, int]]:
        vec = CountVectorizer(ngram_range=ngram_range, min_df=3)
        X = vec.fit_transform(texts.dropna())
        counts = np.asarray(X.sum(axis=0)).ravel()
        vocab = vec.get_feature_names_out()
        top_idx = np.argsort(counts)[::-1][:top_k]
        return [(mask_profanity(str(vocab[i])), int(counts[i])) for i in top_idx]

    top_uni_0 = _get_top_ngrams(g0["text_nostop"], (1, 1), 20)
    top_uni_1 = _get_top_ngrams(g1["text_nostop"], (1, 1), 20)
    top_bi_0 = _get_top_ngrams(g0["text_nostop"], (2, 2), 20)
    top_bi_1 = _get_top_ngrams(g1["text_nostop"], (2, 2), 20)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), dpi=150)
    axes[0].barh([w for w, _ in top_uni_0][::-1], [c for _, c in top_uni_0][::-1], color="#4caf50", edgecolor="black")
    axes[0].set_title("Top-20 Unigram — Lớp 0 (Bình thường)", fontweight="bold")
    axes[0].set_xlabel("Tần suất")

    axes[1].barh([w for w, _ in top_uni_1][::-1], [c for _, c in top_uni_1][::-1], color="#e53935", edgecolor="black")
    axes[1].set_title("Top-20 Unigram — Lớp 1 (Phản cảm, đã che từ tục)", fontweight="bold")
    axes[1].set_xlabel("Tần suất")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "P2_05_top_unigrams.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.5), dpi=150)
    axes[0].barh([w for w, _ in top_bi_0][::-1], [c for _, c in top_bi_0][::-1], color="#2e7d32", edgecolor="black")
    axes[0].set_title("Top-20 Bigram — Lớp 0 (Bình thường)", fontweight="bold")
    axes[0].set_xlabel("Tần suất")

    axes[1].barh([w for w, _ in top_bi_1][::-1], [c for _, c in top_bi_1][::-1], color="#c62828", edgecolor="black")
    axes[1].set_title("Top-20 Bigram — Lớp 1 (Phản cảm, đã che từ tục)", fontweight="bold")
    axes[1].set_xlabel("Tần suất")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "P2_05b_top_bigrams.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # 6. WordCloud theo nhãn (đã che từ tục)
    # =========================================================================
    freq_0 = dict(_get_top_ngrams(g0["text_nostop"], (1, 1), 150))
    freq_1 = dict(_get_top_ngrams(g1["text_nostop"], (1, 1), 150))

    wc0 = WordCloud(width=800, height=450, background_color="white", colormap="Greens", random_state=42).generate_from_frequencies(freq_0)
    wc1 = WordCloud(width=800, height=450, background_color="white", colormap="Reds", random_state=42).generate_from_frequencies(freq_1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=150)
    axes[0].imshow(wc0, interpolation="bilinear")
    axes[0].axis("off")
    axes[0].set_title("WordCloud — Lớp 0: Tweet bình thường", fontsize=13, fontweight="bold")

    axes[1].imshow(wc1, interpolation="bilinear")
    axes[1].axis("off")
    axes[1].set_title("WordCloud — Lớp 1: Tweet phản cảm (Đã che từ tục)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "P2_06_wordcloud_by_label.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # 7. Từ phân biệt nhất: Dirichlet-smoothed Log-Odds Ratio & Chi-square Top-30
    # =========================================================================
    cv_disc = CountVectorizer(ngram_range=(1, 1), min_df=10)
    X_disc = cv_disc.fit_transform(hate_df["text_nostop"].fillna(""))
    vocab_disc = cv_disc.get_feature_names_out()
    y_disc = hate_df["label"].to_numpy()

    c1 = np.asarray(X_disc[y_disc == 1].sum(axis=0)).ravel()
    c0 = np.asarray(X_disc[y_disc == 0].sum(axis=0)).ravel()
    t1 = float(c1.sum())
    t0 = float(c0.sum())

    # Smoothed log-odds ratio with prior alpha = 1.0
    log_odds = np.log((c1 + 1.0) / (t1 + 1.0)) - np.log((c0 + 1.0) / (t0 + 1.0))
    chi2_scores, chi2_pvals = chi2(X_disc, y_disc)

    disc_df = pd.DataFrame(
        {
            "token_masked": [mask_profanity(str(w)) for w in vocab_disc],
            "count_label_1": c1.astype(int),
            "count_label_0": c0.astype(int),
            "log_odds_ratio": np.round(log_odds, 4),
            "chi2_score": np.round(chi2_scores, 2),
            "chi2_pvalue": chi2_pvals,
        }
    )
    disc_top30 = disc_df.sort_values("log_odds_ratio", ascending=False).head(30).reset_index(drop=True)
    disc_top30.to_csv(TAB_DIR / "P2_discriminative_terms.csv", index=False, encoding="utf-8")

    chi2_pos = disc_df[disc_df["log_odds_ratio"] > 0].sort_values("chi2_score", ascending=False).head(25)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7), dpi=150)
    top_lo = disc_top30.head(25)
    axes[0].barh(
        top_lo["token_masked"].iloc[::-1],
        top_lo["log_odds_ratio"].iloc[::-1],
        color="#d32f2f",
        edgecolor="black",
    )
    axes[0].set_title("Top-25 từ đặc trưng Lớp 1 theo Smoothed Log-Odds Ratio", fontweight="bold")
    axes[0].set_xlabel("Log-Odds Ratio (Dirichlet prior +1)")

    axes[1].barh(
        chi2_pos["token_masked"].iloc[::-1],
        chi2_pos["chi2_score"].iloc[::-1],
        color="#1976d2",
        edgecolor="black",
    )
    axes[1].set_title("Top-25 từ phân biệt Lớp 1 theo kiểm định Chi-Square (χ²)", fontweight="bold")
    axes[1].set_xlabel("Giá trị thống kê χ²")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "P2_07_discriminative_log_odds_chi2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # 8. Hashtag phổ biến theo nhãn
    # =========================================================================
    hashtag_re = re.compile(r"#(\w+)")

    def _extract_hashtags(series: pd.Series, top_k=20) -> list[tuple[str, int]]:
        counter: Counter = Counter()
        for txt in series.dropna():
            tags = [mask_profanity(t.lower()) for t in hashtag_re.findall(str(txt))]
            counter.update(tags)
        return counter.most_common(top_k)

    tags_0 = _extract_hashtags(g0["text"], 20)
    tags_1 = _extract_hashtags(g1["text"], 20)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), dpi=150)
    axes[0].barh([f"#{t}" for t, _ in tags_0][::-1], [c for _, c in tags_0][::-1], color="#388e3c", edgecolor="black")
    axes[0].set_title("Top-20 Hashtag — Lớp 0 (Bình thường)", fontweight="bold")
    axes[0].set_xlabel("Số lần xuất hiện")

    axes[1].barh([f"#{t}" for t, _ in tags_1][::-1], [c for _, c in tags_1][::-1], color="#d32f2f", edgecolor="black")
    axes[1].set_title("Top-20 Hashtag — Lớp 1 (Phản cảm, đã che từ tục)", fontweight="bold")
    axes[1].set_xlabel("Số lần xuất hiện")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "P2_08_top_hashtags.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # 9. Phân tích bộ được giao Twitter_Data.csv (Sentiment chính trị -1/0/1)
    # =========================================================================
    vc_tw = tw_valid["sentiment"].value_counts().sort_index()
    tw_sample = tw_valid.sample(n=min(30000, len(tw_valid)), random_state=42)
    tw_sample["text_nostop"] = tw_sample["text"].map(lambda s: clean_text(s, remove_stopwords=True))
    top_tw_neg = _get_top_ngrams(tw_sample[tw_sample["sentiment"] == -1]["text_nostop"], (1, 1), 15)
    top_tw_pos = _get_top_ngrams(tw_sample[tw_sample["sentiment"] == 1]["text_nostop"], (1, 1), 15)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=150)
    sent_labels = ["-1 (Tiêu cực)", "0 (Trung tính)", "+1 (Tích cực)"]
    sent_vals = [int(vc_tw.get(-1, 0)), int(vc_tw.get(0, 0)), int(vc_tw.get(1, 0))]
    b_tw = axes[0].bar(sent_labels, sent_vals, color=["#e53935", "#90a4ae", "#43a047"], edgecolor="black")
    axes[0].set_title("Twitter_Data.csv: Phân bố Sentiment (-1/0/+1)", fontweight="bold")
    axes[0].set_ylabel("Số dòng")
    for b in b_tw:
        axes[0].text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 1000,
            f"{int(b.get_height()):,}",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    axes[0].set_ylim(0, max(sent_vals) * 1.15)

    sns.boxplot(
        data=tw_valid,
        x="sentiment",
        y="n_words",
        hue="sentiment",
        palette={-1: "#ef5350", 0: "#b0bec5", 1: "#66bb6a"},
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Độ dài số từ theo Sentiment (Twitter_Data)", fontweight="bold")
    axes[1].set_xlabel("Sentiment (-1 / 0 / +1)")
    axes[1].set_ylabel("Số từ")

    axes[2].barh([w for w, _ in top_tw_neg][::-1], [c for _, c in top_tw_neg][::-1], color="#fb8c00", edgecolor="black")
    axes[2].set_title("Top-15 từ ở Sentiment = -1 (Chủ đề chính trị Ấn Độ / Modi)", fontweight="bold")
    axes[2].set_xlabel("Tần suất (trên mẫu 30k)")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "P2_09_assigned_twitter_eda.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # 10. So sánh miền từ vựng (Jaccard Similarity Top-5000 từ giữa 2 bộ)
    # =========================================================================
    vec_hate_5k = CountVectorizer(max_features=5000)
    vec_hate_5k.fit(hate_df["text_nostop"].dropna())
    vocab_hate_5k = set(vec_hate_5k.get_feature_names_out())

    vec_tw_5k = CountVectorizer(max_features=5000)
    vec_tw_5k.fit(tw_sample["text_nostop"].dropna())
    vocab_tw_5k = set(vec_tw_5k.get_feature_names_out())

    shared_vocab = vocab_hate_5k & vocab_tw_5k
    union_vocab = vocab_hate_5k | vocab_tw_5k
    only_hate = vocab_hate_5k - vocab_tw_5k
    only_tw = vocab_tw_5k - vocab_hate_5k
    jaccard_5k = round(len(shared_vocab) / max(len(union_vocab), 1), 4)

    # Also compute Jaccard across Top-K = [500, 1000, 2000, 5000]
    hate_word_counts = Counter(" ".join(hate_df["text_nostop"].dropna()).split())
    tw_word_counts = Counter(" ".join(tw_sample["text_nostop"].dropna()).split())
    ks = [500, 1000, 2000, 3000, 5000]
    jaccards_by_k = []
    for k in ks:
        s1 = {w for w, _ in hate_word_counts.most_common(k)}
        s2 = {w for w, _ in tw_word_counts.most_common(k)}
        jaccards_by_k.append(round(len(s1 & s2) / max(len(s1 | s2), 1), 4))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), dpi=150)
    comp_labels = ["Chỉ có ở\nHate Speech", "Từ vựng chung\n(Giao nhau)", "Chỉ có ở\nTwitter_Data"]
    comp_vals = [len(only_hate), len(shared_vocab), len(only_tw)]
    bars_dom = axes[0].bar(comp_labels, comp_vals, color=["#e53935", "#7e57c2", "#26a69a"], edgecolor="black")
    axes[0].set_title(
        f"Giao thoa Top-5000 từ vựng hai miền (Jaccard = {jaccard_5k:.3f})",
        fontweight="bold",
    )
    axes[0].set_ylabel("Số lượng từ (Unique tokens)")
    for b in bars_dom:
        axes[0].text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 50,
            f"{int(b.get_height()):,}",
            ha="center",
            fontweight="bold",
        )
    axes[0].set_ylim(0, max(comp_vals) * 1.18)

    axes[1].plot(ks, jaccards_by_k, marker="o", lw=2.5, color="#5e35b1")
    for k_val, j_val in zip(ks, jaccards_by_k):
        axes[1].annotate(f"{j_val:.3f}", (k_val, j_val), textcoords="offset points", xytext=(0, 8), ha="center")
    axes[1].set_title("Hệ số tương đồng Jaccard theo ngưỡng Top-K từ vựng", fontweight="bold")
    axes[1].set_xlabel("Top-K từ phổ biến nhất mỗi bộ")
    axes[1].set_ylabel("Jaccard Similarity")
    axes[1].set_ylim(0, max(jaccards_by_k) * 1.25)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "P2_10_domain_vocab_overlap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    findings = {
        "hate_train_rows": n_hate_total,
        "hate_label_0": n_label0,
        "hate_label_1": n_label1,
        "hate_positive_pct": pos_pct,
        "imbalance_ratio": imbalance_ratio,
        "hate_exact_duplicates": n_hate_exact_dup,
        "hate_clean_duplicates": n_hate_clean_dup,
        "assigned_twitter_rows_raw": n_tw_total,
        "assigned_twitter_null_rows": n_tw_null,
        "assigned_twitter_valid_rows": int(len(tw_valid)),
        "assigned_twitter_sentiment_counts": {str(k): int(v) for k, v in vc_tw.items()},
        "assigned_reddit_rows_raw": n_rd_total,
        "assigned_reddit_null_rows": n_rd_null,
        "length_stats": len_stats_rows,
        "surface_tests": surface_tests,
        "top_unigrams_label_1_masked": top_uni_1[:15],
        "top_bigrams_label_1_masked": top_bi_1[:15],
        "top_hashtags_label_1_masked": tags_1[:15],
        "top_discriminative_log_odds_masked": disc_top30.head(20).to_dict(orient="records"),
        "domain_jaccard_top5000": jaccard_5k,
        "domain_shared_words_top5000": len(shared_vocab),
        "domain_jaccards_by_k": dict(zip([str(k) for k in ks], jaccards_by_k)),
    }

    with open(PROJECT_ROOT / "reports" / "p2_findings.json", "w", encoding="utf-8") as f:
        json.dump(findings, f, indent=2, ensure_ascii=False)

    return findings


if __name__ == "__main__":
    res = run_p2_eda()
    print("P2 EDA finished. Jaccard top-5000:", res["domain_jaccard_top5000"])
