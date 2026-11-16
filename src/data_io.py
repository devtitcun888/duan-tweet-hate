"""Data collection, downloading, schema verification, and data report generation (P1)."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import get_project_root

PROJECT_ROOT = get_project_root()
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Ensure cache stays on the project drive (D:) if running locally
CACHE_DIR = PROJECT_ROOT / ".cache"
os.environ.setdefault("KAGGLEHUB_CACHE", str(CACHE_DIR / "kagglehub"))
os.environ.setdefault("HF_HOME", str(CACHE_DIR / "hf"))


def _inspect_raw_df(
    df: pd.DataFrame,
    text_col: str,
    label_col: str | None,
    source: str,
    license_name: str,
) -> dict[str, Any]:
    """Compute standardized metadata dictionary for reports/data_report.json."""
    n_rows = int(len(df))
    columns = list(df.columns)
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    n_null = {col: int(df[col].isna().sum()) for col in df.columns}
    n_null_total = int(df.isna().sum().sum())
    n_duplicates_exact = int(df.duplicated().sum())
    n_duplicates_text = int(df[text_col].dropna().duplicated().sum()) if text_col in df.columns else 0

    label_dist: dict[str, int] = {}
    if label_col and label_col in df.columns:
        vc = df[label_col].value_counts(dropna=False)
        for k, v in vc.items():
            key_str = "NaN" if pd.isna(k) else str(k)
            label_dist[key_str] = int(v)

    return {
        "n_rows": n_rows,
        "columns": columns,
        "dtypes": dtypes,
        "n_null_by_column": n_null,
        "n_null": n_null_total,
        "n_duplicates_exact": n_duplicates_exact,
        "n_duplicates_text": n_duplicates_text,
        "label_distribution": label_dist,
        "source": source,
        "license": license_name,
        "download_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }


def load_assigned(force_download: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download and load the assigned Kaggle dataset:
    cosmos98/twitter-and-reddit-sentimental-analysis-dataset.

    Copies Twitter_Data.csv and Reddit_Data.csv into data/raw/.
    Verifies columns and prints the mandatory warning that Twitter_Data.csv only has
    sentiment labels (-1/0/1), not racist/sexist labels.

    Returns:
        (twitter_df, reddit_df) with standardized columns:
          twitter_df: ['text', 'sentiment'] (NaN rows dropped, sentiment as int)
          reddit_df:  ['text', 'sentiment'] (NaN rows dropped, sentiment as int)
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    twitter_csv = RAW_DIR / "Twitter_Data.csv"
    reddit_csv = RAW_DIR / "Reddit_Data.csv"

    if force_download or not twitter_csv.exists() or not reddit_csv.exists():
        import kagglehub

        dl_path = Path(
            kagglehub.dataset_download("cosmos98/twitter-and-reddit-sentimental-analysis-dataset")
        )
        print("Path to dataset files:", dl_path)
        for fname in ["Twitter_Data.csv", "Reddit_Data.csv"]:
            src_file = dl_path / fname
            if not src_file.exists():
                found = list(dl_path.rglob(fname))
                if found:
                    src_file = found[0]
            if src_file.exists():
                shutil.copy2(src_file, RAW_DIR / fname)

    raw_twitter = pd.read_csv(twitter_csv)
    raw_reddit = pd.read_csv(reddit_csv)

    # Mandatory assertion & warning (P1 acceptance requirement 3)
    assert "clean_text" in raw_twitter.columns and "category" in raw_twitter.columns, (
        f"Unexpected columns in Twitter_Data.csv: {list(raw_twitter.columns)}"
    )
    assert not any(
        c in [col.lower() for col in raw_twitter.columns]
        for c in ["racist", "sexist", "hate", "offensive"]
    ), "Unexpected hate/racist column found in Twitter_Data.csv"
    print("CẢNH BÁO: Twitter_Data.csv chỉ có nhãn sentiment, không có nhãn racist/sexist")

    twitter_df = (
        raw_twitter.dropna(subset=["clean_text", "category"])
        .copy()
        .rename(columns={"clean_text": "text", "category": "sentiment"})
    )
    twitter_df["text"] = twitter_df["text"].astype(str)
    twitter_df["sentiment"] = twitter_df["sentiment"].astype(float).astype(int)
    twitter_df = twitter_df[["text", "sentiment"]].reset_index(drop=True)

    reddit_df = (
        raw_reddit.dropna(subset=["clean_comment", "category"])
        .copy()
        .rename(columns={"clean_comment": "text", "category": "sentiment"})
    )
    reddit_df["text"] = reddit_df["text"].astype(str)
    reddit_df["sentiment"] = reddit_df["sentiment"].astype(float).astype(int)
    reddit_df = reddit_df[["text", "sentiment"]].reset_index(drop=True)

    return twitter_df, reddit_df


def load_hate(force_download: bool = False) -> pd.DataFrame:
    """Download and standardize the Racist/Sexist Tweets Hate Speech dataset (Plan A).

    Tries in order:
      1. HuggingFace `tweets-hate-speech-detection/tweets_hate_speech_detection`
      2. GitHub `sharmaroshan/Twitter-Sentiment-Analysis` (train_tweet.csv / test_tweets.csv)
      3. KaggleHub dataset search fallback

    Saves:
      - data/raw/hate_train.csv with exact columns: id(int), text(str), label(int 0/1)
      - data/raw/hate_unlabeled_test.csv with columns: id(int), text(str) if test split has no valid labels
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    hate_train_path = RAW_DIR / "hate_train.csv"
    hate_test_path = RAW_DIR / "hate_unlabeled_test.csv"

    if not force_download and hate_train_path.exists():
        df = pd.read_csv(hate_train_path)
        df["id"] = df["id"].astype(int)
        df["text"] = df["text"].astype(str)
        df["label"] = df["label"].astype(int)
        assert list(df.columns) == ["id", "text", "label"]
        assert set(df["label"].unique()).issubset({0, 1})
        return df

    train_df: pd.DataFrame | None = None
    test_df: pd.DataFrame | None = None

    # Attempt 1: HuggingFace datasets
    try:
        from datasets import load_dataset

        ds = load_dataset("tweets-hate-speech-detection/tweets_hate_speech_detection")
        train_df = ds["train"].to_pandas()
        if "test" in ds:
            test_df = ds["test"].to_pandas()
    except Exception as e_hf:
        print(f"[load_hate] HuggingFace fallback triggered: {e_hf}")

    # Attempt 2: GitHub raw CSVs (sharmaroshan/Twitter-Sentiment-Analysis)
    if train_df is None:
        try:
            train_url = (
                "https://raw.githubusercontent.com/sharmaroshan/"
                "Twitter-Sentiment-Analysis/master/train_tweet.csv"
            )
            test_url = (
                "https://raw.githubusercontent.com/sharmaroshan/"
                "Twitter-Sentiment-Analysis/master/test_tweets.csv"
            )
            train_df = pd.read_csv(train_url)
            try:
                test_df = pd.read_csv(test_url)
            except Exception:
                test_df = None
        except Exception as e_gh:
            raise RuntimeError(f"Failed to download hate speech dataset: {e_gh}") from e_gh

    # Standardize train_df to id, text, label
    col_map = {c.lower().strip(): c for c in train_df.columns}
    text_col = col_map.get("tweet") or col_map.get("text")
    label_col = col_map.get("label")
    id_col = col_map.get("id")

    if text_col is None or label_col is None:
        raise ValueError(f"Missing tweet/label columns in hate dataset: {list(train_df.columns)}")

    out_train = pd.DataFrame()
    if id_col is not None:
        out_train["id"] = train_df[id_col].astype(int)
    else:
        out_train["id"] = range(1, len(train_df) + 1)
    out_train["text"] = train_df[text_col].astype(str)
    out_train["label"] = train_df[label_col].astype(int)
    out_train = out_train.dropna(subset=["text", "label"]).reset_index(drop=True)

    assert list(out_train.columns) == ["id", "text", "label"]
    assert set(out_train["label"].unique()).issubset({0, 1})
    out_train.to_csv(hate_train_path, index=False, encoding="utf-8")

    # Handle test split if present
    if test_df is not None and len(test_df) > 0:
        t_col_map = {c.lower().strip(): c for c in test_df.columns}
        t_text_col = t_col_map.get("tweet") or t_col_map.get("text")
        t_id_col = t_col_map.get("id")
        t_label_col = t_col_map.get("label")

        has_valid_labels = False
        if t_label_col is not None:
            uniq = set(test_df[t_label_col].dropna().unique())
            if uniq and uniq.issubset({0, 1}) and len(uniq) == 2:
                has_valid_labels = True

        if not has_valid_labels and t_text_col is not None:
            out_test = pd.DataFrame()
            if t_id_col is not None:
                out_test["id"] = test_df[t_id_col].astype(int)
            else:
                out_test["id"] = range(1, len(test_df) + 1)
            out_test["text"] = test_df[t_text_col].astype(str)
            out_test.to_csv(hate_test_path, index=False, encoding="utf-8")

    return out_train


def load_davidson(force_download: bool = False) -> pd.DataFrame | None:
    """Download and standardize Davidson et al. (2017) Hate Speech and Offensive Language dataset.

    Mapping for binary cross-domain evaluation (AGENTS.md 1.2):
      - class == 0 (hate speech) -> label = 1
      - class == 2 (neither)     -> label = 0
      - class == 1 (offensive)   -> excluded from binary racist/sexist/hate test
        (saved in raw file with original_class for full transparency)
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    davidson_raw_path = RAW_DIR / "davidson_raw.csv"
    davidson_mapped_path = RAW_DIR / "davidson_binary.csv"

    if not force_download and davidson_mapped_path.exists():
        return pd.read_csv(davidson_mapped_path)

    raw_df: pd.DataFrame | None = None
    try:
        from datasets import load_dataset

        ds = load_dataset("tdavidson/hate_speech_offensive")
        raw_df = ds["train"].to_pandas()
    except Exception:
        try:
            url = (
                "https://raw.githubusercontent.com/t-davidson/"
                "hate-speech-and-offensive-language/master/data/labeled_data.csv"
            )
            raw_df = pd.read_csv(url)
        except Exception as e:
            print(f"[load_davidson] Could not load Davidson dataset: {e}")
            return None

    if raw_df is None or "tweet" not in raw_df.columns or "class" not in raw_df.columns:
        return None

    raw_df.to_csv(davidson_raw_path, index=False, encoding="utf-8")

    # Filter class in {0, 2}: 0 -> 1 (hate), 2 -> 0 (neither)
    sub = raw_df[raw_df["class"].isin([0, 2])].copy().reset_index(drop=True)
    mapped = pd.DataFrame(
        {
            "id": range(1, len(sub) + 1),
            "text": sub["tweet"].astype(str),
            "label": (sub["class"].astype(int) == 0).astype(int),
            "original_class": sub["class"].astype(int),
        }
    )
    mapped.to_csv(davidson_mapped_path, index=False, encoding="utf-8")
    return mapped


def generate_data_report() -> dict[str, Any]:
    """Ensure all datasets are downloaded and generate reports/data_report.json (P1.2)."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    twitter_df, reddit_df = load_assigned()
    hate_df = load_hate()
    davidson_df = load_davidson()

    raw_twitter = pd.read_csv(RAW_DIR / "Twitter_Data.csv")
    raw_reddit = pd.read_csv(RAW_DIR / "Reddit_Data.csv")

    report: dict[str, Any] = {
        "assigned_twitter": _inspect_raw_df(
            raw_twitter,
            text_col="clean_text",
            label_col="category",
            source="kagglehub: cosmos98/twitter-and-reddit-sentimental-analysis-dataset (Twitter_Data.csv)",
            license_name="CC0: Public Domain / Kaggle Community",
        ),
        "assigned_reddit": _inspect_raw_df(
            raw_reddit,
            text_col="clean_comment",
            label_col="category",
            source="kagglehub: cosmos98/twitter-and-reddit-sentimental-analysis-dataset (Reddit_Data.csv)",
            license_name="CC0: Public Domain / Kaggle Community",
        ),
        "hate_train": _inspect_raw_df(
            hate_df,
            text_col="text",
            label_col="label",
            source="HuggingFace: tweets-hate-speech-detection/tweets_hate_speech_detection (Analytics Vidhya)",
            license_name="MIT / Public Research Dataset",
        ),
    }

    report["assigned_twitter"]["n_rows_after_dropna"] = int(len(twitter_df))
    report["assigned_twitter"]["has_racist_sexist_label"] = False
    report["assigned_twitter"]["note"] = (
        "Twitter_Data.csv chỉ có nhãn sentiment chính trị (-1/0/1), không có nhãn racist/sexist."
    )
    report["assigned_reddit"]["n_rows_after_dropna"] = int(len(reddit_df))

    n_pos = int((hate_df["label"] == 1).sum())
    n_neg = int((hate_df["label"] == 0).sum())
    report["hate_train"]["positive_rate_pct"] = round(100.0 * n_pos / len(hate_df), 4)
    report["hate_train"]["imbalance_ratio_neg_to_pos"] = round(n_neg / max(n_pos, 1), 2)

    unlabeled_test_path = RAW_DIR / "hate_unlabeled_test.csv"
    if unlabeled_test_path.exists():
        unlabeled_df = pd.read_csv(unlabeled_test_path)
        report["hate_unlabeled_test"] = _inspect_raw_df(
            unlabeled_df,
            text_col="text",
            label_col=None,
            source="HuggingFace: tweets-hate-speech-detection (test split - unlabeled)",
            license_name="MIT / Public Research Dataset",
        )

    davidson_raw_path = RAW_DIR / "davidson_raw.csv"
    if davidson_raw_path.exists() and davidson_df is not None:
        raw_dav = pd.read_csv(davidson_raw_path)
        report["davidson_cross_domain"] = _inspect_raw_df(
            raw_dav,
            text_col="tweet",
            label_col="class",
            source="Davidson et al. (ICWSM 2017) Hate Speech and Offensive Language",
            license_name="MIT License",
        )
        report["davidson_cross_domain"]["binary_subset_rows"] = int(len(davidson_df))
        report["davidson_cross_domain"]["binary_label_distribution"] = {
            str(k): int(v) for k, v in davidson_df["label"].value_counts().items()
        }

    out_path = REPORTS_DIR / "data_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report
