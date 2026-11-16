"""Vietnamese Hate Speech Dataset (ViHSD — Luu et al., 2021) downloader and data report generator (V1)."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import get_project_root
from src.vi_clean import TEENCODE, compute_teencode_stats, strip_accents, vi_clean

PROJECT_ROOT = get_project_root()
VI_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "vi"
VI_SPLITS_DIR = PROJECT_ROOT / "data" / "splits_vi"
REPORTS_DIR = PROJECT_ROOT / "reports"

CACHE_DIR = PROJECT_ROOT / ".cache"
os.environ.setdefault("HF_HOME", str(CACHE_DIR / "hf"))

try:
    import emoji as _emoji_pkg
except ImportError:
    _emoji_pkg = None

LABEL_ID_TO_STR = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}
LABEL_STR_TO_ID = {"CLEAN": 0, "OFFENSIVE": 1, "HATE": 2, "0": 0, "1": 1, "2": 2}

_VI_ACCENT_CHARS = set(
    "àáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ"
)


def has_vietnamese_accents(s: str) -> bool:
    """Return True if string contains at least one accented Vietnamese character."""
    s_nfc = unicodedata.normalize("NFC", str(s or "")).lower()
    return any(ch in _VI_ACCENT_CHARS for ch in s_nfc)


def _normalize_label(val: Any) -> tuple[int, str]:
    s = str(val).strip().upper()
    if s in LABEL_STR_TO_ID:
        lid = LABEL_STR_TO_ID[s]
        return lid, LABEL_ID_TO_STR[lid]
    try:
        lid = int(float(val))
        if lid in LABEL_ID_TO_STR:
            return lid, LABEL_ID_TO_STR[lid]
    except Exception:
        pass
    raise ValueError(f"Unrecognized ViHSD label: {val!r}")


def load_vihsd(force_download: bool = False) -> tuple[pd.DataFrame, str]:
    """Download and standardize the ViHSD dataset (Luu et al., 2021).

    Tries in order:
      1. HuggingFace `sonlam1102/vihsd`
      2. HuggingFace `visolex/ViHSD`
      3. HuggingFace `tarudesu/ViHSD`

    Returns:
        (df, source_used) where df has columns:
          ['id', 'text', 'label_id', 'label_3', 'split']
    """
    VI_RAW_DIR.mkdir(parents=True, exist_ok=True)
    VI_SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    vihsd_csv = VI_RAW_DIR / "vihsd.csv"
    meta_path = VI_RAW_DIR / "vihsd_source.json"

    if not force_download and vihsd_csv.exists():
        df = pd.read_csv(vihsd_csv)
        df["text"] = df["text"].fillna("").astype(str)
        src_used = "cached:data/raw/vi/vihsd.csv"
        if meta_path.exists():
            src_used = json.loads(meta_path.read_text(encoding="utf-8")).get("source_used", src_used)
        return df, src_used

    from datasets import load_dataset

    candidates = ["sonlam1102/vihsd", "visolex/ViHSD", "tarudesu/ViHSD"]
    raw_records: list[dict[str, Any]] = []
    source_used = ""

    for repo_id in candidates:
        try:
            print(f"[load_vihsd] Trying HuggingFace dataset: {repo_id} ...")
            ds = load_dataset(repo_id)
            if "train" in ds and ("validation" in ds or "dev" in ds) and "test" in ds:
                dev_key = "dev" if "dev" in ds else "validation"
                for sp_name, hf_key in [("train", "train"), ("dev", dev_key), ("test", "test")]:
                    part = ds[hf_key].to_pandas()
                    t_col = "free_text" if "free_text" in part.columns else ("text" if "text" in part.columns else part.columns[0])
                    l_col = "label_id" if "label_id" in part.columns else ("label" if "label" in part.columns else part.columns[1])
                    for txt, lbl in zip(part[t_col], part[l_col]):
                        lid, lstr = _normalize_label(lbl)
                        raw_records.append(
                            {
                                "text": "" if pd.isna(txt) else str(txt),
                                "label_id": lid,
                                "label_3": lstr,
                                "split": sp_name,
                            }
                        )
                source_used = f"huggingface:{repo_id}"
                break
            elif "train" in ds:
                part = ds["train"].to_pandas()
                t_col = "free_text" if "free_text" in part.columns else ("text" if "text" in part.columns else None)
                l_col = "label_id" if "label_id" in part.columns else ("label" if "label" in part.columns else None)
                if "type" in part.columns and t_col is not None and l_col is not None:
                    for txt, lbl, sp_raw in zip(part[t_col], part[l_col], part["type"]):
                        sp_name = str(sp_raw).strip().lower()
                        if sp_name == "validation":
                            sp_name = "dev"
                        lid, lstr = _normalize_label(lbl)
                        raw_records.append(
                            {
                                "text": "" if pd.isna(txt) else str(txt),
                                "label_id": lid,
                                "label_3": lstr,
                                "split": sp_name,
                            }
                        )
                    source_used = f"huggingface:{repo_id}"
                    break
        except Exception as e:
            print(f"[load_vihsd] Candidate {repo_id} failed: {e}")

    if not raw_records:
        raise RuntimeError("Could not download ViHSD from any configured HuggingFace source.")

    df = pd.DataFrame(raw_records)
    df.insert(0, "id", range(1, len(df) + 1))
    df.to_csv(vihsd_csv, index=False, encoding="utf-8")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"source_used": source_used}, f, indent=2)

    return df, source_used


def generate_vi_data_report(force_download: bool = False) -> dict[str, Any]:
    """Load ViHSD, save preprocessed splits, and generate reports/vi_data_report.json (V1)."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    VI_SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    df, source_used = load_vihsd(force_download=force_download)

    # Check NaN / empty raw text before cleaning
    n_null_or_empty = int((df["text"].isna() | (df["text"].str.strip() == "")).sum())

    # Normalized text (NFC + lowercase + strip) for exact & cross-split duplicate checking
    df["text_norm"] = df["text"].map(
        lambda s: unicodedata.normalize("NFC", str(s or "")).strip().lower()
    )

    # Compute split sizes & label distributions
    split_stats = {}
    for sp_name in ["train", "dev", "test"]:
        sub = df[df["split"] == sp_name]
        vc = sub["label_3"].value_counts()
        split_stats[sp_name] = {
            "n_rows": int(len(sub)),
            "n_null_or_empty": int((sub["text"].str.strip() == "").sum()),
            "n_duplicates_exact": int(sub["text_norm"].duplicated().sum()),
            "label_distribution": {
                lbl: int(vc.get(lbl, 0)) for lbl in ["CLEAN", "OFFENSIVE", "HATE"]
            },
            "label_pct": {
                lbl: round(100.0 * int(vc.get(lbl, 0)) / max(len(sub), 1), 2)
                for lbl in ["CLEAN", "OFFENSIVE", "HATE"]
            },
        }

    overall_vc = df["label_3"].value_counts()
    overall_dist = {lbl: int(overall_vc.get(lbl, 0)) for lbl in ["CLEAN", "OFFENSIVE", "HATE"]}
    overall_pct = {
        lbl: round(100.0 * int(overall_vc.get(lbl, 0)) / max(len(df), 1), 2)
        for lbl in ["CLEAN", "OFFENSIVE", "HATE"]
    }

    # Cross-split overlap check (train ∩ dev, train ∩ test, dev ∩ test)
    train_texts = set(df[df["split"] == "train"]["text_norm"]) - {""}
    dev_texts = set(df[df["split"] == "dev"]["text_norm"]) - {""}
    test_texts = set(df[df["split"] == "test"]["text_norm"]) - {""}

    overlap_train_dev = len(train_texts & dev_texts)
    overlap_train_test = len(train_texts & test_texts)
    overlap_dev_test = len(dev_texts & test_texts)

    if overlap_train_test > 0 or overlap_train_dev > 0:
        print(
            f"CẢNH BÁO TRÙNG CHÉO SPLIT (Official ViHSD): train∩dev={overlap_train_dev}, "
            f"train∩test={overlap_train_test}, dev∩test={overlap_dev_test}"
        )

    # Surface characteristics: % unaccented, % teencode, % emoji, lengths
    non_empty_texts = [t for t in df["text"].tolist() if str(t).strip()]
    n_unaccented = sum(1 for t in non_empty_texts if not has_vietnamese_accents(t))
    pct_unaccented = round(100.0 * n_unaccented / max(len(non_empty_texts), 1), 2)

    tc_stats = compute_teencode_stats(non_empty_texts)

    n_with_emoji = 0
    if _emoji_pkg is not None:
        for t in non_empty_texts:
            try:
                if _emoji_pkg.emoji_count(t) > 0:
                    n_with_emoji += 1
            except Exception:
                pass
    pct_emoji = round(100.0 * n_with_emoji / max(len(non_empty_texts), 1), 2)

    # Clean & word-segment texts and save official splits to data/splits_vi/{train,dev,test}.csv
    print("[generate_vi_data_report] Running vi_clean (segmented & unsegmented) on ViHSD...")
    df["text_clean_seg"] = [
        vi_clean(t, segment=True, expand_teencode=True) for t in df["text"]
    ]
    df["text_clean_raw"] = [
        vi_clean(t, segment=False, expand_teencode=True) for t in df["text"]
    ]

    for sp_name in ["train", "dev", "test"]:
        sub = df[df["split"] == sp_name][
            ["id", "text", "text_clean_seg", "text_clean_raw", "label_id", "label_3", "split"]
        ].reset_index(drop=True)
        sub.to_csv(VI_SPLITS_DIR / f"{sp_name}.csv", index=False, encoding="utf-8")

    char_lens = df["text"].str.len()
    word_lens = df["text"].str.split().str.len()

    report = {
        "dataset_name": "ViHSD (Vietnamese Hate Speech Detection — Luu et al., 2021)",
        "source_used": source_used,
        "citation": "Luu, S.T., Nguyen, K.V., Nguyen, N.L.-T. (2021). A Large-Scale Dataset for Hate Speech Detection on Vietnamese Social Media Texts. IEA/AIE 2021, pp. 415-426.",
        "license_note": "Dùng cho mục đích học tập/nghiên cứu, trích dẫn Luu et al. (2021); không phân phối lại dữ liệu thô.",
        "download_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "n_total_rows": int(len(df)),
        "published_reference_counts": {
            "total": 33400,
            "train": 24048,
            "dev": 2672,
            "test": 6680,
            "CLEAN": 27624,
            "HATE": 3514,
            "OFFENSIVE": 2262,
        },
        "n_null_or_empty": n_null_or_empty,
        "overall_label_distribution": overall_dist,
        "overall_label_pct": overall_pct,
        "splits": split_stats,
        "cross_split_overlap_unique_texts": {
            "train_intersect_dev": overlap_train_dev,
            "train_intersect_test": overlap_train_test,
            "dev_intersect_test": overlap_dev_test,
            "note": (
                "Chia chuẩn chính thức của tác giả ViHSD có một số câu bình luận ngắn phổ biến "
                "(vd: 'hay quá', 'đúng rồi') xuất hiện ở nhiều split; giữ nguyên chia chuẩn để so sánh công bằng với bài báo gốc."
            ),
        },
        "linguistic_characteristics": {
            "pct_unaccented_comments": pct_unaccented,
            "n_unaccented_comments": n_unaccented,
            "pct_comments_with_teencode": tc_stats["pct_comments_with_teencode"],
            "pct_tokens_teencode": tc_stats["pct_tokens_teencode"],
            "top_teencode_terms": tc_stats["top_teencode_terms"],
            "pct_comments_with_emoji": pct_emoji,
            "char_length_mean": round(float(char_lens.mean()), 2),
            "char_length_median": round(float(char_lens.median()), 2),
            "word_length_mean": round(float(word_lens.mean()), 2),
            "word_length_median": round(float(word_lens.median()), 2),
        },
    }

    out_path = REPORTS_DIR / "vi_data_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report


if __name__ == "__main__":
    rep = generate_vi_data_report()
    print("ViHSD Report generated:", json.dumps(rep["splits"], indent=2))
