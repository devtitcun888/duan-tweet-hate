"""Data leakage prevention tests (P4.2)."""

from pathlib import Path
import pandas as pd
import pytest

from src.utils import get_project_root

SPLITS_DIR = get_project_root() / "data" / "splits"


@pytest.mark.skipif(
    not (SPLITS_DIR / "train.csv").exists(),
    reason="Splits have not been generated yet (run P4 first).",
)
def test_no_text_clean_overlap_across_splits():
    """Verify train, val, and test have zero overlapping text_clean strings."""
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "val.csv")
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")

    train_set = set(train_df["text_clean"].dropna().astype(str))
    val_set = set(val_df["text_clean"].dropna().astype(str))
    test_set = set(test_df["text_clean"].dropna().astype(str))

    assert len(train_set & val_set) == 0, "Overlap found between train and val!"
    assert len(train_set & test_set) == 0, "Overlap found between train and test!"
    assert len(val_set & test_set) == 0, "Overlap found between val and test!"


@pytest.mark.skipif(
    not (SPLITS_DIR / "train.csv").exists(),
    reason="Splits have not been generated yet (run P4 first).",
)
def test_stratified_positive_class_rate_consistency():
    """Verify positive class rate across train, val, and test differs by < 0.5 percentage points."""
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "val.csv")
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")

    r_train = 100.0 * (train_df["label"] == 1).mean()
    r_val = 100.0 * (val_df["label"] == 1).mean()
    r_test = 100.0 * (test_df["label"] == 1).mean()

    assert abs(r_train - r_val) < 0.5, f"Train ({r_train:.3f}%) vs Val ({r_val:.3f}%) diff >= 0.5%"
    assert abs(r_train - r_test) < 0.5, f"Train ({r_train:.3f}%) vs Test ({r_test:.3f}%) diff >= 0.5%"
    assert abs(r_val - r_test) < 0.5, f"Val ({r_val:.3f}%) vs Test ({r_test:.3f}%) diff >= 0.5%"
