"""Feature representation builders: Word TF-IDF, Char TF-IDF, Word+Char Union, and Bag-of-Words (P4.3)."""

from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from src.utils import load_config


def build_vectorizer(kind: str = "word_tfidf", cfg: dict[str, Any] | None = None):
    """Build a scikit-learn text vectorizer by kind.

    Supported kinds:
      - 'word_tfidf': TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.9, max_features=50000, sublinear_tf=True)
      - 'char_tfidf': TfidfVectorizer(analyzer='char_wb', ngram_range=(2,5), min_df=3, max_features=100000, sublinear_tf=True)
      - 'word+char' : FeatureUnion of word_tfidf and char_tfidf
      - 'bow'       : CountVectorizer(ngram_range=(1,2), min_df=3, max_df=0.9, max_features=50000)
    """
    if cfg is None:
        cfg = load_config()
    tfidf_cfg = cfg.get("tfidf", {})
    ngram_range = tuple(tfidf_cfg.get("ngram_range", [1, 2]))
    min_df = int(tfidf_cfg.get("min_df", 3))
    max_df = float(tfidf_cfg.get("max_df", 0.9))
    max_features = int(tfidf_cfg.get("max_features", 50000))
    sublinear_tf = bool(tfidf_cfg.get("sublinear_tf", True))

    if kind == "word_tfidf":
        return TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
        )
    elif kind == "char_tfidf":
        return TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=min_df,
            max_df=0.95,
            max_features=100000,
            sublinear_tf=sublinear_tf,
        )
    elif kind == "word+char":
        return FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        ngram_range=ngram_range,
                        min_df=min_df,
                        max_df=max_df,
                        max_features=max_features,
                        sublinear_tf=sublinear_tf,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(2, 5),
                        min_df=min_df,
                        max_df=0.95,
                        max_features=50000,
                        sublinear_tf=sublinear_tf,
                    ),
                ),
            ]
        )
    elif kind == "bow":
        return CountVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
        )
    else:
        raise ValueError(f"Unknown vectorizer kind: {kind!r}")
