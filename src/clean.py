"""Text cleaning and surface meta-feature extraction for tweets (P4.1)."""

from __future__ import annotations

import html
import math
import os
import re
from pathlib import Path
from typing import Any

# Configure local NLTK data directory inside project .cache/nltk if available
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_NLTK_CACHE = _PROJECT_ROOT / ".cache" / "nltk"
_NLTK_CACHE.mkdir(parents=True, exist_ok=True)
import nltk

if str(_NLTK_CACHE) not in nltk.data.path:
    nltk.data.path.insert(0, str(_NLTK_CACHE))


def _ensure_nltk_resources() -> None:
    """Download required NLTK corpora if not already present."""
    for pkg, subpath in [
        ("stopwords", "corpora/stopwords"),
        ("wordnet", "corpora/wordnet.zip"),
        ("omw-1.4", "corpora/omw-1.4.zip"),
    ]:
        try:
            nltk.data.find(subpath)
        except LookupError:
            try:
                nltk.download(pkg, download_dir=str(_NLTK_CACHE), quiet=True)
            except Exception:
                pass


_ensure_nltk_resources()

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Words that MUST be preserved even when remove_stopwords=True (P4.1 rule 11)
PRESERVED_WORDS: set[str] = {
    "no",
    "not",
    "nor",
    "never",
    "you",
    "your",
    "he",
    "she",
    "they",
    "them",
    "we",
    "us",
    "i",
    "me",
    "my",
    "his",
    "her",
    "their",
    "women",
    "men",
    "cannot",
    "cant",
    "dont",
    "doesnt",
    "didnt",
    "wont",
    "wouldnt",
    "shouldnt",
    "isnt",
    "arent",
    "wasnt",
    "werent",
}

try:
    _RAW_STOPWORDS = set(stopwords.words("english"))
except Exception:
    _RAW_STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "if", "while", "of", "at", "by",
        "for", "with", "about", "against", "between", "into", "through", "during",
        "before", "after", "above", "below", "to", "from", "up", "down", "in",
        "out", "on", "off", "over", "under", "again", "further", "then", "once",
        "here", "there", "when", "where", "why", "how", "all", "any", "both",
        "each", "few", "more", "most", "other", "some", "such", "only", "own",
        "same", "so", "than", "too", "very", "can", "will", "just", "should", "now",
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "having", "do", "does", "did", "doing", "this", "that", "these", "those",
    }

STOPWORDS_SET: set[str] = _RAW_STOPWORDS - PRESERVED_WORDS
_LEMMATIZER = WordNetLemmatizer()

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_RT_RE = re.compile(r"\brt\b", re.IGNORECASE)
_REPEAT_CHARS_RE = re.compile(r"([a-z])\1{2,}")
_NON_ALPHA_RE = re.compile(r"[^a-z'\s]+")
_APOSTROPHE_EDGE_RE = re.compile(r"(^'+|'+$|\s'+|'+\s)")

try:
    import emoji as _emoji_pkg
except ImportError:
    _emoji_pkg = None


def clean_text(
    text: Any,
    remove_stopwords: bool = False,
    lemmatize: bool = True,
    handle_emoji: str = "demojize",
) -> str:
    """Clean a raw tweet following the 12-step specification in AGENTS.md P4.1.

    Steps:
      1. Convert to str & html.unescape (handle None/NaN safely)
      2. Lowercase
      3. Remove URLs (https?://\\S+|www.\\S+)
      4. Remove @mentions (@\\w+, including @user)
      5. Remove standalone 'rt' token
      6. Convert #tag -> tag
      7. Demojize or remove emoji
      8. Collapse repeated characters >=3 to 2 (soooo -> soo)
      9. Strip non-alphabetic characters (keep intra-word apostrophe)
     10. Tokenize, drop single-character tokens except 'i' and 'a'
     11. Optional stopword removal (keeping negations, pronouns, women/men)
     12. Optional WordNet lemmatization
    """
    if text is None:
        return ""
    if isinstance(text, float) and math.isnan(text):
        return ""
    s = str(text).strip()
    if not s:
        return ""

    # 1. HTML unescape
    s = html.unescape(s)
    # Remove residual broken html entities like &amp; if double-encoded
    s = html.unescape(s)

    # 7 (early pass if emoji present before lowercasing, or after)
    if _emoji_pkg is not None:
        if handle_emoji == "demojize":
            s = _emoji_pkg.demojize(s, delimiters=(" ", " "))
            s = s.replace("_", " ")
        else:
            s = _emoji_pkg.replace_emoji(s, replace=" ")

    # 2. Lowercase
    s = s.lower()

    # 3. Remove URLs
    s = _URL_RE.sub(" ", s)

    # 4. Remove @mentions (including @user)
    s = _MENTION_RE.sub(" ", s)

    # 5. Remove standalone 'rt'
    s = _RT_RE.sub(" ", s)

    # 6. #tag -> tag
    s = _HASHTAG_RE.sub(r"\1", s)

    # 8. Collapse >=3 repeated letters to 2 (e.g., sooooo -> soo)
    s = _REPEAT_CHARS_RE.sub(r"\1\1", s)

    # 9. Remove non-alphabetic characters except intra-word apostrophe
    s = _NON_ALPHA_RE.sub(" ", s)
    # Clean stray apostrophes not inside words
    s = re.sub(r"(?<![a-z])'|'(?![a-z])", " ", s)

    # 10. Tokenize and drop length-1 tokens except 'i' and 'a'
    tokens = [tok for tok in s.split() if len(tok) > 1 or tok in ("i", "a")]

    # 11. Stopwords removal (preserving negations & pronouns)
    if remove_stopwords:
        tokens = [tok for tok in tokens if tok not in STOPWORDS_SET]

    # 12. Lemmatization
    if lemmatize:
        lemmatized = []
        for tok in tokens:
            try:
                tok_lemma = _LEMMATIZER.lemmatize(tok, pos="n")
                if tok_lemma == tok:
                    tok_lemma = _LEMMATIZER.lemmatize(tok, pos="v")
                lemmatized.append(tok_lemma)
            except Exception:
                lemmatized.append(tok)
        tokens = lemmatized

    return " ".join(tokens)


def extract_meta_features(raw: Any) -> dict[str, float | int]:
    """Extract surface-level metadata features from raw tweet text before cleaning."""
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        s = ""
    else:
        s = str(raw)

    n_chars = len(s)
    words = s.split()
    n_words = len(words)
    n_exclaim = s.count("!")
    n_question = s.count("?")
    n_mentions = len(_MENTION_RE.findall(s))
    n_hashtags = len(_HASHTAG_RE.findall(s))
    n_urls = len(_URL_RE.findall(s))

    alpha_chars = [c for c in s if c.isalpha()]
    upper_chars = [c for c in alpha_chars if c.isupper()]
    caps_ratio = (len(upper_chars) / len(alpha_chars)) if alpha_chars else 0.0

    n_emojis = 0
    if _emoji_pkg is not None and s:
        try:
            n_emojis = _emoji_pkg.emoji_count(s)
        except Exception:
            n_emojis = 0

    return {
        "n_exclaim": n_exclaim,
        "n_question": n_question,
        "caps_ratio": round(caps_ratio, 4),
        "n_mentions": n_mentions,
        "n_hashtags": n_hashtags,
        "n_urls": n_urls,
        "n_chars": n_chars,
        "n_words": n_words,
        "n_emojis": n_emojis,
    }
