"""Utility functions: seed control, profanity masking, configuration loading, and experiment logging."""

from __future__ import annotations

import csv
import json
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_project_root() -> Path:
    """Return the root directory of the project."""
    return PROJECT_ROOT


def set_seed(seed: int = 42) -> None:
    """Set random seeds across random, numpy, and torch (if installed) for full reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML configuration from project root."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    else:
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


# Comprehensive categorized lexicon of offensive, racist, sexist, and profane English terms
# Used for SQL lexicon analysis (P3) and display masking (Rule 6).
PROFANITY_LEXICON: dict[str, str] = {
    # Racial / ethnic slurs & derogatory terms
    "nigger": "racial_slur",
    "niggers": "racial_slur",
    "nigga": "racial_slur",
    "niggas": "racial_slur",
    "negro": "racial_slur",
    "chink": "racial_slur",
    "chinks": "racial_slur",
    "spic": "racial_slur",
    "spics": "racial_slur",
    "wetback": "racial_slur",
    "kike": "racial_slur",
    "kikes": "racial_slur",
    "gook": "racial_slur",
    "coon": "racial_slur",
    "coons": "racial_slur",
    "paki": "racial_slur",
    "raghead": "racial_slur",
    "towelhead": "racial_slur",
    "beaner": "racial_slur",
    "redskin": "racial_slur",
    "whitey": "racial_slur",
    "cracker": "racial_slur",
    "honky": "racial_slur",
    "nazis": "racial_slur",
    "nazi": "racial_slur",
    "racist": "racial_slur",
    "racism": "racial_slur",
    "supremacist": "racial_slur",
    "bigot": "racial_slur",
    "bigotry": "racial_slur",
    "islamophobia": "religious_slur",
    "antisemitic": "religious_slur",
    "jihadist": "religious_slur",
    # Sexist / gender-based slurs & derogatory terms
    "bitch": "sexist_slur",
    "bitches": "sexist_slur",
    "whore": "sexist_slur",
    "whores": "sexist_slur",
    "slut": "sexist_slur",
    "sluts": "sexist_slur",
    "slutty": "sexist_slur",
    "cunt": "sexist_slur",
    "cunts": "sexist_slur",
    "hoe": "sexist_slur",
    "hoes": "sexist_slur",
    "skank": "sexist_slur",
    "bimbo": "sexist_slur",
    "feminazi": "sexist_slur",
    "misogyny": "sexist_slur",
    "misogynist": "sexist_slur",
    "sexist": "sexist_slur",
    "sexism": "sexist_slur",
    "pussy": "sexist_slur",
    "pussies": "sexist_slur",
    "twat": "sexist_slur",
    # Homophobic / ableist slurs
    "faggot": "homophobic_slur",
    "faggots": "homophobic_slur",
    "fag": "homophobic_slur",
    "fags": "homophobic_slur",
    "dyke": "homophobic_slur",
    "tranny": "homophobic_slur",
    "retard": "ableist_slur",
    "retards": "ableist_slur",
    "retarded": "ableist_slur",
    # General profanity / vulgarity
    "fuck": "general_profanity",
    "fucking": "general_profanity",
    "fucked": "general_profanity",
    "fucker": "general_profanity",
    "fuckers": "general_profanity",
    "motherfucker": "general_profanity",
    "motherfucking": "general_profanity",
    "shit": "general_profanity",
    "shitty": "general_profanity",
    "bullshit": "general_profanity",
    "asshole": "general_profanity",
    "assholes": "general_profanity",
    "ass": "general_profanity",
    "bastard": "general_profanity",
    "bastards": "general_profanity",
    "dick": "general_profanity",
    "dicks": "general_profanity",
    "cock": "general_profanity",
    "prick": "general_profanity",
    "damn": "general_profanity",
    "piss": "general_profanity",
    "crap": "general_profanity",
}

# Words that should strictly be masked when displayed in charts/tables/text (vulgar/slur words)
# Note: analytical terms like "racist", "racism", "sexist", "sexism", "bigot", "nazi", "misogyny"
# are kept readable in academic discussion while slurs and vulgarities are masked.
WORDS_TO_MASK: set[str] = {
    w
    for w, cat in PROFANITY_LEXICON.items()
    if w
    not in {
        "racist",
        "racism",
        "sexist",
        "sexism",
        "bigot",
        "bigotry",
        "nazi",
        "nazis",
        "supremacist",
        "islamophobia",
        "antisemitic",
        "jihadist",
        "misogyny",
        "misogynist",
        "damn",
        "crap",
    }
}

_MASK_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in sorted(WORDS_TO_MASK, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def _mask_single_word(word: str) -> str:
    """Keep first and last character, replace middle characters with '*'."""
    n = len(word)
    if n <= 2:
        return word[0] + "*" * (n - 1)
    return word[0] + ("*" * (n - 2)) + word[-1]


def mask_profanity(text: str | None) -> str:
    """Mask profanity and slurs in text by keeping first/last character and replacing middle with '*'."""
    if text is None:
        return ""
    s = str(text)
    return _MASK_PATTERN.sub(lambda m: _mask_single_word(m.group(0)), s)


def get_profanity_lexicon() -> dict[str, str]:
    """Return mapping of offensive/profane terms to their category for SQL analysis."""
    return dict(PROFANITY_LEXICON)


def log_experiment(
    model: str,
    vectorizer: str,
    balance: str,
    params: dict[str, Any] | str,
    cv_f1_mean: float,
    cv_f1_std: float,
    val_f1: float | None = None,
    val_pr_auc: float | None = None,
    fit_seconds: float = 0.0,
    experiments_path: str | Path | None = None,
) -> None:
    """Append one experiment record to reports/experiments.csv."""
    if experiments_path is None:
        experiments_path = PROJECT_ROOT / "reports" / "experiments.csv"
    else:
        experiments_path = Path(experiments_path)
    experiments_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = experiments_path.exists() and experiments_path.stat().st_size > 0
    params_json = json.dumps(params, ensure_ascii=False, sort_keys=True) if isinstance(params, dict) else str(params)

    row = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": model,
        "vectorizer": vectorizer,
        "balance": balance,
        "params_json": params_json,
        "cv_f1_mean": round(float(cv_f1_mean), 6) if cv_f1_mean is not None else "",
        "cv_f1_std": round(float(cv_f1_std), 6) if cv_f1_std is not None else "",
        "val_f1": round(float(val_f1), 6) if val_f1 is not None else "",
        "val_pr_auc": round(float(val_pr_auc), 6) if val_pr_auc is not None else "",
        "fit_seconds": round(float(fit_seconds), 3),
    }

    fieldnames = [
        "timestamp",
        "model",
        "vectorizer",
        "balance",
        "params_json",
        "cv_f1_mean",
        "cv_f1_std",
        "val_f1",
        "val_pr_auc",
        "fit_seconds",
    ]
    with open(experiments_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
