# src/preprocessing.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import re
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype, is_object_dtype, is_categorical_dtype


# -----------------------------
# Config
# -----------------------------

@dataclass
class PreprocessConfig:
    id_col: str = "ID"
    target_col: str = "Target"

    # Columns we expect to be numeric in this dataset
    numeric_cols: Tuple[str, ...] = (
        "owner_age",
        "personal_income",
        "business_expenses",
        "business_turnover",
        "business_age_years",
        "business_age_months",
    )

    # Money columns for log transforms
    money_cols: Tuple[str, ...] = (
        "personal_income",
        "business_expenses",
        "business_turnover",
    )

    # If a column is meant to be Yes/No but sometimes has 0/1
    yes_no_like_cols: Tuple[str, ...] = (
        # Add to this list if you discover more
        "has_cellphone",
        "current_problem_cash_flow",
        "marketing_word_of_mouth",
        "problem_sourcing_money",
        "uses_friends_family_savings",
        "uses_informal_lender",
        "has_mobile_money",
        "has_insurance",
        "has_internet_banking",
        "has_debit_card",
        "has_credit_card",
        "compliance_income_tax",
    )

    # Create missing flags for these numeric columns (very helpful for tree models)
    numeric_missing_flags: Tuple[str, ...] = (
        "personal_income",
        "business_expenses",
        "business_turnover",
    )

    # For LightGBM, fill missing categoricals with this token
    lgb_missing_token: str = "missing"


# -----------------------------
# Text cleaning helpers
# -----------------------------

_WEIRD_APOSTROPHES = {
    "’": "'",
    "‘": "'",
    "´": "'",
    "`": "'",
    "â€™": "'",   # common mojibake
    "â€˜": "'",
}

# Sometimes mojibake turns apostrophes into question marks
# Example: Don?t know, doesn?t
# We'll fix "don?t" -> "don't", "doesn?t" -> "doesn't" etc.
def _fix_mojibake_question_apostrophe(s: str) -> str:
    # Replace patterns like don?t / doesn?t / can?t
    # Only when ? sits between letters
    return re.sub(r"([A-Za-z])\?([A-Za-z])", r"\1'\2", s)


def _normalize_whitespace(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _to_lower(s: str) -> str:
    return s.lower()


def _standardize_common_tokens(s: str) -> str:
    """
    Collapse yes/no/don't know variants to canonical forms:
    - yes
    - no
    - don't know
    Also handles variants combined with slashes like:
      "Don?t know / doesn't apply"
      "Don?t know/doesn?t apply"
    We map anything that contains a don't-know-like phrase to "don't know"
    if it's basically a don't-know family.
    """
    s_low = s.lower()

    # Common don't-know family
    dont_know_patterns = [
        "don't know", "dont know", "do not know", "don t know",
        "don’t know",  # curly apostrophe
        "dont't know",  # weird typos
        "don?t know",   # mojibake
        "doesn't know", "doesnt know", "doesn’t know", "doesn?t know",
    ]

    # If string contains any don't-know-like pattern, treat as don't know
    if any(p in s_low for p in dont_know_patterns):
        return "don't know"

    # Some surveys use "refused" similar to missing; keep as "refused"
    if s_low in {"refused", "prefer not to say", "prefer not say"}:
        return "refused"

    # Normalize yes/no short forms
    if s_low in {"yes", "y", "1", "true", "t"}:
        return "yes"
    if s_low in {"no", "n", "0", "false", "f"}:
        return "no"

    # Common N/A variants (keep separate from don't know; can be informative)
    if s_low in {"n/a", "na", "not applicable", "doesn't apply", "doesnt apply", "doesn’t apply", "doesn?t apply"}:
        return "not applicable"

    return s_low


def clean_categorical_value(x) -> object:
    """
    Cleans a single categorical cell:
    - Leaves NaN as NaN
    - Converts to string
    - Fixes mojibake, apostrophes, weird characters
    - Normalizes whitespace
    - Lowercases
    - Collapses yes/no/don't know variants
    """
    if pd.isna(x):
        return np.nan

    s = str(x)

    # If a numeric slipped into a categorical column (e.g., 0), keep it as string for mapping
    s = s.strip()

    # Replace weird apostrophes
    for bad, good in _WEIRD_APOSTROPHES.items():
        s = s.replace(bad, good)

    # Fix "don?t" -> "don't"
    s = _fix_mojibake_question_apostrophe(s)

    # Normalize whitespace
    s = _normalize_whitespace(s)

    # Lowercase + standardize
    s = _standardize_common_tokens(s)

    return s


# -----------------------------
# Numeric helpers
# -----------------------------

def coerce_numeric(series: pd.Series) -> pd.Series:
    """
    Converts a Series to numeric safely.
    - Removes commas
    - Converts scientific notation fine
    - Non-numeric -> NaN
    """
    if is_numeric_dtype(series):
        return series

    s = series.astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(s, errors="coerce")


def safe_log1p(series: pd.Series) -> pd.Series:
    """
    log1p for non-negative values.
    If negatives exist, we set them to NaN (or you can clip to 0).
    """
    s = series.copy()
    s = pd.to_numeric(s, errors="coerce")
    s = s.where(s >= 0, np.nan)
    return np.log1p(s)


# -----------------------------
# Main preprocessing pipeline
# -----------------------------

def detect_categorical_columns(df: pd.DataFrame, cfg: PreprocessConfig) -> List[str]:
    """Find columns that should be treated as categoricals (excluding ID and Target)."""
    cat_cols = []
    for c in df.columns:
        if c in {cfg.id_col, cfg.target_col}:
            continue
        if is_string_dtype(df[c]) or is_object_dtype(df[c]) or is_categorical_dtype(df[c]):
            cat_cols.append(c)
    return cat_cols


def preprocess_dataframe(
    df: pd.DataFrame,
    cfg: PreprocessConfig,
    *,
    is_train: bool,
    for_model: str = "catboost",  # "catboost" or "lightgbm"
) -> pd.DataFrame:
    """
    Preprocess a dataframe.
    - Cleans categorical text
    - Coerces numeric columns
    - Creates missing flags
    - Creates log features
    - Creates business_age_total_months
    - Optionally fills categorical missing for LightGBM
    """
    out = df.copy()

    # 1) Clean categorical columns robustly
    cat_cols = detect_categorical_columns(out, cfg)

    # Some columns might be numeric but loaded as str; don't clean those as categoricals if in numeric_cols
    numeric_set = set([c for c in cfg.numeric_cols if c in out.columns])
    cat_cols = [c for c in cat_cols if c not in numeric_set]

    for c in cat_cols:
        out[c] = out[c].apply(clean_categorical_value)

    # 2) Coerce numeric columns
    for c in cfg.numeric_cols:
        if c in out.columns:
            out[c] = coerce_numeric(out[c])

    # 3) Fix yes/no columns that may contain 0/1 etc.
    # We force clean_categorical_value on these cols even if they were numeric or mixed
    for c in cfg.yes_no_like_cols:
        if c in out.columns:
            out[c] = out[c].apply(clean_categorical_value)

    # 4) Missingness indicators for key numeric vars
    for c in cfg.numeric_missing_flags:
        if c in out.columns:
            out[f"{c}_missing"] = out[c].isna().astype("int8")

    # 5) Log transforms for skewed money columns
    for c in cfg.money_cols:
        if c in out.columns:
            out[f"log_{c}"] = safe_log1p(out[c])

    # 6) Business age consistency: total months
    if "business_age_years" in out.columns and "business_age_months" in out.columns:
        # business_age_months looks like "additional months beyond full years"
        # Use 0 if missing months
        out["business_age_total_months"] = (out["business_age_years"] * 12) + out["business_age_months"].fillna(0)

        # Optional: contradiction flag (months >= 12 is suspicious if it's "additional months")
        out["business_age_months_ge_12"] = (out["business_age_months"] >= 12).fillna(False).astype("int8")

    # 7) LightGBM categorical missing token
    if for_model.lower() == "lightgbm":
        # Fill missing categorical values with explicit token
        cat_cols2 = detect_categorical_columns(out, cfg)
        cat_cols2 = [c for c in cat_cols2 if c not in numeric_set]
        for c in cat_cols2:
            out[c] = out[c].fillna(cfg.lgb_missing_token)

    return out


def preprocess_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: Optional[PreprocessConfig] = None,
    *,
    for_model: str = "catboost",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper to preprocess both train and test consistently.
    """
    if cfg is None:
        cfg = PreprocessConfig()

    train_clean = preprocess_dataframe(train_df, cfg, is_train=True, for_model=for_model)
    test_clean  = preprocess_dataframe(test_df,  cfg, is_train=False, for_model=for_model)

    # Ensure columns match between train and test (except target)
    if cfg.target_col in train_clean.columns and cfg.target_col in test_clean.columns:
        # should not happen; test shouldn't have target
        test_clean = test_clean.drop(columns=[cfg.target_col], errors="ignore")

    # Align columns (sometimes train has extra columns from target)
    train_cols = [c for c in train_clean.columns if c != cfg.target_col]
    test_clean = test_clean.reindex(columns=train_cols, fill_value=np.nan)

    return train_clean, test_clean



