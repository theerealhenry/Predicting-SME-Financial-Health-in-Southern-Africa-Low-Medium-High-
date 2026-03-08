from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple
import numpy as np
import pandas as pd


# =========================================================
# CONFIG
# =========================================================

@dataclass
class FeatureConfig:
    id_col: str = "ID"
    target_col: str = "Target"

    financial_access_cols: Tuple[str, ...] = (
        "has_mobile_money",
        "has_debit_card",
        "has_credit_card",
        "has_internet_banking",
        "has_loan_account",
    )

    insurance_coverage_cols: Tuple[str, ...] = (
        "has_insurance",
        "motor_vehicle_insurance",
        "medical_insurance",
        "funeral_insurance",
    )

    resilience_risk_cols: Tuple[str, ...] = (
        "current_problem_cash_flow",
        "problem_sourcing_money",
        "attitude_worried_shutdown",
        "future_risk_theft_stock",
    )

    formality_cols: Tuple[str, ...] = (
        "compliance_income_tax",
        "keeps_financial_records",
    )

    business_confidence_cols: Tuple[str, ...] = (
        "attitude_stable_business_environment",
        "attitude_satisfied_with_achievement",
        "attitude_more_successful_next_year",
        "motivation_make_more_money",
    )

    insurance_friction_cols: Tuple[str, ...] = (
        "perception_insurance_doesnt_cover_losses",
        "perception_cannot_afford_insurance",
        "perception_insurance_companies_dont_insure_businesses_like_yours",
    )

    informal_coping_cols: Tuple[str, ...] = (
        "uses_friends_family_savings",
        "uses_informal_lender",
    )

    rare_min_count: int = 20
    rare_fill_value: str = "other"


# =========================================================
# HELPERS
# =========================================================

YES_VALUES = {
    "yes", "y", "1", "true", "have now", "yes, always", "yes, sometimes"
}

NO_VALUES = {
    "no", "n", "0", "false", "never had", "used to have but don't have now"
}

AFFIRMATIVE_VALUES = {
    "yes", "y", "1", "true", "agree", "strongly agree"
}

NEGATIVE_VALUES = {
    "no", "n", "0", "false", "disagree", "strongly disagree",
    "never had", "used to have but don't have now"
}


def _safe_string(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.lower()


def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    out = a / b.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def _binary_from_category(s: pd.Series) -> pd.Series:
    """
    Convert categorical yes/no-like or agree/disagree-like values to 1/0.
    Unknown values remain NaN.

    Important:
    - For beneficial columns, 1 means presence of a positive attribute.
    - For risk/friction columns, 1 means presence of a risk/friction attribute.
    The interpretation depends on the feature group, not on this function.
    """
    s2 = _safe_string(s).fillna("missing")
    out = pd.Series(np.nan, index=s.index, dtype="float64")

    out[s2.isin(AFFIRMATIVE_VALUES)] = 1.0
    out[s2.isin(NEGATIVE_VALUES)] = 0.0

    # Broader matching for messy text
    out[s2.str.contains("yes", na=False)] = 1.0
    out[s2.str.contains("agree", na=False)] = 1.0
    out[s2.str.contains("never had", na=False)] = 0.0
    out[s2.str.contains("used to have", na=False)] = 0.0
    out[s2.str.fullmatch(r"no", na=False)] = 0.0
    out[s2.str.contains("disagree", na=False)] = 0.0

    return out


def _make_score(df: pd.DataFrame, cols: Iterable[str], score_name: str, mode: str = "yes") -> pd.DataFrame:
    df = df.copy()
    available = [c for c in cols if c in df.columns]

    bin_cols = []
    for c in available:
        bin_col = f"{c}__bin"
        df[bin_col] = _binary_from_category(df[c])
        bin_cols.append(bin_col)

    if bin_cols:
        df[score_name] = df[bin_cols].fillna(0).sum(axis=1)
        df[f"{score_name}_missing_count"] = df[bin_cols].isna().sum(axis=1)

    return df


def _combine_cat(df: pd.DataFrame, c1: str, c2: str, new_col: str) -> pd.DataFrame:
    if c1 in df.columns and c2 in df.columns:
        a = df[c1].astype("string").fillna("missing")
        b = df[c2].astype("string").fillna("missing")
        df[new_col] = a + "__x__" + b
    return df


def _combine_cat_with_score(df: pd.DataFrame, cat_col: str, score_col: str, new_col: str) -> pd.DataFrame:
    if cat_col in df.columns and score_col in df.columns:
        a = df[cat_col].astype("string").fillna("missing")
        b = df[score_col].fillna(-1).astype(int).astype("string")
        df[new_col] = a + "__x__" + b
    return df


def _make_age_bucket(age_years: pd.Series) -> pd.Series:
    age_years = _safe_numeric(age_years)
    bucket = pd.cut(
        age_years,
        bins=[-np.inf, 0.5, 2, 5, 10, np.inf],
        labels=["under_6m", "y0_5_to_2", "y2_to_5", "y5_to_10", "y10_plus"]
    )
    return bucket.astype("string").fillna("missing")


def _collapse_rare_categories(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: Iterable[str],
    min_count: int = 20,
    fill_value: str = "other",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.copy()
    test_df = test_df.copy()

    for c in cols:
        if c not in train_df.columns or c not in test_df.columns:
            continue

        tr = train_df[c].astype("string").fillna("missing")
        te = test_df[c].astype("string").fillna("missing")

        vc = tr.value_counts(dropna=False)
        keepers = set(vc[vc >= min_count].index.tolist())

        train_df[c] = tr.where(tr.isin(keepers), fill_value)
        test_df[c] = te.where(te.isin(keepers), fill_value)

    return train_df, test_df


# =========================================================
# MAIN FEATURE ENGINEERING
# =========================================================

def engineer_features(df: pd.DataFrame, cfg: Optional[FeatureConfig] = None) -> pd.DataFrame:
    if cfg is None:
        cfg = FeatureConfig()

    df = df.copy()

    # -----------------------------------------------------
    # 1) Core FHI-aligned scores
    # These aggregate survey indicators into interpretable
    # dimensions of financial health:
    # - access to financial services
    # - insurance protection
    # - resilience and risk exposure
    # - business formality
    # - business confidence
    # - insurance market friction
    # - informal coping mechanisms
    # -----------------------------------------------------
    df = _make_score(df, cfg.financial_access_cols, "financial_access_score")
    df = _make_score(df, cfg.insurance_coverage_cols, "insurance_coverage_score")
    df = _make_score(df, cfg.resilience_risk_cols, "resilience_risk_score")
    df = _make_score(df, cfg.formality_cols, "formality_score")
    df = _make_score(df, cfg.business_confidence_cols, "business_confidence_score")
    df = _make_score(df, cfg.insurance_friction_cols, "insurance_friction_score")
    df = _make_score(df, cfg.informal_coping_cols, "informal_coping_score")

    # perception_insurance_important as separate binary
    if "perception_insurance_important" in df.columns:
        df["perception_insurance_important__bin"] = _binary_from_category(
            df["perception_insurance_important"], positive_mode="yes"
        )

    # -----------------------------------------------------
    # 2) Numeric business health proxies
    # -----------------------------------------------------
    for c in ["personal_income", "business_expenses", "business_turnover", "owner_age",
              "business_age_years", "business_age_months"]:
        if c in df.columns:
            df[c] = _safe_numeric(df[c])

    if "business_turnover" in df.columns and "business_expenses" in df.columns:
        df["margin_proxy"] = df["business_turnover"] - df["business_expenses"]
        df["expense_ratio"] = _safe_divide(df["business_expenses"], df["business_turnover"] + 1)
        df["turnover_to_expenses"] = _safe_divide(df["business_turnover"], df["business_expenses"] + 1)

    if "business_turnover" in df.columns and "personal_income" in df.columns:
        df["turnover_to_personal_income"] = _safe_divide(df["business_turnover"], df["personal_income"] + 1)

    if "business_expenses" in df.columns and "personal_income" in df.columns:
        df["expenses_to_personal_income"] = _safe_divide(df["business_expenses"], df["personal_income"] + 1)

    if "business_turnover" in df.columns and "log_business_turnover" not in df.columns:
        df["log_business_turnover"] = np.log1p(df["business_turnover"].clip(lower=0))

    if "business_expenses" in df.columns and "log_business_expenses" not in df.columns:
        df["log_business_expenses"] = np.log1p(df["business_expenses"].clip(lower=0))

    if "personal_income" in df.columns and "log_personal_income" not in df.columns:
        df["log_personal_income"] = np.log1p(df["personal_income"].clip(lower=0))

    if "margin_proxy" in df.columns:
        df["log_margin_proxy_pos"] = np.log1p(df["margin_proxy"].clip(lower=0))

    # -----------------------------------------------------
    # 3) Business maturity
    # -----------------------------------------------------
    if "business_age_years" in df.columns or "business_age_months" in df.columns:
        years = df["business_age_years"] if "business_age_years" in df.columns else 0
        months = df["business_age_months"] if "business_age_months" in df.columns else 0

        years = _safe_numeric(pd.Series(years))
        months = _safe_numeric(pd.Series(months))

        df["business_age_total_months"] = years * 12 + months.fillna(0)
        df["business_age_years_derived"] = df["business_age_total_months"] / 12.0
        df["business_age_bucket"] = _make_age_bucket(df["business_age_years_derived"])

    # -----------------------------------------------------
    # 4) Additional practical binary business features
    # -----------------------------------------------------
    extra_binary_cols = [
        "has_cellphone",
        "offers_credit_to_customers",
        "covid_essential_service",
        "marketing_word_of_mouth",
    ]

    for c in extra_binary_cols:
        if c in df.columns:
            df[f"{c}__bin"] = _binary_from_category(df[c], positive_mode="yes")

    # composite operational behavior score
    op_cols = [f"{c}__bin" for c in extra_binary_cols if f"{c}__bin" in df.columns]
    if op_cols:
        df["operational_behavior_score"] = df[op_cols].fillna(0).sum(axis=1)

    # -----------------------------------------------------
    # 5) FHI-inspired composite proxy
    # -----------------------------------------------------
    needed = [
        "financial_access_score",
        "insurance_coverage_score",
        "formality_score",
        "business_confidence_score",
        "resilience_risk_score",
        "informal_coping_score",
    ]
    if all(c in df.columns for c in needed):
        df["financial_health_proxy_score"] = (
            df["financial_access_score"].fillna(0)
            + df["insurance_coverage_score"].fillna(0)
            + df["formality_score"].fillna(0)
            + df["business_confidence_score"].fillna(0)
            - df["resilience_risk_score"].fillna(0)
            - df["informal_coping_score"].fillna(0)
        )

    # -----------------------------------------------------
    # 6) Interaction features
    # -----------------------------------------------------
    df = _combine_cat(df, "country", "owner_sex", "country_x_owner_sex")

    for score_col in [
        "financial_access_score",
        "insurance_coverage_score",
        "resilience_risk_score",
        "formality_score",
        "business_confidence_score",
        "informal_coping_score",
    ]:
        df = _combine_cat_with_score(df, "country", score_col, f"country_x_{score_col}")

    df = _combine_cat_with_score(df, "owner_sex", "financial_access_score", "owner_sex_x_financial_access_score")

    # optional raw-category interactions
    if "country" in df.columns and "has_mobile_money" in df.columns:
        df = _combine_cat(df, "country", "has_mobile_money", "country_x_has_mobile_money")

    if "country" in df.columns and "has_loan_account" in df.columns:
        df = _combine_cat(df, "country", "has_loan_account", "country_x_has_loan_account")

    # -----------------------------------------------------
    # 7) Bucketized numeric proxies
    # -----------------------------------------------------
    if "expense_ratio" in df.columns:
        df["expense_ratio_bucket"] = pd.cut(
            df["expense_ratio"],
            bins=[-np.inf, 0.25, 0.5, 0.75, 1.0, np.inf],
            labels=["very_low", "low", "moderate", "high", "very_high"]
        ).astype("string").fillna("missing")

    if "turnover_to_expenses" in df.columns:
        df["turnover_to_expenses_bucket"] = pd.cut(
            df["turnover_to_expenses"],
            bins=[-np.inf, 0.5, 1.0, 1.5, 2.5, np.inf],
            labels=["very_weak", "weak", "balanced", "strong", "very_strong"]
        ).astype("string").fillna("missing")

    # -----------------------------------------------------
    # 8) Missingness flags for important engineered metrics
    # -----------------------------------------------------
    for c in [
        "business_turnover",
        "business_expenses",
        "personal_income",
        "margin_proxy",
        "expense_ratio",
        "turnover_to_expenses",
        "turnover_to_personal_income",
    ]:
        if c in df.columns:
            df[f"{c}_missing_flag"] = df[c].isna().astype("int8")

    return df


def engineer_train_test_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: Optional[FeatureConfig] = None,
    collapse_rare_for_non_catboost: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cfg is None:
        cfg = FeatureConfig()

    train_fe = engineer_features(train_df, cfg=cfg)
    test_fe = engineer_features(test_df, cfg=cfg)

    if collapse_rare_for_non_catboost:
        cat_cols = [
            c for c in train_fe.columns
            if train_fe[c].dtype == "object" or str(train_fe[c].dtype).startswith("string")
        ]
        cat_cols = [c for c in cat_cols if c not in {cfg.id_col, cfg.target_col}]

        train_fe, test_fe = _collapse_rare_categories(
            train_fe,
            test_fe,
            cols=cat_cols,
            min_count=cfg.rare_min_count,
            fill_value=cfg.rare_fill_value,
        )

    return train_fe, test_fe