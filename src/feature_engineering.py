"""
Derived predictors for the leakage-safe IBM engagement experiment.

Row-wise ratios only. No train-set statistics (no target encoding, no
global quantiles). The five EngagementLevel ingredients are never used.

This module does not train a model and does not change the source CSV.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.target import ENGAGEMENT_INPUT_COLUMNS, LEVEL_COLUMN, SCORE_COLUMN

FORBIDDEN_COLUMNS = ENGAGEMENT_INPUT_COLUMNS + [
    SCORE_COLUMN,
    LEVEL_COLUMN,
    "OverallSatisfactionIndex",
]

# (output_name, required_source_columns, description)
FEATURE_SPECS = [
    (
        "IncomePerYearExperience",
        ("MonthlyIncome", "TotalWorkingYears"),
        "MonthlyIncome / (TotalWorkingYears + 1). Pay relative to career length.",
    ),
    (
        "IncomePerCompanyYear",
        ("MonthlyIncome", "YearsAtCompany"),
        "MonthlyIncome / (YearsAtCompany + 1). Pay relative to tenure here.",
    ),
    (
        "CurrentRoleRatio",
        ("YearsInCurrentRole", "YearsAtCompany"),
        "YearsInCurrentRole / (YearsAtCompany + 1). Share of company tenure in the current role.",
    ),
    (
        "ManagerTenureRatio",
        ("YearsWithCurrManager", "YearsAtCompany"),
        "YearsWithCurrManager / (YearsAtCompany + 1). Share of company tenure with the current manager.",
    ),
    (
        "PromotionGap",
        ("YearsSinceLastPromotion", "YearsAtCompany"),
        "YearsSinceLastPromotion / (YearsAtCompany + 1). How much of company tenure has passed since the last promotion.",
    ),
    (
        "CompanyTenureRatio",
        ("YearsAtCompany", "Age"),
        "YearsAtCompany / Age. Company tenure as a fraction of age.",
    ),
    (
        "CareerProgressionRate",
        ("JobLevel", "TotalWorkingYears"),
        "JobLevel / (TotalWorkingYears + 1). Job level relative to years of experience.",
    ),
    (
        "ExperienceAtCompanyRatio",
        ("YearsAtCompany", "TotalWorkingYears"),
        "YearsAtCompany / (TotalWorkingYears + 1). Share of working life spent at this company.",
    ),
    (
        "YearsWithoutPromotion",
        ("YearsSinceLastPromotion",),
        "Same values as YearsSinceLastPromotion, named explicitly as time since last promotion.",
    ),
    (
        "OvertimeIndicator",
        ("OverTime",),
        "1 if OverTime is Yes, else 0.",
    ),
]


def _safe_div(numerator, denominator):
    return numerator.astype(float) / (denominator.astype(float) + 1.0)


def add_derived_features(X):
    """
    Return a copy of X with extra columns that can be built from X.

    Skips a feature if any source column is missing.
    Refuses to run if a forbidden leakage column is present in X.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("add_derived_features expects a pandas DataFrame.")

    leaked = [c for c in FORBIDDEN_COLUMNS if c in X.columns]
    if leaked:
        raise ValueError(
            "Derived features must not be built from a table that still contains "
            "target-defining columns: " + ", ".join(leaked)
        )

    out = X.copy()
    created = []
    skipped = []

    def need(cols):
        return all(c in out.columns for c in cols)

    if need(("MonthlyIncome", "TotalWorkingYears")):
        out["IncomePerYearExperience"] = _safe_div(out["MonthlyIncome"], out["TotalWorkingYears"])
        created.append("IncomePerYearExperience")
    else:
        skipped.append("IncomePerYearExperience")

    if need(("MonthlyIncome", "YearsAtCompany")):
        out["IncomePerCompanyYear"] = _safe_div(out["MonthlyIncome"], out["YearsAtCompany"])
        created.append("IncomePerCompanyYear")
    else:
        skipped.append("IncomePerCompanyYear")

    if need(("YearsInCurrentRole", "YearsAtCompany")):
        out["CurrentRoleRatio"] = _safe_div(out["YearsInCurrentRole"], out["YearsAtCompany"])
        created.append("CurrentRoleRatio")
    else:
        skipped.append("CurrentRoleRatio")

    if need(("YearsWithCurrManager", "YearsAtCompany")):
        out["ManagerTenureRatio"] = _safe_div(out["YearsWithCurrManager"], out["YearsAtCompany"])
        created.append("ManagerTenureRatio")
    else:
        skipped.append("ManagerTenureRatio")

    if need(("YearsSinceLastPromotion", "YearsAtCompany")):
        out["PromotionGap"] = _safe_div(out["YearsSinceLastPromotion"], out["YearsAtCompany"])
        created.append("PromotionGap")
    else:
        skipped.append("PromotionGap")

    if need(("YearsAtCompany", "Age")):
        age = out["Age"].astype(float).replace(0, np.nan)
        out["CompanyTenureRatio"] = out["YearsAtCompany"].astype(float) / age
        created.append("CompanyTenureRatio")
    else:
        skipped.append("CompanyTenureRatio")

    if need(("JobLevel", "TotalWorkingYears")):
        out["CareerProgressionRate"] = _safe_div(out["JobLevel"], out["TotalWorkingYears"])
        created.append("CareerProgressionRate")
    else:
        skipped.append("CareerProgressionRate")

    if need(("YearsAtCompany", "TotalWorkingYears")):
        out["ExperienceAtCompanyRatio"] = _safe_div(out["YearsAtCompany"], out["TotalWorkingYears"])
        created.append("ExperienceAtCompanyRatio")
    else:
        skipped.append("ExperienceAtCompanyRatio")

    if need(("YearsSinceLastPromotion",)):
        out["YearsWithoutPromotion"] = out["YearsSinceLastPromotion"].astype(float)
        created.append("YearsWithoutPromotion")
    else:
        skipped.append("YearsWithoutPromotion")

    if need(("OverTime",)):
        out["OvertimeIndicator"] = (
            out["OverTime"].astype(str).str.strip().str.lower().eq("yes").astype(int)
        )
        created.append("OvertimeIndicator")
    else:
        skipped.append("OvertimeIndicator")

    out.attrs["derived_created"] = created
    out.attrs["derived_skipped"] = skipped
    return out


def feature_engineering_table(created, skipped):
    rows = []
    created_set = set(created)
    for name, sources, description in FEATURE_SPECS:
        rows.append(
            {
                "feature": name,
                "source_columns": ", ".join(sources),
                "created": name in created_set,
                "reason_if_skipped": "" if name in created_set else "source column missing",
                "description": description,
            }
        )
    extra_skip = [s for s in skipped if s not in {spec[0] for spec in FEATURE_SPECS}]
    for name in extra_skip:
        rows.append(
            {
                "feature": name,
                "source_columns": "",
                "created": False,
                "reason_if_skipped": "not in FEATURE_SPECS",
                "description": "",
            }
        )
    return pd.DataFrame(rows)


class DerivedFeatureAdder(BaseEstimator, TransformerMixin):
    """sklearn step so a saved pipeline can rebuild ratios from original HR columns."""

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DerivedFeatureAdder requires a pandas DataFrame.")
        leaked = [c for c in FORBIDDEN_COLUMNS if c in X.columns]
        if leaked:
            raise ValueError("Forbidden leakage columns in X: " + ", ".join(leaked))
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = int(X.shape[1])
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        return add_derived_features(X)


def build_dynamic_preprocessor():
    """
    ColumnTransformer that infers types after derived columns are added.

    Numeric: StandardScaler. Categorical: OneHotEncoder (unknown levels ignored).
    """
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("scaler", StandardScaler())]),
                make_column_selector(dtype_include=np.number),
            ),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        )
                    ]
                ),
                make_column_selector(dtype_include=["object", "category", "bool"]),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
