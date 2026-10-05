"""
Preprocessing pipeline for EngagementLevel classification.

Fit this transformer on the training features only, then transform train
and test with the same fitted object. That keeps the experiment honest:
test rows never influence scaling or one-hot categories.

Leakage rule:
    EngagementLevel is built from five survey items (and EngagementScore
    is their mean). Those columns, plus OverallSatisfactionIndex, are
    dropped so the model cannot "cheat" by reconstructing the label.

This module does not train a classifier.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.target import ENGAGEMENT_INPUT_COLUMNS, LEVEL_COLUMN, SCORE_COLUMN

# Extra derived/index columns that repeat the label construction.
LEAKAGE_COLUMNS = ENGAGEMENT_INPUT_COLUMNS + [
    SCORE_COLUMN,
    LEVEL_COLUMN,
    "OverallSatisfactionIndex",
]

# IDs and constants that cannot help a model generalize.
NON_FEATURE_COLUMNS = [
    "EmployeeNumber",
    "EmployeeCount",
    "Over18",
    "StandardHours",
]


def feature_exclusion_columns():
    """Columns that must not be used as predictors."""
    return list(dict.fromkeys(LEAKAGE_COLUMNS + NON_FEATURE_COLUMNS))


def select_model_features(df):
    """
    Return the predictor table (X) and the list of dropped column names.

    Original survey items stay in the full labeled DataFrame; they are
    only excluded from X.
    """
    drop_cols = [col for col in feature_exclusion_columns() if col in df.columns]
    X = df.drop(columns=drop_cols)
    return X, drop_cols


def split_numeric_categorical(X):
    """Detect column types from the feature table actually used for modeling."""
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    other_cols = [c for c in X.columns if c not in numeric_cols + categorical_cols]
    return numeric_cols, categorical_cols, other_cols


def build_preprocessor(X):
    """
    Build an unfitted sklearn ColumnTransformer.

    - Numeric columns: StandardScaler (helps Logistic Regression and SVM).
    - Categorical columns: OneHotEncoder (unknown test levels are ignored).
    """
    numeric_cols, categorical_cols, other_cols = split_numeric_categorical(X)
    if other_cols:
        raise ValueError(
            "Feature columns have unsupported dtypes and were not assigned: "
            + ", ".join(other_cols)
        )

    numeric_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )
    # dense output so every classifier (including SVM) receives a normal array
    categorical_pipeline = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor, numeric_cols, categorical_cols


def preprocessor_feature_names(fitted_preprocessor):
    """Names after scaling / one-hot encoding (sklearn >= 1.0)."""
    return fitted_preprocessor.get_feature_names_out().tolist()
