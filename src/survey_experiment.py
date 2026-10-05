"""
Independent employee-survey engagement experiment (Experiment 3).

The IBM HR demo is unchanged. This module trains only when a real survey
CSV with employee rows is present. It never invents responses or metrics.

EngagementScore is the mean of six 1–5 items that are collected independently
of the workplace/HR predictors. Those six items are excluded from X.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    RANDOM_STATE,
    SURVEY_DATA_DIR,
    SURVEY_DATA_FILENAME,
    SURVEY_DATA_PATH,
    SURVEY_MODELS_DIR,
    SURVEY_PLOTS_DIR,
    SURVEY_REPORTS_DIR,
    SURVEY_TEMPLATE_FILENAME,
    SURVEY_TEMPLATE_PATH,
    TEST_SIZE,
)

PREDICTOR_COLUMNS = [
    "Age",
    "Department",
    "JobRole",
    "JobLevel",
    "YearsExperience",
    "YearsAtCompany",
    "MonthlyIncome",
    "Overtime",
    "WorkHoursPerWeek",
    "TrainingOpportunities",
    "CareerGrowth",
    "ManagerSupport",
    "Recognition",
    "WorkLifeBalance",
    "WorkArrangement",
]

ENGAGEMENT_QUESTION_COLUMNS = [
    "Motivation",
    "ValuedByOrganization",
    "OrganizationalConnection",
    "ExtraEffort",
    "RecommendOrganization",
    "IntentionToStay",
]

TEMPLATE_COLUMNS = PREDICTOR_COLUMNS + ENGAGEMENT_QUESTION_COLUMNS

SCORE_COLUMN = "EngagementScore"
LEVEL_COLUMN = "EngagementLevel"
LEVEL_ORDER = ["Low", "Medium", "High"]

# Project-defined equal-width thirds of a 1–5 Likert mean.
# Not a validated psychological scale.
LEVEL_BINS = [1.00, 2.33, 3.67, 5.00]
LEVEL_RULES = (
    "Low: 1.00–2.33; Medium: 2.34–3.67; High: 3.68–5.00 "
    "(pd.cut bins [1.00, 2.33, 3.67, 5.00], right-closed, include_lowest=True)"
)

TARGET_DISCLAIMER = (
    "EngagementScore and EngagementLevel are a student-defined operationalization "
    "for the independent survey experiment. They are not an official HR standard "
    "and not a clinically validated engagement scale. The six engagement questions "
    "must not be used as predictors."
)

MIN_ROWS_FOR_MODELING = 30

SURVEY_NOT_READY_MESSAGE = (
    "No real survey responses were found.\n"
    "The file in data/survey/ is only a column template, or the data CSV is missing.\n"
    "No fake employee rows were created and no survey-model metrics were invented.\n"
    "\n"
    f"1. Copy {SURVEY_TEMPLATE_PATH} to {SURVEY_DATA_PATH}\n"
    "2. Add one row per consenting employee (do not invent answers).\n"
    "3. Re-run notebooks/survey_engagement_model.ipynb.\n"
)


def resolve_survey_csv_path():
    """Prefer the real survey file; ignore the empty template as a data source."""
    if SURVEY_DATA_PATH.is_file():
        return SURVEY_DATA_PATH
    if not SURVEY_DATA_DIR.is_dir():
        return None
    for path in sorted(SURVEY_DATA_DIR.glob("*.csv")):
        if path.name == SURVEY_TEMPLATE_FILENAME:
            continue
        return path
    return None


def load_survey_table():
    """
    Load a copy of the survey CSV.

    Returns (dataframe or None, status dict).
    Zero-row files (the template) are not treated as collected data.
    """
    path = resolve_survey_csv_path()
    if path is None:
        if SURVEY_TEMPLATE_PATH.is_file():
            template = pd.read_csv(SURVEY_TEMPLATE_PATH)
            return None, {
                "available": False,
                "path": str(SURVEY_TEMPLATE_PATH),
                "n_rows": int(len(template)),
                "message": SURVEY_NOT_READY_MESSAGE,
            }
        return None, {
            "available": False,
            "path": None,
            "n_rows": 0,
            "message": SURVEY_NOT_READY_MESSAGE,
        }

    try:
        raw = pd.read_csv(path)
    except Exception as exc:
        return None, {
            "available": False,
            "path": str(path),
            "n_rows": 0,
            "message": f"Could not read {path}: {exc}",
        }

    if raw.empty:
        return None, {
            "available": False,
            "path": str(path),
            "n_rows": 0,
            "message": SURVEY_NOT_READY_MESSAGE,
        }

    return raw.copy(), {
        "available": True,
        "path": str(path),
        "n_rows": int(len(raw)),
        "message": f"Loaded {len(raw)} survey row(s) from {path}",
    }


def missing_required_columns(df):
    return [c for c in TEMPLATE_COLUMNS if c not in df.columns]


def validate_survey_frame(df):
    """Column presence, Likert range, and missingness. Does not fill values."""
    missing_cols = missing_required_columns(df)
    likert_cols = [
        c
        for c in (
            "TrainingOpportunities",
            "CareerGrowth",
            "ManagerSupport",
            "Recognition",
            "WorkLifeBalance",
        )
        + tuple(ENGAGEMENT_QUESTION_COLUMNS)
        if c in df.columns
    ]
    out_of_range = {}
    for col in likert_cols:
        numeric = pd.to_numeric(df[col], errors="coerce")
        bad = numeric.dropna()
        n_bad = int(((bad < 1) | (bad > 5)).sum())
        if n_bad:
            out_of_range[col] = n_bad

    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "missing_required_columns": missing_cols,
        "missing_values_by_column": df.isna().sum().astype(int).to_dict(),
        "missing_values_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "likert_out_of_range_1_to_5": out_of_range,
        "ready_for_target": len(missing_cols) == 0 and df.shape[0] > 0,
    }


def add_survey_engagement_target(df):
    missing = [c for c in ENGAGEMENT_QUESTION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing engagement-question columns: " + ", ".join(missing))
    out = df.copy()
    items = out[ENGAGEMENT_QUESTION_COLUMNS].apply(pd.to_numeric, errors="coerce")
    out[SCORE_COLUMN] = items.mean(axis=1)
    out[LEVEL_COLUMN] = pd.cut(
        out[SCORE_COLUMN],
        bins=LEVEL_BINS,
        labels=LEVEL_ORDER,
        include_lowest=True,
        right=True,
    )
    n_unlabeled = int(out[LEVEL_COLUMN].isna().sum())
    if n_unlabeled:
        raise ValueError(
            f"{n_unlabeled} row(s) could not be assigned EngagementLevel. "
            "Check for missing Likert answers or scores outside 1.00–5.00."
        )
    return out


def select_survey_features(df):
    """Workplace/HR predictors only. Engagement questions and derived labels stay out of X."""
    drop = [
        c
        for c in ENGAGEMENT_QUESTION_COLUMNS + [SCORE_COLUMN, LEVEL_COLUMN]
        if c in df.columns
    ]
    extra = [c for c in df.columns if c not in PREDICTOR_COLUMNS and c not in drop]
    drop = drop + extra
    X = df.drop(columns=[c for c in drop if c in df.columns])
    keep = [c for c in PREDICTOR_COLUMNS if c in X.columns]
    X = X[keep]
    return X, drop


def class_distribution(y):
    counts = pd.Series(y).astype(str).value_counts().reindex(LEVEL_ORDER).fillna(0).astype(int)
    total = int(counts.sum())
    return pd.DataFrame(
        {
            "EngagementLevel": LEVEL_ORDER,
            "count": [int(counts.loc[k]) for k in LEVEL_ORDER],
            "percent": [
                round(100.0 * float(counts.loc[k]) / total, 2) if total else 0.0
                for k in LEVEL_ORDER
            ],
        }
    )


def make_preprocessor(X):
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("scaler", StandardScaler())]), numeric_cols),
            (
                "categorical",
                Pipeline(
                    [("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def multiclass_metrics(y_true, y_pred):
    y_true = pd.Series(y_true).astype(str)
    y_pred = pd.Series(y_pred).astype(str)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, labels=LEVEL_ORDER, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, labels=LEVEL_ORDER, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, labels=LEVEL_ORDER, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def _estimators():
    return [
        (
            "Logistic Regression",
            LogisticRegression(max_iter=4000, class_weight="balanced", random_state=RANDOM_STATE),
            {"model__C": [0.25, 1.0, 4.0], "model__class_weight": ["balanced", None]},
        ),
        (
            "Decision Tree",
            DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            {
                "model__max_depth": [4, 8, None],
                "model__min_samples_leaf": [1, 4],
                "model__class_weight": ["balanced", None],
            },
        ),
        (
            "Random Forest",
            RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
            {
                "model__n_estimators": [150, 300],
                "model__max_depth": [6, None],
                "model__min_samples_leaf": [1, 4],
                "model__class_weight": ["balanced", None],
            },
        ),
        (
            "Extra Trees",
            ExtraTreesClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
            {
                "model__n_estimators": [150, 300],
                "model__max_depth": [6, None],
                "model__class_weight": ["balanced", None],
            },
        ),
        (
            "Gradient Boosting",
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [80, 160],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
            },
        ),
    ]


def can_model(y):
    counts = pd.Series(y).astype(str).value_counts()
    if int(counts.sum()) < MIN_ROWS_FOR_MODELING:
        return False, (
            f"Need at least {MIN_ROWS_FOR_MODELING} labeled rows to train. "
            f"Found {int(counts.sum())}. No metrics were invented."
        )
    if int((counts < 2).sum()) > 0:
        return False, (
            "Every class that appears must have at least 2 rows for a stratified split. "
            f"Counts: {counts.to_dict()}. No metrics were invented."
        )
    if counts.size < 2:
        return False, "Need at least two EngagementLevel classes. No metrics were invented."
    return True, "ok"


def train_survey_models(labeled_df, save_artifacts=True):
    """
    Fit on train only. HPO is RandomizedSearchCV with scoring=f1_macro.
    Returns None-like skip info if the table is too small.
    """
    y = labeled_df[LEVEL_COLUMN].astype(str)
    ok, reason = can_model(y)
    if not ok:
        return {"skipped": True, "reason": reason}

    X, dropped = select_survey_features(labeled_df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    n_train = int(len(X_train))
    cv_splits = 3 if n_train >= 45 else 2
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

    majority = y_train.value_counts().idxmax()
    dummy_pred = np.full(len(y_test), majority, dtype=object)
    dummy_metrics = multiclass_metrics(y_test, dummy_pred)
    dummy_metrics["model"] = "Majority baseline (reference)"

    rows = [dummy_metrics]
    reports = {
        "Majority baseline (reference)": classification_report(
            y_test, dummy_pred, labels=LEVEL_ORDER, zero_division=0, digits=4
        )
    }
    cv_scores = {}
    fitted = {}
    confusion_paths = {}
    SURVEY_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for name, estimator, param_grid in _estimators():
        prep = make_preprocessor(X_train)
        pipe = Pipeline([("preprocess", prep), ("model", estimator)])
        n_iter = min(8, int(np.prod([len(v) for v in param_grid.values()])))
        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_grid,
            n_iter=max(1, n_iter),
            scoring="f1_macro",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_
        y_pred = best.predict(X_test)
        metrics = multiclass_metrics(y_test, y_pred)
        metrics["model"] = name
        metrics["cv_macro_f1"] = float(search.best_score_)
        rows.append(metrics)
        reports[name] = classification_report(
            y_test, y_pred, labels=LEVEL_ORDER, zero_division=0, digits=4
        )
        cv_scores[name] = float(search.best_score_)
        fitted[name] = best

        cm = confusion_matrix(y_test.astype(str), y_pred, labels=LEVEL_ORDER)
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_title(f"Confusion matrix (test): {name}")
        ax.set_xlabel("Predicted EngagementLevel")
        ax.set_ylabel("True EngagementLevel")
        path = SURVEY_PLOTS_DIR / f"confusion_{name.lower().replace(' ', '_')}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        confusion_paths[name] = str(path)

        if save_artifacts:
            SURVEY_MODELS_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(best, SURVEY_MODELS_DIR / f"{name.lower().replace(' ', '_')}.joblib")

    comparison = pd.DataFrame(rows)
    candidates = comparison[comparison["model"] != "Majority baseline (reference)"]
    best_row = candidates.sort_values(
        by=["f1_macro", "balanced_accuracy", "accuracy"], ascending=False
    ).iloc[0]
    best_name = str(best_row["model"])
    best_pipe = fitted[best_name]

    if save_artifacts:
        SURVEY_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(SURVEY_REPORTS_DIR / "model_comparison.csv", index=False)
        joblib.dump(best_pipe, SURVEY_MODELS_DIR / "final_survey_model.joblib")
        (SURVEY_REPORTS_DIR / "best_model.txt").write_text(
            f"{best_name}\nmacro F1={best_row['f1_macro']:.4f}\n", encoding="utf-8"
        )

    return {
        "skipped": False,
        "comparison": comparison,
        "reports": reports,
        "cv_macro_f1": cv_scores,
        "best_model_name": best_name,
        "best_pipeline": best_pipe,
        "confusion_paths": confusion_paths,
        "dropped_columns": dropped,
        "n_train": n_train,
        "n_test": int(len(X_test)),
        "cv_folds": cv_splits,
        "majority_class": str(majority),
        "X_test": X_test,
        "y_test": y_test,
        "y_pred_best": best_pipe.predict(X_test),
        "X_train": X_train,
        "y_train": y_train,
    }


def gini_importance_table(pipeline):
    tree = pipeline.named_steps["model"]
    if not hasattr(tree, "feature_importances_"):
        return None
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    table = pd.DataFrame(
        {"feature": names, "gini_importance": tree.feature_importances_}
    ).sort_values("gini_importance", ascending=False)
    table["rank"] = range(1, len(table) + 1)
    return table.reset_index(drop=True)


def error_analysis_table(X_test, y_test, y_pred):
    frame = X_test.copy()
    frame["true_EngagementLevel"] = pd.Series(y_test).astype(str).values
    frame["predicted_EngagementLevel"] = pd.Series(y_pred).astype(str).values
    frame["correct"] = frame["true_EngagementLevel"] == frame["predicted_EngagementLevel"]
    return frame


def load_ibm_experiment_comparison():
    """Read already-computed IBM investigation numbers. Do not invent them."""
    path = Path(__file__).resolve().parent.parent / "reports" / "accuracy_investigation" / "comparison_table.csv"
    if not path.is_file():
        fallback = Path(__file__).resolve().parent.parent / "reports" / "model_comparison.csv"
        if fallback.is_file():
            return pd.read_csv(fallback), str(fallback)
        return None, None
    return pd.read_csv(path), str(path)
