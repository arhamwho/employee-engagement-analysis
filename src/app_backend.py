"""
Helpers for the Streamlit app.

Loads the saved Decision Tree pipeline, the HR table, and the metric CSV.
Does not train a model.
"""

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.analysis import COMPARISON_CSV, FINAL_MODEL_COPY, FINAL_MODEL_SOURCE
from src.config import MODELS_DIR, REPORTS_DIR
from src.data_loader import load_raw_dataset, validate_dataset_file
from src.modeling import make_train_test_split, prepare_xy
from src.preprocess import select_model_features, split_numeric_categorical
from src.target import LEVEL_COLUMN, LEVEL_ORDER, add_engagement_target

FINAL_MODEL_DISPLAY_NAME = "Decision Tree"


def age_group_from_age(age):
    age = int(age)
    if age < 25:
        return "Under 25"
    if age <= 34:
        return "25-34"
    if age <= 44:
        return "35-44"
    if age <= 54:
        return "45-54"
    return "55+"


def tenure_group_from_years(years):
    years = int(years)
    if years <= 2:
        return "New (0-2 yrs)"
    if years <= 5:
        return "Early (3-5 yrs)"
    if years <= 10:
        return "Established (6-10 yrs)"
    return "Veteran (10+ yrs)"


def income_bracket_from_monthly(income):
    """Cut-points follow the observed ranges in this project's CSV."""
    income = float(income)
    if income <= 2911:
        return "Low"
    if income <= 4908:
        return "Medium"
    if income <= 8376:
        return "High"
    return "Very High"


def resolve_final_model_path():
    for name in (FINAL_MODEL_COPY, FINAL_MODEL_SOURCE):
        path = MODELS_DIR / name
        if path.is_file():
            return path
    return None


def load_labeled_frame():
    """Working copy with EngagementLevel. Original CSV is not modified."""
    status = validate_dataset_file()
    if not status["available"]:
        return None, status
    raw = load_raw_dataset()
    labeled = add_engagement_target(raw)
    return labeled, status


def load_final_pipeline():
    path = resolve_final_model_path()
    if path is None:
        return None, None
    return joblib.load(path), path


def load_comparison_table():
    if not COMPARISON_CSV.exists():
        return None
    return pd.read_csv(COMPARISON_CSV)


def final_model_metrics(comparison):
    """Return the Decision Tree row from the saved experiment CSV."""
    if comparison is None:
        return None
    match = comparison[comparison["model"] == FINAL_MODEL_DISPLAY_NAME]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def load_feature_importance_table():
    path = REPORTS_DIR / "feature_importance.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_classification_report_text():
    path = REPORTS_DIR / "classification_reports" / "decision_tree.txt"
    if not path.exists():
        alt = REPORTS_DIR / "classification_reports" / "final_model_decision_tree.txt"
        path = alt if alt.exists() else path
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def model_feature_lists(labeled_df):
    X, dropped = select_model_features(labeled_df)
    numeric_cols, categorical_cols, _other = split_numeric_categorical(X)
    return {
        "X": X,
        "dropped": dropped,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "category_options": {
            col: sorted(X[col].astype(str).unique().tolist())
            for col in categorical_cols
        },
        "numeric_bounds": {
            col: {
                "min": float(X[col].min()),
                "max": float(X[col].max()),
                "median": float(X[col].median()),
            }
            for col in numeric_cols
        },
    }


def engagement_share_table(labeled_df):
    counts = labeled_df[LEVEL_COLUMN].value_counts().reindex(LEVEL_ORDER).fillna(0)
    total = int(counts.sum())
    return pd.DataFrame(
        {
            "EngagementLevel": LEVEL_ORDER,
            "count": [int(counts.loc[level]) for level in LEVEL_ORDER],
            "percent": [
                round(100.0 * float(counts.loc[level]) / total, 2) if total else 0.0
                for level in LEVEL_ORDER
            ],
        }
    )


def test_confusion_for_saved_pipeline(labeled_df, pipeline):
    """
    Confusion matrix on the same stratified test fold used in training.
    Uses the saved pipeline. Does not fit a new model.
    """
    X, y, _dropped = prepare_xy(labeled_df)
    _X_train, X_test, _y_train, y_test = make_train_test_split(X, y)
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test.astype(str), y_pred, labels=LEVEL_ORDER)
    return pd.DataFrame(cm, index=LEVEL_ORDER, columns=LEVEL_ORDER)


def prediction_input_warnings(feature_row, labeled_df=None):
    """
    Flag inconsistent HR profiles. Does not invent values or change the row.
    """
    row = feature_row.iloc[0]
    warnings = []
    pairs = [
        ("YearsAtCompany", "TotalWorkingYears", "years at company exceed total working years"),
        ("YearsInCurrentRole", "YearsAtCompany", "years in current role exceed years at company"),
        ("YearsWithCurrManager", "YearsAtCompany", "years with current manager exceed years at company"),
        ("YearsSinceLastPromotion", "YearsAtCompany", "years since promotion exceed years at company"),
    ]
    for left, right, text in pairs:
        if left in row.index and right in row.index:
            if float(row[left]) > float(row[right]):
                warnings.append(text)
    if "Age" in row.index and "TotalWorkingYears" in row.index:
        if float(row["Age"]) - float(row["TotalWorkingYears"]) < 14:
            warnings.append(
                "age minus total working years is under 14 (unusual work-start age)"
            )
    if (
        labeled_df is not None
        and "Department" in row.index
        and "JobRole" in row.index
        and "Department" in labeled_df.columns
        and "JobRole" in labeled_df.columns
    ):
        combo = labeled_df[
            (labeled_df["Department"].astype(str) == str(row["Department"]))
            & (labeled_df["JobRole"].astype(str) == str(row["JobRole"]))
        ]
        if combo.empty:
            warnings.append(
                "this Department and JobRole pair does not appear in the loaded table"
            )
    return warnings


def predict_engagement(pipeline, feature_row, labeled_df=None):
    """
    feature_row: one-row DataFrame with the same column names the pipeline saw.
    """
    if "preprocess" not in getattr(pipeline, "named_steps", {}):
        raise ValueError(
            "Saved artifact is not a full sklearn Pipeline with a 'preprocess' step."
        )
    prep = pipeline.named_steps["preprocess"]
    expected = list(getattr(prep, "feature_names_in_", []))
    if not expected:
        raise ValueError(
            "The saved preprocessor has no feature_names_in_. Re-run the notebook "
            "training cells so the pipeline is saved with column names."
        )
    missing = [col for col in expected if col not in feature_row.columns]
    if missing:
        raise ValueError("Prediction input is missing required columns: " + ", ".join(missing))

    use = feature_row[expected]
    if use.isna().any().any():
        bad = use.columns[use.isna().any()].tolist()
        raise ValueError(
            "Prediction input has missing values in: "
            + ", ".join(bad)
            + ". Fill those fields before predicting."
        )

    pred = pipeline.predict(use)[0]
    result = {
        "label": str(pred),
        "probabilities": None,
        "warnings": prediction_input_warnings(use, labeled_df=labeled_df),
    }
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(use)[0]
        classes = [str(c) for c in pipeline.classes_]
        result["probabilities"] = {cls: float(p) for cls, p in zip(classes, proba)}
    return result
