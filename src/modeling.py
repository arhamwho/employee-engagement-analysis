"""
Train and evaluate classifiers for EngagementLevel.

All four models use:
- the same stratified train/test split
- the same fitted preprocessor
- random_state for reproducibility
- class_weight="balanced" so rare classes get more weight without
  copying rows (no random oversampling)

Metrics are computed on the held-out test set only.
This module does not invent scores.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.config import MODEL_PLOTS_DIR, MODELS_DIR, RANDOM_STATE, REPORTS_DIR, TEST_SIZE
from src.preprocess import build_preprocessor, select_model_features
from src.target import LEVEL_COLUMN, LEVEL_ORDER, add_engagement_target


MODEL_SPECS = [
    (
        "Logistic Regression",
        "logistic_regression.joblib",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    ),
    (
        "Decision Tree",
        "decision_tree.joblib",
        DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    ),
    (
        "Random Forest",
        "random_forest.joblib",
        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    ),
    (
        "Support Vector Machine",
        "svm.joblib",
        SVC(
            kernel="rbf",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    ),
]


def _display_figure():
    backend = plt.get_backend().lower()
    if "agg" not in backend:
        plt.show()


def prepare_xy(labeled_df):
    """Features X and target y from a table that already has EngagementLevel."""
    if LEVEL_COLUMN not in labeled_df.columns:
        raise ValueError("Run add_engagement_target before prepare_xy.")
    y = labeled_df[LEVEL_COLUMN].astype(str)
    X, dropped = select_model_features(labeled_df)
    return X, y, dropped


def make_train_test_split(X, y):
    """One stratified split shared by every model."""
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def multiclass_metrics(y_true, y_pred):
    """
    Test-set scores for one model.

    For a 3-class problem, Precision / Recall / F1 need an averaging rule.
    Macro = unweighted mean over Low, Medium, High (each class counts equally).
    Weighted = mean weighted by how many test rows are in each class.
    """
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
        "precision_weighted": float(
            precision_score(
                y_true, y_pred, labels=LEVEL_ORDER, average="weighted", zero_division=0
            )
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, labels=LEVEL_ORDER, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, labels=LEVEL_ORDER, average="weighted", zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, y_pred, labels=LEVEL_ORDER, average="macro", zero_division=0)
        ),
        "precision": float(
            precision_score(y_true, y_pred, labels=LEVEL_ORDER, average="macro", zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, labels=LEVEL_ORDER, average="macro", zero_division=0)
        ),
    }


def plot_confusion_matrix(y_true, y_pred, model_name, filename):
    """Heatmap of true vs predicted labels on the test set."""
    MODEL_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=LEVEL_ORDER)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LEVEL_ORDER,
        yticklabels=LEVEL_ORDER,
        ax=ax,
        cbar=False,
    )
    ax.set_title(f"Confusion matrix (test set): {model_name}")
    ax.set_xlabel("Predicted EngagementLevel")
    ax.set_ylabel("True EngagementLevel")
    path = MODEL_PLOTS_DIR / filename
    fig.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    _display_figure()
    plt.close(fig)
    return path, cm


def plot_metric_comparison(comparison, filename="model_metric_comparison.png"):
    """Grouped bars of accuracy vs macro F1 for the four fitted models."""
    MODEL_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_df = comparison[comparison["model"] != "Majority baseline (reference)"].copy()
    long = plot_df.melt(
        id_vars="model",
        value_vars=["accuracy", "f1_macro"],
        var_name="metric",
        value_name="score",
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=long, x="model", y="score", hue="metric", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Test-set accuracy vs macro F1 (do not rank by accuracy alone)")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.legend(title="")
    ax.tick_params(axis="x", rotation=15)
    path = MODEL_PLOTS_DIR / filename
    fig.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    _display_figure()
    plt.close(fig)
    return path


def _safe_filename(model_name):
    return model_name.lower().replace(" ", "_")


def train_and_evaluate_models(labeled_df, save_artifacts=True):
    """
    Fit the preprocessor on training rows only, train four classifiers
    on the same split, and return metrics computed from test predictions.
    """
    X, y, dropped = prepare_xy(labeled_df)
    X_train, X_test, y_train, y_test = make_train_test_split(X, y)

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X_train)
    preprocessor.fit(X_train)

    X_train_p = preprocessor.transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    if save_artifacts:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")

    split_info = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "dropped_columns": dropped,
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "train_class_counts": y_train.value_counts().reindex(LEVEL_ORDER).fillna(0).astype(int).to_dict(),
        "test_class_counts": y_test.value_counts().reindex(LEVEL_ORDER).fillna(0).astype(int).to_dict(),
    }

    result_rows = []
    reports = {}
    confusion_paths = {}

    # Reference only: always predict the most common training class.
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train_p, y_train)
    dummy_pred = dummy.predict(X_test_p)
    dummy_metrics = multiclass_metrics(y_test, dummy_pred)
    dummy_metrics["model"] = "Majority baseline (reference)"
    result_rows.append(dummy_metrics)
    reports["Majority baseline (reference)"] = classification_report(
        y_test,
        dummy_pred,
        labels=LEVEL_ORDER,
        zero_division=0,
        digits=4,
    )

    fitted_models = {}
    for model_name, artifact_name, estimator in MODEL_SPECS:
        estimator.fit(X_train_p, y_train)
        y_pred = estimator.predict(X_test_p)
        metrics = multiclass_metrics(y_test, y_pred)
        metrics["model"] = model_name
        result_rows.append(metrics)
        reports[model_name] = classification_report(
            y_test,
            y_pred,
            labels=LEVEL_ORDER,
            zero_division=0,
            digits=4,
        )
        cm_name = f"confusion_matrix_{_safe_filename(model_name)}.png"
        path, _cm = plot_confusion_matrix(y_test, y_pred, model_name, cm_name)
        confusion_paths[model_name] = str(path)
        fitted_models[model_name] = estimator

        if save_artifacts:
            full_pipeline = Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    ("model", estimator),
                ]
            )
            joblib.dump(full_pipeline, MODELS_DIR / artifact_name)

    comparison = pd.DataFrame(result_rows)
    # Keep a stable column order for the notebook table.
    column_order = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "f1_macro",
        "precision_macro",
        "recall_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
    ]
    comparison = comparison[column_order]
    comparison_plot = plot_metric_comparison(comparison)

    if save_artifacts:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
        reports_dir = REPORTS_DIR / "classification_reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        for name, text in reports.items():
            (reports_dir / f"{_safe_filename(name)}.txt").write_text(text, encoding="utf-8")

    return {
        "comparison": comparison,
        "reports": reports,
        "confusion_paths": confusion_paths,
        "comparison_plot": str(comparison_plot),
        "split_info": split_info,
        "fitted_models": fitted_models,
        "preprocessor": preprocessor,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def generate_modeling_notes(run_result):
    """
    Write discussion from the comparison table after metrics exist.

    Does not pick a winner from accuracy alone. Macro F1 is emphasized
    because Low is a rare class in this dataset.
    """
    comparison = run_result["comparison"].copy()
    split_info = run_result["split_info"]
    candidates = comparison[comparison["model"] != "Majority baseline (reference)"]

    # Rank by macro F1, then weighted F1, then accuracy as a tie-breaker only.
    ranked = candidates.sort_values(
        by=["f1_macro", "f1_weighted", "accuracy"],
        ascending=False,
    )
    top = ranked.iloc[0]
    second = ranked.iloc[1] if len(ranked) > 1 else None

    baseline = comparison.loc[
        comparison["model"] == "Majority baseline (reference)"
    ].iloc[0]

    lines = [
        "## Classification experiment notes",
        "",
        "These numbers come from the **held-out test set** after one stratified "
        f"split (test_size={split_info['test_size']}, "
        f"random_state={split_info['random_state']}). "
        "They are not typed in by hand.",
        "",
        "### Why these metrics",
        "",
        "- **Accuracy** is the share of test employees whose `EngagementLevel` "
        "was predicted correctly. With a large Medium class, a model can look "
        "accurate while missing Low and High.",
        "- **Precision** (macro) asks, for each class, how often a predicted "
        "label was actually that class, then averages the three classes equally.",
        "- **Recall** (macro) asks, for each class, how many of the true members "
        "were found, then averages equally. This matters for the small Low group.",
        "- **F1-score / macro F1** is the balance of precision and recall, "
        "averaged equally across Low, Medium, and High. In this multiclass, "
        "imbalanced problem, **macro F1 is more informative than accuracy**.",
        "- **Weighted** precision / recall / F1 weight each class by its test "
        "support. They stay closer to accuracy when Medium dominates.",
        "",
        "Do **not** choose a model from accuracy alone.",
        "",
        "### Split and features",
        "",
        f"- Training rows: {split_info['n_train']}",
        f"- Test rows: {split_info['n_test']}",
        f"- Train class counts: {split_info['train_class_counts']}",
        f"- Test class counts: {split_info['test_class_counts']}",
        "- Columns excluded from X (leakage, IDs, constants): "
        + ", ".join(f"`{c}`" for c in split_info["dropped_columns"]),
        "",
        "Imbalance was handled with `class_weight='balanced'` (no duplicate-row "
        "oversampling). The split is stratified so each class appears in train and test.",
        "",
        "### Majority-class reference",
        "",
        f"- Always predicting the training majority class scores "
        f"accuracy={baseline['accuracy']:.4f} and "
        f"macro F1={baseline['f1_macro']:.4f} on this test set. "
        "A useful model should beat this on **macro F1**, not only match accuracy.",
        "",
        "### Comparison (test set)",
        "",
    ]
    for _, row in candidates.iterrows():
        lines.append(
            f"- **{row['model']}**: accuracy={row['accuracy']:.4f}, "
            f"precision (macro)={row['precision']:.4f}, "
            f"recall (macro)={row['recall']:.4f}, "
            f"F1 (macro)={row['f1_macro']:.4f}, "
            f"F1 (weighted)={row['f1_weighted']:.4f}"
        )

    lines.extend(
        [
            "",
            "### Reading the ranking",
            "",
            f"Using **macro F1 first** (then weighted F1, then accuracy), "
            f"the strongest of the four fitted models on this test split is "
            f"**{top['model']}** (macro F1={top['f1_macro']:.4f}, "
            f"accuracy={top['accuracy']:.4f}).",
        ]
    )
    if second is not None:
        lines.append(
            f"Next is **{second['model']}** "
            f"(macro F1={second['f1_macro']:.4f}, accuracy={second['accuracy']:.4f})."
        )
    acc_leader = candidates.sort_values("accuracy", ascending=False).iloc[0]
    if acc_leader["model"] != top["model"]:
        lines.append(
            f"**Accuracy would have picked `{acc_leader['model']}`** "
            f"(accuracy={acc_leader['accuracy']:.4f}), which is a different "
            "choice than macro F1. That is why accuracy is not used alone."
        )
    else:
        lines.append(
            f"On this split, `{top['model']}` also has the highest accuracy "
            f"among the four ({top['accuracy']:.4f}), but ranking still used "
            "macro F1 as the primary metric because of class imbalance."
        )

    lines.extend(
        [
            "",
            "This ranking is for **this split and this feature set**. "
            "It is not a claim that the student-defined engagement label is an "
            "official HR instrument.",
            "",
            "Saved artifacts (if saving was enabled): `models/preprocessor.joblib` "
            "and one full pipeline `.joblib` file per classifier.",
            "",
        ]
    )
    return "\n".join(lines)


def save_modeling_notes(text, filename="modeling_notes.md"):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(REPORTS_DIR) / filename
    path.write_text(text, encoding="utf-8")
    return path


def run_classification_experiment(raw_or_labeled_df, save_artifacts=True):
    """
    Convenience entry point from the notebook.

    Accepts either a raw copy or a table that already has EngagementLevel.
    """
    if LEVEL_COLUMN not in raw_or_labeled_df.columns:
        labeled = add_engagement_target(raw_or_labeled_df)
    else:
        labeled = raw_or_labeled_df.copy()
    run_result = train_and_evaluate_models(labeled, save_artifacts=save_artifacts)
    notes = generate_modeling_notes(run_result)
    if save_artifacts:
        save_modeling_notes(notes)
    run_result["notes"] = notes
    run_result["labeled_df"] = labeled
    return run_result
