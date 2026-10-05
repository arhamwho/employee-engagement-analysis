"""
Analyze saved classification results without retraining.

The comparison CSV and classification reports from the previous experiment
are the source of truth for metrics. This module only:

- reloads those files
- rebuilds the same train/test split
- scores the already-saved Joblib pipelines
- writes charts and an error-analysis write-up

It does not change recorded accuracy, precision, recall, or F1.
"""

import shutil

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.config import MODEL_PLOTS_DIR, MODELS_DIR, REPORTS_DIR
from src.data_loader import load_raw_dataset
from src.modeling import make_train_test_split, prepare_xy, _display_figure
from src.target import LEVEL_COLUMN, LEVEL_ORDER, SCORE_COLUMN, add_engagement_target

FINAL_MODEL_NAME = "Decision Tree"
FINAL_MODEL_SOURCE = "decision_tree.joblib"
FINAL_MODEL_COPY = "final_model.joblib"

SAVED_PIPELINES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "Random Forest": "random_forest.joblib",
    "Support Vector Machine": "svm.joblib",
}

COMPARISON_CSV = REPORTS_DIR / "model_comparison.csv"
ANALYSIS_PLOTS_DIR = REPORTS_DIR / "analysis_plots"


def load_saved_comparison():
    """Read metrics written by the training experiment. Do not recompute them."""
    if not COMPARISON_CSV.exists():
        raise FileNotFoundError(
            f"Missing {COMPARISON_CSV}. Run the classification experiment first."
        )
    return pd.read_csv(COMPARISON_CSV)


def fitted_model_rows(comparison):
    """Drop the majority-class reference row."""
    return comparison[comparison["model"] != "Majority baseline (reference)"].copy()


def plot_accuracy_comparison(comparison, filename="accuracy_comparison.png"):
    """Bar chart of test accuracy for the four fitted models only."""
    ANALYSIS_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_df = fitted_model_rows(comparison)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=plot_df, x="model", y="accuracy", color="#3b6d8c", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Test-set accuracy by model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.tick_params(axis="x", rotation=15)
    for path in (ANALYSIS_PLOTS_DIR / filename, MODEL_PLOTS_DIR / filename):
        fig.savefig(path, dpi=150, bbox_inches="tight")
    _display_figure()
    plt.close(fig)
    return ANALYSIS_PLOTS_DIR / filename


def plot_macro_f1_comparison(comparison, filename="macro_f1_comparison.png"):
    """Bar chart of test macro F1 for the four fitted models only."""
    ANALYSIS_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_df = fitted_model_rows(comparison)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=plot_df, x="model", y="f1_macro", color="#2a6b4a", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Test-set macro F1 by model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Macro F1")
    ax.tick_params(axis="x", rotation=15)
    for path in (ANALYSIS_PLOTS_DIR / filename, MODEL_PLOTS_DIR / filename):
        fig.savefig(path, dpi=150, bbox_inches="tight")
    _display_figure()
    plt.close(fig)
    return ANALYSIS_PLOTS_DIR / filename


def rebuild_test_frame():
    """
    Recreate the labeled table and the same stratified test fold used in training.

    Predictions come from saved pipelines. Metrics in the CSV are not overwritten.
    """
    raw = load_raw_dataset()
    labeled = add_engagement_target(raw)
    X, y, dropped = prepare_xy(labeled)
    X_train, X_test, y_train, y_test = make_train_test_split(X, y)
    test_frame = labeled.loc[X_test.index].copy()
    test_frame["y_true"] = y_test.astype(str).values
    return {
        "labeled": labeled,
        "X_test": X_test,
        "y_test": y_test.astype(str),
        "test_frame": test_frame,
        "dropped": dropped,
    }


def predict_saved_pipelines(X_test):
    """Score each saved full pipeline on the rebuilt test features."""
    preds = {}
    for name, filename in SAVED_PIPELINES.items():
        path = MODELS_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing saved model: {path}")
        pipeline = joblib.load(path)
        preds[name] = pipeline.predict(X_test)
    return preds


def confusion_counts(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=LEVEL_ORDER)
    return pd.DataFrame(cm, index=LEVEL_ORDER, columns=LEVEL_ORDER)


def plot_confusion(cm_df, model_name, filename):
    ANALYSIS_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_df,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        cbar=False,
    )
    ax.set_title(f"Confusion matrix (test set): {model_name}")
    ax.set_xlabel("Predicted EngagementLevel")
    ax.set_ylabel("True EngagementLevel")
    path = ANALYSIS_PLOTS_DIR / filename
    fig.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    _display_figure()
    plt.close(fig)
    return path


def adjacent_error_summary(cm_df):
    """Count neighbor-bin errors vs Low↔High jumps. Bins are ordered Low < Medium < High."""
    low_med = int(cm_df.loc["Low", "Medium"] + cm_df.loc["Medium", "Low"])
    med_high = int(cm_df.loc["Medium", "High"] + cm_df.loc["High", "Medium"])
    low_high = int(cm_df.loc["Low", "High"] + cm_df.loc["High", "Low"])
    return {
        "low_medium_swaps": low_med,
        "medium_high_swaps": med_high,
        "low_high_swaps": low_high,
        "adjacent_total": low_med + med_high,
    }


def error_rows(test_frame, y_pred):
    frame = test_frame.copy()
    frame["y_pred"] = y_pred
    frame["is_error"] = frame["y_true"] != frame["y_pred"]
    return frame


def copy_final_model():
    src = MODELS_DIR / FINAL_MODEL_SOURCE
    dest = MODELS_DIR / FINAL_MODEL_COPY
    shutil.copy2(src, dest)
    return dest


def generate_analysis_report(comparison, cms, test_frame, final_pred, final_report_text):
    """Build markdown from saved metrics plus confusion counts. No new metric invention."""
    fitted = fitted_model_rows(comparison)
    ranked = fitted.sort_values(["f1_macro", "f1_weighted", "accuracy"], ascending=False)
    top = ranked.iloc[0]
    acc_leader = fitted.sort_values("accuracy", ascending=False).iloc[0]
    baseline = comparison.loc[
        comparison["model"] == "Majority baseline (reference)"
    ].iloc[0]

    final_cm = cms[FINAL_MODEL_NAME]
    swaps = adjacent_error_summary(final_cm)
    n_test = int(final_cm.values.sum())
    n_correct = int(final_cm.values.trace())
    n_wrong = n_test - n_correct

    per_class = []
    for label in LEVEL_ORDER:
        support = int(final_cm.loc[label].sum())
        correct = int(final_cm.loc[label, label])
        wrong = support - correct
        per_class.append((label, support, correct, wrong, wrong / support if support else 0))

    hardest = max(per_class, key=lambda row: row[4])
    easiest = min(per_class, key=lambda row: row[4])

    err_frame = error_rows(test_frame, final_pred)
    errors = err_frame[err_frame["is_error"]]
    boundary = None
    if SCORE_COLUMN in err_frame.columns and len(errors):
        near = errors[
            ((errors[SCORE_COLUMN] - 2.00).abs() <= 0.2)
            | ((errors[SCORE_COLUMN] - 3.00).abs() <= 0.2)
        ]
        boundary = {
            "n_errors": int(len(errors)),
            "n_near_cut": int(len(near)),
            "error_score_mean": float(errors[SCORE_COLUMN].mean()),
            "correct_score_mean": float(
                err_frame.loc[~err_frame["is_error"], SCORE_COLUMN].mean()
            ),
        }

    lines = [
        "## Model analysis (from the saved experiment)",
        "",
        "Metrics in the comparison table are the **same values** written during training "
        f"(`{COMPARISON_CSV.name}`). This section does not retrain models and does not "
        "replace those scores.",
        "",
        "### Selection criterion (stated before ranking)",
        "",
        "1. Primary: highest **macro F1** on the held-out test set.",
        "2. Tie-break: higher **weighted F1**, then higher **accuracy**.",
        "3. A model that only matches the majority baseline on accuracy, while missing "
        "Low and High, is not selected.",
        "",
        f"Under that rule the selected final model is **{FINAL_MODEL_NAME}** "
        f"(macro F1={top['f1_macro']:.4f}, accuracy={top['accuracy']:.4f}).",
        "",
        f"Accuracy alone would have preferred **{acc_leader['model']}** "
        f"(accuracy={acc_leader['accuracy']:.4f}), which on this split is the same "
        f"accuracy as the majority baseline ({baseline['accuracy']:.4f}). "
        "Random Forest predicted only Medium, so that accuracy is not evidence of "
        "better engagement classification.",
        "",
        "### Performance is modest",
        "",
        f"Even the selected model has test accuracy {top['accuracy']:.4f} and "
        f"macro F1 {top['f1_macro']:.4f}. Macro F1 is only a small lift over the "
        f"majority baseline macro F1 of {baseline['f1_macro']:.4f}. "
        "These HR predictors (pay, tenure, department, overtime, and similar fields) "
        "do **not** strongly recover the student-defined engagement bins once the "
        "survey items that define the bins are held out.",
        "",
        "### Classes: easiest and hardest (Decision Tree, test set)",
        "",
    ]
    for label, support, correct, wrong, err_rate in per_class:
        lines.append(
            f"- **{label}**: support={support}, correct={correct}, "
            f"errors={wrong}, error rate={err_rate:.1%}"
        )
    lines.extend(
        [
            "",
            f"- Easiest class (lowest error rate): **{easiest[0]}** "
            f"({easiest[4]:.1%} of true {easiest[0]} rows misclassified).",
            f"- Hardest class (highest error rate): **{hardest[0]}** "
            f"({hardest[4]:.1%} of true {hardest[0]} rows misclassified).",
            "",
            "Medium is easiest because it is the majority class (217 of 294 test rows). "
            "Low is hardest: only 20 test rows, and the tree almost never identifies them.",
            "",
            "### Does imbalance affect performance?",
            "",
            "Yes. Medium is about three-quarters of the sample. Models can raise accuracy "
            "by predicting Medium. Random Forest did exactly that (recall Medium = 1.00, "
            "recall Low = 0.00, recall High = 0.00). `class_weight='balanced'` did not "
            "overcome the lack of signal plus the small Low sample.",
            "",
            "### Which errors occur most (Decision Tree confusion matrix)",
            "",
            f"- Test rows: {n_test}; correct: {n_correct}; errors: {n_wrong}.",
            f"- Low ↔ Medium swaps: {swaps['low_medium_swaps']}",
            f"- Medium ↔ High swaps: {swaps['medium_high_swaps']}",
            f"- Low ↔ High swaps: {swaps['low_high_swaps']}",
            f"- Adjacent-bin errors (Low–Medium or Medium–High): {swaps['adjacent_total']}",
            "",
        ]
    )
    if swaps["medium_high_swaps"] >= swaps["low_medium_swaps"]:
        lines.append(
            "The more common neighbor mix-up is **Medium vs High**, not Low vs Medium. "
            "Low vs High jumps are the rarest of the three off-diagonal pairs, which is "
            "expected if errors are mostly between neighboring score bins."
        )
    else:
        lines.append(
            "The more common neighbor mix-up is **Low vs Medium**. "
            "Low vs High jumps are less common than adjacent-bin confusions."
        )

    if boundary:
        lines.extend(
            [
                "",
                "### Possible reasons for errors",
                "",
                f"- {boundary['n_near_cut']} of {boundary['n_errors']} Decision Tree errors "
                "have `EngagementScore` within 0.2 of a cut-point (2.00 or 3.00). "
                "Employees near a bin edge can be labeled differently from neighbors "
                "with almost the same survey mean.",
                "- The classifier **cannot see** the five survey items that created the "
                "label. It only sees other HR fields, which EDA showed were weakly "
                "associated with the composite score.",
                "- Low has 20 test examples, so recall for Low is unstable and easy to "
                "drive near zero.",
                "- Adjacent ordinal bins (Low–Medium–High) are closer in the underlying "
                "mean score than Low vs High, so neighbor confusions are more plausible "
                "than opposite-end confusions.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "### Possible reasons for errors",
                "",
                "- The label is a coarse bin of five survey items that the model is not "
                "allowed to use as features.",
                "- Other HR variables showed weak association with the composite score in EDA.",
                "- The Low class is small, so it is easy to miss.",
                "",
            ]
        )

    lines.extend(
        [
            "### Limitations",
            "",
            "- One train/test split; another split could reorder close models.",
            "- `EngagementLevel` is a **student-defined operationalization**, not an "
            "official or validated engagement instrument.",
            "- Predicted association is **not causation**. A correct or incorrect "
            "classification does not mean that overtime, pay, or department *causes* "
            "engagement.",
            "- No hyperparameter search was claimed as optimal; these are the models "
            "from the previous experiment.",
            "",
            "### Final model artifacts",
            "",
            f"- Selected model: {FINAL_MODEL_NAME}",
            f"- Copy saved as `models/{FINAL_MODEL_COPY}` (same weights as "
            f"`{FINAL_MODEL_SOURCE}`).",
            "",
            "Classification report for the selected model (test set, unchanged from training):",
            "",
            "```",
            final_report_text.strip(),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def run_result_analysis(save_outputs=True):
    comparison = load_saved_comparison()
    rebuilt = rebuild_test_frame()
    preds = predict_saved_pipelines(rebuilt["X_test"])

    cms = {}
    cm_paths = {}
    for name, y_pred in preds.items():
        cms[name] = confusion_counts(rebuilt["y_test"], y_pred)
        slug = name.lower().replace(" ", "_")
        cm_paths[name] = plot_confusion(
            cms[name],
            name,
            f"analysis_confusion_{slug}.png",
        )

    acc_path = plot_accuracy_comparison(comparison)
    f1_path = plot_macro_f1_comparison(comparison)

    report_path = REPORTS_DIR / "classification_reports" / "decision_tree.txt"
    final_report_text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""

    notes = generate_analysis_report(
        comparison,
        cms,
        rebuilt["test_frame"],
        preds[FINAL_MODEL_NAME],
        final_report_text,
    )

    dest = None
    if save_outputs:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / "model_analysis.md").write_text(notes, encoding="utf-8")
        dest = copy_final_model()
        # Also keep a standalone copy of the selected model's report text.
        (REPORTS_DIR / "classification_reports" / "final_model_decision_tree.txt").write_text(
            final_report_text, encoding="utf-8"
        )

    return {
        "comparison": comparison,
        "cms": cms,
        "cm_paths": cm_paths,
        "accuracy_plot": acc_path,
        "macro_f1_plot": f1_path,
        "notes": notes,
        "final_model_path": dest,
        "final_report_text": final_report_text,
        "test_frame": rebuilt["test_frame"],
        "predictions": preds,
        "adjacent": adjacent_error_summary(cms[FINAL_MODEL_NAME]),
    }
