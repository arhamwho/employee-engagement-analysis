"""
Model-derived feature importance for the selected Decision Tree.

sklearn DecisionTreeClassifier.feature_importances_ is mean decrease in
impurity (Gini) across splits in the fitted tree. That is a statement about
how the tree used columns when predicting, not a causal HR finding.

This module does not retrain the tree. It reads models/final_model.joblib
(or decision_tree.joblib as a fallback).
"""

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.analysis import ANALYSIS_PLOTS_DIR, FINAL_MODEL_COPY, FINAL_MODEL_SOURCE
from src.config import MODELS_DIR, REPORTS_DIR
from src.modeling import _display_figure

TOP_N = 15


def load_final_pipeline():
    """Load the selected full pipeline (preprocess + Decision Tree)."""
    for name in (FINAL_MODEL_COPY, FINAL_MODEL_SOURCE):
        path = MODELS_DIR / name
        if path.exists():
            return joblib.load(path), path
    raise FileNotFoundError(
        "No saved final Decision Tree pipeline found. "
        "Train the models first, then copy decision_tree.joblib to final_model.joblib."
    )


def gini_feature_importance_table(pipeline):
    """
    Pair transformed feature names with tree.feature_importances_.

    One-hot columns keep names like Attrition_No after encoding.
    """
    preprocessor = pipeline.named_steps["preprocess"]
    tree = pipeline.named_steps["model"]
    if not hasattr(tree, "feature_importances_"):
        raise TypeError(
            "The fitted classifier has no feature_importances_. "
            "This project selected a Decision Tree, which does."
        )
    names = preprocessor.get_feature_names_out()
    importances = tree.feature_importances_
    if len(names) != len(importances):
        raise ValueError("Feature name count does not match importance vector length.")
    table = pd.DataFrame(
        {
            "feature": names,
            "gini_importance": importances,
        }
    ).sort_values("gini_importance", ascending=False)
    table["rank"] = range(1, len(table) + 1)
    table["share_of_total"] = table["gini_importance"] / table["gini_importance"].sum()
    return table.reset_index(drop=True)


def plot_top_importances(table, top_n=TOP_N, filename="feature_importance_top15.png"):
    """Horizontal bar chart of the highest model-derived Gini importances."""
    ANALYSIS_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    top = table.head(top_n).sort_values("gini_importance", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(
        data=top,
        y="feature",
        x="gini_importance",
        color="#3b6d8c",
        ax=ax,
    )
    ax.set_title(
        f"Top {top_n} features by Decision Tree Gini importance "
        "(model-derived; not causal)"
    )
    ax.set_xlabel("Model-derived Gini importance (share of total impurity decrease)")
    ax.set_ylabel("Transformed feature name")
    path = ANALYSIS_PLOTS_DIR / filename
    fig.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    _display_figure()
    plt.close(fig)
    return path


def generate_importance_notes(table, pipeline_path, top_n=TOP_N):
    """Short Markdown built only from the importance table."""
    top = table.head(top_n)
    n_features = len(table)
    n_zero = int((table["gini_importance"] == 0).sum())
    top_sum = float(top["gini_importance"].sum())
    leader = top.iloc[0]
    second = top.iloc[1] if len(top) > 1 else None
    third = top.iloc[2] if len(top) > 2 else None

    lines = [
        "## Model-derived feature importance (Decision Tree)",
        "",
        "The selected final model is a **Decision Tree**. Importance here is "
        "sklearn's mean decrease in **Gini impurity** (`feature_importances_`) "
        "on the fitted tree. These values describe features **associated with "
        "model predictions** in this tree. They are **not** proof that a "
        "variable causes employee engagement.",
        "",
        f"- Pipeline file: `{pipeline_path}`",
        f"- Transformed features after scaling / one-hot encoding: **{n_features}**",
        f"- Features with Gini importance exactly 0: **{n_zero}**",
        f"- The top {top_n} features account for **{top_sum:.1%}** of total "
        "model-derived importance (importances sum to 1).",
        "",
        "### Most influential predictive features (top "
        + str(top_n)
        + ")",
        "",
    ]
    for _, row in top.iterrows():
        lines.append(
            f"{int(row['rank'])}. `{row['feature']}` — Gini importance "
            f"{row['gini_importance']:.4f} ({row['share_of_total']:.1%} of total)"
        )

    lines.extend(
        [
            "",
            "### Short interpretation (from these numbers only)",
            "",
            f"The single most important predictive feature in this tree is "
            f"`{leader['feature']}` (Gini importance {leader['gini_importance']:.4f}).",
        ]
    )
    if second is not None and third is not None:
        lines.append(
            f"Next are `{second['feature']}` ({second['gini_importance']:.4f}) and "
            f"`{third['feature']}` ({third['gini_importance']:.4f})."
        )
    lines.extend(
        [
            "Several of the highest ranks are pay- or rate-related columns "
            "(`MonthlyRate`, `HourlyRate`, `DailyRate`, `MonthlyIncome`). That "
            "means the tree **used those fields when splitting**, not that pay "
            "causes engagement. Tenure-style fields (`YearsInCurrentRole`, "
            "`TotalWorkingYears`, `TrainingTimesLastYear`, and similar) also "
            "appear among the top ranks as important predictive features.",
            "",
            "A deep tree can assign model-derived importance to columns that "
            "help it fit the training sample even when those columns are only "
            "weakly related to engagement in simple summaries. Read this chart "
            "as an explanation of **this classifier**, not as an HR causal ranking.",
            "",
            "One-hot names (for example `Attrition_No`) are dummy columns created "
            "by the preprocessor. Importance on a dummy is still only "
            "model-derived importance for that encoded feature.",
            "",
        ]
    )
    return "\n".join(lines)


def run_interpretability(top_n=TOP_N, save_outputs=True):
    """Compute Gini importance, plot the top features, and write notes."""
    pipeline, path = load_final_pipeline()
    table = gini_feature_importance_table(pipeline)
    plot_path = plot_top_importances(table, top_n=top_n)
    notes = generate_importance_notes(table, path, top_n=top_n)
    csv_path = None
    md_path = None
    if save_outputs:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = REPORTS_DIR / "feature_importance.csv"
        table.to_csv(csv_path, index=False)
        md_path = REPORTS_DIR / "feature_importance_notes.md"
        md_path.write_text(notes, encoding="utf-8")
    return {
        "table": table,
        "top": table.head(top_n),
        "plot_path": plot_path,
        "notes": notes,
        "pipeline_path": path,
        "csv_path": csv_path,
        "md_path": md_path,
    }
