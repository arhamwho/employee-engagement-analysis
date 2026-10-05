"""
Student-defined engagement target for Case Study 90.

The IBM HR table does not contain a ready-made "Employee Engagement" label.
This module builds a transparent operationalization from five existing
1–4 survey items, then bins the mean into Low / Medium / High.

This is NOT an official HR standard and NOT a validated psychological scale.
It exists so this case study has a 3-class target that can be reproduced.

The five original columns are never dropped. The source CSV is never written.
This module does not train a model.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import REPORTS_DIR, TARGET_PLOTS_DIR

# Five survey items used in this case study's engagement operationalization.
# Order matches the assignment statement.
ENGAGEMENT_INPUT_COLUMNS = [
    "JobInvolvement",
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "RelationshipSatisfaction",
    "WorkLifeBalance",
]

SCORE_COLUMN = "EngagementScore"
LEVEL_COLUMN = "EngagementLevel"
LEVEL_ORDER = ["Low", "Medium", "High"]

# Project-defined bins (right-closed):
#   [1.00, 2.00] -> Low
#   (2.00, 3.00] -> Medium   (covers 2.01–3.00)
#   (3.00, 4.00] -> High     (covers 3.01–4.00)
LEVEL_BINS = [1.00, 2.00, 3.00, 4.00]

TARGET_DISCLAIMER = (
    "EngagementScore and EngagementLevel are a **student-defined operationalization** "
    "for Case Study 90. The IBM HR Analytics dataset does not include an official "
    "employee-engagement target. These thresholds are **not** an HR industry standard "
    "and the score is **not** a clinically or scientifically validated engagement scale. "
    "They are a transparent, reproducible rule so this project can train a "
    "3-class classifier later."
)


class MissingEngagementColumnsError(ValueError):
    """Raised when one or more required survey columns are absent."""


def required_columns_report(df):
    """
    Check that every input column exists.

    Returns a dict the notebook can print. Does not invent columns.
    """
    present = [col for col in ENGAGEMENT_INPUT_COLUMNS if col in df.columns]
    missing = [col for col in ENGAGEMENT_INPUT_COLUMNS if col not in df.columns]
    return {
        "required": list(ENGAGEMENT_INPUT_COLUMNS),
        "present": present,
        "missing": missing,
        "all_present": len(missing) == 0,
    }


def verify_engagement_input_columns(df):
    """Raise a clear error if any required column is missing."""
    report = required_columns_report(df)
    if not report["all_present"]:
        missing = ", ".join(report["missing"])
        raise MissingEngagementColumnsError(
            "Cannot build EngagementScore. Missing required column(s): "
            f"{missing}. Original variables were not changed."
        )
    return report


def engagement_score_from_row_mean(df):
    """
    EngagementScore = mean of the five survey items (equal weights).

    pandas mean(axis=1) averages across columns for each employee.
    Original columns are left as they are.
    """
    verify_engagement_input_columns(df)
    return df[ENGAGEMENT_INPUT_COLUMNS].mean(axis=1)


def engagement_level_from_score(score):
    """
    Map EngagementScore to Low / Medium / High using project thresholds.

    pd.cut with right=True and include_lowest=True implements:
    1.00–2.00 Low, 2.01–3.00 Medium, 3.01–4.00 High.
    """
    return pd.cut(
        score,
        bins=LEVEL_BINS,
        labels=LEVEL_ORDER,
        include_lowest=True,
        right=True,
    )


def add_engagement_target(df):
    """
    Return a copy with EngagementScore and EngagementLevel added.

    The DataFrame passed in is not modified in place.
    The five source columns stay in the table.
    """
    out = df.copy()
    verify_engagement_input_columns(out)
    out[SCORE_COLUMN] = engagement_score_from_row_mean(out)
    out[LEVEL_COLUMN] = engagement_level_from_score(out[SCORE_COLUMN])
    n_unlabeled = int(out[LEVEL_COLUMN].isna().sum())
    if n_unlabeled:
        raise ValueError(
            f"{n_unlabeled} row(s) could not be assigned EngagementLevel. "
            "Survey values may be missing or the mean score falls outside 1.00–4.00."
        )
    return out


def target_distribution_table(df):
    """Counts, percentages, and a simple imbalance summary from the labeled table."""
    if LEVEL_COLUMN not in df.columns:
        raise ValueError(f"Column {LEVEL_COLUMN} is missing. Run add_engagement_target first.")

    counts = df[LEVEL_COLUMN].value_counts(dropna=False)
    counts = counts.reindex(LEVEL_ORDER)
    # If a level has 0 people, reindex keeps it with NaN — fill with 0.
    counts = counts.fillna(0).astype(int)
    total = int(counts.sum())
    percent = (100.0 * counts / total).round(2) if total else counts.astype(float)

    majority_level = str(counts.idxmax()) if total else None
    minority_level = str(counts.idxmin()) if total else None
    majority_n = int(counts.max()) if total else 0
    minority_n = int(counts.min()) if total else 0
    majority_pct = float(percent.loc[majority_level]) if majority_level is not None else 0.0
    imbalance_ratio = (
        float(majority_n) / float(minority_n) if minority_n else float("inf")
    )

    table = pd.DataFrame(
        {
            "EngagementLevel": LEVEL_ORDER,
            "count": [int(counts.loc[level]) for level in LEVEL_ORDER],
            "percent": [float(percent.loc[level]) for level in LEVEL_ORDER],
        }
    )

    return {
        "table": table,
        "total": total,
        "majority_level": majority_level,
        "majority_count": majority_n,
        "majority_percent": majority_pct,
        "minority_level": minority_level,
        "minority_count": minority_n,
        "imbalance_ratio_majority_to_minority": round(imbalance_ratio, 2),
        "classes_are_imbalanced": majority_pct >= 50.0 or imbalance_ratio >= 2.0,
    }


def _ensure_target_plot_dir():
    TARGET_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    return TARGET_PLOTS_DIR


def _save_target_figure(filename):
    path = _ensure_target_plot_dir() / filename
    plt.savefig(path, dpi=150, bbox_inches="tight")
    return path


def _display_figure():
    backend = plt.get_backend().lower()
    if "agg" not in backend:
        plt.show()


def plot_engagement_score_distribution(df, filename="engagement_score_distribution.png"):
    """Histogram of the continuous student-defined score, with threshold lines."""
    if SCORE_COLUMN not in df.columns:
        print("Skipped: EngagementScore is not in the table yet.")
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df[SCORE_COLUMN].dropna(), bins=16, kde=True, color="#3b6d8c", ax=ax)
    ax.axvline(2.00, color="#b35c2a", linestyle="--", linewidth=1.4, label="Low | Medium (2.00)")
    ax.axvline(3.00, color="#2a6b4a", linestyle="--", linewidth=1.4, label="Medium | High (3.00)")
    ax.set_title("Distribution of student-defined EngagementScore")
    ax.set_xlabel("EngagementScore (mean of five 1–4 survey items)")
    ax.set_ylabel("Count")
    ax.legend()
    path = _save_target_figure(filename)
    _display_figure()
    plt.close(fig)
    return path


def plot_engagement_level_distribution(df, filename="engagement_level_distribution.png"):
    """Bar chart of Low / Medium / High counts."""
    if LEVEL_COLUMN not in df.columns:
        print("Skipped: EngagementLevel is not in the table yet.")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(
        data=df,
        x=LEVEL_COLUMN,
        order=LEVEL_ORDER,
        color="#3b6d8c",
        ax=ax,
    )
    ax.set_title("Distribution of student-defined EngagementLevel")
    ax.set_xlabel("EngagementLevel (project-defined bins)")
    ax.set_ylabel("Number of employees")
    path = _save_target_figure(filename)
    _display_figure()
    plt.close(fig)
    return path


def plot_engagement_level_percent(df, filename="engagement_level_percent.png"):
    """Bar chart of class percentages, useful when checking imbalance."""
    summary = target_distribution_table(df)
    table = summary["table"]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=table,
        x="EngagementLevel",
        y="percent",
        order=LEVEL_ORDER,
        color="#8aa8b8",
        ax=ax,
    )
    ax.set_title("Share of employees in each EngagementLevel")
    ax.set_xlabel("EngagementLevel (project-defined bins)")
    ax.set_ylabel("Percent of employees")
    ax.set_ylim(0, 100)
    path = _save_target_figure(filename)
    _display_figure()
    plt.close(fig)
    return path


def generate_target_notes(df):
    """Markdown notes built from the labeled DataFrame (no hard-coded class counts)."""
    summary = target_distribution_table(df)
    table = summary["table"]
    score = df[SCORE_COLUMN]
    lines = [
        "## Target construction notes",
        "",
        TARGET_DISCLAIMER,
        "",
        "### Formula",
        "",
        "EngagementScore = mean(JobInvolvement, JobSatisfaction, "
        "EnvironmentSatisfaction, RelationshipSatisfaction, WorkLifeBalance)",
        "",
        "Each input is used as stored (typically 1–4). All five items have equal weight.",
        "",
        "### Project-defined thresholds",
        "",
        "- 1.00–2.00 → Low",
        "- 2.01–3.00 → Medium",
        "- 3.01–4.00 → High",
        "",
        "### Observed score range in this table",
        "",
        f"- EngagementScore min = {float(score.min()):.2f}, "
        f"max = {float(score.max()):.2f}, mean = {float(score.mean()):.3f}, "
        f"median = {float(score.median()):.2f}",
        "",
        "### Class counts (computed from this table)",
        "",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"- **{row['EngagementLevel']}**: {int(row['count'])} employees "
            f"({row['percent']:.2f}%)"
        )
    lines.extend(
        [
            "",
            "### Class imbalance check",
            "",
            f"- Majority class: **{summary['majority_level']}** "
            f"({summary['majority_count']} employees, "
            f"{summary['majority_percent']:.2f}%).",
            f"- Smallest class: **{summary['minority_level']}** "
            f"({summary['minority_count']} employees).",
            f"- Majority-to-minority count ratio: "
            f"{summary['imbalance_ratio_majority_to_minority']}.",
        ]
    )
    if summary["classes_are_imbalanced"]:
        lines.append(
            "- **The classes are imbalanced** in this sample: one level is much "
            "more common than the others. Later modeling should account for that "
            "(for example with stratified splitting). No model is trained here."
        )
    else:
        lines.append(
            "- Class sizes are relatively even in this sample. Re-check after any "
            "row filtering. No model is trained here."
        )
    lines.extend(
        [
            "",
            "### What was preserved",
            "",
            "- The five original survey columns are still in the working table.",
            "- The CSV in `data/` was not overwritten.",
            "",
        ]
    )
    return "\n".join(lines)


def save_target_notes(text, filename="target_construction.md"):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(REPORTS_DIR) / filename
    path.write_text(text, encoding="utf-8")
    return path
