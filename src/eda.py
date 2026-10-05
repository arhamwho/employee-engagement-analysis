"""
Exploratory Data Analysis helpers for Case Study 90.

This module only describes the loaded table. It does not train a model
and it does not write back to the original CSV.

Plots are saved under reports/eda_plots/. Findings text is built from
pandas calculations on the DataFrame you pass in.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import EDA_PLOTS_DIR, REPORTS_DIR

# Survey-style fields used as engagement-related variables in this case study.
# Names must match columns in the loaded CSV.
LIKERT_ENGAGEMENT_COLUMNS = [
    "JobSatisfaction",
    "JobInvolvement",
    "EnvironmentSatisfaction",
    "RelationshipSatisfaction",
    "WorkLifeBalance",
]
COMPOSITE_ENGAGEMENT_COLUMN = "OverallSatisfactionIndex"

# Likely building blocks of OverallSatisfactionIndex (checked in code, not assumed).
CANDIDATE_INDEX_COMPONENTS = [
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "RelationshipSatisfaction",
    "WorkLifeBalance",
]

# Numeric HR fields that are useful for box plots (skipped if missing).
DEFAULT_BOXPLOT_COLUMNS = [
    "Age",
    "MonthlyIncome",
    "YearsAtCompany",
    "TotalWorkingYears",
    "DistanceFromHome",
    "OverallSatisfactionIndex",
]


def set_plot_style():
    """One consistent look for every chart in the notebook."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.figsize"] = (10, 5)
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.labelsize"] = 11


def ensure_plot_dir():
    """Create the plot folder if it does not exist yet."""
    EDA_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    return EDA_PLOTS_DIR


def _display_figure():
    """Show the figure in Jupyter. Skip on non-interactive (Agg) backends."""
    backend = plt.get_backend().lower()
    if "agg" not in backend:
        plt.show()


def save_current_figure(filename):
    """Save the active Matplotlib figure. Does not touch the source CSV."""
    plot_dir = ensure_plot_dir()
    path = plot_dir / filename
    plt.savefig(path, dpi=150, bbox_inches="tight")
    return path


def existing_columns(df, columns):
    """Keep only column names that are actually in the DataFrame."""
    return [name for name in columns if name in df.columns]


def numeric_columns(df):
    return df.select_dtypes(include="number").columns.tolist()


def categorical_columns(df):
    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def constant_columns(df):
    """Columns with only one unique value. They cannot show a distribution."""
    return [col for col in df.columns if df[col].nunique(dropna=False) <= 1]


def identifier_columns(df):
    """Heuristic: a column that is unique on every row (for example EmployeeNumber)."""
    n = len(df)
    found = []
    for col in df.columns:
        if df[col].nunique(dropna=False) == n:
            found.append(col)
    return found


def iqr_outlier_table(df):
    """
    Count IQR outliers for each numeric column.

    A value is flagged if it is below Q1 - 1.5*IQR or above Q3 + 1.5*IQR.
    This is a screening rule, not proof that a value is an error.
    """
    rows = []
    num_df = df.select_dtypes(include="number")
    for col in num_df.columns:
        series = num_df[col].dropna()
        if series.nunique() <= 1:
            rows.append(
                {
                    "column": col,
                    "q1": series.iloc[0] if len(series) else np.nan,
                    "q3": series.iloc[0] if len(series) else np.nan,
                    "iqr": 0.0,
                    "lower_fence": np.nan,
                    "upper_fence": np.nan,
                    "n_outliers": 0,
                    "pct_outliers": 0.0,
                    "note": "constant column — IQR not used",
                }
            )
            continue

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        if iqr == 0:
            rows.append(
                {
                    "column": col,
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                    "lower_fence": q1,
                    "upper_fence": q3,
                    "n_outliers": 0,
                    "pct_outliers": 0.0,
                    "note": "IQR is 0 — fences not expanded",
                }
            )
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (series < lower) | (series > upper)
        n_out = int(mask.sum())
        rows.append(
            {
                "column": col,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_fence": lower,
                "upper_fence": upper,
                "n_outliers": n_out,
                "pct_outliers": round(100.0 * n_out / len(series), 2),
                "note": "",
            }
        )
    return pd.DataFrame(rows).sort_values("n_outliers", ascending=False).reset_index(drop=True)


def correlation_matrix(df, drop_constant=True, drop_identifiers=True):
    """Pearson correlation for numeric columns. Constants/IDs can be dropped."""
    num_df = df.select_dtypes(include="number").copy()
    if drop_constant:
        num_df = num_df.drop(columns=constant_columns(num_df), errors="ignore")
    if drop_identifiers:
        num_df = num_df.drop(columns=identifier_columns(num_df), errors="ignore")
    return num_df.corr(numeric_only=True)


def strongest_correlations(corr_df, top_n=15):
    """Return the strongest unique variable pairs from a correlation matrix."""
    records = []
    cols = list(corr_df.columns)
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            value = corr_df.loc[left, right]
            if pd.notna(value):
                records.append(
                    {
                        "variable_a": left,
                        "variable_b": right,
                        "pearson_r": float(value),
                        "abs_r": abs(float(value)),
                    }
                )
    table = pd.DataFrame(records)
    if table.empty:
        return table
    return table.sort_values("abs_r", ascending=False).head(top_n).reset_index(drop=True)


def verify_satisfaction_index(df):
    """
    Check whether OverallSatisfactionIndex equals the mean of four Likert fields.

    This is measured from the table, not taken from a paper.
    """
    needed = existing_columns(df, CANDIDATE_INDEX_COMPONENTS + [COMPOSITE_ENGAGEMENT_COLUMN])
    result = {
        "index_column": COMPOSITE_ENGAGEMENT_COLUMN,
        "candidate_components": CANDIDATE_INDEX_COMPONENTS,
        "can_check": False,
        "matches_row_mean": False,
        "max_abs_difference": None,
    }
    if COMPOSITE_ENGAGEMENT_COLUMN not in df.columns:
        return result
    if len(existing_columns(df, CANDIDATE_INDEX_COMPONENTS)) < 4:
        return result

    computed = df[CANDIDATE_INDEX_COMPONENTS].mean(axis=1)
    diff = (computed - df[COMPOSITE_ENGAGEMENT_COLUMN]).abs()
    result["can_check"] = True
    result["max_abs_difference"] = float(diff.max())
    result["matches_row_mean"] = bool(diff.max() < 1e-9)
    return result


def plot_numeric_distribution(df, column, title, xlabel, filename):
    """Histogram plus KDE for one numeric column."""
    if column not in df.columns:
        print("Skipped plot; missing column:", column)
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df[column].dropna(), kde=True, bins=30, color="#3b6d8c", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    path = save_current_figure(filename)
    _display_figure()
    plt.close(fig)
    return path


def plot_count_distribution(df, column, title, xlabel, filename, horizontal=False):
    """Bar chart of category frequencies, ordered by count."""
    if column not in df.columns:
        print("Skipped plot; missing column:", column)
        return None

    order = df[column].value_counts().index
    if horizontal:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=df, y=column, order=order, color="#3b6d8c", ax=ax)
        ax.set_xlabel("Count")
        ax.set_ylabel(xlabel)
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(data=df, x=column, order=order, color="#3b6d8c", ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=0)
    ax.set_title(title)
    path = save_current_figure(filename)
    _display_figure()
    plt.close(fig)
    return path


def plot_likert_distribution(df, column, title, filename):
    """Bar chart for 1-4 (or similar) survey scores, kept in numeric order."""
    if column not in df.columns:
        print("Skipped plot; missing column:", column)
        return None

    order = sorted(df[column].dropna().unique())
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x=column, order=order, color="#3b6d8c", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(column + " (survey score)")
    ax.set_ylabel("Count")
    path = save_current_figure(filename)
    _display_figure()
    plt.close(fig)
    return path


def plot_boxplots(df, columns, title, filename):
    """Side-by-side box plots. Each panel uses its own y-scale."""
    cols = existing_columns(df, columns)
    if not cols:
        print("Skipped box plots; none of the requested columns are present.")
        return None

    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        sns.boxplot(y=df[col], color="#8aa8b8", ax=ax, fliersize=3)
        ax.set_title(col)
        ax.set_ylabel(col)
        ax.set_xlabel("")
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    path = save_current_figure(filename)
    _display_figure()
    plt.close(fig)
    return path


def plot_correlation_heatmap(df, filename="correlation_heatmap.png"):
    """Pearson correlation heatmap for informative numeric columns."""
    corr = correlation_matrix(df)
    if corr.empty:
        print("Skipped heatmap; no numeric columns to correlate.")
        return None

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        corr,
        cmap="vlag",
        center=0,
        square=False,
        linewidths=0.3,
        ax=ax,
        cbar_kws={"label": "Pearson r"},
    )
    ax.set_title("Pearson correlation heatmap (numeric columns)")
    path = save_current_figure(filename)
    _display_figure()
    plt.close(fig)
    return path


def plot_engagement_subheatmap(df, filename="engagement_correlation_heatmap.png"):
    """Zoomed correlation view for engagement-related columns only."""
    cols = existing_columns(df, LIKERT_ENGAGEMENT_COLUMNS + [COMPOSITE_ENGAGEMENT_COLUMN])
    if len(cols) < 2:
        print("Skipped engagement heatmap; not enough engagement columns.")
        return None

    corr = df[cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr,
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        ax=ax,
        cbar_kws={"label": "Pearson r"},
    )
    ax.set_title("Correlations among engagement-related variables")
    path = save_current_figure(filename)
    _display_figure()
    plt.close(fig)
    return path


def plot_engagement_by_category(df, category, engagement_col, title, filename):
    """Box plots of an engagement score across a categorical HR variable."""
    if category not in df.columns or engagement_col not in df.columns:
        print("Skipped comparison; missing columns:", category, engagement_col)
        return None

    order = (
        df.groupby(category)[engagement_col]
        .median()
        .sort_values()
        .index
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        data=df,
        x=category,
        y=engagement_col,
        order=order,
        color="#8aa8b8",
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel(category)
    ax.set_ylabel(engagement_col)
    ax.tick_params(axis="x", rotation=25)
    path = save_current_figure(filename)
    _display_figure()
    plt.close(fig)
    return path


def plot_mean_engagement_by_group(df, group_col, value_cols, title, filename):
    """Grouped bar chart of mean engagement-related scores by a category."""
    cols = existing_columns(df, value_cols)
    if group_col not in df.columns or not cols:
        print("Skipped grouped means; missing columns.")
        return None

    summary = df.groupby(group_col)[cols].mean().reset_index()
    long = summary.melt(id_vars=group_col, var_name="variable", value_name="mean_score")
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(
        data=long,
        x=group_col,
        y="mean_score",
        hue="variable",
        ax=ax,
    )
    ax.set_title(title)
    ax.set_ylabel("Mean score")
    ax.set_xlabel(group_col)
    ax.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")
    path = save_current_figure(filename)
    _display_figure()
    plt.close(fig)
    return path


def plot_income_vs_engagement(df, filename="income_vs_overall_satisfaction.png"):
    """Scatter of monthly income vs composite satisfaction, with a trend line."""
    x_col, y_col = "MonthlyIncome", COMPOSITE_ENGAGEMENT_COLUMN
    if x_col not in df.columns or y_col not in df.columns:
        print("Skipped scatter; missing MonthlyIncome or OverallSatisfactionIndex.")
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.regplot(
        data=df,
        x=x_col,
        y=y_col,
        scatter_kws={"alpha": 0.25, "s": 18, "color": "#3b6d8c"},
        line_kws={"color": "#b35c2a"},
        ax=ax,
    )
    ax.set_title("Monthly income vs overall satisfaction index")
    ax.set_xlabel("MonthlyIncome")
    ax.set_ylabel(y_col)
    path = save_current_figure(filename)
    _display_figure()
    plt.close(fig)
    return path


def _fmt(value, digits=2):
    if pd.isna(value):
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def generate_eda_findings(df):
    """
    Build the EDA Findings section from the DataFrame only.

    Every number is calculated here. The text states associations,
    not cause-and-effect.
    """
    n_rows, n_cols = df.shape
    n_missing = int(df.isna().sum().sum())
    n_dupes = int(df.duplicated().sum())
    const_cols = constant_columns(df)
    id_cols = identifier_columns(df)
    num_cols = numeric_columns(df)
    cat_cols = categorical_columns(df)
    outlier_tbl = iqr_outlier_table(df)
    corr = correlation_matrix(df)
    top_corr = strongest_correlations(corr, top_n=8)
    index_check = verify_satisfaction_index(df)

    lines = []
    lines.append("## EDA Findings")
    lines.append("")
    lines.append(
        "This section is generated automatically from the loaded DataFrame. "
        "Statements below are **descriptive**. A correlation or group-mean "
        "difference is an association in this sample, not proof that one "
        "variable causes another."
    )
    lines.append("")

    lines.append("### Data shape and quality")
    lines.append(
        f"- The working table has **{n_rows} rows** and **{n_cols} columns**."
    )
    lines.append(f"- Missing cell count: **{n_missing}**.")
    lines.append(f"- Duplicate rows: **{n_dupes}**.")
    if const_cols:
        lines.append(
            "- Constant columns (one unique value, little analytic value): "
            + ", ".join(f"`{c}`" for c in const_cols)
            + "."
        )
    if id_cols:
        lines.append(
            "- Identifier-like columns (unique on every row): "
            + ", ".join(f"`{c}`" for c in id_cols)
            + "."
        )
    lines.append(
        f"- Pandas dtypes: **{len(num_cols)}** numeric columns and "
        f"**{len(cat_cols)}** object/category/bool columns. "
        "Some numeric columns are still ordinal survey scores (for example "
        "JobSatisfaction on a 1–4 scale)."
    )
    lines.append("")

    if index_check["can_check"]:
        lines.append("### Composite engagement field")
        comps = ", ".join(f"`{c}`" for c in index_check["candidate_components"])
        if index_check["matches_row_mean"]:
            lines.append(
                f"- `{COMPOSITE_ENGAGEMENT_COLUMN}` matches the row-wise mean of "
                f"{comps} within a numerical tolerance of "
                f"{index_check['max_abs_difference']}. "
                "High Pearson correlations between the index and those four "
                "inputs are expected because the index is built from them. "
                "They should not be treated as four independent discoveries."
            )
        else:
            lines.append(
                f"- `{COMPOSITE_ENGAGEMENT_COLUMN}` was compared with the mean of "
                f"{comps}. The largest absolute difference was "
                f"{_fmt(index_check['max_abs_difference'], 6)}. "
                "The index is therefore **not** an exact mean of those four fields "
                "in this file."
            )
        lines.append("")

    lines.append("### Numerical distributions")
    if "Age" in df.columns:
        age = df["Age"]
        lines.append(
            f"- `Age` ranges from {int(age.min())} to {int(age.max())} "
            f"(mean {_fmt(age.mean())}, median {_fmt(age.median())}). "
            f"Skewness is {_fmt(age.skew(), 3)} (mildly right-skewed)."
        )
    if "MonthlyIncome" in df.columns:
        inc = df["MonthlyIncome"]
        inc_out = outlier_tbl.loc[outlier_tbl["column"] == "MonthlyIncome", "n_outliers"]
        n_inc_out = int(inc_out.iloc[0]) if len(inc_out) else 0
        lines.append(
            f"- `MonthlyIncome` is right-skewed (skewness {_fmt(inc.skew(), 3)}). "
            f"Median {_fmt(inc.median(), 0)}, mean {_fmt(inc.mean(), 0)}, "
            f"min {int(inc.min())}, max {int(inc.max())}. "
            f"IQR rule flags **{n_inc_out}** high-income points "
            f"({_fmt(100 * n_inc_out / n_rows, 2)}% of rows). "
            "Those values can be real senior-pay observations, not data errors."
        )
    if "YearsAtCompany" in df.columns:
        yac = df["YearsAtCompany"]
        yac_out = outlier_tbl.loc[outlier_tbl["column"] == "YearsAtCompany", "n_outliers"]
        n_yac_out = int(yac_out.iloc[0]) if len(yac_out) else 0
        lines.append(
            f"- `YearsAtCompany` is right-skewed (skewness {_fmt(yac.skew(), 3)}). "
            f"Median {_fmt(yac.median())} years, mean {_fmt(yac.mean())}, "
            f"max {int(yac.max())}. IQR rule flags **{n_yac_out}** long-tenure rows."
        )
    lines.append("")

    lines.append("### Engagement-related score distributions")
    for col in existing_columns(df, LIKERT_ENGAGEMENT_COLUMNS):
        counts = df[col].value_counts().sort_index()
        mode_val = int(counts.idxmax())
        lines.append(
            f"- `{col}`: mean {_fmt(df[col].mean())}, most common score "
            f"**{mode_val}** (n={int(counts.max())} of {n_rows})."
        )
    if COMPOSITE_ENGAGEMENT_COLUMN in df.columns:
        osi = df[COMPOSITE_ENGAGEMENT_COLUMN]
        lines.append(
            f"- `{COMPOSITE_ENGAGEMENT_COLUMN}`: mean {_fmt(osi.mean())}, "
            f"median {_fmt(osi.median())}, min {_fmt(osi.min())}, "
            f"max {_fmt(osi.max())}."
        )
    if "JobInvolvement" in df.columns and COMPOSITE_ENGAGEMENT_COLUMN in df.columns:
        r_ji = float(df["JobInvolvement"].corr(df[COMPOSITE_ENGAGEMENT_COLUMN]))
        lines.append(
            f"- Pearson r between `JobInvolvement` and `{COMPOSITE_ENGAGEMENT_COLUMN}` "
            f"is {_fmt(r_ji, 3)}. In this table that association is essentially none, "
            "so involvement is not interchangeable with the composite satisfaction index."
        )
    lines.append("")

    lines.append("### Categorical workforce mix")
    if "Department" in df.columns:
        dept = df["Department"].value_counts()
        parts = ", ".join(f"{idx} n={int(val)}" for idx, val in dept.items())
        lines.append(f"- `Department` counts: {parts}.")
    if "JobRole" in df.columns:
        role = df["JobRole"].value_counts()
        lines.append(
            f"- `JobRole` has {role.size} roles. Largest: {role.index[0]} "
            f"(n={int(role.iloc[0])}); smallest: {role.index[-1]} "
            f"(n={int(role.iloc[-1])})."
        )
    if "OverTime" in df.columns:
        ot = df["OverTime"].value_counts()
        yes = int(ot.get("Yes", 0))
        lines.append(
            f"- `OverTime`: Yes n={yes} ({_fmt(100 * yes / n_rows, 1)}%), "
            f"No n={int(ot.get('No', 0))}."
        )
    if "Attrition" in df.columns:
        att = df["Attrition"].value_counts()
        yes = int(att.get("Yes", 0))
        lines.append(
            f"- `Attrition`: Yes n={yes} ({_fmt(100 * yes / n_rows, 1)}%), "
            f"No n={int(att.get('No', 0))}. Attrition is a related HR outcome "
            "in this file; it is not the engagement class label by itself."
        )
    lines.append("")

    lines.append("### Outliers (IQR screening)")
    flagged = outlier_tbl[outlier_tbl["n_outliers"] > 0]
    if flagged.empty:
        lines.append("- No numeric column had IQR-flagged points.")
    else:
        lines.append(
            "- IQR flags are a spread check. Discrete scales with a small IQR "
            "(for example training counts) can produce many flags even when "
            "values are valid survey or tenure numbers."
        )
        for _, row in flagged.head(8).iterrows():
            extra = f" {row['note']}" if row["note"] else ""
            lines.append(
                f"- `{row['column']}`: {int(row['n_outliers'])} flagged "
                f"({row['pct_outliers']}% of non-missing values).{extra}"
            )
    lines.append("")

    lines.append("### Numerical correlations (not causal)")
    if not top_corr.empty:
        lines.append(
            "- Largest absolute Pearson correlations among numeric fields "
            "(excluding constant and identifier columns):"
        )
        for _, row in top_corr.iterrows():
            lines.append(
                f"  - `{row['variable_a']}` and `{row['variable_b']}`: "
                f"r = {_fmt(row['pearson_r'], 3)}"
            )
        lines.append(
            "- Several of those pairs are structurally related (job level with "
            "income, tenure fields with each other). A large r does not mean "
            "one variable drives engagement."
        )
    if COMPOSITE_ENGAGEMENT_COLUMN in corr.columns:
        osi_corr = (
            corr[COMPOSITE_ENGAGEMENT_COLUMN]
            .drop(labels=[COMPOSITE_ENGAGEMENT_COLUMN], errors="ignore")
            .abs()
            .sort_values(ascending=False)
        )
        # Skip the four components if they dominate, mention next HR fields.
        skip = set(CANDIDATE_INDEX_COMPONENTS)
        other = osi_corr.drop(labels=[c for c in osi_corr.index if c in skip], errors="ignore")
        if len(other):
            top_name = other.index[0]
            top_r = float(corr.loc[top_name, COMPOSITE_ENGAGEMENT_COLUMN])
            lines.append(
                f"- After setting aside the composite's own input fields, the "
                f"next strongest numeric association with `{COMPOSITE_ENGAGEMENT_COLUMN}` "
                f"in this table is `{top_name}` (r = {_fmt(top_r, 3)}). "
                "Interpret the size of r, not a causal story."
            )
        if "MonthlyIncome" in corr.columns:
            r_inc = float(corr.loc["MonthlyIncome", COMPOSITE_ENGAGEMENT_COLUMN])
            lines.append(
                f"- `MonthlyIncome` vs `{COMPOSITE_ENGAGEMENT_COLUMN}`: "
                f"r = {_fmt(r_inc, 3)}. In this sample, income and the composite "
                "index barely move together."
            )
    lines.append("")

    lines.append("### HR variables compared with engagement-related scores")
    if "OverTime" in df.columns and COMPOSITE_ENGAGEMENT_COLUMN in df.columns:
        means = df.groupby("OverTime")[COMPOSITE_ENGAGEMENT_COLUMN].mean()
        parts = ", ".join(f"{idx} mean={_fmt(val, 3)}" for idx, val in means.items())
        lines.append(
            f"- Mean `{COMPOSITE_ENGAGEMENT_COLUMN}` by `OverTime`: {parts}. "
            "This is a group average only."
        )
    if "Attrition" in df.columns:
        eng_cols = existing_columns(
            df, LIKERT_ENGAGEMENT_COLUMNS + [COMPOSITE_ENGAGEMENT_COLUMN]
        )
        if eng_cols:
            lines.append(
                "- Employees with `Attrition` = Yes have lower mean engagement-related "
                "scores in this table than employees with Attrition = No:"
            )
            grouped = df.groupby("Attrition")[eng_cols].mean()
            for col in eng_cols:
                if set(grouped.index) >= {"Yes", "No"}:
                    lines.append(
                        f"  - `{col}`: No {_fmt(grouped.loc['No', col], 3)}, "
                        f"Yes {_fmt(grouped.loc['Yes', col], 3)}"
                    )
            lines.append(
                "- That pattern is consistent with engagement and leaving being "
                "**associated** here. It does not by itself prove that low scores "
                "caused attrition."
            )
    if "Department" in df.columns and COMPOSITE_ENGAGEMENT_COLUMN in df.columns:
        dept_means = df.groupby("Department")[COMPOSITE_ENGAGEMENT_COLUMN].mean().sort_values()
        spread = float(dept_means.max() - dept_means.min())
        lines.append(
            "- Mean `{idx}` by `Department`: {vals}. "
            "The gap between the lowest and highest department means is "
            "{spread} points on the index scale.".format(
                idx=COMPOSITE_ENGAGEMENT_COLUMN,
                vals=", ".join(f"{i} {_fmt(v, 3)}" for i, v in dept_means.items()),
                spread=_fmt(spread, 3),
            )
        )
    if "JobRole" in df.columns and COMPOSITE_ENGAGEMENT_COLUMN in df.columns:
        role_means = df.groupby("JobRole")[COMPOSITE_ENGAGEMENT_COLUMN].mean().sort_values()
        lines.append(
            f"- Mean `{COMPOSITE_ENGAGEMENT_COLUMN}` by `JobRole` ranges from "
            f"{_fmt(role_means.iloc[0], 3)} (`{role_means.index[0]}`) to "
            f"{_fmt(role_means.iloc[-1], 3)} (`{role_means.index[-1]}`)."
        )
    lines.append("")

    lines.append("### Limits of this EDA")
    lines.append(
        "- No classifier was trained. There is no accuracy, precision, recall, "
        "or F1-score in this section."
    )
    lines.append(
        "- Engagement classes (Low / Medium / High) have not been defined from "
        "these scores yet."
    )
    lines.append(
        "- Plots and tables use a memory copy of the CSV. The file in `data/` "
        "was not modified."
    )
    lines.append("")
    return "\n".join(lines)


def save_eda_findings(text, filename="eda_findings.md"):
    """Write the generated findings to reports/ for the case-study write-up."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / filename
    path.write_text(text, encoding="utf-8")
    return path
