"""
Leakage-safe hyperparameter optimization for EngagementLevel.

- Does not use the five target-defining survey items as predictors.
- Does not change labels.
- Does not duplicate rows or train on the test set.
- Hyperparameters are chosen with StratifiedKFold on the training fold only.
- Each CV-selected candidate is scored on the held-out test set once.

Does not overwrite models/final_model.joblib. Candidates are saved under
models/optimization/. A recommendation is written after the test evaluation.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
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
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import MODELS_DIR, RANDOM_STATE, REPORTS_DIR, TEST_SIZE  # noqa: E402
from src.data_loader import inspect_dataset, load_raw_dataset  # noqa: E402
from src.feature_engineering import (  # noqa: E402
    add_derived_features,
    build_dynamic_preprocessor,
    feature_engineering_table,
)
from src.modeling import make_train_test_split, prepare_xy  # noqa: E402
from src.preprocess import feature_exclusion_columns  # noqa: E402
from src.target import (  # noqa: E402
    ENGAGEMENT_INPUT_COLUMNS,
    LEVEL_COLUMN,
    LEVEL_ORDER,
    SCORE_COLUMN,
    add_engagement_target,
)

OUT_DIR = REPORTS_DIR / "optimization"
PLOTS_DIR = OUT_DIR / "plots"
CANDIDATE_DIR = MODELS_DIR / "optimization"
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
# Use sklearn's built-in multiclass names. A custom make_scorer(f1_score)
# in sklearn 1.6 defaults to pos_label=1, which breaks 3-class labels.
SCORING = {"accuracy": "accuracy", "f1_macro": "f1_macro"}


def _safe_name(text):
    return (
        text.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("—", "-")
    )


def test_metrics(y_true, y_pred):
    y_true = pd.Series(y_true).astype(str)
    y_pred = pd.Series(y_pred).astype(str)
    return {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "test_precision_macro": float(
            precision_score(y_true, y_pred, labels=LEVEL_ORDER, average="macro", zero_division=0)
        ),
        "test_recall_macro": float(
            recall_score(y_true, y_pred, labels=LEVEL_ORDER, average="macro", zero_division=0)
        ),
        "test_f1_macro": float(
            f1_score(y_true, y_pred, labels=LEVEL_ORDER, average="macro", zero_division=0)
        ),
        "test_f1_weighted": float(
            f1_score(y_true, y_pred, labels=LEVEL_ORDER, average="weighted", zero_division=0)
        ),
    }


def cv_both_metrics(estimator, X, y):
    cv_res = cross_validate(
        estimator,
        X,
        y,
        cv=CV,
        scoring=SCORING,
        n_jobs=-1,
        return_train_score=False,
    )
    return {
        "cv_accuracy_mean": float(cv_res["test_accuracy"].mean()),
        "cv_accuracy_std": float(cv_res["test_accuracy"].std()),
        "cv_f1_macro_mean": float(cv_res["test_f1_macro"].mean()),
        "cv_f1_macro_std": float(cv_res["test_f1_macro"].std()),
    }


def drop_model_prefix(grid):
    """HPO runs on the classifier; grids in this script use model__ from Pipeline style."""
    if isinstance(grid, list):
        return [drop_model_prefix(g) for g in grid]
    return {k.replace("model__", ""): v for k, v in grid.items()}


def wrap_fitted(preprocessor, classifier):
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", classifier),
        ]
    )


def wrap_unfitted(classifier):
    return Pipeline(
        steps=[
            ("preprocess", build_dynamic_preprocessor()),
            ("model", classifier),
        ]
    )


def search_grid(name, estimator, param_grid, scoring_name, X, y, n_jobs=-1):
    search = GridSearchCV(
        estimator,
        param_grid=drop_model_prefix(param_grid),
        scoring="f1_macro" if scoring_name == "f1_macro" else "accuracy",
        cv=CV,
        n_jobs=n_jobs,
        refit=True,
        error_score="raise",
    )
    t0 = time.time()
    search.fit(X, y)
    elapsed = time.time() - t0
    n_cells = len(search.cv_results_["params"])
    print(
        f"  GridSearchCV {name} [{scoring_name}] "
        f"best={search.best_score_:.4f}  {elapsed:.1f}s  "
        f"fits={n_cells * 5}"
    )
    return search, elapsed


def search_random(name, estimator, param_dist, scoring_name, X, y, n_iter, n_jobs=-1):
    search = RandomizedSearchCV(
        estimator,
        param_distributions=drop_model_prefix(param_dist),
        n_iter=n_iter,
        scoring="f1_macro" if scoring_name == "f1_macro" else "accuracy",
        cv=CV,
        n_jobs=n_jobs,
        refit=True,
        random_state=RANDOM_STATE,
        error_score="raise",
    )
    t0 = time.time()
    search.fit(X, y)
    elapsed = time.time() - t0
    print(
        f"  RandomizedSearchCV {name} [{scoring_name}] "
        f"best={search.best_score_:.4f}  {elapsed:.1f}s  "
        f"n_iter={n_iter}"
    )
    return search, elapsed


def pack_candidate(family, scoring_name, search_kind, search, elapsed, preprocessor, X_train_df, y_train):
    clf = search.best_estimator_
    full = wrap_fitted(preprocessor, clf)
    honest = wrap_unfitted(clf)
    cv_extra = cv_both_metrics(honest, X_train_df, y_train)
    params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    return {
        "family": family,
        "optimized_for": scoring_name,
        "search": search_kind,
        "model": f"{family} (opt {scoring_name})",
        "best_params": params,
        "search_best_score": float(search.best_score_),
        "search_seconds": float(elapsed),
        "estimator": full,
        **cv_extra,
    }


def plot_confusion(y_true, y_pred, title, path):
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
    ax.set_title(title)
    ax.set_xlabel("Predicted EngagementLevel")
    ax.set_ylabel("True EngagementLevel")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return cm


def gini_or_linear_importance(pipeline, top_n=15):
    model = pipeline.named_steps["model"]
    prep = pipeline.named_steps["preprocess"]
    try:
        names = np.asarray(prep.get_feature_names_out())
    except Exception:
        return None, "Could not read transformed feature names."

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        kind = "gini_importance"
        note = (
            "Gini / impurity decrease from the fitted tree ensemble. "
            "Feature importance indicates contribution to model predictions "
            "and does not establish causation."
        )
    elif hasattr(model, "coef_"):
        coef = np.abs(np.asarray(model.coef_, dtype=float))
        values = coef.mean(axis=0) if coef.ndim == 2 else coef
        kind = "mean_abs_coefficient"
        note = (
            "Mean absolute coefficient across classes. "
            "Feature importance indicates contribution to model predictions "
            "and does not establish causation."
        )
    else:
        return None, (
            "This estimator has neither feature_importances_ nor coef_. "
            "Importance was not computed."
        )

    table = pd.DataFrame({"feature": names, kind: values})
    total = float(table[kind].sum())
    table["share_of_total"] = table[kind] / total if total else 0.0
    table = table.sort_values(kind, ascending=False).reset_index(drop=True)
    table["rank"] = np.arange(1, len(table) + 1)
    table.attrs["kind"] = kind
    table.attrs["note"] = note
    return table.head(top_n), note


def write_pipeline_audit(raw, labeled, X, y, X_train, X_test, y_train, y_test, dropped):
    info = inspect_dataset(raw)
    leaked_in_x = [
        c
        for c in ENGAGEMENT_INPUT_COLUMNS
        + [SCORE_COLUMN, LEVEL_COLUMN, "OverallSatisfactionIndex"]
        if c in X.columns
    ]
    lines = [
        "# Pipeline audit (leakage-safe optimization)",
        "",
        "Target definition was **not** changed. The IBM CSV was **not** modified.",
        "",
        "## Dataset",
        "",
        f"- Rows: {info['n_rows']}",
        f"- Columns: {info['n_columns']}",
        f"- Missing values (cells): {info['missing_values_total']}",
        f"- Duplicate rows: {info['duplicate_rows']}",
        "",
        "Missing-value handling: none required (zero missing cells). "
        "Duplicate handling: none required (zero duplicate rows). "
        "Rows were not copied or dropped.",
        "",
        "## Target",
        "",
        "EngagementLevel is the student-defined 3-class label from the mean of "
        "JobInvolvement, JobSatisfaction, EnvironmentSatisfaction, "
        "RelationshipSatisfaction, WorkLifeBalance.",
        "",
        f"- Train counts: {y_train.value_counts().reindex(LEVEL_ORDER).fillna(0).astype(int).to_dict()}",
        f"- Test counts: {y_test.value_counts().reindex(LEVEL_ORDER).fillna(0).astype(int).to_dict()}",
        "",
        "## Split",
        "",
        f"- `test_size={TEST_SIZE}`, `random_state={RANDOM_STATE}`, `stratify=y`",
        f"- Train rows: {len(X_train)}",
        f"- Test rows: {len(X_test)}",
        f"- Index overlap between train and test: {len(set(X_train.index) & set(X_test.index))}",
        "",
        "The test fold is held out until every hyperparameter search has finished. "
        "It is not used inside GridSearchCV or RandomizedSearchCV.",
        "",
        "## Preprocessing (fit on train only)",
        "",
        "- Numeric: `StandardScaler`",
        "- Categorical: `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`",
        "- Encoding and scaling are inside the sklearn Pipeline, fitted on training rows.",
        "",
        "## Leakage control",
        "",
        "Columns excluded from X: "
        + ", ".join(f"`{c}`" for c in dropped),
        "",
        f"- Forbidden columns still in X: {leaked_in_x if leaked_in_x else 'none'}",
        f"- Exclusion helper agrees: {dropped == [c for c in feature_exclusion_columns() if c in labeled.columns]}",
        "",
        "Cross-validation uses `StratifiedKFold(n_splits=5, shuffle=True, "
        f"random_state={RANDOM_STATE})` on the **training** table only.",
        "",
    ]
    path = OUT_DIR / "pipeline_audit.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data…")
    raw = load_raw_dataset()
    labeled = add_engagement_target(raw)
    X, y, dropped = prepare_xy(labeled)

    leaked = [c for c in ENGAGEMENT_INPUT_COLUMNS + [SCORE_COLUMN, LEVEL_COLUMN] if c in X.columns]
    if leaked:
        raise RuntimeError("Leakage columns present in X: " + ", ".join(leaked))

    X_eng = add_derived_features(X)
    created = X_eng.attrs.get("derived_created", [])
    skipped = X_eng.attrs.get("derived_skipped", [])
    fe_table = feature_engineering_table(created, skipped)
    fe_table.to_csv(OUT_DIR / "derived_features.csv", index=False)
    fe_lines = [
        "# Derived features (leakage-safe)",
        "",
        "Built only from columns that remain after dropping the five "
        "target-defining items, EngagementScore, EngagementLevel, "
        "OverallSatisfactionIndex, IDs, and constants.",
        "",
        "Denominators add 1 so a zero tenure/experience year does not divide by zero.",
        "",
        "| feature | source_columns | created | description |",
        "| --- | --- | --- | --- |",
    ]
    for _, row in fe_table.iterrows():
        fe_lines.append(
            f"| {row['feature']} | {row['source_columns']} | {row['created']} | {row['description']} |"
        )
    fe_lines.extend(
        [
            "",
            "These ratios are row-wise. They do not use other employees' values "
            "and do not use the test fold.",
            "",
        ]
    )
    (OUT_DIR / "feature_engineering.md").write_text("\n".join(fe_lines), encoding="utf-8")
    print("Derived features created:", created)
    print("Derived features skipped:", skipped)

    X_train, X_test, y_train, y_test = make_train_test_split(X_eng, y)
    write_pipeline_audit(raw, labeled, X_eng, y, X_train, X_test, y_train, y_test, dropped)
    print(f"Split: train={len(X_train)} test={len(X_test)}")

    preprocessor = build_dynamic_preprocessor()
    preprocessor.fit(X_train)
    X_train_p = preprocessor.transform(X_train)
    print("Preprocessor fitted on train only. Test transform happens after HPO.")

    candidates = []

    print("Majority baseline (train majority, evaluated on test once)…")
    dummy = DummyClassifier(strategy="most_frequent")
    dummy_pipe = wrap_unfitted(dummy)
    dummy_cv = cv_both_metrics(dummy_pipe, X_train, y_train)
    dummy_pipe.fit(X_train, y_train)
    dummy_pred = dummy_pipe.predict(X_test)
    dummy_test = test_metrics(y_test, dummy_pred)
    majority_row = {
        "family": "Majority baseline",
        "optimized_for": "none",
        "search": "most_frequent",
        "model": "Majority baseline (reference)",
        "best_params": {"strategy": "most_frequent"},
        "search_best_score": dummy_cv["cv_accuracy_mean"],
        "search_seconds": 0.0,
        "estimator": dummy_pipe,
        **dummy_cv,
        **dummy_test,
        "n_test": int(len(y_test)),
        "y_pred": dummy_pred,
    }
    candidates.append(majority_row)
    print(
        f"  majority test acc={dummy_test['test_accuracy']:.4f} "
        f"macro F1={dummy_test['test_f1_macro']:.4f}"
    )

    print("Logistic Regression grids…")
    lr = LogisticRegression(max_iter=4000, random_state=RANDOM_STATE)
    lr_grid = {
        "model__C": [0.25, 1.0, 4.0],
        "model__class_weight": [None, "balanced"],
        "model__solver": ["lbfgs"],
    }
    for metric in ("accuracy", "f1_macro"):
        search, elapsed = search_grid("Logistic Regression", lr, lr_grid, metric, X_train_p, y_train)
        candidates.append(
            pack_candidate(
                "Logistic Regression", metric, "GridSearchCV", search, elapsed, preprocessor, X_train, y_train
            )
        )

    print("Decision Tree grids…")
    dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
    dt_grid = {
        "model__class_weight": [None, "balanced"],
        "model__max_depth": [None, 4, 8, 12, 20],
        "model__min_samples_split": [2, 10, 20],
        "model__min_samples_leaf": [1, 4, 8],
        "model__max_features": [None, "sqrt", "log2"],
        "model__criterion": ["gini", "entropy"],
        "model__ccp_alpha": [0.0, 0.001, 0.01],
    }
    for metric in ("accuracy", "f1_macro"):
        search, elapsed = search_grid("Decision Tree", dt, dt_grid, metric, X_train_p, y_train)
        candidates.append(
            pack_candidate(
                "Decision Tree", metric, "GridSearchCV", search, elapsed, preprocessor, X_train, y_train
            )
        )

    print("Random Forest randomized searches (same parameter space as specified)…")
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)
    rf_dist = {
        "model__n_estimators": [200, 400, 800],
        "model__max_depth": [None, 5, 10, 15, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", None],
        "model__class_weight": [None, "balanced", "balanced_subsample"],
    }
    # Full factorial is 3*5*3*3*3*3 = 1215 cells × 5 folds × 2 scorings.
    # RandomizedSearchCV samples that same space (n_iter=36 per scoring).
    for metric in ("accuracy", "f1_macro"):
        search, elapsed = search_random(
            "Random Forest", rf, rf_dist, metric, X_train_p, y_train, n_iter=36, n_jobs=-1
        )
        candidates.append(
            pack_candidate(
                "Random Forest", metric, "RandomizedSearchCV", search, elapsed, preprocessor, X_train, y_train
            )
        )

    print("Extra Trees randomized searches…")
    et = ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=1)
    et_dist = {
        "model__n_estimators": [200, 400, 800],
        "model__max_depth": [None, 5, 10, 15, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", None],
        "model__class_weight": [None, "balanced", "balanced_subsample"],
    }
    for metric in ("accuracy", "f1_macro"):
        search, elapsed = search_random(
            "Extra Trees", et, et_dist, metric, X_train_p, y_train, n_iter=36, n_jobs=-1
        )
        candidates.append(
            pack_candidate(
                "Extra Trees", metric, "RandomizedSearchCV", search, elapsed, preprocessor, X_train, y_train
            )
        )

    print("Gradient Boosting grids…")
    gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
    gb_grid = {
        "model__n_estimators": [80, 160],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [2, 3],
        "model__min_samples_leaf": [1, 4],
    }
    for metric in ("accuracy", "f1_macro"):
        search, elapsed = search_grid("Gradient Boosting", gb, gb_grid, metric, X_train_p, y_train)
        candidates.append(
            pack_candidate(
                "Gradient Boosting", metric, "GridSearchCV", search, elapsed, preprocessor, X_train, y_train
            )
        )

    print("HistGradientBoosting grids…")
    hgb = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    hgb_grid = {
        "model__max_iter": [100, 200],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [3, 6],
        "model__min_samples_leaf": [20, 50],
        "model__class_weight": [None, "balanced"],
    }
    for metric in ("accuracy", "f1_macro"):
        search, elapsed = search_grid(
            "HistGradientBoosting", hgb, hgb_grid, metric, X_train_p, y_train
        )
        candidates.append(
            pack_candidate(
                "HistGradientBoosting", metric, "GridSearchCV", search, elapsed, preprocessor, X_train, y_train
            )
        )

    print("SVM grids…")
    svm = SVC(random_state=RANDOM_STATE)
    svm_grid = [
        {
            "model__kernel": ["linear"],
            "model__C": [0.1, 1, 10],
            "model__class_weight": [None, "balanced"],
        },
        {
            "model__kernel": ["rbf"],
            "model__C": [0.1, 1, 10],
            "model__gamma": ["scale", 0.01],
            "model__class_weight": [None, "balanced"],
        },
    ]
    for metric in ("accuracy", "f1_macro"):
        search, elapsed = search_grid("SVM", svm, svm_grid, metric, X_train_p, y_train)
        candidates.append(
            pack_candidate("SVM", metric, "GridSearchCV", search, elapsed, preprocessor, X_train, y_train)
        )

    print("Evaluating CV-selected candidates ONCE on the held-out test set…")
    rows = []
    reports = {}
    for cand in candidates:
        if cand["model"] == "Majority baseline (reference)":
            y_pred = cand["y_pred"]
            metrics = {k: cand[k] for k in dummy_test}
        else:
            y_pred = cand["estimator"].predict(X_test)
            metrics = test_metrics(y_test, y_pred)
            cand.update(metrics)
            cand["y_pred"] = y_pred
        report = classification_report(
            y_test, y_pred, labels=LEVEL_ORDER, zero_division=0, digits=4
        )
        reports[cand["model"]] = report
        cm_path = PLOTS_DIR / f"cm_{_safe_name(cand['model'])}.png"
        plot_confusion(y_test, y_pred, f"Test confusion: {cand['model']}", cm_path)
        joblib.dump(cand["estimator"], CANDIDATE_DIR / f"{_safe_name(cand['model'])}.joblib")
        rows.append(
            {
                "model": cand["model"],
                "family": cand["family"],
                "optimized_for": cand["optimized_for"],
                "search": cand["search"],
                "cv_accuracy_mean": cand["cv_accuracy_mean"],
                "cv_accuracy_std": cand["cv_accuracy_std"],
                "cv_f1_macro_mean": cand["cv_f1_macro_mean"],
                "cv_f1_macro_std": cand["cv_f1_macro_std"],
                "test_accuracy": metrics["test_accuracy"],
                "test_f1_macro": metrics["test_f1_macro"],
                "test_balanced_accuracy": metrics["test_balanced_accuracy"],
                "test_precision_macro": metrics["test_precision_macro"],
                "test_recall_macro": metrics["test_recall_macro"],
                "test_f1_weighted": metrics["test_f1_weighted"],
                "best_params": json.dumps(cand["best_params"], default=str),
                "search_seconds": cand["search_seconds"],
            }
        )
        print(
            f"  {cand['model']}: test acc={metrics['test_accuracy']:.4f} "
            f"macro F1={metrics['test_f1_macro']:.4f} "
            f"CV acc={cand['cv_accuracy_mean']:.4f} CV F1={cand['cv_f1_macro_mean']:.4f}"
        )

    results = pd.DataFrame(rows)
    results_path = REPORTS_DIR / "optimization_results.csv"
    results.to_csv(results_path, index=False)
    results.to_csv(OUT_DIR / "optimization_results.csv", index=False)

    by_acc = results.sort_values(
        by=["test_accuracy", "cv_accuracy_mean", "test_f1_macro"], ascending=False
    )
    by_f1 = results.sort_values(
        by=["test_f1_macro", "cv_f1_macro_mean", "test_accuracy"], ascending=False
    )
    by_acc.to_csv(OUT_DIR / "comparison_by_test_accuracy.csv", index=False)
    by_f1.to_csv(OUT_DIR / "comparison_by_test_macro_f1.csv", index=False)

    reports_dir = OUT_DIR / "classification_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    for name, text in reports.items():
        (reports_dir / f"{_safe_name(name)}.txt").write_text(text, encoding="utf-8")

    fitted_only = results[results["family"] != "Majority baseline"]
    majority_acc = float(results.loc[results["family"] == "Majority baseline", "test_accuracy"].iloc[0])
    majority_f1 = float(results.loc[results["family"] == "Majority baseline", "test_f1_macro"].iloc[0])
    majority_cv_acc = float(
        results.loc[results["family"] == "Majority baseline", "cv_accuracy_mean"].iloc[0]
    )

    # Selection uses CV first. Test is reported, not used to retune.
    acc_pool = fitted_only[fitted_only["optimized_for"] == "accuracy"]
    f1_pool = fitted_only[fitted_only["optimized_for"] == "f1_macro"]
    best_acc_cv = acc_pool.sort_values(
        by=["cv_accuracy_mean", "cv_f1_macro_mean"], ascending=False
    ).iloc[0]
    best_f1_cv = f1_pool.sort_values(
        by=["cv_f1_macro_mean", "cv_accuracy_mean"], ascending=False
    ).iloc[0]

    best_acc_cand = next(c for c in candidates if c["model"] == best_acc_cv["model"])
    best_f1_cand = next(c for c in candidates if c["model"] == best_f1_cv["model"])

    acc_table, acc_note = gini_or_linear_importance(best_acc_cand["estimator"])
    f1_table, f1_note = gini_or_linear_importance(best_f1_cand["estimator"])
    if acc_table is not None:
        acc_table.to_csv(OUT_DIR / "feature_importance_accuracy_model.csv", index=False)
    if f1_table is not None:
        f1_table.to_csv(OUT_DIR / "feature_importance_macro_f1_model.csv", index=False)

    def beats(row, threshold):
        return float(row["test_accuracy"]) >= threshold - 1e-12

    best_test_acc_row = fitted_only.sort_values("test_accuracy", ascending=False).iloc[0]
    best_test_f1_row = fitted_only.sort_values("test_f1_macro", ascending=False).iloc[0]

    reached_80 = bool((fitted_only["test_accuracy"] >= 0.80).any())
    reached_85 = bool((fitted_only["test_accuracy"] >= 0.85).any())
    reached_90 = bool((fitted_only["test_accuracy"] >= 0.90).any())
    beat_majority = bool((fitted_only["test_accuracy"] > majority_acc + 1e-12).any())

    dummy_like = (
        abs(float(best_acc_cv["test_accuracy"]) - majority_acc) < 1e-9
        and abs(float(best_acc_cv["test_f1_macro"]) - majority_f1) < 1e-9
    )
    cv_supports_acc = float(best_acc_cv["cv_accuracy_mean"]) >= majority_cv_acc - 0.01

    # Do not replace Streamlit with a majority clone.
    replace_streamlit = (
        (not dummy_like)
        and cv_supports_acc
        and float(best_acc_cv["test_accuracy"]) > majority_acc + 0.005
        and float(best_acc_cv["test_f1_macro"]) >= 0.35
    )
    recommend = (
        best_acc_cv["model"]
        if replace_streamlit
        else (
            best_f1_cv["model"]
            if float(best_f1_cv["test_f1_macro"]) >= 0.387
            else "Keep current Streamlit Decision Tree (untuned, class_weight=balanced)"
        )
    )

    summary = {
        "majority_test_accuracy": majority_acc,
        "majority_test_macro_f1": majority_f1,
        "best_cv_accuracy_model": best_acc_cv["model"],
        "best_cv_accuracy_cv_mean": float(best_acc_cv["cv_accuracy_mean"]),
        "best_cv_accuracy_test_accuracy": float(best_acc_cv["test_accuracy"]),
        "best_cv_accuracy_test_macro_f1": float(best_acc_cv["test_f1_macro"]),
        "best_cv_macro_f1_model": best_f1_cv["model"],
        "best_cv_macro_f1_cv_mean": float(best_f1_cv["cv_f1_macro_mean"]),
        "best_cv_macro_f1_test_accuracy": float(best_f1_cv["test_accuracy"]),
        "best_cv_macro_f1_test_macro_f1": float(best_f1_cv["test_f1_macro"]),
        "highest_observed_test_accuracy_model": best_test_acc_row["model"],
        "highest_observed_test_accuracy": float(best_test_acc_row["test_accuracy"]),
        "highest_observed_test_macro_f1_model": best_test_f1_row["model"],
        "highest_observed_test_macro_f1": float(best_test_f1_row["test_f1_macro"]),
        "beats_majority_baseline": beat_majority,
        "reached_80": reached_80,
        "reached_85": reached_85,
        "reached_90": reached_90,
        "replaced_streamlit_model": False,
        "streamlit_recommendation": recommend,
        "dummy_like_accuracy_winner": dummy_like,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def fmt_row(row):
        return (
            f"**{row['model']}**: CV acc={row['cv_accuracy_mean']:.4f}±{row['cv_accuracy_std']:.4f}, "
            f"CV macro F1={row['cv_f1_macro_mean']:.4f}±{row['cv_f1_macro_std']:.4f}, "
            f"test acc={row['test_accuracy']:.4f}, test macro F1={row['test_f1_macro']:.4f}, "
            f"balanced acc={row['test_balanced_accuracy']:.4f}"
        )

    md = [
        "# Leakage-safe optimization results",
        "",
        "Target definition unchanged. Five label-defining items excluded from X. "
        "Hyperparameters chosen on training StratifiedKFold only. Test set scored once.",
        "",
        "## Success questions (actual test-set numbers)",
        "",
        f"1. Beat majority baseline {majority_acc:.2%}? **{'Yes' if beat_majority else 'No'}**",
        f"2. Reach 80%? **{'Yes' if reached_80 else 'No'}**",
        f"3. Reach 85%? **{'Yes' if reached_85 else 'No'}**",
        f"4. Reach 90%? **{'Yes' if reached_90 else 'No'}**",
        "",
        "If 90% is not reached here, it is **not** forced. Prior work showed 90%+ "
        "only when the five target-defining items were put back into X (target leakage).",
        "",
        "## Majority baseline",
        "",
        f"- Test accuracy: {majority_acc:.4f}",
        f"- Test macro F1: {majority_f1:.4f}",
        f"- CV accuracy mean: {majority_cv_acc:.4f}",
        "",
        "This row is not hidden.",
        "",
        "## CV-selected specialists",
        "",
        "These two models were chosen from **cross-validation**, then evaluated on test.",
        "",
        "### Highest CV accuracy",
        "",
        fmt_row(best_acc_cv),
        "",
        f"Params: `{best_acc_cv['best_params']}`",
        "",
        "### Highest CV macro F1",
        "",
        fmt_row(best_f1_cv),
        "",
        f"Params: `{best_f1_cv['best_params']}`",
        "",
        "## Trade-off",
        "",
        "Medium is about 74% of employees. An accuracy-optimized tree or forest "
        "often predicts Medium for almost everyone. That can match or slightly "
        "exceed 73.81% accuracy while **macro F1 stays near the dummy (0.28)** "
        "because Low and High are missed. A balanced tree usually has **lower "
        "accuracy** and **higher macro F1** because it spends capacity on the rare classes.",
        "",
        f"Accuracy-optimized winner dummy-like (same as majority on test)? **{dummy_like}**",
        "",
        "## Highest observed test scores (one-shot; not used to retune)",
        "",
        f"- Highest test accuracy: {best_test_acc_row['model']} "
        f"({float(best_test_acc_row['test_accuracy']):.4f})",
        f"- Highest test macro F1: {best_test_f1_row['model']} "
        f"({float(best_test_f1_row['test_f1_macro']):.4f})",
        "",
        "## Streamlit",
        "",
        f"- `models/final_model.joblib` was **not** overwritten by this script.",
        f"- Recommendation: {recommend}",
        "",
        "Replace the demo only if a candidate beats majority accuracy with CV support "
        "and does not collapse to always-Medium (macro F1 near 0.28).",
        "",
        "## Feature importance",
        "",
        "Feature importance indicates contribution to model predictions and does not establish causation.",
        "",
        f"Accuracy-oriented model: {acc_note}",
        "",
        f"Macro-F1-oriented model: {f1_note}",
        "",
        "Tables: `reports/optimization/feature_importance_accuracy_model.csv` and "
        "`feature_importance_macro_f1_model.csv`.",
        "",
        "## Files",
        "",
        "- `reports/optimization_results.csv`",
        "- `reports/optimization/comparison_by_test_accuracy.csv`",
        "- `reports/optimization/comparison_by_test_macro_f1.csv`",
        "- Candidate pipelines: `models/optimization/`",
        "",
    ]
    (OUT_DIR / "optimization_notes.md").write_text("\n".join(md), encoding="utf-8")
    print("Wrote", results_path)
    print("Best CV accuracy model:", best_acc_cv["model"], float(best_acc_cv["test_accuracy"]))
    print("Best CV macro F1 model:", best_f1_cv["model"], float(best_f1_cv["test_f1_macro"]))
    print("Beat majority?", beat_majority, "80/85/90", reached_80, reached_85, reached_90)
    print("Streamlit recommendation:", recommend)


if __name__ == "__main__":
    main()
