"""
Honest investigation of whether >90% test accuracy is achievable.

Does not overwrite models/final_model.joblib or the Streamlit demo.
Does not invent metrics. Test rows are never used for fitting or HPO.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
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
    f1_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import RANDOM_STATE, REPORTS_DIR, TEST_SIZE  # noqa: E402
from src.data_loader import load_raw_dataset  # noqa: E402
from src.preprocess import NON_FEATURE_COLUMNS  # noqa: E402
from src.target import (  # noqa: E402
    ENGAGEMENT_INPUT_COLUMNS,
    LEVEL_COLUMN,
    LEVEL_ORDER,
    SCORE_COLUMN,
    add_engagement_target,
)

OUT_DIR = REPORTS_DIR / "accuracy_investigation"
CV = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
ALWAYS_DROP = list(dict.fromkeys(NON_FEATURE_COLUMNS + [SCORE_COLUMN, LEVEL_COLUMN, "OverallSatisfactionIndex"]))


def metrics_dict(y_true, y_pred):
    y_true = pd.Series(y_true).astype(str)
    y_pred = pd.Series(y_pred).astype(str)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=LEVEL_ORDER, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def majority_baseline(y_train, y_test):
    maj = y_train.value_counts().idxmax()
    pred = np.full(len(y_test), maj, dtype=object)
    out = metrics_dict(y_test, pred)
    out["majority_class"] = str(maj)
    return out, pred


def split_xy(X, y):
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


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


def features_for(df, include_survey_items):
    drop = list(ALWAYS_DROP)
    if not include_survey_items:
        drop = list(dict.fromkeys(drop + ENGAGEMENT_INPUT_COLUMNS))
    drop = [c for c in drop if c in df.columns]
    X = df.drop(columns=drop)
    return X, drop


def evaluate_estimator(name, estimator, param_distributions, X_train, y_train, X_test, y_test, n_iter):
    prep = make_preprocessor(X_train)
    pipe = Pipeline([("preprocess", prep), ("model", estimator)])
    if param_distributions:
        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring="f1_macro",
            cv=CV,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_
        cv_macro_f1 = float(search.best_score_)
        best_params = {k: (None if v is None else v) for k, v in search.best_params_.items()}
        # json-friendly
        best_params = {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in best_params.items()}
        best_params = {k: (int(v) if isinstance(v, (np.integer,)) else v) for k, v in best_params.items()}
    else:
        pipe.fit(X_train, y_train)
        best = pipe
        cv_macro_f1 = None
        best_params = {}
    pred = best.predict(X_test)
    row = {"model": name, "cv_macro_f1": cv_macro_f1, "best_params": best_params}
    row.update(metrics_dict(y_test, pred))
    return row


def search_space():
    return [
        (
            "Logistic Regression (tuned)",
            LogisticRegression(max_iter=4000, random_state=RANDOM_STATE),
            {
                "model__C": [0.1, 0.5, 1.0, 2.0, 5.0],
                "model__class_weight": ["balanced", None],
            },
            8,
        ),
        (
            "Random Forest (tuned)",
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "model__n_estimators": [150, 300, 500],
                "model__max_depth": [6, 10, 16, None],
                "model__min_samples_leaf": [1, 2, 4, 8],
                "model__max_features": ["sqrt", 0.4, 0.7],
                "model__class_weight": ["balanced", "balanced_subsample", None],
            },
            12,
        ),
        (
            "Extra Trees (tuned)",
            ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "model__n_estimators": [150, 300, 500],
                "model__max_depth": [6, 10, 16, None],
                "model__min_samples_leaf": [1, 2, 4, 8],
                "model__class_weight": ["balanced", None],
            },
            10,
        ),
        (
            "Gradient Boosting (tuned)",
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [80, 150, 250],
                "model__learning_rate": [0.03, 0.06, 0.1],
                "model__max_depth": [2, 3, 4],
                "model__subsample": [0.7, 1.0],
            },
            10,
        ),
        (
            "HistGradientBoosting (tuned)",
            HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "model__max_iter": [120, 250, 400],
                "model__learning_rate": [0.03, 0.06, 0.1],
                "model__max_depth": [3, 6, None],
                "model__l2_regularization": [0.0, 0.1, 1.0],
                "model__class_weight": ["balanced", None],
            },
            10,
        ),
        (
            "SVM (tuned)",
            SVC(kernel="rbf", random_state=RANDOM_STATE),
            {
                "model__C": [0.5, 1.0, 2.0, 4.0],
                "model__gamma": ["scale", 0.03, 0.01],
                "model__class_weight": ["balanced", None],
            },
            8,
        ),
    ]


def run_model_suite(X_train, y_train, X_test, y_test):
    rows = []
    for name, est, params, n_iter in search_space():
        print(f"  fitting {name} ...", flush=True)
        row = evaluate_estimator(name, est, params, X_train, y_train, X_test, y_test, n_iter)
        print(
            f"    test acc={row['accuracy']:.4f}  macro F1={row['f1_macro']:.4f}  "
            f"bal acc={row['balanced_accuracy']:.4f}",
            flush=True,
        )
        rows.append(row)
    return rows


def score_histogram(score, bin_width=0.2):
    edges = np.arange(1.0, 4.0 + bin_width, bin_width)
    counts, edges = np.histogram(score.dropna(), bins=edges)
    return {
        "bin_left": [float(x) for x in edges[:-1]],
        "bin_right": [float(x) for x in edges[1:]],
        "count": [int(c) for c in counts],
    }


def class_table(y):
    counts = y.astype(str).value_counts().reindex(LEVEL_ORDER).fillna(0).astype(int)
    total = int(counts.sum())
    return {
        "level": LEVEL_ORDER,
        "count": [int(counts.loc[k]) for k in LEVEL_ORDER],
        "percent": [round(100.0 * float(counts.loc[k]) / total, 2) if total else 0.0 for k in LEVEL_ORDER],
        "total": total,
    }


def tertile_labels(train_score, full_score):
    q1, q2 = train_score.quantile([1.0 / 3.0, 2.0 / 3.0]).tolist()
    bins = [float(full_score.min()) - 1e-6, q1, q2, float(full_score.max()) + 1e-6]
    labels = pd.cut(full_score, bins=bins, labels=LEVEL_ORDER, include_lowest=True)
    return labels.astype(str), {"train_q33": float(q1), "train_q66": float(q2)}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labeled = add_engagement_target(load_raw_dataset())
    y = labeled[LEVEL_COLUMN].astype(str)
    score = labeled[SCORE_COLUMN]

    # --- Why 65% vs 74% on the original leakage-safe split ---
    X_safe, dropped_safe = features_for(labeled, include_survey_items=False)
    Xtr, Xte, ytr, yte = split_xy(X_safe, y)
    dummy_metrics, _ = majority_baseline(ytr, yte)

    dt = Pipeline(
        [
            ("preprocess", make_preprocessor(Xtr)),
            ("model", DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE)),
        ]
    )
    dt.fit(Xtr, ytr)
    dt_pred = dt.predict(Xte)
    dt_metrics = metrics_dict(yte, dt_pred)
    rf = Pipeline(
        [
            ("preprocess", make_preprocessor(Xtr)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
                ),
            ),
        ]
    )
    rf.fit(Xtr, ytr)
    rf_pred = rf.predict(Xte)
    rf_metrics = metrics_dict(yte, rf_pred)

    current = {
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
        "dropped_from_X": dropped_safe,
        "train_class_counts": class_table(ytr),
        "test_class_counts": class_table(yte),
        "majority_baseline": dummy_metrics,
        "decision_tree_balanced": dt_metrics,
        "random_forest_balanced": rf_metrics,
        "dt_pred_distribution": pd.Series(dt_pred).value_counts().reindex(LEVEL_ORDER).fillna(0).astype(int).to_dict(),
        "rf_pred_distribution": pd.Series(rf_pred).value_counts().reindex(LEVEL_ORDER).fillna(0).astype(int).to_dict(),
        "note": (
            "class_weight='balanced' makes the tree trade Medium accuracy for rare-class recall. "
            "The untuned Random Forest on this split predicted only Medium, matching majority accuracy. "
            "After dropping the five survey items that define the label, remaining HR fields are a weak signal."
        ),
    }

    # --- Experiment 1: leakage-safe HPO ---
    print("EXPERIMENT 1 — leakage-safe HPO", flush=True)
    exp1_models = run_model_suite(Xtr, ytr, Xte, yte)
    exp1_best = max(exp1_models, key=lambda r: r["f1_macro"])

    # --- Experiment 2: five survey items allowed ---
    print("EXPERIMENT 2 — engagement-indicator features", flush=True)
    X_ind, dropped_ind = features_for(labeled, include_survey_items=True)
    Xtr2, Xte2, ytr2, yte2 = split_xy(X_ind, y)
    dummy2, _ = majority_baseline(ytr2, yte2)
    exp2_models = run_model_suite(Xtr2, ytr2, Xte2, yte2)
    exp2_best = max(exp2_models, key=lambda r: r["f1_macro"])

    from src.target import engagement_level_from_score, engagement_score_from_row_mean

    test_rows = labeled.loc[Xte2.index]
    formula_level = engagement_level_from_score(engagement_score_from_row_mean(test_rows)).astype(str)
    oracle_metrics = metrics_dict(yte2, formula_level)

    # Five items only (still a leakage / reconstruction experiment)
    X_only = labeled[ENGAGEMENT_INPUT_COLUMNS].copy()
    Xtr3, Xte3, ytr3, yte3 = split_xy(X_only, y)
    print("  fitting Decision Tree on five survey items only ...", flush=True)
    only_dt = evaluate_estimator(
        "Decision Tree (five survey items only)",
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        {"model__max_depth": [4, 6, 8, None], "model__min_samples_leaf": [1, 2, 4]},
        Xtr3,
        ytr3,
        Xte3,
        yte3,
        6,
    )

    # --- Experiment 3: target sensitivity ---
    print("EXPERIMENT 3 — target sensitivity", flush=True)
    hist = score_histogram(score)
    current_classes = class_table(y)
    tertile_y, tertile_cuts = tertile_labels(score.loc[Xtr.index], score)
    tertile_counts = class_table(tertile_y)
    ytr_t = tertile_y.loc[Xtr.index]
    yte_t = tertile_y.loc[Xte.index]
    dummy_t, _ = majority_baseline(ytr_t, yte_t)
    print("  fitting Random Forest on train-quantile tertile labels (leakage-safe X, same split rows) ...", flush=True)
    tertile_rf = evaluate_estimator(
        "Random Forest (tertile labels, leakage-safe X)",
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        {
            "model__n_estimators": [200, 400],
            "model__max_depth": [8, 16, None],
            "model__class_weight": ["balanced", None],
            "model__min_samples_leaf": [1, 4],
        },
        Xtr,
        ytr_t,
        Xte,
        yte_t,
        8,
    )

    payload = {
        "split": {"test_size": TEST_SIZE, "random_state": RANDOM_STATE, "cv_folds": 3},
        "current_project": current,
        "experiment_1_leakage_safe": {
            "features": "HR fields only; five survey items, score, level, OverallSatisfactionIndex, IDs/constants excluded",
            "target": "project bins 1.00-2.00 Low, 2.01-3.00 Medium, 3.01-4.00 High",
            "majority_baseline": dummy_metrics,
            "models": exp1_models,
            "best_by_macro_f1": exp1_best,
        },
        "experiment_2_engagement_indicators": {
            "label": (
                "Classification using the same engagement-related indicators from which "
                "the project-defined label was constructed."
            ),
            "features": "five survey items plus other HR fields; score, level, OverallSatisfactionIndex, IDs/constants still excluded",
            "dropped": dropped_ind,
            "majority_baseline": dummy2,
            "models": exp2_models,
            "best_by_macro_f1": exp2_best,
            "oracle_reconstruction_same_formula": {
                "model": "Oracle: re-apply mean + cut on the five survey items (not a trained classifier)",
                **oracle_metrics,
            },
            "five_items_only": only_dt,
        },
        "experiment_3_target_sensitivity": {
            "current_thresholds": "1.00-2.00 Low, 2.01-3.00 Medium, 3.01-4.00 High",
            "current_class_counts_full_table": current_classes,
            "score_min": float(score.min()),
            "score_max": float(score.max()),
            "score_mean": float(score.mean()),
            "score_median": float(score.median()),
            "histogram": hist,
            "tertile_cutpoints_from_training_scores": tertile_cuts,
            "tertile_class_counts_full_table": tertile_counts,
            "majority_baseline_tertile_test": dummy_t,
            "random_forest_tertile_leakage_safe": tertile_rf,
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "results.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    table_rows = []

    def add_row(experiment, target, features, metrics, baseline_acc, leakage):
        table_rows.append(
            {
                "Experiment": experiment,
                "Target Definition": target,
                "Features Used": features,
                "Accuracy": metrics["accuracy"],
                "Macro F1": metrics["f1_macro"],
                "Balanced Accuracy": metrics["balanced_accuracy"],
                "Majority Baseline": baseline_acc,
                "Leakage Risk": leakage,
            }
        )

    add_row(
        "Current project (Decision Tree, no HPO)",
        "Project bins on mean of five 1-4 items",
        "Leakage-safe HR fields only",
        dt_metrics,
        dummy_metrics["accuracy"],
        "Controlled: five items excluded",
    )
    add_row(
        "Current project majority baseline",
        "Project bins",
        "Predict training majority class",
        dummy_metrics,
        dummy_metrics["accuracy"],
        "None (baseline)",
    )
    add_row(
        f"Experiment 1 best by macro F1: {exp1_best['model']}",
        "Project bins",
        "Leakage-safe HR fields only; 3-fold stratified CV HPO on train",
        exp1_best,
        dummy_metrics["accuracy"],
        "Controlled: five items excluded",
    )
    add_row(
        f"Experiment 2 best by macro F1: {exp2_best['model']}",
        "Project bins",
        "Five survey items + other HR fields",
        exp2_best,
        dummy2["accuracy"],
        "High: predictors include label components",
    )
    add_row(
        "Experiment 2 oracle (formula, not trained)",
        "Project bins",
        "Re-apply mean + cut on the five survey items",
        oracle_metrics,
        dummy2["accuracy"],
        "Definitional: this is the label rule",
    )
    add_row(
        "Experiment 2: Decision Tree on five items only",
        "Project bins",
        "Only the five survey items",
        only_dt,
        dummy2["accuracy"],
        "High: predictors are the label components",
    )
    add_row(
        "Experiment 3: RF on train-quantile tertiles",
        "Tertiles of EngagementScore using training quantiles only",
        "Leakage-safe HR fields only",
        tertile_rf,
        dummy_t["accuracy"],
        "Controlled: five items excluded; bins from train scores",
    )

    table = pd.DataFrame(table_rows)
    csv_path = OUT_DIR / "comparison_table.csv"
    table.to_csv(csv_path, index=False)
    print("\n=== COMPARISON TABLE ===", flush=True)
    print(table.to_string(index=False), flush=True)
    print(f"\nWrote {json_path}", flush=True)
    print(f"Wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
