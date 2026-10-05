"""
Unsupervised employee engagement profiling (Case Study 90).

Groups employees from observed HR and satisfaction fields. There is no
supervised target. EngagementLevel and EngagementScore are never used.

This module does not invent cluster metrics. sklearn computes them.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CLUSTERING_PLOTS_DIR,
    CLUSTERING_REPORTS_DIR,
    MODELS_DIR,
    RANDOM_STATE,
)

try:
    from src.config import CLUSTERING_PIPELINE_PATH
except ImportError:
    CLUSTERING_PIPELINE_PATH = MODELS_DIR / "engagement_clustering_pipeline.joblib"

CANDIDATE_COLUMNS = [
    "JobInvolvement",
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "RelationshipSatisfaction",
    "WorkLifeBalance",
    "OverTime",
    "Age",
    "MonthlyIncome",
    "JobLevel",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
    "TrainingTimesLastYear",
    "JobRole",
    "BusinessTravel",
    "MaritalStatus",
    "DistanceFromHome",
]

FORBIDDEN_COLUMNS = [
    "EngagementScore",
    "EngagementLevel",
    "OverallSatisfactionIndex",
    "EmployeeNumber",
    "EmployeeCount",
    "Over18",
    "StandardHours",
]

SATISFACTION_COLUMNS = [
    "JobInvolvement",
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "RelationshipSatisfaction",
    "WorkLifeBalance",
]

K_RANGE = range(2, 7)

PROFILE_PALETTE = [
    "#1e4d6b",
    "#b45309",
    "#2a6b4a",
    "#7c3aed",
    "#be123c",
    "#0f766e",
]


def available_clustering_columns(df):
    present = [c for c in CANDIDATE_COLUMNS if c in df.columns]
    missing = [c for c in CANDIDATE_COLUMNS if c not in df.columns]
    leaked = [c for c in FORBIDDEN_COLUMNS if c in present]
    if leaked:
        raise ValueError("Forbidden columns selected for clustering: " + ", ".join(leaked))
    return present, missing


def select_clustering_frame(df):
    present, missing = available_clustering_columns(df)
    if not present:
        raise ValueError("None of the clustering candidate columns exist in the table.")
    X = df[present].copy()
    return X, present, missing


def split_feature_types(X):
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return numeric_cols, categorical_cols


def build_cluster_preprocessor(X):
    numeric_cols, categorical_cols = split_feature_types(X)
    transformers = []
    if numeric_cols:
        transformers.append(("numeric", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            )
        )
    if not transformers:
        raise ValueError("No numeric or categorical columns to preprocess.")
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor, numeric_cols, categorical_cols


def make_clustering_pipeline(X, n_clusters):
    preprocessor, _, _ = build_cluster_preprocessor(X)
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "cluster",
                KMeans(
                    n_clusters=int(n_clusters),
                    random_state=RANDOM_STATE,
                    n_init=10,
                ),
            ),
        ]
    )


def _cluster_metrics(X_processed, labels):
    labels = np.asarray(labels)
    n_labels = len(set(labels) - {-1})
    if n_labels < 2:
        return {
            "silhouette": float("nan"),
            "davies_bouldin": float("nan"),
            "calinski_harabasz": float("nan"),
            "n_clusters_found": int(n_labels),
        }
    return {
        "silhouette": float(silhouette_score(X_processed, labels)),
        "davies_bouldin": float(davies_bouldin_score(X_processed, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X_processed, labels)),
        "n_clusters_found": int(n_labels),
    }


def evaluate_kmeans_k(X, k_range=K_RANGE):
    preprocessor, numeric_cols, categorical_cols = build_cluster_preprocessor(X)
    X_p = preprocessor.fit_transform(X)
    rows = []
    inertias = []
    for k in k_range:
        model = KMeans(n_clusters=int(k), random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(X_p)
        scores = _cluster_metrics(X_p, labels)
        sizes = pd.Series(labels).value_counts().sort_index()
        min_share = float(sizes.min() / len(labels))
        rows.append(
            {
                "k": int(k),
                "inertia": float(model.inertia_),
                "silhouette": scores["silhouette"],
                "davies_bouldin": scores["davies_bouldin"],
                "calinski_harabasz": scores["calinski_harabasz"],
                "min_cluster_share": min_share,
                "min_cluster_n": int(sizes.min()),
            }
        )
        inertias.append(float(model.inertia_))
    table = pd.DataFrame(rows)
    return table, X_p, preprocessor, numeric_cols, categorical_cols


def choose_k(k_table):
    """
    Pick K using internal metrics plus a minimum-size rule.

    Highest silhouette among solutions whose smallest cluster is at least 8%
    of the sample. If that filter is empty, use the global silhouette max.
    Davies-Bouldin (lower) is the tie-breaker.
    """
    usable = k_table[k_table["min_cluster_share"] >= 0.08].copy()
    pool = usable if not usable.empty else k_table
    pool = pool.sort_values(
        by=["silhouette", "calinski_harabasz"],
        ascending=False,
    )
    # Prefer lower Davies-Bouldin when silhouette is within 0.01 of the best.
    best_sil = float(pool["silhouette"].iloc[0])
    close = pool[pool["silhouette"] >= best_sil - 0.01]
    chosen = close.sort_values(by=["davies_bouldin", "k"]).iloc[0]
    reason = (
        f"K={int(chosen['k'])} selected from K=2–6. "
        f"Silhouette={chosen['silhouette']:.4f} "
        f"(highest among solutions with smallest cluster ≥ 8% of rows"
        f"{'' if not usable.empty else '; size filter was empty, used all K'}). "
        f"Davies-Bouldin={chosen['davies_bouldin']:.4f} used as tie-breaker "
        f"among silhouette values within 0.01 of the best. "
        f"This is an internal clustering criterion, not a claim that K is a "
        f"true number of psychological types."
    )
    return int(chosen["k"]), reason, chosen


def evaluate_agglomerative(X_processed, k_range=K_RANGE):
    rows = []
    for k in k_range:
        model = AgglomerativeClustering(n_clusters=int(k), linkage="ward")
        labels = model.fit_predict(X_processed)
        scores = _cluster_metrics(X_processed, labels)
        rows.append({"k": int(k), "algorithm": "Agglomerative (ward)", **scores})
    return pd.DataFrame(rows)


def evaluate_dbscan(X_processed):
    """
    Probe a few eps values. DBSCAN is often a poor fit for this mixed,
    one-hot encoded table; results are reported rather than forced.
    """
    nn = NearestNeighbors(n_neighbors=10)
    nn.fit(X_processed)
    distances, _ = nn.kneighbors(X_processed)
    typical = float(np.median(distances[:, 9]))
    rows = []
    for eps in [typical * 0.7, typical, typical * 1.4, typical * 2.0]:
        model = DBSCAN(eps=float(eps), min_samples=15)
        labels = model.fit_predict(X_processed)
        n_noise = int((labels == -1).sum())
        scores = _cluster_metrics(X_processed, labels)
        rows.append(
            {
                "algorithm": "DBSCAN",
                "eps": float(eps),
                "min_samples": 15,
                "n_clusters_found": scores["n_clusters_found"],
                "n_noise": n_noise,
                "noise_share": float(n_noise / len(labels)),
                "silhouette": scores["silhouette"],
                "davies_bouldin": scores["davies_bouldin"],
                "calinski_harabasz": scores["calinski_harabasz"],
            }
        )
    return pd.DataFrame(rows), typical


def overtime_share(series):
    return 100.0 * float(series.astype(str).str.strip().str.lower().eq("yes").mean())


def cluster_profile_table(X, labels):
    labels = np.asarray(labels)
    out_rows = []
    n = len(X)
    for cid in sorted(set(labels.tolist())):
        mask = labels == cid
        part = X.loc[mask]
        row = {
            "cluster": int(cid),
            "n_employees": int(mask.sum()),
            "percent": round(100.0 * float(mask.sum()) / n, 2),
        }
        for col in SATISFACTION_COLUMNS:
            if col in part.columns:
                row[f"mean_{col}"] = float(part[col].mean())
        sat_present = [c for c in SATISFACTION_COLUMNS if c in part.columns]
        if sat_present:
            row["mean_satisfaction_items"] = float(part[sat_present].mean(axis=1).mean())
        if "OverTime" in part.columns:
            row["overtime_percent"] = overtime_share(part["OverTime"])
        if "MonthlyIncome" in part.columns:
            row["mean_MonthlyIncome"] = float(part["MonthlyIncome"].mean())
        if "Age" in part.columns:
            row["mean_Age"] = float(part["Age"].mean())
        if "YearsAtCompany" in part.columns:
            row["mean_YearsAtCompany"] = float(part["YearsAtCompany"].mean())
        if "YearsInCurrentRole" in part.columns:
            row["mean_YearsInCurrentRole"] = float(part["YearsInCurrentRole"].mean())
        if "YearsSinceLastPromotion" in part.columns:
            row["mean_YearsSinceLastPromotion"] = float(part["YearsSinceLastPromotion"].mean())
        if "JobLevel" in part.columns:
            row["mean_JobLevel"] = float(part["JobLevel"].mean())
        if "DistanceFromHome" in part.columns:
            row["mean_DistanceFromHome"] = float(part["DistanceFromHome"].mean())
        if "TrainingTimesLastYear" in part.columns:
            row["mean_TrainingTimesLastYear"] = float(part["TrainingTimesLastYear"].mean())
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def overall_reference(X):
    ref = {"n_employees": int(len(X)), "percent": 100.0}
    for col in SATISFACTION_COLUMNS:
        if col in X.columns:
            ref[f"mean_{col}"] = float(X[col].mean())
    sat_present = [c for c in SATISFACTION_COLUMNS if c in X.columns]
    if sat_present:
        ref["mean_satisfaction_items"] = float(X[sat_present].mean(axis=1).mean())
    if "OverTime" in X.columns:
        ref["overtime_percent"] = overtime_share(X["OverTime"])
    for col, key in [
        ("MonthlyIncome", "mean_MonthlyIncome"),
        ("Age", "mean_Age"),
        ("YearsAtCompany", "mean_YearsAtCompany"),
        ("YearsInCurrentRole", "mean_YearsInCurrentRole"),
        ("YearsSinceLastPromotion", "mean_YearsSinceLastPromotion"),
        ("JobLevel", "mean_JobLevel"),
    ]:
        if col in X.columns:
            ref[key] = float(X[col].mean())
    return ref


def name_clusters(profile_df, overall):
    """
    Descriptive names from this table's cluster statistics only.
    Not a validated psychological taxonomy.
    """
    names = {}
    descriptions = {}
    used = set()
    sat_all = float(overall.get("mean_satisfaction_items", 0.0))
    ot_all = float(overall.get("overtime_percent", 0.0))
    pay_all = float(overall.get("mean_MonthlyIncome", 0.0))
    ten_all = float(overall.get("mean_YearsAtCompany", 0.0))

    for _, row in profile_df.iterrows():
        cid = int(row["cluster"])
        sat = float(row.get("mean_satisfaction_items", sat_all))
        ot = float(row.get("overtime_percent", ot_all))
        pay = float(row.get("mean_MonthlyIncome", pay_all))
        ten = float(row.get("mean_YearsAtCompany", ten_all))

        if sat >= sat_all + 0.12:
            sat_part = "High Satisfaction"
        elif sat <= sat_all - 0.12:
            sat_part = "Lower Satisfaction"
        else:
            sat_part = "Balanced Satisfaction"

        extras = []
        if ot >= ot_all + 6:
            extras.append("Higher Overtime")
        elif ot <= ot_all - 6:
            extras.append("Lower Overtime")
        if pay_all and pay >= pay_all * 1.25:
            extras.append("Higher Pay")
        elif pay_all and pay <= pay_all * 0.80:
            extras.append("Lower Pay")
        if ten_all and ten >= ten_all * 1.35:
            extras.append("Longer Tenure")
        elif ten_all and ten <= ten_all * 0.70:
            extras.append("Shorter Tenure")

        # If satisfaction barely differs, name the cluster by the HR
        # characteristics that actually separate the groups.
        if abs(sat - sat_all) < 0.12 and extras:
            name = " / ".join(extras[:3]) + " Profile"
        elif extras:
            name = f"{sat_part} / {' / '.join(extras[:2])} Profile"
        else:
            name = f"{sat_part} Profile"

        base = name
        suffix = 2
        while name in used:
            name = f"{base} ({suffix})"
            suffix += 1
        used.add(name)
        names[str(cid)] = name

        descriptions[str(cid)] = (
            f"{int(row['n_employees'])} employees ({row['percent']:.1f}% of the table). "
            f"Mean of the five workplace satisfaction items = {sat:.2f} "
            f"(table mean {sat_all:.2f}). Overtime {ot:.1f}% "
            f"(table {ot_all:.1f}%). Mean monthly income "
            f"{pay:.0f}; mean years at company {ten:.1f}. "
            "These are observed group averages, not a clinical engagement type."
        )
    return names, descriptions


def cluster_driver_table(X, labels, top_n=15):
    """
    How much each numeric feature differs across clusters (between-cluster
    range divided by overall standard deviation). Not causal importance.
    """
    labels = np.asarray(labels)
    numeric_cols, _ = split_feature_types(X)
    rows = []
    for col in numeric_cols:
        overall_std = float(X[col].std(ddof=0))
        if overall_std == 0:
            continue
        means = [float(X.loc[labels == cid, col].mean()) for cid in sorted(set(labels))]
        stretch = (max(means) - min(means)) / overall_std
        rows.append(
            {
                "feature": col,
                "between_cluster_range_over_std": stretch,
                "min_cluster_mean": min(means),
                "max_cluster_mean": max(means),
                "overall_mean": float(X[col].mean()),
            }
        )
    if "OverTime" in X.columns:
        ot_flag = (
            X["OverTime"].astype(str).str.strip().str.lower().eq("yes").astype(float) * 100.0
        )
        ot_std = float(ot_flag.std(ddof=0))
        shares = [
            overtime_share(X.loc[labels == cid, "OverTime"])
            for cid in sorted(set(labels))
        ]
        stretch = (max(shares) - min(shares)) / ot_std if ot_std else 0.0
        rows.append(
            {
                "feature": "OverTime_yes_percent",
                "between_cluster_range_over_std": stretch,
                "min_cluster_mean": min(shares),
                "max_cluster_mean": max(shares),
                "overall_mean": overtime_share(X["OverTime"]),
            }
        )
    table = pd.DataFrame(rows).sort_values(
        "between_cluster_range_over_std", ascending=False
    )
    table["rank"] = range(1, len(table) + 1)
    return table.head(top_n).reset_index(drop=True)


def _save_fig(fig, filename):
    CLUSTERING_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = CLUSTERING_PLOTS_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_elbow(k_table):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(k_table["k"], k_table["inertia"], marker="o", color="#1e4d6b")
    ax.set_xlabel("K (number of clusters)")
    ax.set_ylabel("Inertia (within-cluster sum of squares)")
    ax.set_title("K-Means elbow curve")
    ax.set_xticks(list(k_table["k"]))
    return _save_fig(fig, "elbow_curve.png")


def plot_silhouette_by_k(k_table):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(k_table["k"].astype(str), k_table["silhouette"], color="#3b6d8c")
    ax.set_xlabel("K")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Silhouette score by K (K-Means)")
    return _save_fig(fig, "silhouette_by_k.png")


def plot_cluster_sizes(profile_df, names):
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    labels = [names.get(str(int(c)), f"Cluster {int(c)}") for c in profile_df["cluster"]]
    colors = [PROFILE_PALETTE[int(c) % len(PROFILE_PALETTE)] for c in profile_df["cluster"]]
    ax.barh(labels, profile_df["n_employees"], color=colors)
    ax.set_xlabel("Number of employees")
    ax.set_title("Cluster sizes")
    return _save_fig(fig, "cluster_sizes.png")


def plot_pca(X_processed, labels, names):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_processed)
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    for cid in sorted(set(labels.tolist())):
        mask = labels == cid
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=18,
            alpha=0.7,
            color=PROFILE_PALETTE[int(cid) % len(PROFILE_PALETTE)],
            label=names.get(str(int(cid)), f"Cluster {cid}"),
        )
    ax.set_xlabel(f"PC1 ({100 * pca.explained_variance_ratio_[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({100 * pca.explained_variance_ratio_[1]:.1f}% variance)")
    ax.set_title("PCA of clustered employees (2D)")
    ax.legend(frameon=False, fontsize=8)
    path = _save_fig(fig, "pca_clusters.png")
    return path, pca, coords


def plot_satisfaction_by_cluster(profile_df, names):
    cols = [f"mean_{c}" for c in SATISFACTION_COLUMNS if f"mean_{c}" in profile_df.columns]
    long = profile_df.melt(
        id_vars=["cluster"],
        value_vars=cols,
        var_name="item",
        value_name="mean",
    )
    long["item"] = long["item"].str.replace("mean_", "", regex=False)
    long["profile"] = long["cluster"].map(lambda c: names.get(str(int(c)), f"Cluster {c}"))
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    sns.barplot(data=long, x="item", y="mean", hue="profile", ax=ax)
    ax.set_ylim(1, 4)
    ax.set_ylabel("Cluster mean (1–4 items)")
    ax.set_xlabel("")
    ax.set_title("Satisfaction-item means by cluster")
    ax.legend(frameon=False, fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    return _save_fig(fig, "satisfaction_by_cluster.png")


def plot_wlb_by_cluster(profile_df, names):
    if "mean_WorkLifeBalance" not in profile_df.columns:
        return None
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    labels = [names.get(str(int(c)), f"Cluster {int(c)}") for c in profile_df["cluster"]]
    colors = [PROFILE_PALETTE[int(c) % len(PROFILE_PALETTE)] for c in profile_df["cluster"]]
    ax.bar(labels, profile_df["mean_WorkLifeBalance"], color=colors)
    ax.set_ylabel("Mean WorkLifeBalance (1–4)")
    ax.set_title("Work-life balance by cluster")
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
    return _save_fig(fig, "worklife_by_cluster.png")


def plot_overtime_by_cluster(profile_df, names):
    if "overtime_percent" not in profile_df.columns:
        return None
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    labels = [names.get(str(int(c)), f"Cluster {int(c)}") for c in profile_df["cluster"]]
    colors = [PROFILE_PALETTE[int(c) % len(PROFILE_PALETTE)] for c in profile_df["cluster"]]
    ax.bar(labels, profile_df["overtime_percent"], color=colors)
    ax.set_ylabel("Employees with overtime (%)")
    ax.set_title("Overtime share by cluster")
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
    return _save_fig(fig, "overtime_by_cluster.png")


def plot_profile_heatmap(profile_df, names):
    value_cols = [
        c
        for c in profile_df.columns
        if c.startswith("mean_") or c == "overtime_percent"
    ]
    heat = profile_df.set_index("cluster")[value_cols].copy()
    heat.index = [names.get(str(int(i)), f"Cluster {int(i)}") for i in heat.index]
    heat.columns = [c.replace("mean_", "") for c in heat.columns]
    # z-score columns so mixed units are comparable in the picture only
    z = (heat - heat.mean()) / heat.std(ddof=0).replace(0, 1)
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    sns.heatmap(z, cmap="RdBu_r", center=0, ax=ax, annot=heat.round(2), fmt="")
    ax.set_title("Cluster profile heatmap (cell text = raw mean; color = column z-score)")
    return _save_fig(fig, "cluster_profile_heatmap.png")


def fit_and_save(raw_df, save_artifacts=True):
    """
    Fit K-Means on all rows (unsupervised profiling). Returns a bundle dict.
    """
    inspect_missing = int(raw_df.isna().sum().sum())
    inspect_dups = int(raw_df.duplicated().sum())
    X, present, missing = select_clustering_frame(raw_df)
    leaked_in_x = [c for c in FORBIDDEN_COLUMNS if c in X.columns]
    if leaked_in_x:
        raise RuntimeError("Leakage/target columns in clustering X: " + ", ".join(leaked_in_x))

    k_table, X_p, _, numeric_cols, categorical_cols = evaluate_kmeans_k(X)
    k, reason, chosen_row = choose_k(k_table)
    agg_table = evaluate_agglomerative(X_p)
    db_table, db_typical = evaluate_dbscan(X_p)

    pipeline = make_clustering_pipeline(X, k)
    labels = pipeline.fit_predict(X)
    X_proc = pipeline.named_steps["preprocess"].transform(X)
    final_scores = _cluster_metrics(X_proc, labels)

    profiles = cluster_profile_table(X, labels)
    overall = overall_reference(X)
    names, descriptions = name_clusters(profiles, overall)
    profiles["profile_name"] = profiles["cluster"].map(lambda c: names[str(int(c))])
    drivers = cluster_driver_table(X, labels)

    plot_paths = {}
    if save_artifacts:
        CLUSTERING_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        CLUSTERING_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        plot_paths["elbow"] = str(plot_elbow(k_table))
        plot_paths["silhouette"] = str(plot_silhouette_by_k(k_table))
        plot_paths["sizes"] = str(plot_cluster_sizes(profiles, names))
        pca_path, pca, coords = plot_pca(X_proc, labels, names)
        plot_paths["pca"] = str(pca_path)
        plot_paths["satisfaction"] = str(plot_satisfaction_by_cluster(profiles, names))
        wlb = plot_wlb_by_cluster(profiles, names)
        if wlb:
            plot_paths["worklife"] = str(wlb)
        ot = plot_overtime_by_cluster(profiles, names)
        if ot:
            plot_paths["overtime"] = str(ot)
        plot_paths["heatmap"] = str(plot_profile_heatmap(profiles, names))
        pca_df = pd.DataFrame(
            {
                "pc1": coords[:, 0],
                "pc2": coords[:, 1],
                "cluster": labels,
                "profile_name": [names[str(int(c))] for c in labels],
            }
        )
        pca_df.to_csv(CLUSTERING_REPORTS_DIR / "pca_coordinates.csv", index=False)
        k_table.to_csv(CLUSTERING_REPORTS_DIR / "k_selection.csv", index=False)
        agg_table.to_csv(CLUSTERING_REPORTS_DIR / "agglomerative_comparison.csv", index=False)
        db_table.to_csv(CLUSTERING_REPORTS_DIR / "dbscan_probe.csv", index=False)
        profiles.to_csv(CLUSTERING_REPORTS_DIR / "cluster_profiles.csv", index=False)
        drivers.to_csv(CLUSTERING_REPORTS_DIR / "cluster_drivers.csv", index=False)
        pd.DataFrame([overall]).to_csv(CLUSTERING_REPORTS_DIR / "overall_reference.csv", index=False)
    else:
        pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(X_proc)

    metadata = {
        "k": int(k),
        "k_selection_reason": reason,
        "silhouette": final_scores["silhouette"],
        "davies_bouldin": final_scores["davies_bouldin"],
        "calinski_harabasz": final_scores["calinski_harabasz"],
        "inertia": float(pipeline.named_steps["cluster"].inertia_),
        "n_employees": int(len(X)),
        "features_used": present,
        "features_missing_from_candidates": missing,
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "forbidden_excluded": FORBIDDEN_COLUMNS,
        "missing_value_cells": inspect_missing,
        "duplicate_rows": inspect_dups,
        "names": names,
        "descriptions": descriptions,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "dbscan_typical_10nn_distance": db_typical,
        "no_supervised_target": True,
        "disclaimer": (
            "Clusters are data-driven employee engagement profiles based on "
            "observed HR and satisfaction characteristics. They are not "
            "scientifically validated psychological engagement categories."
        ),
    }
    if save_artifacts:
        (CLUSTERING_REPORTS_DIR / "cluster_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    bundle = {
        "pipeline": pipeline,
        "pca": pca,
        "columns": present,
        "k": int(k),
        "names": names,
        "descriptions": descriptions,
        "profiles": profiles.to_dict(orient="records"),
        "overall": overall,
        "metrics": {
            "silhouette": final_scores["silhouette"],
            "davies_bouldin": final_scores["davies_bouldin"],
            "calinski_harabasz": final_scores["calinski_harabasz"],
        },
        "k_table": k_table.to_dict(orient="records"),
        "metadata": metadata,
        "plot_paths": plot_paths,
        "drivers": drivers.to_dict(orient="records"),
        "labels": labels.tolist(),
    }

    if save_artifacts:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, CLUSTERING_PIPELINE_PATH)

    return bundle


def attach_cluster_labels(df, bundle):
    X = df[bundle["columns"]]
    labels = bundle["pipeline"].predict(X)
    out = df.copy()
    out["ClusterId"] = labels
    out["EngagementProfile"] = [bundle["names"][str(int(c))] for c in labels]
    return out


def load_clustering_bundle(path=None):
    path = Path(path) if path is not None else CLUSTERING_PIPELINE_PATH
    if not path.is_file():
        return None
    return joblib.load(path)


def assign_profile(bundle, feature_row):
    """
    feature_row: one-row DataFrame with the original clustering columns.
    Uses the saved Pipeline so preprocessing matches training.
    """
    pipeline = bundle["pipeline"]
    columns = bundle["columns"]
    missing = [c for c in columns if c not in feature_row.columns]
    if missing:
        raise ValueError("Missing clustering columns: " + ", ".join(missing))
    use = feature_row[columns]
    if use.isna().any().any():
        bad = use.columns[use.isna().any()].tolist()
        raise ValueError("Missing values in: " + ", ".join(bad))
    cluster_id = int(pipeline.predict(use)[0])
    key = str(cluster_id)
    profiles = {int(p["cluster"]): p for p in bundle["profiles"]}
    profile = profiles[cluster_id]
    comparison = []
    overall = bundle["overall"]
    mapping = [
        ("mean_JobSatisfaction", "JobSatisfaction", "Job satisfaction (1–4)"),
        ("mean_JobInvolvement", "JobInvolvement", "Job involvement (1–4)"),
        ("mean_EnvironmentSatisfaction", "EnvironmentSatisfaction", "Environment satisfaction (1–4)"),
        ("mean_RelationshipSatisfaction", "RelationshipSatisfaction", "Relationship satisfaction (1–4)"),
        ("mean_WorkLifeBalance", "WorkLifeBalance", "Work-life balance (1–4)"),
        ("mean_MonthlyIncome", "MonthlyIncome", "Monthly income"),
        ("mean_Age", "Age", "Age"),
        ("mean_YearsAtCompany", "YearsAtCompany", "Years at company"),
        ("mean_YearsInCurrentRole", "YearsInCurrentRole", "Years in current role"),
        ("mean_YearsSinceLastPromotion", "YearsSinceLastPromotion", "Years since last promotion"),
    ]
    row = use.iloc[0]
    for profile_key, col, label in mapping:
        if col in use.columns and profile_key in profile:
            comparison.append(
                {
                    "measure": label,
                    "this_employee": float(row[col]),
                    "cluster_mean": float(profile[profile_key]),
                    "table_mean": float(overall.get(profile_key, float("nan"))),
                }
            )
    if "OverTime" in use.columns:
        comparison.append(
            {
                "measure": "Overtime (Yes=100, No=0)",
                "this_employee": 100.0 if str(row["OverTime"]).strip().lower() == "yes" else 0.0,
                "cluster_mean": float(profile.get("overtime_percent", float("nan"))),
                "table_mean": float(overall.get("overtime_percent", float("nan"))),
            }
        )
    return {
        "cluster": cluster_id,
        "profile_name": bundle["names"][key],
        "description": bundle["descriptions"][key],
        "percent": float(profile["percent"]),
        "n_employees": int(profile["n_employees"]),
        "profile": profile,
        "comparison": pd.DataFrame(comparison),
    }
