"""
Employee Engagement Profiling and Segmentation — Streamlit app
Case Study No. 90

Loads the saved K-Means clustering pipeline. Does not retrain on refresh.

    streamlit run app.py
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from src.clustering import (
    PROFILE_PALETTE,
    assign_profile,
    attach_cluster_labels,
    load_clustering_bundle,
)
from src.config import MODELS_DIR, REPORTS_DIR
from src.data_loader import load_raw_dataset, validate_dataset_file

try:
    from src.config import CLASSIFICATION_BASELINE_DIR, CLUSTERING_PLOTS_DIR
except ImportError:
    CLASSIFICATION_BASELINE_DIR = REPORTS_DIR / "classification_baseline"
    CLUSTERING_PLOTS_DIR = REPORTS_DIR / "clustering" / "plots"

st.set_page_config(
    page_title="Employee Engagement Profiling",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f4f6f8;
            color: #1e293b;
            --text-color: #1e293b;
            --background-color: #f4f6f8;
            --secondary-background-color: #ffffff;
            --header-height: 0rem;
        }
        .block-container {
            padding-top: 1.05rem;
            padding-bottom: 2rem;
            padding-left: 1.2rem;
            padding-right: 1.2rem;
            max-width: 1180px;
            color: #1e293b;
        }
        /* Streamlit's toolbar is position:fixed and sat on top of st.tabs.
           Hide it instead of pushing the whole page down with a large padding-top. */
        header[data-testid="stHeader"] {
            display: none;
        }

        .main [data-testid="stHeading"] h1,
        .main [data-testid="stHeading"] h2,
        .main [data-testid="stHeading"] h3,
        .main h1, .main h2, .main h3 {
            color: #1e293b;
            letter-spacing: -0.02em;
        }
        .main .stMarkdown p,
        .main .stMarkdown li,
        .main [data-testid="stText"] p {
            color: #334155;
        }
        .main [data-testid="stCaption"] p { color: #475569; }
        .main .stMarkdown strong { color: #1e293b; }
        .main .stMarkdown code {
            background: #e8eef2;
            color: #1e4d6b;
        }

        .main .hero {
            background: #1e4d6b;
            color: #f8fafc;
            padding: 1.15rem 1.35rem;
            border-radius: 12px;
            margin-bottom: 0.75rem;
        }
        .main .hero h1 {
            color: #f8fafc;
            margin: 0 0 0.3rem 0;
            font-size: 1.55rem;
            line-height: 1.25;
        }
        .main .hero p { color: #dbe7ef; margin: 0; font-size: 0.95rem; }

        .main .stMarkdown table {
            width: 100%;
            background: #ffffff;
            border-collapse: collapse;
            border: 1px solid #d5dee7;
        }
        .main .stMarkdown table th {
            background: #1e4d6b;
            color: #f8fafc;
            padding: 0.5rem 0.7rem;
            text-align: left;
            font-weight: 600;
        }
        .main .stMarkdown table td {
            background: #ffffff;
            color: #1e293b;
            padding: 0.45rem 0.7rem;
            border-top: 1px solid #d5dee7;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #d5dee7;
            border-radius: 10px;
            padding: 0.7rem 0.85rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        div[data-testid="stMetricLabel"] { color: #475569; }
        div[data-testid="stMetricValue"] { color: #1e293b; }
        div[data-testid="stMetricDelta"] { color: #334155; }

        div[data-testid="stTabs"] {
            overflow: visible;
        }
        div[data-testid="stTabs"] [data-baseweb="tab-list"] {
            background: transparent;
            gap: 0.5rem 0.55rem;
            flex-wrap: wrap;
            align-items: stretch;
            height: auto !important;
            min-height: 2.75rem;
            overflow: visible !important;
        }
        div[data-testid="stTabs"] button[data-baseweb="tab"] {
            background: #e2eaf0;
            color: #1e293b;
            border: 1px solid #c5d3de;
            border-radius: 8px;
            padding: 0.55rem 0.9rem;
            height: auto !important;
            min-height: 2.5rem;
            line-height: 1.3;
            overflow: visible !important;
            white-space: nowrap;
        }
        div[data-testid="stTabs"] button[data-baseweb="tab"] p,
        div[data-testid="stTabs"] button[data-baseweb="tab"] span {
            color: #1e293b;
            font-size: 0.92rem;
            line-height: 1.3;
            overflow: visible;
        }
        div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
            background: #ffffff;
            border-color: #1e4d6b;
            box-shadow: inset 0 -2px 0 #1e4d6b;
        }
        div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] p,
        div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] span {
            color: #1e4d6b;
            font-weight: 600;
        }

        .main label p, .main [data-testid="stWidgetLabel"] p { color: #1e293b; }
        .main .stSelectbox, .main .stRadio, .main .stSlider,
        .main .stNumberInput, .main .stTextInput { color: #1e293b; }
        .main [data-testid="stDataFrame"] { background: #ffffff; color: #1e293b; }

        .main .stButton > button {
            background-color: #1e4d6b;
            color: #f8fafc;
            border: 1px solid #16384f;
            padding: 0.55rem 1.35rem;
            font-weight: 600;
            border-radius: 8px;
        }
        .main .stButton > button:hover {
            background-color: #16384f;
            color: #f8fafc;
            border-color: #122e40;
        }

        .main [data-testid="stCode"] {
            background: #ffffff;
            border: 1px solid #d5dee7;
            color: #1e293b;
        }
        .main [data-testid="stAlert"] p { color: #1e293b; }
        .main [data-testid="stAlert"] a { color: #1e4d6b; }

        .main div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #ffffff;
            border: 1px solid #d5dee7;
            border-radius: 12px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
            padding: 0.15rem 0.1rem;
        }

        .main .level-list { margin: 0.15rem 0 0.35rem 0; }
        .main .level-item {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            padding: 0.45rem 0;
            border-bottom: 1px solid #e6edf2;
            color: #1e293b;
        }
        .main .level-item:last-child { border-bottom: 0; }
        .main .pill {
            display: inline-block;
            min-width: 4.6rem;
            text-align: center;
            border-radius: 999px;
            padding: 0.18rem 0.55rem;
            font-size: 0.82rem;
            font-weight: 650;
        }
        .main .pill-low { background: #fef3e8; color: #9a3412; border: 1px solid #f0d3b5; }
        .main .pill-medium { background: #e8f0f5; color: #1e4d6b; border: 1px solid #c5d6e2; }
        .main .pill-high { background: #e8f4ee; color: #166534; border: 1px solid #b7d4c2; }

        .main .result-card {
            border-radius: 12px;
            padding: 0.9rem 1.05rem;
            margin: 0.4rem 0 0.75rem 0;
            border: 1px solid #d5dee7;
        }
        .main .result-Low { background: #fef6ee; border-left: 5px solid #b45309; }
        .main .result-Medium { background: #eef4f8; border-left: 5px solid #3b6d8c; }
        .main .result-High { background: #eef6f1; border-left: 5px solid #2a6b4a; }
        .main .result-c0 { background: #eef6f1; border-left: 5px solid #2a6b4a; }
        .main .result-c1 { background: #eef4f8; border-left: 5px solid #1e4d6b; }
        .main .result-card h3 { margin: 0 0 0.25rem 0; color: #1e293b; }
        .main .result-card p { margin: 0; color: #334155; }

        .main .metric-note {
            font-size: 0.86rem;
            color: #475569;
            margin: 0.15rem 0 0.55rem 0;
        }

        section[data-testid="stSidebar"] {
            background-color: #1b3a4b;
            color: #e8eef2;
            --text-color: #e8eef2;
            --background-color: #1b3a4b;
            --secondary-background-color: #163445;
        }
        section[data-testid="stSidebar"] > div { padding-top: 0.85rem; }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] small,
        section[data-testid="stSidebar"] [data-testid="stCaption"] p {
            color: #e8eef2;
        }
        section[data-testid="stSidebar"] .status-box {
            border-radius: 8px;
            padding: 0.65rem 0.75rem;
            margin-top: 0.35rem;
        }
        section[data-testid="stSidebar"] .status-box.ok {
            background: #dcefe3;
            border: 1px solid #8fbf9c;
        }
        section[data-testid="stSidebar"] .status-box.ok,
        section[data-testid="stSidebar"] .status-box.ok p,
        section[data-testid="stSidebar"] .status-box.ok span,
        section[data-testid="stSidebar"] .status-box.ok div {
            color: #14532d;
        }
        section[data-testid="stSidebar"] .status-box.ok .status-title {
            font-weight: 700;
            font-size: 0.92rem;
            color: #14532d;
        }
        section[data-testid="stSidebar"] .status-box.ok .status-file {
            margin-top: 0.22rem;
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 0.82rem;
            color: #1e3a2a;
        }
        section[data-testid="stSidebar"] .status-box.bad {
            background: #fde8e8;
            border: 1px solid #e3b6b6;
        }
        section[data-testid="stSidebar"] .status-box.bad,
        section[data-testid="stSidebar"] .status-box.bad div {
            color: #7f1d1d;
        }

        @media (max-width: 1100px) {
            .block-container { max-width: 100%; }
            div[data-testid="stTabs"] button[data-baseweb="tab"] { padding: 0.5rem 0.75rem; }
        }
        @media (max-width: 900px) {
            .block-container { padding-left: 0.75rem; padding-right: 0.75rem; }
            .main .hero h1 { font-size: 1.32rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def cached_bundle():
    return load_clustering_bundle()


@st.cache_data(show_spinner=False)
def cached_raw():
    status = validate_dataset_file()
    if not status["available"]:
        return None, status
    return load_raw_dataset(), status


@st.cache_data(show_spinner=False)
def cached_baseline_table():
    path = CLASSIFICATION_BASELINE_DIR / "model_comparison.csv"
    if not path.is_file():
        return None
    return pd.read_csv(path)


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("#ffffff")
    ax.figure.set_facecolor("#ffffff")
    ax.tick_params(colors="#1e293b")
    ax.xaxis.label.set_color("#1e293b")
    ax.yaxis.label.set_color("#1e293b")
    ax.title.set_color("#1e293b")
    legend = ax.get_legend()
    if legend is not None:
        legend.get_frame().set_facecolor("#ffffff")
        legend.get_frame().set_edgecolor("#ffffff")
        for text in legend.get_texts():
            text.set_color("#1e293b")


def show_pyplot(fig):
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def show_saved_plot(filename, caption):
    path = CLUSTERING_PLOTS_DIR / filename
    if path.is_file():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.caption(f"Chart not found: `{filename}`. Re-run `scripts/train_engagement_clustering.py`.")


def profile_color(cluster_id):
    return PROFILE_PALETTE[int(cluster_id) % len(PROFILE_PALETTE)]


def page_home(bundle):
    st.markdown(
        """
        <div class="hero">
          <h1>Employee Engagement Profiling and Segmentation</h1>
          <p>Case Study No. 90 · Unsupervised K-Means profiles from observed HR and satisfaction fields</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns((1.35, 1), gap="medium")
    with left:
        with st.container(border=True):
            st.markdown("**Problem**")
            st.write(
                "The IBM HR table has no official Employee Engagement score. "
                "A supervised Low / Medium / High classifier could not beat the "
                "73.81% majority baseline without target leakage. This app groups "
                "employees instead of forcing a 90% accuracy claim."
            )
        with st.container(border=True):
            st.markdown("**Objective**")
            st.write(
                "Discover data-driven employee engagement profiles based on "
                "observed HR and satisfaction characteristics, then assign a "
                "new employee row to the nearest learned cluster."
            )
    with right:
        with st.container(border=True):
            st.markdown("**Selected grouping (from saved model)**")
            if bundle is None:
                st.warning("Clustering pipeline is not on disk.")
            else:
                m = bundle["metrics"]
                st.metric("K (clusters)", bundle["k"])
                st.metric("Silhouette score", f"{m['silhouette']:.3f}")
                st.metric("Davies-Bouldin", f"{m['davies_bouldin']:.3f}")
                st.caption(
                    "These are internal clustering scores, not prediction accuracy."
                )
    st.info(
        "This application is an **analytical / academic prototype**. "
        "Clusters are not a validated psychological instrument and must not "
        "be used for hiring, promotion, or discipline."
    )


def page_dashboard(raw_df, bundle):
    st.header("HR Analytics Dashboard")
    st.caption("Descriptive charts from the loaded IBM HR table, grouped by the fitted K-Means profiles.")
    if raw_df is None or bundle is None:
        st.warning("Dataset or clustering model is missing.")
        return
    labeled = attach_cluster_labels(raw_df, bundle)
    profiles = pd.DataFrame(bundle["profiles"])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Employees", f"{len(labeled):,}")
    m2.metric("Profiles (K)", str(bundle["k"]))
    m3.metric("Silhouette", f"{bundle['metrics']['silhouette']:.3f}")
    ot = 100.0 * labeled["OverTime"].astype(str).str.lower().eq("yes").mean()
    m4.metric("Overtime share", f"{ot:.1f}%")

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        with st.container(border=True):
            st.markdown("**Profile sizes**")
            fig, ax = plt.subplots(figsize=(6.0, 3.6))
            ax.bar(
                profiles["profile_name"],
                profiles["n_employees"],
                color=[profile_color(c) for c in profiles["cluster"]],
            )
            ax.set_ylabel("Employees")
            plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
            style_axes(ax)
            fig.tight_layout()
            show_pyplot(fig)
    with c2:
        with st.container(border=True):
            st.markdown("**Overtime by profile**")
            fig, ax = plt.subplots(figsize=(6.0, 3.6))
            ax.bar(
                profiles["profile_name"],
                profiles["overtime_percent"],
                color=[profile_color(c) for c in profiles["cluster"]],
            )
            ax.set_ylabel("Overtime (%)")
            plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
            style_axes(ax)
            fig.tight_layout()
            show_pyplot(fig)

    with st.container(border=True):
        st.markdown("**Monthly income by profile**")
        fig, ax = plt.subplots(figsize=(8.4, 3.8))
        sns.boxplot(
            data=labeled,
            x="EngagementProfile",
            y="MonthlyIncome",
            hue="ClusterId",
            palette={int(c): profile_color(c) for c in profiles["cluster"]},
            legend=False,
            ax=ax,
        )
        ax.set_xlabel("Profile")
        style_axes(ax)
        fig.tight_layout()
        show_pyplot(fig)

    with st.container(border=True):
        st.markdown("**Department mix by profile**")
        ct = pd.crosstab(labeled["Department"], labeled["EngagementProfile"], normalize="index") * 100
        fig, ax = plt.subplots(figsize=(8.4, 3.8))
        ct.plot(kind="bar", ax=ax, rot=15, color=[profile_color(i) for i in range(len(ct.columns))])
        ax.set_ylabel("Percent of department")
        ax.legend(title="", frameon=False)
        style_axes(ax)
        fig.tight_layout()
        show_pyplot(fig)
    st.caption("Charts describe this sample. They do not show that pay or tenure causes engagement.")


def page_profiles(bundle):
    st.header("Employee Engagement Profiling")
    st.caption(
        "Assign an employee row to the nearest K-Means cluster using the saved "
        "preprocessing + clustering pipeline. This is not a Low/Medium/High accuracy prediction."
    )
    if bundle is None:
        st.warning("Clustering pipeline is not on disk.")
        return

    st.markdown("**Learned profiles**")
    for p in bundle["profiles"]:
        cid = int(p["cluster"])
        name = bundle["names"][str(cid)]
        st.markdown(
            f"""
            <div class="result-card result-c{cid % 2}">
              <h3>{name}</h3>
              <p>{bundle["descriptions"][str(cid)]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    raw_df, _status = cached_raw()
    if raw_df is None:
        st.warning("Cannot build the input form without the HR table.")
        return

    st.subheader("Assign a profile")
    cols = bundle["columns"]
    med = raw_df[cols].median(numeric_only=True)

    def opts(col):
        return sorted(raw_df[col].astype(str).unique().tolist())

    a, b, c = st.columns(3)
    with a:
        age = st.slider("Age", 18, 65, int(med.get("Age", 36)))
        job_level = st.slider("Job level", 1, 5, int(med.get("JobLevel", 2)))
        income = st.number_input(
            "Monthly income",
            min_value=1000,
            max_value=20000,
            value=int(med.get("MonthlyIncome", 5000)),
            step=100,
        )
        distance = st.slider("Distance from home", 1, 30, int(med.get("DistanceFromHome", 8)))
    with b:
        years_co = st.slider("Years at company", 0, 40, int(med.get("YearsAtCompany", 5)))
        years_role = st.slider("Years in current role", 0, 20, int(med.get("YearsInCurrentRole", 2)))
        years_promo = st.slider("Years since last promotion", 0, 15, int(med.get("YearsSinceLastPromotion", 1)))
        years_mgr = st.slider("Years with current manager", 0, 20, int(med.get("YearsWithCurrManager", 2)))
        training = st.slider("Training times last year", 0, 6, int(med.get("TrainingTimesLastYear", 3)))
    with c:
        job_inv = st.slider("Job involvement (1–4)", 1, 4, int(med.get("JobInvolvement", 3)))
        job_sat = st.slider("Job satisfaction (1–4)", 1, 4, int(med.get("JobSatisfaction", 3)))
        env_sat = st.slider("Environment satisfaction (1–4)", 1, 4, int(med.get("EnvironmentSatisfaction", 3)))
        rel_sat = st.slider("Relationship satisfaction (1–4)", 1, 4, int(med.get("RelationshipSatisfaction", 3)))
        wlb = st.slider("Work-life balance (1–4)", 1, 4, int(med.get("WorkLifeBalance", 3)))

    d1, d2, d3, d4 = st.columns(4)
    overtime = d1.selectbox("Overtime", opts("OverTime"))
    job_role = d2.selectbox("Job role", opts("JobRole"))
    travel = d3.selectbox("Business travel", opts("BusinessTravel"))
    marital = d4.selectbox("Marital status", opts("MaritalStatus"))

    row = pd.DataFrame(
        [
            {
                "JobInvolvement": job_inv,
                "JobSatisfaction": job_sat,
                "EnvironmentSatisfaction": env_sat,
                "RelationshipSatisfaction": rel_sat,
                "WorkLifeBalance": wlb,
                "OverTime": overtime,
                "Age": age,
                "MonthlyIncome": income,
                "JobLevel": job_level,
                "YearsAtCompany": years_co,
                "YearsInCurrentRole": years_role,
                "YearsSinceLastPromotion": years_promo,
                "YearsWithCurrManager": years_mgr,
                "TrainingTimesLastYear": training,
                "JobRole": job_role,
                "BusinessTravel": travel,
                "MaritalStatus": marital,
                "DistanceFromHome": distance,
            }
        ]
    )

    if st.button("Assign engagement profile", type="primary"):
        try:
            result = assign_profile(bundle, row)
        except ValueError as exc:
            st.error(str(exc))
            return
        cid = result["cluster"]
        st.markdown(
            f"""
            <div class="result-card result-c{cid % 2}">
              <h3>Assigned profile: {result["profile_name"]}</h3>
              <p>{result["description"]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        k1, k2, k3 = st.columns(3)
        k1.metric("Cluster id", cid)
        k2.metric("Share of table", f"{result['percent']:.1f}%")
        k3.metric("Employees in profile", result["n_employees"])
        st.markdown("**This employee vs cluster averages vs table**")
        st.dataframe(result["comparison"], hide_index=True, width="stretch")
        st.caption(
            "Assignment uses the fitted scaler, one-hot encoder, and K-Means "
            "centroids. It is a nearest-cluster lookup, not a supervised prediction."
        )


def page_cluster_analysis(bundle):
    st.header("Cluster Analysis")
    st.caption("Internal metrics and charts written by the clustering training script.")
    if bundle is None:
        st.warning("Clustering pipeline is not on disk.")
        return
    m = bundle["metrics"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Silhouette score", f"{m['silhouette']:.3f}")
    c2.metric("Davies-Bouldin", f"{m['davies_bouldin']:.3f}")
    c3.metric("Calinski-Harabasz", f"{m['calinski_harabasz']:.1f}")
    st.caption("Do not read these as classification accuracy.")

    k_table = pd.DataFrame(bundle["k_table"])
    st.markdown("**K selection (K-Means, train on all rows)**")
    st.dataframe(
        k_table.style.format(
            {
                "inertia": "{:.0f}",
                "silhouette": "{:.3f}",
                "davies_bouldin": "{:.3f}",
                "calinski_harabasz": "{:.1f}",
                "min_cluster_share": "{:.1%}",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.write(bundle["metadata"]["k_selection_reason"])

    g1, g2 = st.columns(2)
    with g1:
        show_saved_plot("elbow_curve.png", "Elbow curve (inertia vs K)")
        show_saved_plot("cluster_sizes.png", "Cluster sizes")
        show_saved_plot("satisfaction_by_cluster.png", "Satisfaction items by cluster")
        show_saved_plot("overtime_by_cluster.png", "Overtime by cluster")
    with g2:
        show_saved_plot("silhouette_by_k.png", "Silhouette by K")
        show_saved_plot("pca_clusters.png", "PCA of clustered employees")
        show_saved_plot("worklife_by_cluster.png", "Work-life balance by cluster")
        show_saved_plot("cluster_profile_heatmap.png", "Profile heatmap")


def page_performance(bundle, baseline):
    st.header("Model Performance")
    st.markdown("**Clustering (final method)**")
    st.write(
        "Clustering has no ground-truth accuracy. The app reports silhouette, "
        "Davies-Bouldin, and cluster shares from the saved fit."
    )
    if bundle is None:
        st.warning("Clustering pipeline is not on disk.")
    else:
        m = bundle["metrics"]
        st.write(
            f"- Algorithm: K-Means, K={bundle['k']}, `random_state=42`\n"
            f"- Silhouette: **{m['silhouette']:.4f}**\n"
            f"- Davies-Bouldin: **{m['davies_bouldin']:.4f}**\n"
            f"- Calinski-Harabasz: **{m['calinski_harabasz']:.1f}**"
        )
        st.dataframe(pd.DataFrame(bundle["profiles"]), hide_index=True, width="stretch")

    st.markdown("---")
    st.markdown("**Supervised classification baseline (not the final model)**")
    st.write(
        "The first experiment predicted a student-defined `EngagementLevel`. "
        "Leakage-safe models did not beat the 73.81% majority baseline. "
        "Those results are kept in `reports/classification_baseline/`."
    )
    if baseline is None:
        st.caption("Baseline comparison CSV was not found.")
    else:
        show = baseline.copy()
        for col in show.columns:
            if col != "model" and pd.api.types.is_numeric_dtype(show[col]):
                show[col] = show[col].map(lambda x: f"{x:.3f}")
        st.dataframe(show, hide_index=True, width="stretch")
        st.caption(
            "Do not present 90%+ as the IBM result. That figure required putting "
            "the five label-defining items back into X (target leakage)."
        )


def page_drivers(bundle):
    st.header("Feature Importance / Cluster Drivers")
    st.write(
        "For clustering, “importance” is how much a feature’s mean differs "
        "across profiles (range of cluster means divided by the overall "
        "standard deviation). **This is not causation** and not Decision-Tree Gini "
        "importance from the old classifier."
    )
    if bundle is None:
        st.warning("Clustering pipeline is not on disk.")
        return
    drivers = pd.DataFrame(bundle["drivers"])
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    plot = drivers.sort_values("rank")
    ax.barh(plot["feature"][::-1], plot["between_cluster_range_over_std"][::-1], color="#3b6d8c")
    ax.set_xlabel("Between-cluster range / overall SD")
    ax.set_title("Which fields differ most across profiles")
    style_axes(ax)
    fig.tight_layout()
    show_pyplot(fig)
    st.dataframe(drivers, hide_index=True, width="stretch")
    st.caption(
        "On this table the split is driven by tenure, job level, and pay. "
        "The five satisfaction items barely differ between clusters."
    )


def page_about(bundle):
    st.header("About the Project")
    st.markdown(
        """
        **Case Study No. 90 — Employee Engagement Profiling and Segmentation
        Using Machine Learning.**

        The final method is **unsupervised K-Means**. The earlier supervised
        classifier is documented as a limitation, not hidden.

        Clusters are data-driven employee engagement profiles based on observed
        HR and satisfaction characteristics. They are **not** scientifically
        validated psychological engagement categories.
        """
    )
    st.markdown("**How to retrain clustering**")
    st.code("python scripts/train_engagement_clustering.py", language="bash")
    st.markdown("**How to run this app**")
    st.code("streamlit run app.py", language="bash")
    if bundle is not None:
        st.markdown("**Features used in clustering**")
        st.write(", ".join(f"`{c}`" for c in bundle["columns"]))
        st.markdown("**Excluded (no supervised target / no IDs)**")
        st.write(", ".join(f"`{c}`" for c in bundle["metadata"]["forbidden_excluded"]))


def main():
    inject_styles()
    bundle = cached_bundle()
    raw_df, data_status = cached_raw()
    baseline = cached_baseline_table()

    st.sidebar.markdown("### Case Study No. 90")
    st.sidebar.caption("Employee engagement profiling (K-Means)")
    if not data_status["available"]:
        st.sidebar.markdown(
            '<div class="status-box bad"><div class="status-title">Dataset missing</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            '<div class="status-box ok"><div class="status-title">HR table loaded</div></div>',
            unsafe_allow_html=True,
        )
    if bundle is None:
        st.sidebar.markdown(
            '<div class="status-box bad"><div class="status-title">Clustering model missing</div></div>',
            unsafe_allow_html=True,
        )
        st.error(
            "Save `models/engagement_clustering_pipeline.joblib` by running "
            "`python scripts/train_engagement_clustering.py`."
        )
    else:
        st.sidebar.markdown(
            """
            <div class="status-box ok">
              <div class="status-title">Clustering pipeline loaded</div>
              <div class="status-file">engagement_clustering_pipeline.joblib</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    tabs = st.tabs(
        [
            "Home",
            "HR Analytics Dashboard",
            "Employee Engagement Profiles",
            "Cluster Analysis",
            "Model Performance",
            "Feature Importance / Cluster Drivers",
            "About Project",
        ]
    )
    with tabs[0]:
        page_home(bundle)
    with tabs[1]:
        page_dashboard(raw_df, bundle)
    with tabs[2]:
        page_profiles(bundle)
    with tabs[3]:
        page_cluster_analysis(bundle)
    with tabs[4]:
        page_performance(bundle, baseline)
    with tabs[5]:
        page_drivers(bundle)
    with tabs[6]:
        page_about(bundle)


main()
