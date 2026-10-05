# Employee Engagement Profiling and Segmentation Using Machine Learning

**Case Study No. 90**

The IBM HR table has **no official Employee Engagement score**. Supervised
classification of a student-defined Low / Medium / High label could not beat
the **73.81% majority baseline** without target leakage. The **final method**
is unsupervised **K-Means profiling**.

The Streamlit app assigns an employee to the nearest learned cluster. It does
**not** display a fake clustering accuracy. It reports silhouette, Davies-Bouldin,
and cluster shares.

Clusters are **data-driven employee engagement profiles based on observed HR
and satisfaction characteristics**. They are not validated psychological types.

## Current status

- **Final model:** K-Means (`K=2`) in `models/engagement_clustering_pipeline.joblib`
- **Silhouette:** 0.175 · **Davies-Bouldin:** 2.155 (computed, not invented)
- **Profiles:** Higher Pay / Longer Tenure (34%); Lower Pay / Shorter Tenure (66%)
- **Supervised experiment:** preserved under `reports/classification_baseline/`
- **Report:** `reports/clustering/clustering_report.md`

Do not continue trying to force 80%, 85%, or 90% classification accuracy on
this leakage-safe IBM feature set.

## Dataset

The loader reads a **copy** of the CSV. The original file is not modified.

- **Folder:** `employee-engagement-analysis/data/`
- **Filename used now:** `IBM_HR_Attrition_Enhanced.csv`

Alternate names still accepted if you swap files later: `ibm_hr_employee_attrition.csv`, `WA_Fn-UseC_-HR-Employee-Attrition.csv`.

## Student-defined target (not an official scale)

The IBM HR file has **no** Employee Engagement column. This project therefore defines:

- `EngagementScore` = mean of `JobInvolvement`, `JobSatisfaction`, `EnvironmentSatisfaction`, `RelationshipSatisfaction`, `WorkLifeBalance`
- `EngagementLevel`: 1.00–2.00 Low, 2.01–3.00 Medium, 3.01–4.00 High

These cut-points are for this case study only. They are **not** an HR standard and **not** a validated clinical or scientific engagement instrument. The original five columns are kept. The CSV in `data/` is not modified.

## Classification (test-set scores)

One stratified split: `test_size=0.20`, `random_state=42` (1,176 train / 294 test). The same split and the same fitted preprocessor are used for every model. Imbalance is handled with `class_weight="balanced"` (no duplicate-row oversampling).

The five survey items that **define** `EngagementLevel` are excluded from X so the models cannot reconstruct the label.

Precision, recall, and F1 in the table below are **macro-averaged** (each of Low / Medium / High counts equally). That is the useful default for this imbalanced 3-class problem. Accuracy can look high if a model mostly predicts Medium.

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
| --- | ---: | ---: | ---: | ---: |
| Majority baseline (reference) | 0.738 | 0.246 | 0.333 | 0.283 |
| Logistic Regression | 0.367 | 0.354 | 0.380 | 0.304 |
| Decision Tree | 0.653 | 0.394 | 0.383 | 0.387 |
| Random Forest | 0.738 | 0.246 | 0.333 | 0.283 |
| Support Vector Machine | 0.486 | 0.360 | 0.375 | 0.348 |

These values were computed with scikit-learn on the held-out test set. Re-run the notebook to refresh them. Full reports: `reports/model_comparison.csv` and `reports/modeling_notes.md`.

**Ranking for this split used macro F1, not accuracy.** This Decision Tree is
**not** the Streamlit model anymore. Accuracy of 65.31% is **below** the 73.81%
majority baseline. Do not present it as a high-performing 90% model.

90%+ accuracy appeared only when the five survey items that **define**
`EngagementLevel` were put back into X (**target leakage**).

The Streamlit app now loads **K-Means clustering**. Classification files remain
under `reports/classification_baseline/` and the original Joblib files under
`models/`.

### Model-derived feature importance (old classifier)

The Decision Tree's sklearn Gini importances (`feature_importances_`) rank features **associated with model predictions**. They do **not** prove that a variable causes engagement. In this fitted tree the highest model-derived importances include `MonthlyRate`, `HourlyRate`, `DailyRate`, `MonthlyIncome`, and `Age`. Chart and notes: `reports/analysis_plots/feature_importance_top15.png` and `reports/feature_importance_notes.md`.

## Workflow

1. Keep the IBM HR CSV in `data/`.
2. Train clustering: `python scripts/train_engagement_clustering.py`
3. Run the app: `streamlit run app.py`
4. The supervised notebook remains available as the documented baseline.

## Project structure

```text
employee-engagement-analysis/
├── data/IBM_HR_Attrition_Enhanced.csv
├── notebooks/
├── src/clustering.py
├── models/engagement_clustering_pipeline.joblib
├── reports/clustering/
├── reports/classification_baseline/
├── app.py
└── README.md
```

## Tech stack

| Package | Role |
| --- | --- |
| Pandas | Load and clean tabular HR data |
| NumPy | Numerical arrays and calculations |
| Scikit-learn | Classification models and evaluation |
| Matplotlib | Charts |
| Seaborn | Statistical plots |
| Joblib | Save and load trained models |
| Streamlit | Simple web interface |
| Jupyter | Interactive analysis notebook |

## How to run

### 1. Open a terminal

Move into the project folder:

```bash
cd employee-engagement-analysis
```

### 2. Create a virtual environment (recommended)

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install packages

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

Opens a six-tab demo: Home, HR Analytics Dashboard, Prediction, Model Performance, Feature Importance, and About. The app **loads** `models/final_model.joblib` and `reports/model_comparison.csv`; it does **not** retrain on refresh.

### 5. Open the Jupyter notebook

```bash
jupyter notebook notebooks/employee_engagement_analysis.ipynb
```

Alternatively, open the same notebook from VS Code / Cursor after selecting the project virtual environment as the kernel. Run all cells to refresh EDA, target plots, and the classification experiment.

### 6. Independent survey model (Experiment 3)

The IBM Streamlit demo is **not** replaced by this experiment. Do not claim 90%+ accuracy for the survey model until real responses exist and sklearn has scored a held-out test set.

#### How to collect survey data

1. Obtain ethics / instructor approval and informed consent.
2. Use the template columns exactly (names are case-sensitive).
3. One row per consenting employee. Do **not** invent answers.
4. Engagement items are independent 1–5 Likert questions. They are **not** copied from IBM `JobSatisfaction` / `JobInvolvement` fields.

#### Where to put the CSV

Copy the template and add rows:

```bash
cp data/survey/employee_engagement_survey_template.csv data/survey/employee_engagement_survey.csv
```

Place the filled file at:

`data/survey/employee_engagement_survey.csv`

The empty template is **not** treated as collected data.

#### Required column names

**Workplace / HR predictors (used in X):**
`Age`, `Department`, `JobRole`, `JobLevel`, `YearsExperience`, `YearsAtCompany`, `MonthlyIncome`, `Overtime`, `WorkHoursPerWeek`, `TrainingOpportunities`, `CareerGrowth`, `ManagerSupport`, `Recognition`, `WorkLifeBalance`, `WorkArrangement`

Suggested encodings (document whatever you actually collect):

- `Overtime`: Yes / No
- `WorkArrangement`: Onsite / Hybrid / Remote
- `TrainingOpportunities`, `CareerGrowth`, `ManagerSupport`, `Recognition`, `WorkLifeBalance`: 1–5
- `JobLevel`: integer job level used by your organization

**Independent engagement questions (1–5 Likert, used only to build the target):**
`Motivation`, `ValuedByOrganization`, `OrganizationalConnection`, `ExtraEffort`, `RecommendOrganization`, `IntentionToStay`

#### How EngagementScore is calculated

```text
EngagementScore = mean(
  Motivation,
  ValuedByOrganization,
  OrganizationalConnection,
  ExtraEffort,
  RecommendOrganization,
  IntentionToStay
)
```

**Project-defined levels** (not a validated scale):

- **Low:** 1.00–2.33
- **Medium:** 2.34–3.67
- **High:** 3.68–5.00

(`pd.cut` bins `[1.00, 2.33, 3.67, 5.00]`, right-closed, `include_lowest=True`.)

#### Which columns are predictors / excluded

- **Predictors:** the workplace/HR list above.
- **Excluded from the final predictive model (leakage control):** the six engagement questions, plus constructed `EngagementScore` and `EngagementLevel`.

#### How to train and evaluate

```bash
jupyter notebook notebooks/survey_engagement_model.ipynb
```

Run all cells. If the CSV has no real rows, training is skipped and **no metrics are invented**. With enough labeled rows (at least 30, and at least two rows per class that appears), the notebook:

- uses a stratified train/test split (`test_size=0.20`, `random_state=42`)
- runs stratified cross-validation / RandomizedSearchCV
- ranks models by **macro F1**
- reports accuracy, macro precision, macro recall, macro F1, balanced accuracy, a majority baseline, confusion matrices, feature importance, and error analysis
- writes artifacts under `models/survey/` and `reports/survey_model/`

## Three experiments (do not mix their claims)

| Experiment | Data | Features | How to talk about it |
| --- | --- | --- | --- |
| 1 | IBM HR classification (baseline) | Leakage-safe HR fields | Archived. DT accuracy **65.31%**, below majority **73.81%**. Macro F1 **0.387**. Not a 90% model. Not the Streamlit model. |
| 2 | IBM HR classification with label items in X | Five items that **define** the IBM label | **TARGET LEAKAGE / DEMONSTRATION ONLY.** Not used in Streamlit. |
| Clustering (current IBM app) | IBM HR, unsupervised | Satisfaction items + HR fields; **no** EngagementLevel target | K-Means **K=2**, silhouette **0.175**, Davies-Bouldin **2.155**. No clustering accuracy %. |
| Independent survey | Survey CSV when collected | Workplace predictors only | Still has **no responses**. Do not invent survey accuracy. |

## What is intentionally missing

- No invented metrics: scores come from `sklearn` on the test split
- Survey items used to **build** `EngagementLevel` are not used as features

## Academic note

Use only data you are allowed to use for coursework (public datasets with a clear license, or data provided by your instructor). Do not upload confidential employee records from a workplace unless you have explicit permission.
