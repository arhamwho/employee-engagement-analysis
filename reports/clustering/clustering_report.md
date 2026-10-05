# Employee Engagement Profiling and Segmentation

Case Study No. 90 — clustering report.

Clusters are **data-driven employee engagement profiles based on observed HR
and satisfaction characteristics**. They are not scientifically validated
psychological engagement categories.

---

## 1. Problem definition

The IBM HR Analytics Employee Attrition table records workplace conditions,
pay, tenure, and several 1–4 satisfaction items. It does **not** contain an
official Employee Engagement score. The assignment allows **grouping** as an
acceptable machine-learning output. This experiment groups employees into
interpretable profiles instead of predicting a 3-class label.

---

## 2. Why supervised classification was inadequate

The first approach built a student-defined `EngagementLevel` (mean of five
1–4 items, then Low / Medium / High) and trained classifiers on other HR
fields.

Preserved results (held-out test set, leakage-safe features):

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Majority baseline (always Medium) | 73.81% | 0.283 |
| Decision Tree (`class_weight=balanced`) | 65.31% | 0.387 |
| Random Forest | 73.81% | 0.283 |
| Logistic Regression | 36.73% | 0.304 |
| SVM | 48.64% | 0.348 |

Accuracy-oriented models matched the majority dummy. Hyperparameter search,
derived ratios, Extra Trees, and Gradient Boosting did not beat 73.81%
without putting the five label-defining items back into X (target leakage).
That experiment is archived under `reports/classification_baseline/`.

This project does **not** continue chasing 80%, 85%, or 90% classification
accuracy.

---

## 3. Why clustering is appropriate

Grouping does not need a ground-truth engagement class. K-Means can find
recurring combinations of satisfaction items and HR attributes. The output
is a **profile assignment**, not a claim that the model “knows” engagement
better than 90% of the time. There is no clustering accuracy percentage in
the app.

---

## 4. Dataset

- File: `data/IBM_HR_Attrition_Enhanced.csv` (read-only copy)
- Rows: 1,470 employees
- Source family: IBM HR Analytics Employee Attrition & Performance
- The original CSV is not modified

---

## 5. Variables

Used (all present in the table):

JobInvolvement, JobSatisfaction, EnvironmentSatisfaction,
RelationshipSatisfaction, WorkLifeBalance, OverTime, Age, MonthlyIncome,
JobLevel, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion,
YearsWithCurrManager, TrainingTimesLastYear, JobRole, BusinessTravel,
MaritalStatus, DistanceFromHome.

**Not used:** EngagementLevel, EngagementScore, OverallSatisfactionIndex,
EmployeeNumber, EmployeeCount, Over18, StandardHours.

Using the five satisfaction items here is **not** target leakage: there is
no supervised target. They are observed inputs to grouping.

---

## 6. Data cleaning

- Missing value cells: **0** — no imputation
- Duplicate rows: **0** — no rows dropped or copied
- IDs and constant columns excluded as listed above

---

## 7. Preprocessing

Sklearn `ColumnTransformer` inside a `Pipeline`:

- Numeric columns: `StandardScaler`
- Categorical columns (`OverTime`, `JobRole`, `BusinessTravel`,
  `MaritalStatus`): `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`
- Then `KMeans(n_init=10, random_state=42)`

The same fitted pipeline is saved to
`models/engagement_clustering_pipeline.joblib` and used by Streamlit, so a
new employee row is transformed exactly as the training table was.

---

## 8. K-Means methodology

K-Means assigns each employee to the nearest centroid in the scaled /
one-hot space. All 1,470 rows are used (unsupervised profiling; no
classification train/test split). Hierarchical (ward) clustering and DBSCAN
were run on the same transformed matrix for comparison.

---

## 9. Choosing K

K was tested from 2 through 6.

| K | Inertia | Silhouette | Davies-Bouldin | Calinski-Harabasz | Smallest cluster |
| ---: | ---: | ---: | ---: | ---: | ---: |
| **2** | 20087 | **0.1750** | **2.155** | **288.0** | 33.9% (499) |
| 3 | 18786 | 0.1180 | 2.487 | 204.7 | 14.4% |
| 4 | 17769 | 0.1219 | 2.290 | 172.1 | 9.6% |
| 5 | 17109 | 0.0821 | 2.644 | 148.1 | 9.3% |
| 6 | 16621 | 0.0749 | 2.633 | 130.5 | 8.5% |

**Selected K = 2.** It has the highest silhouette, the lowest Davies-Bouldin,
and the highest Calinski-Harabasz in this range. K=3–6 were not chosen to
make the story look more like Low/Medium/High engagement. The elbow
continues to fall, but the internal validity scores worsen.

Agglomerative clustering also peaked at K=2 (silhouette 0.144), below
K-Means.

---

## 10. Evaluation metrics

Final K-Means (K=2) on the fitted space:

- **Silhouette score:** 0.175
- **Davies-Bouldin index:** 2.155
- **Calinski-Harabasz score:** 288.0

Silhouette 0.175 is a **weak-to-moderate** structure. Overlap is expected
after one-hot encoding job role and mixing satisfaction with career-stage
variables. The score is reported honestly; it is not an accuracy.

**DBSCAN** was not appropriate here. Probes around the median 10-NN distance
found either all noise, or a single cluster plus noise. Density clustering
does not replace K-Means on this table.

---

## 11. Cluster profiles

Names come from the statistics that actually differ. The five satisfaction
item means are almost the same in both groups (~2.72 vs ~2.75), so the
clusters are **not** named “high engagement” vs “low engagement.”

| | Higher Pay / Longer Tenure | Lower Pay / Shorter Tenure | Table |
| --- | ---: | ---: | ---: |
| Cluster id | 0 | 1 | — |
| Employees | 499 | 971 | 1,470 |
| Share | 33.95% | 66.05% | 100% |
| Mean JobSatisfaction | 2.71 | 2.74 | 2.73 |
| Mean JobInvolvement | 2.76 | 2.71 | 2.73 |
| Mean EnvironmentSatisfaction | 2.77 | 2.70 | 2.72 |
| Mean RelationshipSatisfaction | 2.73 | 2.70 | 2.71 |
| Mean WorkLifeBalance | 2.79 | 2.75 | 2.76 |
| Overtime % | 26.7% | 29.1% | 28.3% |
| Mean monthly income | 10,609 | 4,393 | 6,503 |
| Mean age | 41.9 | 34.4 | 36.9 |
| Mean years at company | 12.8 | 4.1 | 7.0 |
| Mean years in current role | 7.7 | 2.4 | 4.2 |
| Mean years since promotion | 4.7 | 0.9 | 2.2 |
| Mean job level | 3.08 | 1.54 | 2.06 |

**Higher Pay / Longer Tenure Profile:** later-career employees, higher job
level and pay, longer time in role and with the current manager.

**Lower Pay / Shorter Tenure Profile:** earlier-career employees, lower pay
and job level, shorter tenure. Slightly higher overtime share (29% vs 27%),
not a large gap.

---

## 12. EDA findings

Features that stretch most across clusters (range of cluster means / overall
SD): YearsInCurrentRole, YearsAtCompany, JobLevel, YearsWithCurrManager,
MonthlyIncome, YearsSinceLastPromotion, Age.

Satisfaction items and overtime barely separate the groups. Distance from
home is essentially the same. PCA of the transformed space: PC1 24.4% and
PC2 8.8% of variance — a 2D picture is incomplete but useful for display.

Charts: `reports/clustering/plots/`.

---

## 13. Limitations

- Internal metrics are modest; employees are not cleanly separated.
- K-Means assumes spherical clusters in Euclidean space after scaling.
- One-hot job roles increase dimensionality and can weaken silhouette.
- Satisfaction items were included but did not drive the split; career
  stage did. That is a finding, not a failure to “find engagement types.”
- No hold-out “clustering accuracy” exists. Silhouette is computed on the
  same rows used to fit the centroids.
- Profile names are descriptive labels for this table only.

---

## 14. Ethical considerations

This is an academic prototype. It must not be used for hiring, promotion,
discipline, or surveillance. Cluster assignment can reflect pay and tenure
structures that correlate with protected attributes (for example age). The
app shows group averages, not a judgment of a person. Do not treat a
profile name as a psychological diagnosis.

---

## 15. Conclusion

Supervised classification of a student-defined engagement label could not
beat a 73.81% majority dummy without leakage. The final method is
**unsupervised profiling**.

K-Means with **K=2** produces two data-driven employee engagement profiles
based on observed HR and satisfaction characteristics:

1. Higher Pay / Longer Tenure Profile (34%)
2. Lower Pay / Shorter Tenure Profile (66%)

Those groups differ mainly in career stage and compensation, not in the
five workplace satisfaction items. The project reports silhouette 0.175 and
Davies-Bouldin 2.155, not a fake prediction accuracy.
