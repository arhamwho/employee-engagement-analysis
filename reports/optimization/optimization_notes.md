# Leakage-safe optimization results

Target definition unchanged. IBM CSV unchanged. The five label-defining items
(`JobInvolvement`, `JobSatisfaction`, `EnvironmentSatisfaction`,
`RelationshipSatisfaction`, `WorkLifeBalance`) plus `EngagementScore`,
`EngagementLevel`, and `OverallSatisfactionIndex` stayed out of X.

Hyperparameters were chosen with **StratifiedKFold (5 folds) on the training
fold only**. Each CV-selected candidate was scored **once** on the held-out
test set (294 rows). Test scores were not used to retune.

Full numeric table: `reports/optimization_results.csv`

Pipeline audit: `reports/optimization/pipeline_audit.md`

Derived features: `reports/optimization/feature_engineering.md`

---

## 1. Pipeline verification

| Check | Result |
| --- | --- |
| Split | `test_size=0.20`, `random_state=42`, stratified |
| Sizes | 1,176 train / 294 test; no index overlap |
| Missing values | 0 cells; no imputation |
| Duplicate rows | 0; no rows copied or dropped |
| Numeric preprocess | `StandardScaler`, fit on train only |
| Categorical preprocess | `OneHotEncoder(handle_unknown='ignore')`, fit on train only |
| Leakage in X | none |
| Test set during HPO | unused |

Train class counts: Low 80, Medium 866, High 230.
Test class counts: Low 20, Medium 217, High 57.

---

## 2. Feature engineering

All 10 requested derived features were created because every source column exists.
They are row-wise ratios (denominator + 1). They do not use other employees or
the test fold. They did **not** produce a model that beats the majority baseline
on accuracy.

---

## 3–8. What was tuned (train CV only)

- **Decision Tree:** full GridSearchCV over `class_weight`, `max_depth`,
  `min_samples_split`, `min_samples_leaf`, `max_features`, `criterion`,
  `ccp_alpha` (8,100 fits per scoring).
- **Random Forest / Extra Trees:** RandomizedSearchCV (`n_iter=36`) over the
  specified space (`n_estimators` 200/400/800, depths, split/leaf, `max_features`,
  `class_weight`). A full RF factorial is 1,215 cells × 5 folds × 2 scorings;
  the search sampled that same space.
- **GradientBoosting** and **HistGradientBoosting:** compact grids.
- **SVM:** linear and rbf, `C` / `gamma` / `class_weight`.
- Separate searches for **accuracy** and **macro F1**.

Every candidate below reports CV accuracy mean/std and CV macro F1 mean/std.

---

## 9–10. Test evaluation (one shot) and baselines

**Majority baseline (always Medium):** test accuracy **73.81%**, macro F1 **0.2831**,
balanced accuracy **0.3333**. This row is not hidden.

Accuracy-optimized trees/forests/SVM with `class_weight=None` collapse to the
same majority predictor: **73.81% accuracy and 0.2831 macro F1**.

### Sorted by test accuracy

| Model | CV Accuracy | CV Macro F1 | Test Accuracy | Test Macro F1 | Balanced Acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| Majority baseline | 0.7364 | 0.2827 | **0.7381** | 0.2831 | 0.3333 |
| Logistic Regression (opt accuracy) | 0.7245 | 0.2853 | **0.7381** | 0.2944 | 0.3376 |
| Decision Tree (opt accuracy) | 0.7364 | 0.2827 | **0.7381** | 0.2831 | 0.3333 |
| Random Forest (opt accuracy) | 0.7364 | 0.2882 | **0.7381** | 0.2831 | 0.3333 |
| Extra Trees (opt accuracy) | 0.7364 | 0.2827 | **0.7381** | 0.2831 | 0.3333 |
| Gradient Boosting (opt accuracy) | 0.7330 | 0.2900 | **0.7381** | 0.2831 | 0.3333 |
| SVM (opt accuracy) | 0.7364 | 0.2827 | **0.7381** | 0.2831 | 0.3333 |
| HistGradientBoosting (opt accuracy) | 0.7245 | 0.2985 | 0.7347 | 0.3035 | 0.3404 |
| Gradient Boosting (opt F1) | 0.7075 | 0.3253 | 0.6939 | 0.2914 | 0.3220 |
| Decision Tree (opt F1) | 0.6513 | **0.3786** | 0.6190 | 0.3419 | 0.3421 |
| Extra Trees (opt F1) | 0.6497 | 0.3530 | 0.6190 | 0.3276 | 0.3292 |
| SVM (opt F1) | 0.6250 | 0.3261 | 0.6122 | 0.3218 | 0.3282 |
| Random Forest (opt F1) | 0.5790 | 0.3580 | 0.5578 | **0.3741** | 0.3900 |
| HistGradientBoosting (opt F1) | 0.5731 | 0.3353 | 0.5204 | 0.3227 | 0.3257 |
| Logistic Regression (opt F1) | 0.3869 | 0.3169 | 0.3673 | 0.3032 | 0.3818 |

*Existing Streamlit Decision Tree (previous experiment, same split, no extra
derived features): test accuracy **0.6531**, macro F1 **0.3872**. That row was
not part of this HPO and was not used to pick a new winner.*

### Sorted by test macro F1 (HPO candidates only)

Highest: **Random Forest (opt F1)** 0.3741, then Decision Tree (opt F1) 0.3419.
Both are **below** the current Streamlit Decision Tree (0.3872).

---

## 11. Model selection (CV first)

**Accuracy specialist (highest CV accuracy):** Random Forest (opt accuracy).
CV accuracy 0.7364 (tied with the majority dummy). Test accuracy **73.81%**,
macro F1 **0.2831**. It is a majority clone. Cross-validation **does not**
support a real accuracy gain.

**Macro-F1 specialist (highest CV macro F1):** Decision Tree (opt F1).
CV macro F1 0.3786. Test accuracy **61.90%**, macro F1 **0.3419**.

**Highest one-shot test macro F1 among new candidates:** Random Forest
(opt F1) at **0.3741**, with test accuracy **55.78%**.

### Trade-off

Medium is 73.67% of the table. Optimizing accuracy pushes every flexible model
toward “always Medium.” That matches 73.81% accuracy and **destroys** Low/High
recall (balanced accuracy 0.33). Optimizing macro F1 recovers some minority
predictions and **lowers** accuracy into the 37–62% range.

The accuracy-optimized models do **not** beat the majority baseline; they
**are** the majority baseline.

`models/final_model.joblib` was **not** replaced.

---

## 12. Feature importance

**Feature importance indicates contribution to model predictions and does not establish causation.**

Accuracy-oriented Random Forest still ranks pay/age/tenure ratios highly, but
the fitted model predicts like a dummy, so those importances are not evidence
of a useful engagement signal.

Macro-F1 Decision Tree (this HPO), top 15 Gini:

1. DailyRate
2. ExperienceAtCompanyRatio
3. CareerProgressionRate
4. Age
5. CompanyTenureRatio
6. TotalWorkingYears
7. MonthlyRate
8. IncomePerYearExperience
9. DistanceFromHome
10. HourlyRate
11. NumCompaniesWorked
12. Attrition_No
13. IncomePerCompanyYear
14. PromotionGap
15. CurrentRoleRatio

CSV: `reports/optimization/feature_importance_macro_f1_model.csv`

The **current Streamlit tree** (no extra ratios) still ranks MonthlyRate,
HourlyRate, DailyRate, MonthlyIncome, and Age first
(`reports/feature_importance.csv`).

---

## 13. Artifacts

Saved:

- `reports/optimization_results.csv`
- `reports/optimization/` (audit, notes, CMs, importances, classification reports)
- Candidate pipelines under `models/optimization/` (not wired to Streamlit)

**Not** updated, because nothing beat the legitimate baseline in a useful way:

- `models/final_model.joblib`
- `models/preprocessor.joblib`
- `reports/model_comparison.csv`

---

## Success criteria (actual numbers)

1. **Beat 73.81% majority accuracy?** No.
2. **Reach 80%?** No.
3. **Reach 85%?** No.
4. **Reach 90%?** No.

**90% cannot be achieved on this leakage-safe IBM feature set.** Prior
investigation reached 90%+ only by putting the five target-defining items back
into X (target leakage / demonstration only). This optimization did not use
those columns and did not invent scores.

### Headline numbers

| Question | Answer |
| --- | --- |
| Best accuracy model (CV) | Random Forest (opt accuracy) — majority clone |
| Best Macro F1 model (CV) | Decision Tree (opt F1) |
| Best test accuracy (HPO) | 73.81% (tied with majority; several models) |
| Best test Macro F1 (HPO) | 0.3741 (Random Forest opt F1) |
| Majority baseline | accuracy 73.81%, macro F1 0.2831 |
| Beats the baseline? | No on accuracy. Yes on macro F1 vs dummy (0.37 vs 0.28), but **not** better than the current Streamlit tree (0.3872) |
| 80 / 85 / 90%? | No / No / No |
| Most important features | DailyRate and compensation/tenure fields; **not causation** |
| Recommended Streamlit model | **Keep the current Decision Tree** (`class_weight='balanced'`, test acc 65.31%, macro F1 0.3872) |

A higher-accuracy Streamlit model would just predict Medium for almost every
employee. That is not a better engagement classifier.
