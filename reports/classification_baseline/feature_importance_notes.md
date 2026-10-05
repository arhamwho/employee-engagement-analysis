## Model-derived feature importance (Decision Tree)

The selected final model is a **Decision Tree**. Importance here is sklearn's mean decrease in **Gini impurity** (`feature_importances_`) on the fitted tree. These values describe features **associated with model predictions** in this tree. They are **not** proof that a variable causes employee engagement.

- Pipeline file: `/Users/arham/ml final project/employee-engagement-analysis/models/final_model.joblib`
- Transformed features after scaling / one-hot encoding: **61**
- Features with Gini importance exactly 0: **9**
- The top 15 features account for **80.3%** of total model-derived importance (importances sum to 1).

### Most influential predictive features (top 15)

1. `MonthlyRate` — Gini importance 0.1039 (10.4% of total)
2. `HourlyRate` — Gini importance 0.0898 (9.0% of total)
3. `DailyRate` — Gini importance 0.0791 (7.9% of total)
4. `MonthlyIncome` — Gini importance 0.0727 (7.3% of total)
5. `Age` — Gini importance 0.0699 (7.0% of total)
6. `YearsInCurrentRole` — Gini importance 0.0535 (5.3% of total)
7. `PercentSalaryHike` — Gini importance 0.0527 (5.3% of total)
8. `TrainingTimesLastYear` — Gini importance 0.0501 (5.0% of total)
9. `TotalWorkingYears` — Gini importance 0.0496 (5.0% of total)
10. `NumCompaniesWorked` — Gini importance 0.0415 (4.2% of total)
11. `Attrition_No` — Gini importance 0.0351 (3.5% of total)
12. `YearsSinceLastPromotion` — Gini importance 0.0350 (3.5% of total)
13. `DistanceFromHome` — Gini importance 0.0312 (3.1% of total)
14. `YearsWithCurrManager` — Gini importance 0.0210 (2.1% of total)
15. `YearsAtCompany` — Gini importance 0.0182 (1.8% of total)

### Short interpretation (from these numbers only)

The single most important predictive feature in this tree is `MonthlyRate` (Gini importance 0.1039).
Next are `HourlyRate` (0.0898) and `DailyRate` (0.0791).
Several of the highest ranks are pay- or rate-related columns (`MonthlyRate`, `HourlyRate`, `DailyRate`, `MonthlyIncome`). That means the tree **used those fields when splitting**, not that pay causes engagement. Tenure-style fields (`YearsInCurrentRole`, `TotalWorkingYears`, `TrainingTimesLastYear`, and similar) also appear among the top ranks as important predictive features.

A deep tree can assign model-derived importance to columns that help it fit the training sample even when those columns are only weakly related to engagement in simple summaries. Read this chart as an explanation of **this classifier**, not as an HR causal ranking.

One-hot names (for example `Attrition_No`) are dummy columns created by the preprocessor. Importance on a dummy is still only model-derived importance for that encoded feature.
