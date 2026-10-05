## EDA Findings

This section is generated automatically from the loaded DataFrame. Statements below are **descriptive**. A correlation or group-mean difference is an association in this sample, not proof that one variable causes another.

### Data shape and quality
- The working table has **1470 rows** and **39 columns**.
- Missing cell count: **0**.
- Duplicate rows: **0**.
- Constant columns (one unique value, little analytic value): `EmployeeCount`, `Over18`, `StandardHours`.
- Identifier-like columns (unique on every row): `EmployeeNumber`.
- Pandas dtypes: **27** numeric columns and **12** object/category/bool columns. Some numeric columns are still ordinal survey scores (for example JobSatisfaction on a 1–4 scale).

### Composite engagement field
- `OverallSatisfactionIndex` matches the row-wise mean of `JobSatisfaction`, `EnvironmentSatisfaction`, `RelationshipSatisfaction`, `WorkLifeBalance` within a numerical tolerance of 0.0. High Pearson correlations between the index and those four inputs are expected because the index is built from them. They should not be treated as four independent discoveries.

### Numerical distributions
- `Age` ranges from 18 to 60 (mean 36.92, median 36.00). Skewness is 0.413 (mildly right-skewed).
- `MonthlyIncome` is right-skewed (skewness 1.370). Median 4919, mean 6503, min 1009, max 19999. IQR rule flags **114** high-income points (7.76% of rows). Those values can be real senior-pay observations, not data errors.
- `YearsAtCompany` is right-skewed (skewness 1.765). Median 5.00 years, mean 7.01, max 40. IQR rule flags **104** long-tenure rows.

### Engagement-related score distributions
- `JobSatisfaction`: mean 2.73, most common score **4** (n=459 of 1470).
- `JobInvolvement`: mean 2.73, most common score **3** (n=868 of 1470).
- `EnvironmentSatisfaction`: mean 2.72, most common score **3** (n=453 of 1470).
- `RelationshipSatisfaction`: mean 2.71, most common score **3** (n=459 of 1470).
- `WorkLifeBalance`: mean 2.76, most common score **3** (n=893 of 1470).
- `OverallSatisfactionIndex`: mean 2.73, median 2.75, min 1.00, max 4.00.
- Pearson r between `JobInvolvement` and `OverallSatisfactionIndex` is -0.003. In this table that association is essentially none, so involvement is not interchangeable with the composite satisfaction index.

### Categorical workforce mix
- `Department` counts: Research & Development n=961, Sales n=446, Human Resources n=63.
- `JobRole` has 9 roles. Largest: Sales Executive (n=326); smallest: Human Resources (n=52).
- `OverTime`: Yes n=416 (28.3%), No n=1054.
- `Attrition`: Yes n=237 (16.1%), No n=1233. Attrition is a related HR outcome in this file; it is not the engagement class label by itself.

### Outliers (IQR screening)
- IQR flags are a spread check. Discrete scales with a small IQR (for example training counts) can produce many flags even when values are valid survey or tenure numbers.
- `TrainingTimesLastYear`: 238 flagged (16.19% of non-missing values).
- `MonthlyIncome`: 114 flagged (7.76% of non-missing values).
- `YearsSinceLastPromotion`: 107 flagged (7.28% of non-missing values).
- `YearsAtCompany`: 104 flagged (7.07% of non-missing values).
- `StockOptionLevel`: 85 flagged (5.78% of non-missing values).
- `TotalWorkingYears`: 63 flagged (4.29% of non-missing values).
- `NumCompaniesWorked`: 52 flagged (3.54% of non-missing values).
- `OverallSatisfactionIndex`: 36 flagged (2.45% of non-missing values).

### Numerical correlations (not causal)
- Largest absolute Pearson correlations among numeric fields (excluding constant and identifier columns):
  - `JobLevel` and `MonthlyIncome`: r = 0.950
  - `JobLevel` and `TotalWorkingYears`: r = 0.782
  - `PercentSalaryHike` and `PerformanceRating`: r = 0.774
  - `MonthlyIncome` and `TotalWorkingYears`: r = 0.773
  - `YearsAtCompany` and `YearsWithCurrManager`: r = 0.769
  - `YearsAtCompany` and `YearsInCurrentRole`: r = 0.759
  - `YearsInCurrentRole` and `YearsWithCurrManager`: r = 0.714
  - `Age` and `TotalWorkingYears`: r = 0.680
- Several of those pairs are structurally related (job level with income, tenure fields with each other). A large r does not mean one variable drives engagement.
- After setting aside the composite's own input fields, the next strongest numeric association with `OverallSatisfactionIndex` in this table is `HourlyRate` (r = -0.067). Interpret the size of r, not a causal story.
- `MonthlyIncome` vs `OverallSatisfactionIndex`: r = 0.017. In this sample, income and the composite index barely move together.

### HR variables compared with engagement-related scores
- Mean `OverallSatisfactionIndex` by `OverTime`: No mean=2.709, Yes mean=2.785. This is a group average only.
- Employees with `Attrition` = Yes have lower mean engagement-related scores in this table than employees with Attrition = No:
  - `JobSatisfaction`: No 2.779, Yes 2.468
  - `JobInvolvement`: No 2.770, Yes 2.519
  - `EnvironmentSatisfaction`: No 2.771, Yes 2.464
  - `RelationshipSatisfaction`: No 2.734, Yes 2.599
  - `WorkLifeBalance`: No 2.781, Yes 2.658
  - `OverallSatisfactionIndex`: No 2.766, Yes 2.547
- That pattern is consistent with engagement and leaving being **associated** here. It does not by itself prove that low scores caused attrition.
- Mean `OverallSatisfactionIndex` by `Department`: Research & Development 2.726, Sales 2.735, Human Resources 2.774. The gap between the lowest and highest department means is 0.048 points on the index scale.
- Mean `OverallSatisfactionIndex` by `JobRole` ranges from 2.688 (`Research Director`) to 2.772 (`Manufacturing Director`).

### Limits of this EDA
- No classifier was trained. There is no accuracy, precision, recall, or F1-score in this section.
- Engagement classes (Low / Medium / High) have not been defined from these scores yet.
- Plots and tables use a memory copy of the CSV. The file in `data/` was not modified.
