# Derived features (leakage-safe)

Built only from columns that remain after dropping the five target-defining items, EngagementScore, EngagementLevel, OverallSatisfactionIndex, IDs, and constants.

Denominators add 1 so a zero tenure/experience year does not divide by zero.

| feature | source_columns | created | description |
| --- | --- | --- | --- |
| IncomePerYearExperience | MonthlyIncome, TotalWorkingYears | True | MonthlyIncome / (TotalWorkingYears + 1). Pay relative to career length. |
| IncomePerCompanyYear | MonthlyIncome, YearsAtCompany | True | MonthlyIncome / (YearsAtCompany + 1). Pay relative to tenure here. |
| CurrentRoleRatio | YearsInCurrentRole, YearsAtCompany | True | YearsInCurrentRole / (YearsAtCompany + 1). Share of company tenure in the current role. |
| ManagerTenureRatio | YearsWithCurrManager, YearsAtCompany | True | YearsWithCurrManager / (YearsAtCompany + 1). Share of company tenure with the current manager. |
| PromotionGap | YearsSinceLastPromotion, YearsAtCompany | True | YearsSinceLastPromotion / (YearsAtCompany + 1). How much of company tenure has passed since the last promotion. |
| CompanyTenureRatio | YearsAtCompany, Age | True | YearsAtCompany / Age. Company tenure as a fraction of age. |
| CareerProgressionRate | JobLevel, TotalWorkingYears | True | JobLevel / (TotalWorkingYears + 1). Job level relative to years of experience. |
| ExperienceAtCompanyRatio | YearsAtCompany, TotalWorkingYears | True | YearsAtCompany / (TotalWorkingYears + 1). Share of working life spent at this company. |
| YearsWithoutPromotion | YearsSinceLastPromotion | True | Same values as YearsSinceLastPromotion, named explicitly as time since last promotion. |
| OvertimeIndicator | OverTime | True | 1 if OverTime is Yes, else 0. |

These ratios are row-wise. They do not use other employees' values and do not use the test fold.
