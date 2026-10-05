# Independent survey target (Experiment 3)

This file documents the **project-defined** engagement target for the independent survey experiment. It is **not** a validated HR, psychological, or clinical scale.

The six engagement questions are collected independently of the workplace/HR predictors. They are used **only** to construct the label. They must **not** be used as features in the final predictive model.

## EngagementScore

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

Each item is a 1–5 Likert response.

## EngagementLevel thresholds

`pd.cut` bins `[1.00, 2.33, 3.67, 5.00]`, right-closed, `include_lowest=True`:

| Level | Score range |
| --- | --- |
| Low | 1.00–2.33 |
| Medium | 2.34–3.67 |
| High | 3.68–5.00 |

These cut-points are equal-width thirds of a 1–5 mean, chosen for this case study. They are not an official engagement instrument.

## Leakage control

**Predictors (X):** Age, Department, JobRole, JobLevel, YearsExperience, YearsAtCompany, MonthlyIncome, Overtime, WorkHoursPerWeek, TrainingOpportunities, CareerGrowth, ManagerSupport, Recognition, WorkLifeBalance, WorkArrangement

**Excluded from X:** Motivation, ValuedByOrganization, OrganizationalConnection, ExtraEffort, RecommendOrganization, IntentionToStay, EngagementScore, EngagementLevel

`WorkLifeBalance` on the survey form is a **workplace predictor**. It is not one of the six items that define EngagementScore.

## Status

No real survey responses have been collected yet. The file `data/survey/employee_engagement_survey_template.csv` is a header-only template. Do not report survey-model accuracy until `data/survey/employee_engagement_survey.csv` contains real rows and `notebooks/survey_engagement_model.ipynb` has scored a held-out test set.
