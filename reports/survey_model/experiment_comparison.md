# Three experiments — do not mix their claims

The IBM Streamlit demo (`app.py`, `models/final_model.joblib`) stays on Experiment 1 until a valid survey model exists.

## Experiment 1 — IBM HR, leakage-safe (current demo)

- **Data:** `data/IBM_HR_Attrition_Enhanced.csv`
- **Target:** student-defined mean of five IBM 1–4 items, then Low / Medium / High
- **Features:** workplace/HR fields only. The five items that **define** the IBM label are excluded from X.
- **Held-out Decision Tree:** accuracy **65.31%**, macro F1 **0.387**
- **Majority baseline accuracy:** **73.81%**
- **How to present it:** this is **not** a high-performing model. Accuracy is below the majority class. Do not describe it as 90%+.

## Experiment 2 — IBM HR with label-defining variables in X

- **Data:** the same IBM table
- **What changed:** the five variables used to construct EngagementLevel were put back into X
- **Label:** **TARGET LEAKAGE / DEMONSTRATION ONLY**
- **Finding:** 90%+ accuracy appeared only in this leaked setting. That is reconstruction of the labeling formula, not independent prediction of engagement.
- **How to present it:** demonstration of leakage. Not the Streamlit model. Not a valid 90% result.

## Experiment 3 — Independent employee survey (proposed final model)

- **Data:** real survey responses at `data/survey/employee_engagement_survey.csv` (not collected yet)
- **Template:** `data/survey/employee_engagement_survey_template.csv` (headers only; not training data)
- **Target:** mean of six independent 1–5 Likert items (see `target_definition.md`)
- **Features:** workplace/HR predictors only. The six engagement questions are excluded from X.
- **Primary ranking metric:** macro F1 (classes may be imbalanced). Always report the majority-class baseline.
- **Performance:** **not evaluated.** No accuracy, F1, or other score is claimed until real responses exist and sklearn scores a held-out test set.
- **Streamlit:** do not replace the IBM demo with this experiment until that evaluation exists.

## What was not done

- No fake employee survey rows
- No synthetic data presented as real responses
- No invented survey-model metrics
- No change to the current Streamlit IBM pipeline
