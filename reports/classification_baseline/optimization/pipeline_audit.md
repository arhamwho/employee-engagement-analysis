# Pipeline audit (leakage-safe optimization)

Target definition was **not** changed. The IBM CSV was **not** modified.

## Dataset

- Rows: 1470
- Columns: 39
- Missing values (cells): 0
- Duplicate rows: 0

Missing-value handling: none required (zero missing cells). Duplicate handling: none required (zero duplicate rows). Rows were not copied or dropped.

## Target

EngagementLevel is the student-defined 3-class label from the mean of JobInvolvement, JobSatisfaction, EnvironmentSatisfaction, RelationshipSatisfaction, WorkLifeBalance.

- Train counts: {'Low': 80, 'Medium': 866, 'High': 230}
- Test counts: {'Low': 20, 'Medium': 217, 'High': 57}

## Split

- `test_size=0.2`, `random_state=42`, `stratify=y`
- Train rows: 1176
- Test rows: 294
- Index overlap between train and test: 0

The test fold is held out until every hyperparameter search has finished. It is not used inside GridSearchCV or RandomizedSearchCV.

## Preprocessing (fit on train only)

- Numeric: `StandardScaler`
- Categorical: `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`
- Encoding and scaling are inside the sklearn Pipeline, fitted on training rows.

## Leakage control

Columns excluded from X: `JobInvolvement`, `JobSatisfaction`, `EnvironmentSatisfaction`, `RelationshipSatisfaction`, `WorkLifeBalance`, `EngagementScore`, `EngagementLevel`, `OverallSatisfactionIndex`, `EmployeeNumber`, `EmployeeCount`, `Over18`, `StandardHours`

- Forbidden columns still in X: none
- Exclusion helper agrees: True

Cross-validation uses `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on the **training** table only.
