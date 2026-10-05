## Target construction notes

EngagementScore and EngagementLevel are a **student-defined operationalization** for Case Study 90. The IBM HR Analytics dataset does not include an official employee-engagement target. These thresholds are **not** an HR industry standard and the score is **not** a clinically or scientifically validated engagement scale. They are a transparent, reproducible rule so this project can train a 3-class classifier later.

### Formula

EngagementScore = mean(JobInvolvement, JobSatisfaction, EnvironmentSatisfaction, RelationshipSatisfaction, WorkLifeBalance)

Each input is used as stored (typically 1–4). All five items have equal weight.

### Project-defined thresholds

- 1.00–2.00 → Low
- 2.01–3.00 → Medium
- 3.01–4.00 → High

### Observed score range in this table

- EngagementScore min = 1.20, max = 4.00, mean = 2.731, median = 2.80

### Class counts (computed from this table)

- **Low**: 100 employees (6.80%)
- **Medium**: 1083 employees (73.67%)
- **High**: 287 employees (19.52%)

### Class imbalance check

- Majority class: **Medium** (1083 employees, 73.67%).
- Smallest class: **Low** (100 employees).
- Majority-to-minority count ratio: 10.83.
- **The classes are imbalanced** in this sample: one level is much more common than the others. Later modeling should account for that (for example with stratified splitting). No model is trained here.

### What was preserved

- The five original survey columns are still in the working table.
- The CSV in `data/` was not overwritten.
