## Classification experiment notes

These numbers come from the **held-out test set** after one stratified split (test_size=0.2, random_state=42). They are not typed in by hand.

### Why these metrics

- **Accuracy** is the share of test employees whose `EngagementLevel` was predicted correctly. With a large Medium class, a model can look accurate while missing Low and High.
- **Precision** (macro) asks, for each class, how often a predicted label was actually that class, then averages the three classes equally.
- **Recall** (macro) asks, for each class, how many of the true members were found, then averages equally. This matters for the small Low group.
- **F1-score / macro F1** is the balance of precision and recall, averaged equally across Low, Medium, and High. In this multiclass, imbalanced problem, **macro F1 is more informative than accuracy**.
- **Weighted** precision / recall / F1 weight each class by its test support. They stay closer to accuracy when Medium dominates.

Do **not** choose a model from accuracy alone.

### Split and features

- Training rows: 1176
- Test rows: 294
- Train class counts: {'Low': 80, 'Medium': 866, 'High': 230}
- Test class counts: {'Low': 20, 'Medium': 217, 'High': 57}
- Columns excluded from X (leakage, IDs, constants): `JobInvolvement`, `JobSatisfaction`, `EnvironmentSatisfaction`, `RelationshipSatisfaction`, `WorkLifeBalance`, `EngagementScore`, `EngagementLevel`, `OverallSatisfactionIndex`, `EmployeeNumber`, `EmployeeCount`, `Over18`, `StandardHours`

Imbalance was handled with `class_weight='balanced'` (no duplicate-row oversampling). The split is stratified so each class appears in train and test.

### Majority-class reference

- Always predicting the training majority class scores accuracy=0.7381 and macro F1=0.2831 on this test set. A useful model should beat this on **macro F1**, not only match accuracy.

### Comparison (test set)

- **Logistic Regression**: accuracy=0.3673, precision (macro)=0.3544, recall (macro)=0.3796, F1 (macro)=0.3043, F1 (weighted)=0.4198
- **Decision Tree**: accuracy=0.6531, precision (macro)=0.3940, recall (macro)=0.3834, F1 (macro)=0.3872, F1 (weighted)=0.6434
- **Random Forest**: accuracy=0.7381, precision (macro)=0.2460, recall (macro)=0.3333, F1 (macro)=0.2831, F1 (weighted)=0.6269
- **Support Vector Machine**: accuracy=0.4864, precision (macro)=0.3598, recall (macro)=0.3750, F1 (macro)=0.3480, F1 (weighted)=0.5291

### Reading the ranking

Using **macro F1 first** (then weighted F1, then accuracy), the strongest of the four fitted models on this test split is **Decision Tree** (macro F1=0.3872, accuracy=0.6531).
Next is **Support Vector Machine** (macro F1=0.3480, accuracy=0.4864).
**Accuracy would have picked `Random Forest`** (accuracy=0.7381), which is a different choice than macro F1. That is why accuracy is not used alone.

This ranking is for **this split and this feature set**. It is not a claim that the student-defined engagement label is an official HR instrument.

Saved artifacts (if saving was enabled): `models/preprocessor.joblib` and one full pipeline `.joblib` file per classifier.
