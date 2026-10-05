## Model analysis (from the saved experiment)

Metrics in the comparison table are the **same values** written during training (`model_comparison.csv`). This section does not retrain models and does not replace those scores.

### Selection criterion (stated before ranking)

1. Primary: highest **macro F1** on the held-out test set.
2. Tie-break: higher **weighted F1**, then higher **accuracy**.
3. A model that only matches the majority baseline on accuracy, while missing Low and High, is not selected.

Under that rule the selected final model is **Decision Tree** (macro F1=0.3872, accuracy=0.6531).

Accuracy alone would have preferred **Random Forest** (accuracy=0.7381), which on this split is the same accuracy as the majority baseline (0.7381). Random Forest predicted only Medium, so that accuracy is not evidence of better engagement classification.

### Performance is modest

Even the selected model has test accuracy 0.6531 and macro F1 0.3872. Macro F1 is only a small lift over the majority baseline macro F1 of 0.2831. These HR predictors (pay, tenure, department, overtime, and similar fields) do **not** strongly recover the student-defined engagement bins once the survey items that define the bins are held out.

### Classes: easiest and hardest (Decision Tree, test set)

- **Low**: support=20, correct=1, errors=19, error rate=95.0%
- **Medium**: support=217, correct=174, errors=43, error rate=19.8%
- **High**: support=57, correct=17, errors=40, error rate=70.2%

- Easiest class (lowest error rate): **Medium** (19.8% of true Medium rows misclassified).
- Hardest class (highest error rate): **Low** (95.0% of true Low rows misclassified).

Medium is easiest because it is the majority class (217 of 294 test rows). Low is hardest: only 20 test rows, and the tree almost never identifies them.

### Does imbalance affect performance?

Yes. Medium is about three-quarters of the sample. Models can raise accuracy by predicting Medium. Random Forest did exactly that (recall Medium = 1.00, recall Low = 0.00, recall High = 0.00). `class_weight='balanced'` did not overcome the lack of signal plus the small Low sample.

### Which errors occur most (Decision Tree confusion matrix)

- Test rows: 294; correct: 192; errors: 102.
- Low ↔ Medium swaps: 33
- Medium ↔ High swaps: 65
- Low ↔ High swaps: 4
- Adjacent-bin errors (Low–Medium or Medium–High): 98

The more common neighbor mix-up is **Medium vs High**, not Low vs Medium. Low vs High jumps are the rarest of the three off-diagonal pairs, which is expected if errors are mostly between neighboring score bins.

### Possible reasons for errors

- 26 of 102 Decision Tree errors have `EngagementScore` within 0.2 of a cut-point (2.00 or 3.00). Employees near a bin edge can be labeled differently from neighbors with almost the same survey mean.
- The classifier **cannot see** the five survey items that created the label. It only sees other HR fields, which EDA showed were weakly associated with the composite score.
- Low has 20 test examples, so recall for Low is unstable and easy to drive near zero.
- Adjacent ordinal bins (Low–Medium–High) are closer in the underlying mean score than Low vs High, so neighbor confusions are more plausible than opposite-end confusions.

### Limitations

- One train/test split; another split could reorder close models.
- `EngagementLevel` is a **student-defined operationalization**, not an official or validated engagement instrument.
- Predicted association is **not causation**. A correct or incorrect classification does not mean that overtime, pay, or department *causes* engagement.
- No hyperparameter search was claimed as optimal; these are the models from the previous experiment.

### Final model artifacts

- Selected model: Decision Tree
- Copy saved as `models/final_model.joblib` (same weights as `decision_tree.joblib`).

Classification report for the selected model (test set, unchanged from training):

```
precision    recall  f1-score   support

         Low     0.0526    0.0500    0.0513        20
      Medium     0.7598    0.8018    0.7803       217
        High     0.3696    0.2982    0.3301        57

    accuracy                         0.6531       294
   macro avg     0.3940    0.3834    0.3872       294
weighted avg     0.6361    0.6531    0.6434       294
```
