# Initial supervised classification experiment (baseline)

This folder preserves the **first** machine-learning approach for Case Study No. 90.

It is **not** the final solution. Supervised classification of a student-defined
`EngagementLevel` could not beat the majority baseline in a leakage-safe way.

## Limitation (computed, not invented)

- Majority baseline test accuracy: **73.81%**
- Majority baseline macro F1: **0.283**
- Best leakage-safe Decision Tree: accuracy **65.31%**, macro F1 **0.387**
- Leakage-safe models did **not** reach 80%, 85%, or 90% accuracy
- 90%+ appeared only when the five label-defining survey items were put back
  into X (**target leakage / demonstration only**)

Hyperparameter search, extra derived features, Random Forest, Extra Trees,
Gradient Boosting, and SVM did not change that conclusion. Details:
`optimization/optimization_notes.md`

## What was kept

- `model_comparison.csv` — held-out classification scores
- `modeling_notes.md`, `model_analysis.md`
- `classification_reports/`
- `feature_importance.csv` — Gini importances of the old Decision Tree
- `target_construction.md` — how `EngagementLevel` was built (not used as a
  clustering target)
- `optimization/` — leakage-safe HPO
- `accuracy_investigation/` — 90% leakage check

The original Joblib files remain under `models/` (`final_model.joblib`,
`decision_tree.joblib`) so this baseline can still be inspected. The Streamlit
app now loads **clustering** as the primary model.
