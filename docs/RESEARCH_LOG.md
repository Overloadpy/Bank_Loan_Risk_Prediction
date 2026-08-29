# Machine Learning Research & Experimentation Log

**Author**: Yordanos Andargachew  
**Contact**: `+251 952 190 305`  
**Project**: Bank Loan Risk Prediction System  
**Date**: August 2026  

---

## Experiment Iteration Timeline

```text
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│ Iteration 01    │ ───►  │ Iteration 02    │ ───►  │ Iteration 03    │ ───►  │ Iteration 04    │
│ Raw Data Audit  │       │ Missing Value   │       │ Linear Baseline │       │ RF Ensemble &   │
│ & Distribution  │       │ Imputation Test │       │ & Class Weights │       │ Hyperparameters │
└─────────────────┘       └─────────────────┘       └─────────────────┘       └─────────────────┘
```

---

## Experiment 01: Exploratory Audit & Class Imbalance Discovery
- **Date**: 2026-08-30
- **Hypothesis**: The German Credit dataset has significant class imbalance that will cause standard accuracy metrics to be misleading.
- **Observations**:
  - Target `Risk` contains 700 Good (70%) and 300 Bad (30%).
  - Naive majority baseline achieves 70.0% accuracy but 0.0% recall on defaults.
  - Continuous features (`Credit amount`, `Duration`) exhibit right-skewness requiring normalization.
- **Decision**: Adopt ROC-AUC and minority-class Recall as primary optimization criteria; enforce `class_weight='balanced'`.

---

## Experiment 02: Missing Value Strategy Evaluation
- **Date**: 2026-08-30
- **Hypothesis**: Replacing missing account status with the mode (<100 DM) will distort applicant financial health compared to treating missingness as an independent `'unknown'` state.
- **Experimental Trials**:
  1. *Trial A (Drop Missing)*: Discarded 40% of rows ($N=606$). Result: Severe sample truncation, poor generalization on unbanked applicants.
  2. *Trial B (Mode Imputation)*: Imputed `'little'` for missing. Result: Elevated False Positive rate for prime borrowers with external accounts.
  3. *Trial C (Informative Category `'unknown'`)*: Encoded `'unknown'` as separate one-hot category. Result: Retained 100% sample size ($N=1000$); feature importance confirmed `Checking account_unknown` is a top-10 predictor.
- **Decision**: Standardize on informative category `'unknown'` in `CreditDataPreprocessor`.

---

## Experiment 03: Baseline Logistic Regression Optimization
- **Date**: 2026-08-30
- **Objective**: Establish a calibrated, interpretable linear benchmark.
- **Iterations**:
  - Unweighted Logistic Regression: Accuracy 73.0%, Recall (Bad) 43.3%, ROC-AUC 0.761.
  - Balanced Logistic Regression (`class_weight='balanced'`): Accuracy 71.5%, Recall (Bad) **75.0%**, ROC-AUC **0.7615**.
- **Outcome**: The balanced model caught 75% of defaulting borrowers (45 out of 60 in test fold), making it viable for conservative credit policies.

---

## Experiment 04: Random Forest Hyperparameter Tuning
- **Date**: 2026-08-30
- **Objective**: Maximize non-linear feature interaction capture without overfitting small training sample ($N=800$).
- **Parameter Grid Explored**:
  - `n_estimators`: [50, 100, 200] -> 100 yielded optimal variance stabilization.
  - `max_depth`: [5, 10, 15, None] -> `max_depth=10` prevented leaf memorization while capturing interaction depth.
  - `min_samples_leaf`: [1, 2, 4] -> `min_samples_leaf=2` smoothed out boundary jitter.
- **Final Metrics**: Test Accuracy 70.0%, Recall (Bad) 71.67%, ROC-AUC **0.7844**.
- **Outcome**: Achieved highest ROC-AUC ($0.7844$), providing the most accurate probability calibration across the full spectrum of loan risk tiers.

---

## Experiment 05: Multi-Modal Interface & Latency Testing
- **Date**: 2026-08-30
- **Objective**: Ensure sub-50ms inference latency for CLI and desktop GUI.
- **Outcome**: Single inference latency measured at <8ms on standard CPU, enabling instant live updates in `gui.py` as user modifies spinboxes and dropdowns.
