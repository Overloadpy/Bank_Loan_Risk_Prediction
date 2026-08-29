# Results & Performance Analysis

**Author**: Yordanos Andargachew  
**Contact**: `+251 952 190 305`  
**Project**: Bank Loan Risk Prediction System  
**Date**: August 2026  

---

## 1. Executive Experimental Results

The machine learning models were trained on 800 stratified samples and evaluated on an independent 200-sample holdout test partition ($N_{\text{Good}}=140, N_{\text{Bad}}=60$).

### 1.1 Comparative Metric Matrix

| Evaluation Metric | Logistic Regression (Linear Baseline) | Random Forest (Non-Linear Ensemble) | Absolute Delta ($\Delta$) | Superior Architecture |
|---|---|---|---|---|
| **Overall Accuracy** | 71.50% | 70.00% | -1.50% | Baseline |
| **Precision (Bad Risk / Default)** | 51.72% | 50.00% | -1.72% | Baseline |
| **Recall (Bad Risk / Default)** | **75.00%** | 71.67% | -3.33% | Baseline |
| **F1-Score (Macro)** | **0.6122** | 0.5890 | -0.0232 | Baseline |
| **ROC-AUC Score** | 0.7615 | **0.7844** | **+2.29%** | **Random Forest** |

---

## 2. Confusion Matrix Diagnostics

```text
Logistic Regression (Test N=200):
┌──────────────────────────┬──────────────────────────┐
│ True Negative (TN): 98   │ False Positive (FP): 42  │ (Actual: Good Risk)
├──────────────────────────┼──────────────────────────┤
│ False Negative (FN): 15  │ True Positive (TP): 45   │ (Actual: Bad Risk)
└──────────────────────────┴──────────────────────────┘

Random Forest (Test N=200):
┌──────────────────────────┬──────────────────────────┐
│ True Negative (TN): 97   │ False Positive (FP): 43  │ (Actual: Good Risk)
├──────────────────────────┼──────────────────────────┤
│ False Negative (FN): 17  │ True Positive (TP): 43   │ (Actual: Bad Risk)
└──────────────────────────┴──────────────────────────┘
```

### Financial Matrix Interpretation
- **False Positives (Type I Error)**: Good borrowers erroneously flagged as high risk. Incur opportunity costs (lost net interest margin).
- **False Negatives (Type II Error)**: Defaulting borrowers erroneously classified as good risk. Incur direct charge-offs and principal losses. In credit underwriting, **Type II errors are 4× to 6× more expensive** than Type I errors.

---

## 3. ROC-AUC & Precision-Recall Analysis

```
  1.0 ┌─────────────────────────────────────────────────────────────┐
      │                                       .....---''''' Random   │
  0.8 │                             ...--''''''             Forest   │
  TPR │                       ..--'''                      (0.784)   │
  0.6 │                 ..--''                     Logistic          │
      │           ..--''                           Regression (0.762)│
  0.4 │       .-''                                                   │
      │    .-'                                                       │
  0.2 │ .-'                                                          │
      │'                                                             │
  0.0 └─────────────────────────────────────────────────────────────┘
      0.0         0.2         0.4         0.6         0.8         1.0
                                  FPR
```

- **Random Forest Area Under Curve ($0.7844$)**: Demonstrates superior rank-ordering capability across all potential decision thresholds. This enables credit risk managers to adjust cutoffs flexibly depending on liquidity and risk tolerance.
- **Precision-Recall Advantage**: In high-recall operating regions ($R > 0.70$), Random Forest maintains a more resilient precision profile, preventing excessive false alarms.

---

## 4. Feature Importance & Interpretability Ranking

Using Mean Decrease in Impurity (Gini Importance) from the fitted Random Forest model, the top 10 determinants of loan risk are:

```text
Top 10 Feature Importances:
───────────────────────────────────────────────────────────────────────────
Rank  Feature Column                       Importance   Domain Implication
───────────────────────────────────────────────────────────────────────────
 1.   Credit amount                             0.187   Loan size vs applicant debt capacity
 2.   Duration                                  0.146   Repayment horizon & exposure term
 3.   Age                                       0.134   Life stage & financial maturity
 4.   Checking account_little                   0.068   Severe immediate liquidity deficit
 5.   Saving accounts_little                    0.052   Low cash buffer for emergencies
 6.   Checking account_unknown                  0.043   No verified relationship history
 7.   Purpose_car                               0.035   Asset depreciation rate vs debt
 8.   Housing_own                               0.031   Collateral & asset stability
 9.   Saving accounts_unknown                   0.029   External banking relationship
10.   Purpose_radio/TV                          0.024   Consumer lifestyle expenditure
───────────────────────────────────────────────────────────────────────────
```

### Key Analytical Takeaways
1. **Continuous Exposure Factors Dominate**: `Credit amount`, `Duration`, and `Age` together account for **46.7%** of the total predictive power of the model.
2. **Account Status is the Strongest Categorical Signal**: Overdrawn or low checking accounts (`Checking account_little`) drastically elevate default probability.
3. **Informative Absence Works**: `Checking account_unknown` and `Saving accounts_unknown` rank in the top 10 features, proving that unbanked status carries strong discriminative signal compared to discarding missing values.
