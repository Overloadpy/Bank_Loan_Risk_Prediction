# Problem Definition & Formulation

**Author**: Yordanos Andargachew  
**Contact**: `+251 952 190 305`  
**Project**: Bank Loan Risk Prediction System  
**Date**: August 2026  

---

## 1. Executive Summary & Problem Context
In commercial lending and retail banking, credit underwriting is the core determinant of institutional solvency and profitability. Granting loans to borrowers who ultimately default causes direct write-offs, loan-loss provisioning, legal recovery costs, and capital impairment. Conversely, excessively conservative lending rules result in opportunity cost—rejecting creditworthy applicants and losing net interest margin (NIM) to competitors.

The **Bank Loan Risk Prediction System** solves this core challenge by deploying supervised machine learning models to classify loan applicants into **Good Credit Risk** (high repayment probability) versus **Bad Credit Risk** (elevated default likelihood).

```
   ┌──────────────────────┐        ┌────────────────────────────┐        ┌─────────────────────────┐
   │ Loan Applicant Data  │ ───►  │ ML Scoring Pipeline        │ ───►  │ Automated Underwriting  │
   │ (Financial & Demo)   │        │ (Preprocessing + Ensemble) │        │ (Decision + Tiers + CI) │
   └──────────────────────┘        └────────────────────────────┘        └─────────────────────────┘
```

---

## 2. Business Objectives & Stakeholder Analysis

### 2.1 Core Business Objectives
1. **Minimize Non-Performing Loans (NPLs)**: Detect subprime borrowers with high delinquency propensities prior to loan origination.
2. **Accelerate Underwriting Velocity**: Automate instant decisioning for prime applications while flagging ambiguous cases for manual tier-2 review.
3. **Calibrated Risk Pricing**: Provide calibrated continuous probabilities ($P(\text{Default}|X)$) to inform loan pricing, interest rate spreads, and collateral requirements.
4. **Regulatory Auditability & Fairness**: Maintain interpretable baseline coefficients alongside high-capacity non-linear ensembles.

### 2.2 Key Stakeholders
- **Chief Risk Officer (CRO) & Credit Committee**: Requires portfolio-level default minimization and macroeconomic stress robustness.
- **Loan Underwriting Officers**: Require an intuitive desktop studio UI (`gui.py`) and CLI (`cli.py`) for instantaneous case assessment.
- **Compliance & Fair Lending Auditors**: Require transparent feature attribution and bias auditing across demographic attributes (e.g., sex, age).
- **Borrowers / Applicants**: Benefit from rapid, objective, and unbiased credit turnaround times.

---

## 3. Input & Output Contract Specifications

### 3.1 Input Feature Space
The system ingests 9 multi-modal applicant attributes:

| Feature Name | Data Type | Description & Valid Range / Domain |
|---|---|---|
| `Age` | Integer | Applicant age in years (Range: 19 – 75) |
| `Sex` | Categorical | Biological sex (`male`, `female`) |
| `Job` | Ordinal / Int | Employment classification: `0` (Unskilled non-resident), `1` (Unskilled resident), `2` (Skilled), `3` (Management / Highly skilled) |
| `Housing` | Categorical | Residential status (`own`, `rent`, `free`) |
| `Saving accounts` | Categorical | Total liquid savings balance (`little`, `moderate`, `quite rich`, `rich`, `unknown`) |
| `Checking account` | Categorical | Liquidity in current account (`little`, `moderate`, `rich`, `unknown`) |
| `Credit amount` | Integer | Requested principal loan amount in Deutsche Mark (DM) |
| `Duration` | Integer | Loan repayment term in months (Range: 4 – 72) |
| `Purpose` | Categorical | Economic purpose (`car`, `furniture/equipment`, `radio/TV`, `domestic appliances`, `repairs`, `education`, `business`, `vacation/others`) |

### 3.2 Output Specification
For any applicant vector $x \in \mathcal{X}$, the system outputs a structured decision payload:

```json
{
  "risk_class": "Good Risk | Bad Risk",
  "risk_tier": "LOW RISK | MODERATE RISK | HIGH RISK | CRITICAL RISK",
  "prob_good": 0.8118,
  "prob_bad": 0.1882,
  "confidence_score": 81.18,
  "recommendation": "APPROVE LOAN: Excellent credit profile and high repayment confidence.",
  "model_used": "Random Forest (Advanced Ensemble) | Logistic Regression",
  "evaluated_at": "2026-08-30 00:40:40"
}
```

---

## 4. Formal Research Questions (RQs)

To guide the modeling architecture, this project addresses three central empirical questions:

### ❓ Research Question 1 (RQ1: Non-Linearity vs Linear Baselines)
> *To what extent does a non-linear ensemble (Random Forest) outperform a generalized linear model (Balanced Logistic Regression) on structured credit scoring data with interaction effects?*

### ❓ Research Question 2 (RQ2: Information Value of Account Absence)
> *How does preserving missing account indicators (`unknown`) as informative states compare against naive imputation or listwise deletion in predicting default rates?*

### ❓ Research Question 3 (RQ3: Asymmetric Loss & Decision Boundaries)
> *Given that False Negatives (granting a loan to a borrower who defaults) carry significantly higher monetary loss than False Positives (lost interest margin), how should the classification threshold be calibrated across risk tiers?*
