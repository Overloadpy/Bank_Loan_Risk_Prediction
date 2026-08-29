# Dataset Documentation & ML Data Card

**Author**: Yordanos Andargachew  
**Contact**: `+251 952 190 305`  
**Project**: Bank Loan Risk Prediction System  
**Dataset**: German Credit Risk Dataset (Statlog)  
**File**: `data/dataset.csv`  

---

## 1. Dataset Overview & Provenance

The German Credit Risk dataset (originally curated by Prof. Dr. Hans Hofmann, University of Hamburg, 1994) is the benchmark standard for credit evaluation research in machine learning. It contains real historical retail loan outcomes from a major German financial institution.

- **Total Records**: 1,000 credit applications
- **Total Features**: 9 predictor variables + 1 target variable (`Risk`)
- **Primary Domain**: Retail banking, credit scoring, default risk assessment
- **Format**: Comma-Separated Values (CSV), UTF-8 encoded

---

## 2. Feature Dictionary & Data Types

| Attribute | Storage Type | Cardinality / Range | Semantics & Financial Interpretation |
|---|---|---|---|
| `Age` | `int64` | 19 – 75 | Applicant age in years. Older applicants typically possess longer credit and employment histories. |
| `Sex` | `string` | 2 (`male`, `female`) | Biological sex derived from personal status records. |
| `Job` | `int64` | 4 classes (`0`, `1`, `2`, `3`) | Skill level: `0` (Unemployed/Unskilled non-resident), `1` (Unskilled resident), `2` (Skilled worker/clerk), `3` (Management/Self-employed/High qualification). |
| `Housing` | `string` | 3 classes (`own`, `rent`, `free`) | Tenancy status. Homeowners (`own`) demonstrate lower default risk due to asset stability. |
| `Saving accounts` | `string` | 5 classes | Liquid balance in savings: `little` (<100 DM), `moderate` (100–500 DM), `quite rich` (500–1000 DM), `rich` (≥1000 DM), `unknown` (missing). |
| `Checking account` | `string` | 4 classes | Current account liquidity: `little` (<0 DM / overdrawn), `moderate` (0–200 DM), `rich` (≥200 DM), `unknown` (no checking account at institution). |
| `Credit amount` | `int64` | 250 – 18,424 DM | Requested principal loan balance. Mean = 3,271 DM; Median = 2,319 DM. |
| `Duration` | `int64` | 4 – 72 months | Amortization schedule length. Longer durations correlate with macroeconomic exposure risk. |
| `Purpose` | `string` | 8 classes | Expenditure intent: `car`, `furniture/equipment`, `radio/TV`, `domestic appliances`, `repairs`, `education`, `business`, `vacation/others`. |
| `Risk` | `string` | 2 classes (`good`, `bad`) | **Ground Truth Target**: `good` (Credit repaid according to contract; 700 cases), `bad` (Default or delinquent; 300 cases). |

---

## 3. Missing Value Audit & Informative Imputation

An audit of missing values reveals structural absence patterns:

```text
Missing Value Audit (N=1,000):
─────────────────────────────────────────────
Feature              Missing Count   Percent
─────────────────────────────────────────────
Age                             0      0.0%
Sex                             0      0.0%
Job                             0      0.0%
Housing                         0      0.0%
Saving accounts               183     18.3%
Checking account              394     39.4%
Credit amount                   0      0.0%
Duration                        0      0.0%
Purpose                         0      0.0%
Risk                            0      0.0%
─────────────────────────────────────────────
```

### Missing Data Mechanics (MNAR / MAR)
In the German banking context, missing values in `Checking account` (39.4%) and `Saving accounts` (18.3%) do **not** represent data corruption or unobserved sensor dropouts. Instead, they indicate applicants who maintain no active checking or savings account with this specific originating bank (often holding primary accounts elsewhere or applying as first-time customers).

- **Handling Strategy**: Rather than dropping 40% of observations via listwise deletion or imputing modal values (which would falsely label unbanked customers as having `<100 DM` accounts), we treat missingness as an explicit informative categorical state: `'unknown'`.

---

## 4. Class Imbalance Characteristics

The target distribution displays a **70:30 class imbalance**:
- **Good Risk (`0`)**: 700 instances (70.0%)
- **Bad Risk (`1`)**: 300 instances (30.0%)

```
┌───────────────────────────────────────────────┐
│ Good Risk (70.0% - 700 cases)                 │ Bad Risk (30.0% - 300 cases)  │
└───────────────────────────────────────────────┴───────────────────────────────┘
```

Without compensation, standard loss estimators tend to predict the majority class to minimize cross-entropy loss, resulting in elevated Type II errors (failing to identify defaults). Both models in this system employ `class_weight='balanced'` to enforce inverse frequency penalization:
$$w_0 = \frac{1000}{2 \times 700} \approx 0.714, \quad w_1 = \frac{1000}{2 \times 300} \approx 1.667$$

---

## 5. Algorithmic Bias & Fair Lending Audit

Credit scoring models deployed in production must comply with international fair lending mandates (such as the Equal Credit Opportunity Act - ECOA and EU AI Act Title III high-risk AI obligations).

### Demographic Subgroup Disparities
1. **Sex Disparity**:
   - Female applicants represent 31.0% (310 records) with a historical default rate of 35.2%.
   - Male applicants represent 69.0% (690 records) with a historical default rate of 27.7%.
   - **Mitigation**: The model does not directly penalize gender; one-hot encoding ensures gender weights are audited during post-hoc explainability passes.
2. **Age Disparity**:
   - Applicants under 25 demonstrate higher default rates (42.1%) due to shorter career trajectories.
   - Age is continuous and normalized to prevent step-function discrimination against younger borrowers.
