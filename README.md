# Credit-Default-Risk (Vitto)

This project performs an end-to-end analysis of credit card default behavior for a simulated credit risk team at **Vitto**. Using the UCI Credit Card Default dataset (Taiwan, 2005), we identify predictive signals and build machine learning models to classify high-risk borrowers.

## Setup Guide

1. **Environment Setup**: Ensure that Python 3.8+ is been installed.
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run Analysis**: Open `train.ipynb` in VS Code and run all cells to reproduce the results.

## Dataset Source

- **Source**: [UCI Machine Learning Repository - Default of Credit Card Clients](Here, its present inside data).
- **Scale**: 30,000 records, 25 variables.
- **Timeframe**: April 2005 – September 2005.

## 🧠 Methodology Summary

- **Exploratory Data Analysis (EDA)**: Investigated demographic trends (Age, Sex, Education) and payment behaviors. 
- **Data Quality**: Explicitly handled undocumented categories in `EDUCATION` and `MARRIAGE` and identified overpayment anomalies (`BILL_AMT < 0`).
- **Feature Engineering**: Designed behavioral signals:
    - `AVG_UTIL_RATE`: Average credit utilization over 6 months.
    - `AVG_PAY_RATIO`: Consistency of bill repayment.
    - `TOTAL_DELAY_MONTHS`: Cumulative months of delinquency.
- **Modeling**: Evaluated Logistic Regression (baseline) and Tree-based models (XGBClassifier).
- **Evaluation**: Used Stratified 5-fold Cross-Validation with focus on AUC-ROC and Recall to capture high-risk defaults effectively.

## 📋 Feature List

| Feature Category | Examples |
| :--- | :--- |
| **Demographics** | `AGE`, `SEX`, `EDUCATION`, `MARRIAGE` |
| **Credit Limit** | `LIMIT_BAL` |
| **Payment History** | `PAY_0` to `PAY_6` (Repayment status) |
| **Billing & Payments** | `BILL_AMT`, `PAY_AMT` |
| **Engineered Features** | `AVG_UTIL_RATE`, `AVG_PAY_RATIO`, `TOTAL_DELAY_MONTHS` |

## 💡 Key Findings

- **Top Predictor**: The most recent repayment status (`PAY_0`) is the strongest indicator of immediate default risk.
- **Delinquency Signal**: Customers with `TOTAL_DELAY_MONTHS` > 2 show a significantly higher default rate (~22% baseline vs > 60% for chronic delays).
- **Utilization Threshold**: High utilization rate (above 80% on average) is a secondary but critical risk signal.
- **Model Selection**: Tree-based models provided superior performance in identifying non-linear patterns compared to the linear baseline.

