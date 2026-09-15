# Customer Churn Prediction & Lifetime Value (LTV) Engine

Beginner-friendly end-to-end project that predicts customer churn, estimates lifetime value, explains model decisions with SHAP, and serves results through FastAPI and a dashboard.

## Project Progress

### Day 1 — Completed ✅

- Dataset loading
- Data cleaning
- EDA
- PostgreSQL
- SQL analysis

### Day 2 — Completed ✅

- Feature engineering
- Encoding
- Preprocessing
- Train/test split

### Day 3 — Completed ✅

- Logistic Regression
- Random Forest
- XGBoost
- Model evaluation
- Confusion matrices
- Model comparison
- Best model selection
- Churn probability prediction
- Customer risk classification
- Saved trained churn model

### Day 4 — Completed ✅

- Estimated Lifetime Revenue (LTV Proxy)
- LTV segments (terciles)
- Churn risk + LTV retention priority
- High-risk / high-LTV customer list
- SHAP global feature importance
- SHAP individual customer explanation

## Machine Learning Results (Day 3)

Actual test-set results (5,634 train / 1,409 test). Metrics for churn class (Yes = 1):

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.7970 | 0.6467 | 0.5187 | 0.5757 | 0.8418 |
| Random Forest | 0.7807 | 0.6102 | 0.4813 | 0.5381 | 0.8203 |
| **XGBoost (best)** | **0.8077** | **0.6758** | **0.5294** | **0.5937** | **0.8443** |

## Day 4 — LTV Estimation & SHAP Explainability

### LTV Methodology

The Telco dataset **does not** contain a true future Customer Lifetime Value target.
This project therefore uses an **Estimated Lifetime Revenue / LTV Proxy** — useful for prioritization, **not** actual future revenue.

Formula:

```
estimated_ltv = MonthlyCharges × estimated_lifetime_months
estimated_lifetime_months = tenure + expected_remaining_months

If Churn == Yes:
    expected_remaining_months = 0
Else:
    expected_remaining_months =
        CONTRACT_BASE_REMAINING_MONTHS[Contract] × (1 - churn_probability)
```

Documented contract assumptions (MVP, not proven forecasts):

| Contract | Base remaining months |
|----------|----------------------|
| Month-to-month | 6 |
| One year | 12 |
| Two year | 24 |

### LTV Segmentation

Low / Medium / High segments use **data-driven terciles** of `estimated_ltv` (33rd / 66th percentiles).

From the actual run on all 7,043 customers:

| Segment | Threshold (approx.) | Mean estimated_ltv |
|---------|---------------------|--------------------|
| Low | ≤ 894.00 | 363.97 |
| Medium | ≤ 3206.13 | 1837.73 |
| High | > 3206.13 | 6199.98 |

### Churn + LTV

Day 3 churn probabilities are scored for all customers (no model retraining), then combined with LTV segments into a `retention_priority` field.

Documented MVP priority rules (not universal business law):

- High risk + High LTV → **Critical**
- High risk + Medium LTV → High
- High risk + Low LTV → Medium
- Medium risk + High LTV → High
- Low risk + High LTV → Monitor
- otherwise → Low / Medium as mapped in code

Actual counts from Day 4:

| retention_priority | Customers |
|--------------------|-----------|
| Low | 3180 |
| Monitor | 1927 |
| Medium | 1282 |
| High | 600 |
| Critical | 54 |

**High-risk + High-LTV customers identified: 54** (mean estimated_ltv ≈ 4356.59).
These accounts may deserve higher-priority retention attention. Retention is not guaranteed to prevent churn.

### SHAP Explainability

SHAP explains the saved Day 3 **XGBoost** model (`TreeExplainer`).

- Global importance: which features most influence churn predictions overall
- Individual explanation: which features pushed one customer's score toward/away from churn
- SHAP shows **model contribution / association**, not proof of causation

Actual top SHAP features (mean |SHAP| on a 500-customer sample):

1. `is_month_to_month`
2. `tenure`
3. `OnlineSecurity_No`
4. `MonthlyCharges`
5. `TechSupport_No`

Example individual explanation (customer `1024-GUALD`): Actual Yes, Predicted Yes, probability 0.6748, High risk — driven upward mainly by month-to-month status and short tenure-related signals.

## Project Structure

```
Customer-Churn-Prediction/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_churn_model.ipynb
│   ├── 04_ltv_model.ipynb
│   └── 05_shap_analysis.ipynb
├── src/
│   ├── preprocessing.py
│   ├── churn_model.py
│   ├── ltv_model.py
│   ├── run_day3.py
│   └── run_day4.py
├── models/
│   ├── preprocessor.pkl
│   └── churn_model.pkl
├── reports/
│   ├── retention_priority.csv
│   ├── shap_feature_importance.csv
│   └── figures/
└── requirements.txt
```

## Quick Start (Windows)

```powershell
cd Customer-Churn-Prediction
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Commands

```powershell
python src/run_day1.py   # if using helpers
python src/run_day2.py
python src/run_day3.py
python src/run_day4.py

# Or open notebooks/04_ltv_model.ipynb and notebooks/05_shap_analysis.ipynb
```
