# Customer Churn Prediction & Lifetime Value (LTV) Engine

Beginner-friendly end-to-end project that predicts customer churn, estimates lifetime value, explains model decisions with SHAP, and serves results through FastAPI and a dashboard.

## Project Progress

### Day 1 — Completed ✅

- Dataset loading
- Data cleaning
- Exploratory Data Analysis
- PostgreSQL database
- SQL analysis

#### Day 1 details (real results)

- Dataset: IBM Telco Customer Churn — 7,043 customers, 21 columns
- Overall churn rate: **26.54%** (1,869 churned / 5,174 not churned)
- Month-to-month churn: 42.71% | Two-year: 2.83%
- Fiber optic churn: 41.89% | Electronic check: 45.29%

```
Telco CSV → Pandas → Cleaning → EDA → Cleaned CSV → PostgreSQL → SQL Analysis
```

### Day 2 — Completed ✅

- Feature engineering
- Numerical/categorical feature identification
- Categorical encoding (`OneHotEncoder`, `handle_unknown="ignore"`)
- Numerical preprocessing (`StandardScaler`)
- Train/test split (80/20, stratified, `random_state=42`)
- Class balance analysis
- Reusable preprocessing pipeline (`src/preprocessing.py`)
- Saved preprocessing model (`models/preprocessor.pkl`)

#### Day 2 details (real outputs)

| Item | Value |
|------|-------|
| Features before encoding | 26 (customerID excluded) |
| Train / test sizes | 5,634 / 1,409 |
| Churn rate (train & test) | 26.54% / 26.54% (stratified) |
| ML-ready columns after transform | 55 |
| Missing values after transform | 0 |

```
Cleaned CSV → Feature Engineering → Encoding/Scaling → Train/Test Split → ML-ready data
```

Engineered features: `tenure_group`, `total_services`, `has_security_service`, `has_streaming_service`, `is_month_to_month`, `is_long_term_customer`, `average_monthly_revenue`

## Project Structure

```
Customer-Churn-Prediction/
├── data/raw/                 # Original Telco CSV
├── data/processed/           # Cleaned + featured + train/test splits
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_feature_engineering.ipynb
├── src/
│   ├── check_data.py
│   ├── load_to_postgres.py
│   ├── preprocessing.py
│   ├── run_eda_day1.py
│   └── run_day2.py
├── models/
│   ├── preprocessor.pkl
│   └── feature_columns.pkl
├── reports/
│   ├── figures/              # Day 1 EDA charts
│   └── feature_summary.csv   # Day 2 feature dictionary
├── sql/
├── api/
├── dashboard/
├── tests/
├── .gitignore
├── requirements.txt
└── README.md
```

## Quick Start (Windows)

```powershell
cd Customer-Churn-Prediction
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a local `.env` file (do not commit it):

```env
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/churn_ltv_db
```

If your password contains `@`, URL-encode it as `%40`.

## Dataset

IBM Telco Customer Churn CSV:

`data/raw/telco_churn.csv`

## Day 1 Commands

```powershell
python src/check_data.py
python src/run_eda_day1.py
python src/load_to_postgres.py
```

## Day 2 Commands

```powershell
# Run feature engineering + preprocessing end-to-end
python src/run_day2.py

# Or open notebooks/02_feature_engineering.ipynb and Run All
```

Outputs:

- `data/processed/X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`
- `models/preprocessor.pkl`
- `models/feature_columns.pkl`
- `reports/feature_summary.csv`
