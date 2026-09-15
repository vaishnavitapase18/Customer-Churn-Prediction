# Customer Churn Prediction & Lifetime Value (LTV) Engine

Beginner-friendly end-to-end project that predicts customer churn, estimates lifetime value, explains model decisions with SHAP, and serves results through FastAPI and a dashboard.

## Project Progress

### Day 1 — Completed ✅

- Dataset loaded (IBM Telco Customer Churn — 7,043 customers, 21 columns)
- Data cleaning completed (`TotalCharges` converted to numeric; 11 blank values for `tenure = 0` filled with 0)
- EDA completed (10 visualizations saved in `reports/figures/`)
- Churn analysis completed
- Cleaned dataset created (`data/processed/cleaned_telco.csv`)
- PostgreSQL database created (`churn_ltv_db`)
- Customer table created (`customers`)
- Data loaded into PostgreSQL (7,043 rows)
- SQL analysis completed (`sql/analysis_queries.sql`)

#### Day 1 pipeline

```
Telco CSV → Pandas → Cleaning → EDA → Cleaned CSV → PostgreSQL → SQL Analysis
```

#### Key findings from Day 1 (real results)

| Metric | Value |
|--------|-------|
| Overall churn rate | 26.54% (1,869 churned / 5,174 not churned) |
| Month-to-month contract churn | 42.71% |
| One-year contract churn | 11.27% |
| Two-year contract churn | 2.83% |
| Fiber optic internet churn | 41.89% |
| Electronic check payment churn | 45.29% |
| Avg tenure (churned vs not) | 17.98 vs 37.57 months |
| Avg monthly charges (churned vs not) | $74.44 vs $61.27 |

## Project Structure

```
Customer-Churn-Prediction/
├── data/raw/              # Original Telco CSV
├── data/processed/        # Cleaned dataset
├── notebooks/             # EDA and analysis notebooks
├── src/                   # Python scripts
├── models/                # Saved ML models (later)
├── api/                   # FastAPI app (later)
├── dashboard/             # Dashboard (later)
├── sql/                   # SQL analysis queries
├── tests/                 # Tests (later)
├── reports/figures/       # EDA charts
├── .env                   # Database credentials (local only — not in Git)
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
# Inspect raw data
python src/check_data.py

# Run cleaning + EDA (optional helper; same logic as the notebook)
python src/run_eda_day1.py

# Or open notebooks/01_eda.ipynb and Run All

# Load cleaned data into PostgreSQL (after creating churn_ltv_db and editing .env)
python src/load_to_postgres.py
```
