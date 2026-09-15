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

## Machine Learning Results

Actual test-set results (Day 2 split: 5,634 train / 1,409 test, `random_state=42`, stratified).

Metrics below are for the **churn class (Yes = 1)**.

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.7970 | 0.6467 | 0.5187 | 0.5757 | 0.8418 |
| Random Forest | 0.7807 | 0.6102 | 0.4813 | 0.5381 | 0.8203 |
| **XGBoost (best)** | **0.8077** | **0.6758** | **0.5294** | **0.5937** | **0.8443** |

**Best model selected: XGBoost**

Selection was based primarily on **F1 Score**, then **Recall**, then **ROC-AUC** (not Accuracy alone), because the dataset is imbalanced (~26.54% churn).

**Test-set risk levels** (initial thresholds: Low &lt; 0.30, Medium &lt; 0.60, High ≥ 0.60):

| Risk | Customers |
|------|-----------|
| Low | 872 |
| Medium | 333 |
| High | 204 |

High-risk customers (204 / 1,409) are a practical first list for telecom retention outreach.

```
Cleaned → Features → Split → Preprocess → LR / RF / XGBoost → Best Model → Probability → Risk
```

## Project Structure

```
Customer-Churn-Prediction/
├── data/raw/
├── data/processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_churn_model.ipynb
├── src/
│   ├── check_data.py
│   ├── load_to_postgres.py
│   ├── preprocessing.py
│   ├── churn_model.py
│   ├── run_eda_day1.py
│   ├── run_day2.py
│   └── run_day3.py
├── models/
│   ├── preprocessor.pkl
│   ├── feature_columns.pkl
│   ├── churn_model.pkl
│   └── churn_model_metadata.pkl
├── reports/
│   ├── figures/
│   ├── model_comparison.csv
│   ├── sample_predictions.csv
│   └── feature_summary.csv
├── sql/
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

IBM Telco Customer Churn CSV: `data/raw/telco_churn.csv`

## Commands

```powershell
# Day 1
python src/check_data.py
python src/run_eda_day1.py
python src/load_to_postgres.py

# Day 2
python src/run_day2.py

# Day 3
python src/run_day3.py
# Or open notebooks/03_churn_model.ipynb and Run All
```
