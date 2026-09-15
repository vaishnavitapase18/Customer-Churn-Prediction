"""
Day 1 — Load cleaned Telco data into PostgreSQL.

Prerequisites:
1. Edit .env and replace YOUR_PASSWORD with your real PostgreSQL password.
2. Create the database: churn_ltv_db
3. Make sure data/processed/cleaned_telco.csv exists (run the EDA notebook first).

Run from the project root:
    python src/load_to_postgres.py
"""

from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_telco.csv"
ENV_PATH = PROJECT_ROOT / ".env"


def main() -> None:
    print("=" * 60)
    print("LOAD CLEANED DATA INTO POSTGRESQL")
    print("=" * 60)

    # Load credentials from .env (never hard-code the password)
    load_dotenv(ENV_PATH)
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL not found. Create a .env file in the project root with:\n"
            "DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/churn_ltv_db"
        )

    if "YOUR_PASSWORD" in database_url:
        raise ValueError(
            "Please replace YOUR_PASSWORD in the .env file with your real PostgreSQL password."
        )

    if not CLEANED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at: {CLEANED_DATA_PATH}\n"
            "Run notebooks/01_eda.ipynb first to create cleaned_telco.csv"
        )

    print(f"\nLoading cleaned CSV: {CLEANED_DATA_PATH}")
    df = pd.read_csv(CLEANED_DATA_PATH)
    print(f"Rows: {len(df)} | Columns: {len(df.columns)}")

    print("\nConnecting to PostgreSQL...")
    engine = create_engine(database_url)

    print("Writing table: customers (if_exists='replace')...")
    df.to_sql("customers", engine, if_exists="replace", index=False)

    # Verify the load
    with engine.connect() as conn:
        row_count = conn.execute(text("SELECT COUNT(*) FROM customers")).scalar()
        sample = pd.read_sql(text("SELECT * FROM customers LIMIT 5"), conn)

    print(f"\nSuccess! Rows in customers table: {row_count}")
    print("\nSample rows from PostgreSQL:")
    print(sample)
    print("\n" + "=" * 60)
    print("Load complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
