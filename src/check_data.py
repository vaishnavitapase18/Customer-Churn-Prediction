"""
Day 1 — Inspect the raw Telco Customer Churn dataset.

Run from the project root:
    python src/check_data.py
"""

from pathlib import Path

import pandas as pd

# Project root = parent of the src/ folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "telco_churn.csv"


def main() -> None:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {RAW_DATA_PATH}\n"
            "Please place the Telco CSV at data/raw/telco_churn.csv"
        )

    print("=" * 60)
    print("TELCO CUSTOMER CHURN — DATA INSPECTION")
    print("=" * 60)
    print(f"\nLoading: {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)

    print(f"\n1. Number of rows: {df.shape[0]}")
    print(f"2. Number of columns: {df.shape[1]}")

    print("\n3. Column names:")
    for i, col in enumerate(df.columns, start=1):
        print(f"   {i:2d}. {col}")

    print("\n4. First 5 rows:")
    print(df.head())

    print("\n5. Data types:")
    print(df.dtypes)

    print("\n6. Missing values (null counts):")
    missing = df.isnull().sum()
    print(missing)
    print(f"\n   Total missing cells: {int(missing.sum())}")

    # TotalCharges often has blank strings that look non-null
    if "TotalCharges" in df.columns:
        blank_total_charges = (df["TotalCharges"].astype(str).str.strip() == "").sum()
        print(f"   Blank TotalCharges values: {blank_total_charges}")

    print("\n7. Duplicate rows:")
    print(f"   Duplicate rows: {df.duplicated().sum()}")

    print("\n8. Duplicate customer IDs:")
    if "customerID" in df.columns:
        print(f"   Duplicate customerIDs: {df['customerID'].duplicated().sum()}")
    else:
        print("   customerID column not found.")

    print("\n9. Churn distribution:")
    if "Churn" in df.columns:
        churn_counts = df["Churn"].value_counts(dropna=False)
        churn_pct = df["Churn"].value_counts(normalize=True, dropna=False) * 100
        print(churn_counts)
        print("\n   Churn percentages:")
        print(churn_pct.round(2))
    else:
        print("   Churn column not found.")

    print("\n10. Descriptive statistics (numeric columns):")
    print(df.describe())

    print("\n" + "=" * 60)
    print("Inspection complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
