"""
Day 2 runner — feature engineering, train/test split, fit preprocessor on train only.

Run from project root:
    python src/run_day2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from preprocessing import (  # noqa: E402
    create_features,
    get_feature_matrix,
    get_feature_summary,
    build_preprocessor,
)

CLEANED_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_telco.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def print_balance(name: str, y: pd.Series) -> None:
    counts = y.value_counts().sort_index()
    pcts = (y.value_counts(normalize=True).sort_index() * 100).round(2)
    print(f"\n{name}")
    print("  counts:")
    for label, count in counts.items():
        label_name = "No (0)" if label == 0 else "Yes (1)"
        print(f"    {label_name}: {count}")
    print("  percentages:")
    for label, pct in pcts.items():
        label_name = "No (0)" if label == 0 else "Yes (1)"
        print(f"    {label_name}: {pct}%")


def main() -> None:
    print("=" * 60)
    print("DAY 2 — FEATURE ENGINEERING & PREPROCESSING")
    print("=" * 60)

    # --- Load cleaned Day 1 data ---
    df = pd.read_csv(CLEANED_PATH)
    print(f"\nLoaded: {CLEANED_PATH}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"TotalCharges dtype: {df['TotalCharges'].dtype}")
    print(f"Missing values: {int(df.isnull().sum().sum())}")
    print(df.head())

    # --- Feature engineering (no Churn used) ---
    df_feat = create_features(df)
    print("\nEngineered columns added:")
    for col in [
        "tenure_group",
        "total_services",
        "has_security_service",
        "has_streaming_service",
        "is_month_to_month",
        "is_long_term_customer",
        "average_monthly_revenue",
    ]:
        print(f"  - {col}")
    print(df_feat[
        [
            "customerID",
            "tenure",
            "tenure_group",
            "total_services",
            "has_security_service",
            "has_streaming_service",
            "is_month_to_month",
            "is_long_term_customer",
            "average_monthly_revenue",
            "Churn",
        ]
    ].head())

    # Save featured (pre-encoding) dataset for inspection
    featured_path = PROCESSED_DIR / "featured_telco.csv"
    df_feat.to_csv(featured_path, index=False)
    print(f"\nSaved featured dataset: {featured_path}")

    X, y = get_feature_matrix(df_feat)
    print(f"\nML feature matrix X shape: {X.shape}")
    print(f"Target y shape: {y.shape}")
    print(f"customerID excluded: {'customerID' not in X.columns}")

    # --- Class balance (full data) ---
    print_balance("Full dataset Churn balance", y)

    # --- Train/test split BEFORE fitting preprocessor (prevents leakage) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\nTrain/test split (stratified, random_state=42):")
    print(f"Training features: {X_train.shape}")
    print(f"Testing features:  {X_test.shape}")
    print(f"Training target:   {y_train.shape}")
    print(f"Testing target:    {y_test.shape}")

    print_balance("Training set Churn balance", y_train)
    print_balance("Testing set Churn balance", y_test)

    # --- Fit preprocessor on TRAIN only ---
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = list(preprocessor.get_feature_names_out())
    print(f"\nTransformed training shape: {X_train_processed.shape}")
    print(f"Transformed testing shape:  {X_test_processed.shape}")
    print(f"Output feature count: {len(feature_names)}")
    print(f"NaNs in train transform: {np.isnan(X_train_processed).sum()}")
    print(f"NaNs in test transform:  {np.isnan(X_test_processed).sum()}")

    # Dense arrays (OneHotEncoder sparse_output=False) → CSV is appropriate
    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train_df.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test_df.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False, header=["Churn"])
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False, header=["Churn"])

    # Also keep raw (pre-transform) splits for debugging
    X_train.to_csv(PROCESSED_DIR / "X_train_raw.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test_raw.csv", index=False)

    print("\nSaved processed splits:")
    for name in [
        "X_train.csv",
        "X_test.csv",
        "y_train.csv",
        "y_test.csv",
        "X_train_raw.csv",
        "X_test_raw.csv",
    ]:
        print(f"  - {PROCESSED_DIR / name}")

    # --- Save preprocessor + feature column metadata ---
    preprocessor_path = MODELS_DIR / "preprocessor.pkl"
    feature_columns_path = MODELS_DIR / "feature_columns.pkl"

    joblib.dump(preprocessor, preprocessor_path)
    joblib.dump(
        {
            "input_features": list(X.columns),
            "output_features": feature_names,
            "numeric_features": list(
                __import__("preprocessing").NUMERIC_FEATURES
            ),
            "binary_features": list(
                __import__("preprocessing").BINARY_FEATURES
            ),
            "categorical_features": list(
                __import__("preprocessing").CATEGORICAL_FEATURES
            ),
        },
        feature_columns_path,
    )
    print(f"\nSaved preprocessor: {preprocessor_path}")
    print(f"Saved feature columns: {feature_columns_path}")

    # Reload check
    loaded = joblib.load(preprocessor_path)
    reload_shape = loaded.transform(X_test.head(3)).shape
    print(f"Reload check transform shape: {reload_shape}")

    # Feature summary table
    summary = get_feature_summary()
    summary_path = REPORTS_DIR / "feature_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nFeature summary saved: {summary_path}")
    print(summary.to_string(index=False))

    print("\n" + "=" * 60)
    print("DAY 2 RUNNER COMPLETE — no ML model trained (that is Day 3)")
    print("=" * 60)


if __name__ == "__main__":
    main()
