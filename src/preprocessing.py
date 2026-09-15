"""
Day 2 — Feature engineering and reusable preprocessing for Telco churn.

Important rules:
- Never use Churn to create features (avoids target leakage).
- Fit the sklearn preprocessor ONLY on training data.
- Exclude customerID from ML features (it is an identifier, not a predictor).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------------
# Column groups (after feature engineering)
# ---------------------------------------------------------------------------

ID_COL = "customerID"
TARGET_COL = "Churn"

# Binary / already 0-1 numeric flags (pass through, no scaling needed)
BINARY_FEATURES: List[str] = [
    "SeniorCitizen",
    "has_security_service",
    "has_streaming_service",
    "is_month_to_month",
    "is_long_term_customer",
]

# Continuous numeric features to scale
NUMERIC_FEATURES: List[str] = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "total_services",
    "average_monthly_revenue",
]

# Nominal categorical features to one-hot encode
CATEGORICAL_FEATURES: List[str] = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "tenure_group",
]

SERVICE_YES_COLS: List[str] = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

SECURITY_COLS: List[str] = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
]

STREAMING_COLS: List[str] = [
    "StreamingTV",
    "StreamingMovies",
]


def encode_target(y: pd.Series) -> pd.Series:
    """Encode Churn: No -> 0, Yes -> 1."""
    mapping = {"No": 0, "Yes": 1}
    encoded = y.map(mapping)
    if encoded.isna().any():
        bad = y[encoded.isna()].unique()
        raise ValueError(f"Unexpected Churn values: {bad}")
    return encoded.astype(int)


def _count_yes(row_values: pd.Series) -> int:
    return int((row_values == "Yes").sum())


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create business features from cleaned Telco data.

    Uses only information available about the customer today.
    Does NOT use the Churn column.
    """
    out = df.copy()

    # 1) tenure_group — buckets of how long the customer has stayed
    out["tenure_group"] = pd.cut(
        out["tenure"],
        bins=[-0.1, 12, 24, 48, 72],
        labels=["0-12", "13-24", "25-48", "49-72"],
    ).astype(str)

    # 2) total_services — how many services the customer actively uses
    out["total_services"] = out[SERVICE_YES_COLS].apply(_count_yes, axis=1)

    # 3) has_security_service — any security / support add-on
    out["has_security_service"] = (
        out[SECURITY_COLS].eq("Yes").any(axis=1).astype(int)
    )

    # 4) has_streaming_service — TV and/or movies streaming
    out["has_streaming_service"] = (
        out[STREAMING_COLS].eq("Yes").any(axis=1).astype(int)
    )

    # 5) is_month_to_month — flexible contracts churn more often
    out["is_month_to_month"] = (out["Contract"] == "Month-to-month").astype(int)

    # 6) is_long_term_customer — tenure of 24+ months
    out["is_long_term_customer"] = (out["tenure"] >= 24).astype(int)

    # 7) average_monthly_revenue — spend intensity
    # For tenure=0 (brand-new customers), use MonthlyCharges instead of dividing by 0.
    out["average_monthly_revenue"] = np.where(
        out["tenure"] > 0,
        out["TotalCharges"] / out["tenure"],
        out["MonthlyCharges"],
    )

    return out


def get_feature_matrix(df_featured: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Return X (features for ML) and y (encoded target).

    Drops customerID (identifier) and raw Churn string.
    """
    if TARGET_COL not in df_featured.columns:
        raise KeyError("Churn column missing from dataframe.")

    y = encode_target(df_featured[TARGET_COL])

    feature_cols = BINARY_FEATURES + NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing = [c for c in feature_cols if c not in df_featured.columns]
    if missing:
        raise KeyError(f"Missing engineered/base columns: {missing}")

    X = df_featured[feature_cols].copy()
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """
    Build an unfitted ColumnTransformer:

    - StandardScaler for continuous numeric columns
    - passthrough for binary 0/1 columns
    - OneHotEncoder(handle_unknown='ignore') for categoricals
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def get_feature_summary() -> pd.DataFrame:
    """Human-readable feature dictionary for Day 2 documentation."""
    rows = [
        {
            "Feature": "customerID",
            "Type": "Identifier",
            "Description": "Unique customer identifier",
            "Used for ML": "No",
        },
        {
            "Feature": "Churn",
            "Type": "Target",
            "Description": "Whether the customer left (Yes/No -> 1/0)",
            "Used for ML": "Target only",
        },
        {
            "Feature": "gender",
            "Type": "Categorical",
            "Description": "Customer gender",
            "Used for ML": "Yes",
        },
        {
            "Feature": "SeniorCitizen",
            "Type": "Binary numeric",
            "Description": "1 if senior citizen, else 0",
            "Used for ML": "Yes",
        },
        {
            "Feature": "Partner",
            "Type": "Categorical",
            "Description": "Whether customer has a partner",
            "Used for ML": "Yes",
        },
        {
            "Feature": "Dependents",
            "Type": "Categorical",
            "Description": "Whether customer has dependents",
            "Used for ML": "Yes",
        },
        {
            "Feature": "tenure",
            "Type": "Numerical",
            "Description": "Months the customer has stayed with the company",
            "Used for ML": "Yes",
        },
        {
            "Feature": "PhoneService",
            "Type": "Categorical",
            "Description": "Has phone service",
            "Used for ML": "Yes",
        },
        {
            "Feature": "MultipleLines",
            "Type": "Categorical",
            "Description": "Multiple phone lines status",
            "Used for ML": "Yes",
        },
        {
            "Feature": "InternetService",
            "Type": "Categorical",
            "Description": "Internet service type (DSL / Fiber / None)",
            "Used for ML": "Yes",
        },
        {
            "Feature": "OnlineSecurity",
            "Type": "Categorical",
            "Description": "Online security add-on",
            "Used for ML": "Yes",
        },
        {
            "Feature": "OnlineBackup",
            "Type": "Categorical",
            "Description": "Online backup add-on",
            "Used for ML": "Yes",
        },
        {
            "Feature": "DeviceProtection",
            "Type": "Categorical",
            "Description": "Device protection add-on",
            "Used for ML": "Yes",
        },
        {
            "Feature": "TechSupport",
            "Type": "Categorical",
            "Description": "Tech support add-on",
            "Used for ML": "Yes",
        },
        {
            "Feature": "StreamingTV",
            "Type": "Categorical",
            "Description": "Streaming TV add-on",
            "Used for ML": "Yes",
        },
        {
            "Feature": "StreamingMovies",
            "Type": "Categorical",
            "Description": "Streaming movies add-on",
            "Used for ML": "Yes",
        },
        {
            "Feature": "Contract",
            "Type": "Categorical",
            "Description": "Contract type (month-to-month / 1yr / 2yr)",
            "Used for ML": "Yes",
        },
        {
            "Feature": "PaperlessBilling",
            "Type": "Categorical",
            "Description": "Uses paperless billing",
            "Used for ML": "Yes",
        },
        {
            "Feature": "PaymentMethod",
            "Type": "Categorical",
            "Description": "How the customer pays",
            "Used for ML": "Yes",
        },
        {
            "Feature": "MonthlyCharges",
            "Type": "Numerical",
            "Description": "Current monthly bill amount",
            "Used for ML": "Yes",
        },
        {
            "Feature": "TotalCharges",
            "Type": "Numerical",
            "Description": "Total amount charged to date",
            "Used for ML": "Yes",
        },
        {
            "Feature": "tenure_group",
            "Type": "Categorical (engineered)",
            "Description": "Tenure bucket: 0-12 / 13-24 / 25-48 / 49-72 months",
            "Used for ML": "Yes",
        },
        {
            "Feature": "total_services",
            "Type": "Numerical (engineered)",
            "Description": "Count of services with value Yes",
            "Used for ML": "Yes",
        },
        {
            "Feature": "has_security_service",
            "Type": "Binary (engineered)",
            "Description": "1 if any of OnlineSecurity/Backup/DeviceProtection/TechSupport is Yes",
            "Used for ML": "Yes",
        },
        {
            "Feature": "has_streaming_service",
            "Type": "Binary (engineered)",
            "Description": "1 if StreamingTV or StreamingMovies is Yes",
            "Used for ML": "Yes",
        },
        {
            "Feature": "is_month_to_month",
            "Type": "Binary (engineered)",
            "Description": "1 if Contract is Month-to-month",
            "Used for ML": "Yes",
        },
        {
            "Feature": "is_long_term_customer",
            "Type": "Binary (engineered)",
            "Description": "1 if tenure >= 24 months",
            "Used for ML": "Yes",
        },
        {
            "Feature": "average_monthly_revenue",
            "Type": "Numerical (engineered)",
            "Description": "TotalCharges / tenure (or MonthlyCharges if tenure=0)",
            "Used for ML": "Yes",
        },
    ]
    return pd.DataFrame(rows)
