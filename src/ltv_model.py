"""
Day 4 — Estimated Lifetime Revenue (LTV Proxy) helpers.

IMPORTANT LIMITATION:
The IBM Telco dataset does NOT contain a true future Customer Lifetime Value
target. Values produced here are an ESTIMATE / PROXY based on available
historical fields (MonthlyCharges, tenure, Contract) plus Day 3 churn risk.
They are NOT actual future revenue predictions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Documented business assumption for expected remaining months by contract
# (used only for customers who have NOT already churned).
# These are MVP assumptions — not scientifically optimal forecasts.
CONTRACT_BASE_REMAINING_MONTHS: Dict[str, float] = {
    "Month-to-month": 6.0,
    "One year": 12.0,
    "Two year": 24.0,
}

LTV_METHODOLOGY = """
Estimated Lifetime Revenue (LTV Proxy)
======================================
Formula:
  estimated_ltv = MonthlyCharges x estimated_lifetime_months

  estimated_lifetime_months = tenure + expected_remaining_months

  If Churn == Yes (already left):
      expected_remaining_months = 0
  Else:
      expected_remaining_months =
          CONTRACT_BASE_REMAINING_MONTHS[Contract] x (1 - churn_probability)

Contract base remaining months (documented assumption):
  Month-to-month -> 6
  One year      -> 12
  Two year      -> 24

This is NOT actual future LTV. It is a transparent proxy for prioritization.
"""


def expected_remaining_months(
    contract: str,
    churned: bool,
    churn_probability: float,
) -> float:
    """Compute expected remaining months under the documented LTV proxy rules."""
    if churned:
        return 0.0
    base = CONTRACT_BASE_REMAINING_MONTHS.get(str(contract), 6.0)
    p = float(np.clip(churn_probability, 0.0, 1.0))
    return float(base * (1.0 - p))


def calculate_estimated_ltv(
    monthly_charges: float,
    tenure: float,
    contract: str,
    churned: bool,
    churn_probability: float,
) -> float:
    """
    estimated_ltv = MonthlyCharges × (tenure + expected_remaining_months)
    """
    remaining = expected_remaining_months(contract, churned, churn_probability)
    lifetime_months = float(tenure) + remaining
    return float(monthly_charges) * lifetime_months


def add_ltv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add LTV proxy columns to a customer dataframe.

    Required columns:
      MonthlyCharges, tenure, Contract, Churn, churn_probability
    """
    required = ["MonthlyCharges", "tenure", "Contract", "Churn", "churn_probability"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for LTV calculation: {missing}")

    out = df.copy()
    remaining = []
    lifetimes = []
    ltvs = []
    for _, row in out.iterrows():
        churned = str(row["Churn"]) == "Yes"
        rem = expected_remaining_months(
            row["Contract"], churned, float(row["churn_probability"])
        )
        life = float(row["tenure"]) + rem
        ltv = float(row["MonthlyCharges"]) * life
        remaining.append(rem)
        lifetimes.append(life)
        ltvs.append(ltv)

    out["expected_remaining_months"] = remaining
    out["estimated_lifetime_months"] = lifetimes
    out["estimated_ltv"] = ltvs
    return out


def assign_ltv_segments(df: pd.DataFrame, ltv_col: str = "estimated_ltv") -> pd.DataFrame:
    """
    Assign Low / Medium / High LTV segments using terciles (33rd / 66th percentiles).

    Data-driven: thresholds come from the actual estimated_ltv distribution.
    """
    out = df.copy()
    q33 = out[ltv_col].quantile(1 / 3)
    q66 = out[ltv_col].quantile(2 / 3)

    def _seg(v: float) -> str:
        if v <= q33:
            return "Low"
        if v <= q66:
            return "Medium"
        return "High"

    out["ltv_segment"] = out[ltv_col].apply(_seg)
    out.attrs["ltv_q33"] = float(q33)
    out.attrs["ltv_q66"] = float(q66)
    return out


def assign_retention_priority(risk_level: str, ltv_segment: str) -> str:
    """
    Documented MVP priority rules (not universal business law):

    High risk + High LTV     → Critical
    High risk + Medium LTV   → High
    High risk + Low LTV      → Medium
    Medium risk + High LTV   → High
    Medium risk + Medium LTV → Medium
    Medium risk + Low LTV    → Low
    Low risk + High LTV      → Monitor
    Low risk + Medium/Low    → Low
    """
    risk = str(risk_level)
    ltv = str(ltv_segment)
    key = (risk, ltv)
    mapping = {
        ("High", "High"): "Critical",
        ("High", "Medium"): "High",
        ("High", "Low"): "Medium",
        ("Medium", "High"): "High",
        ("Medium", "Medium"): "Medium",
        ("Medium", "Low"): "Low",
        ("Low", "High"): "Monitor",
        ("Low", "Medium"): "Low",
        ("Low", "Low"): "Low",
    }
    return mapping.get(key, "Low")


def add_retention_priority(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["retention_priority"] = [
        assign_retention_priority(r, l)
        for r, l in zip(out["risk_level"], out["ltv_segment"])
    ]
    return out


def high_risk_high_ltv(df: pd.DataFrame) -> pd.DataFrame:
    """Customers with High churn risk AND High LTV segment."""
    mask = (df["risk_level"] == "High") & (df["ltv_segment"] == "High")
    cols = [
        c
        for c in [
            "customerID",
            "Churn",
            "churn_prediction",
            "churn_probability",
            "risk_level",
            "MonthlyCharges",
            "tenure",
            "Contract",
            "estimated_ltv",
            "ltv_segment",
            "retention_priority",
        ]
        if c in df.columns
    ]
    return df.loc[mask, cols].sort_values("estimated_ltv", ascending=False).reset_index(drop=True)
