"""
Day 4 runner — LTV proxy + SHAP explainability using the saved Day 3 model.

Run from project root:
    python src/run_day4.py

Does NOT retrain the churn model.
Does NOT build FastAPI / dashboard / Docker.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from churn_model import (  # noqa: E402
    load_churn_model,
    load_preprocessor,
    predict_churn,
    risk_level,
)
from ltv_model import (  # noqa: E402
    FIGURES_DIR,
    LTV_METHODOLOGY,
    REPORTS_DIR,
    add_ltv_columns,
    add_retention_priority,
    assign_ltv_segments,
    high_risk_high_ltv,
)
from preprocessing import create_features, get_feature_matrix  # noqa: E402

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def score_all_customers(cleaned: pd.DataFrame, model, preprocessor) -> pd.DataFrame:
    """Score every customer with the saved Day 3 model (no retraining)."""
    featured = create_features(cleaned)
    X, y = get_feature_matrix(featured)
    X_proc = preprocessor.transform(X)
    feature_names = list(preprocessor.get_feature_names_out())
    X_proc_df = pd.DataFrame(X_proc, columns=feature_names)

    preds = predict_churn(model, X_proc_df)
    out = cleaned.copy().reset_index(drop=True)
    out["churn_prediction"] = preds["prediction"]
    out["churn_probability"] = preds["probability"]
    out["risk_level"] = preds["risk_level"]
    out["actual_churn_encoded"] = y.reset_index(drop=True)
    return out, X_proc_df


def plot_ltv_figures(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    sns.histplot(df["estimated_ltv"], bins=40, color="steelblue")
    plt.title("Estimated Lifetime Revenue (LTV Proxy) Distribution")
    plt.xlabel("estimated_ltv")
    plt.ylabel("Customers")
    plt.tight_layout()
    path = FIGURES_DIR / "ltv_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x="Churn", y="estimated_ltv", hue="Churn", palette="Set2", legend=False)
    plt.title("Estimated LTV by Churn Status")
    plt.tight_layout()
    path = FIGURES_DIR / "ltv_by_churn.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Contract", y="estimated_ltv", hue="Contract", palette="Set2", legend=False)
    plt.title("Estimated LTV by Contract")
    plt.tight_layout()
    path = FIGURES_DIR / "ltv_by_contract.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    plt.figure(figsize=(7, 5))
    order = ["Low", "Medium", "High"]
    sns.boxplot(
        data=df,
        x="risk_level",
        y="estimated_ltv",
        order=order,
        hue="risk_level",
        hue_order=order,
        palette="Set2",
        legend=False,
    )
    plt.title("Estimated LTV by Churn Risk Level")
    plt.tight_layout()
    path = FIGURES_DIR / "ltv_by_risk_level.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    plt.figure(figsize=(6, 4))
    seg_counts = df["ltv_segment"].value_counts().reindex(["Low", "Medium", "High"])
    sns.barplot(
        x=seg_counts.index,
        y=seg_counts.values,
        hue=seg_counts.index,
        palette=["#a6cee3", "#1f78b4", "#08306b"],
        legend=False,
    )
    plt.title("LTV Segment Counts (Terciles)")
    plt.ylabel("Customers")
    plt.tight_layout()
    path = FIGURES_DIR / "ltv_segments.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def run_shap(model, X_proc_df: pd.DataFrame, scored: pd.DataFrame, sample_size: int = 500) -> pd.DataFrame:
    """
    Global + one individual SHAP explanation using TreeExplainer (XGBoost).
    """
    model_name = type(model).__name__
    print(f"\nSaved model type: {model_name}")
    if "XGB" not in model_name and "Forest" not in model_name and "Tree" not in model_name:
        raise TypeError(
            f"Expected a tree-based Day 3 model for TreeExplainer, got {model_name}"
        )

    n = min(sample_size, len(X_proc_df))
    # Use a reproducible sample from the full scored set
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_proc_df), size=n, replace=False)
    X_sample = X_proc_df.iloc[idx]
    print(f"Computing SHAP values on sample of {n} customers...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # For binary XGBoost, shap_values is (n_samples, n_features) for the positive class
    if isinstance(shap_values, list):
        shap_matrix = shap_values[1]
    else:
        shap_matrix = shap_values

    assert shap_matrix.shape[1] == X_sample.shape[1], (
        f"SHAP features {shap_matrix.shape[1]} != model features {X_sample.shape[1]}"
    )
    print(f"SHAP matrix shape: {shap_matrix.shape} (matches model inputs)")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure()
    shap.summary_plot(shap_matrix, X_sample, show=False, max_display=20)
    plt.tight_layout()
    summary_path = FIGURES_DIR / "shap_summary.png"
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {summary_path}")

    plt.figure()
    shap.summary_plot(shap_matrix, X_sample, plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    bar_path = FIGURES_DIR / "shap_feature_importance.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {bar_path}")

    mean_abs = np.abs(shap_matrix).mean(axis=0)
    importance = (
        pd.DataFrame(
            {
                "feature": X_sample.columns,
                "mean_absolute_shap_value": mean_abs,
            }
        )
        .sort_values("mean_absolute_shap_value", ascending=False)
        .reset_index(drop=True)
    )
    imp_path = REPORTS_DIR / "shap_feature_importance.csv"
    importance.to_csv(imp_path, index=False)
    print(f"Saved: {imp_path}")
    print("\nTop 10 SHAP features (actual):")
    print(importance.head(10).to_string(index=False))

    # Individual explanation: pick a High-risk customer from the sample if possible
    sample_meta = scored.iloc[idx].reset_index(drop=True)
    high_idx = sample_meta.index[sample_meta["risk_level"] == "High"]
    if len(high_idx) > 0:
        local_i = int(high_idx[0])
    else:
        local_i = 0

    customer = sample_meta.iloc[local_i]
    local_shap = shap_matrix[local_i]
    local_df = pd.DataFrame(
        {
            "feature": X_sample.columns,
            "shap_value": local_shap,
            "feature_value": X_sample.iloc[local_i].values,
        }
    )
    local_df["abs_shap"] = local_df["shap_value"].abs()
    local_df = local_df.sort_values("abs_shap", ascending=False)

    toward_churn = local_df[local_df["shap_value"] > 0].head(5)
    toward_stay = local_df[local_df["shap_value"] < 0].head(5)

    print("\n" + "=" * 60)
    print("INDIVIDUAL CUSTOMER SHAP EXPLANATION")
    print("=" * 60)
    print(f"Customer ID: {customer['customerID']}")
    print(f"Actual Churn: {customer['Churn']}")
    print(f"Predicted Churn: {customer['churn_prediction']}")
    print(f"Churn Probability: {customer['churn_probability']:.4f}")
    print(f"Risk Level: {customer['risk_level']}")
    print("\nTop factors pushing TOWARD churn (positive SHAP):")
    print(toward_churn[["feature", "shap_value"]].to_string(index=False))
    print("\nTop factors pushing TOWARD stay (negative SHAP):")
    print(toward_stay[["feature", "shap_value"]].to_string(index=False))
    print(
        "\nInterpretation: These SHAP values show how features contributed to THIS "
        "model prediction. They are associations with the model's output, not proof "
        "that a feature causes churn."
    )

    # Force / waterfall-style bar for this customer
    top_local = local_df.head(12).sort_values("shap_value")
    plt.figure(figsize=(8, 6))
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in top_local["shap_value"]]
    plt.barh(top_local["feature"], top_local["shap_value"], color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title(f"SHAP contributions — {customer['customerID']}")
    plt.xlabel("SHAP value (→ churn if positive)")
    plt.tight_layout()
    local_path = FIGURES_DIR / "shap_individual_explanation.png"
    plt.savefig(local_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {local_path}")

    # Save individual explanation CSV
    expl_path = REPORTS_DIR / "shap_individual_explanation.csv"
    pd.DataFrame(
        {
            "customerID": customer["customerID"],
            "actual_churn": customer["Churn"],
            "predicted_churn": customer["churn_prediction"],
            "churn_probability": customer["churn_probability"],
            "risk_level": customer["risk_level"],
            "feature": local_df["feature"],
            "shap_value": local_df["shap_value"],
            "feature_value": local_df["feature_value"],
        }
    ).to_csv(expl_path, index=False)
    print(f"Saved: {expl_path}")

    return importance


def main() -> None:
    print("=" * 60)
    print("DAY 4 — LTV PROXY + SHAP")
    print("=" * 60)
    print(LTV_METHODOLOGY)

    model = load_churn_model()
    preprocessor = load_preprocessor()
    meta = joblib.load(MODELS_DIR / "churn_model_metadata.pkl")
    print(f"Loaded Day 3 model: {type(model).__name__} ({meta.get('best_model_name')})")
    print("No retraining performed.")

    cleaned = pd.read_csv(PROCESSED_DIR / "cleaned_telco.csv")
    print(f"\nLoaded cleaned customers: {cleaned.shape}")

    scored, X_proc_df = score_all_customers(cleaned, model, preprocessor)
    print("Scored all customers with Day 3 model.")
    print(scored[["customerID", "churn_probability", "risk_level"]].head())

    # --- LTV ---
    ltv_df = add_ltv_columns(scored)
    ltv_df = assign_ltv_segments(ltv_df)
    ltv_df = add_retention_priority(ltv_df)

    q33 = ltv_df["estimated_ltv"].quantile(1 / 3)
    q66 = ltv_df["estimated_ltv"].quantile(2 / 3)
    print(f"\nLTV tercile thresholds (data-driven): Low <= {q33:.2f}, Medium <= {q66:.2f}, High > {q66:.2f}")
    print("\nSample LTV rows:")
    print(
        ltv_df[
            [
                "customerID",
                "tenure",
                "MonthlyCharges",
                "estimated_ltv",
                "ltv_segment",
                "Churn",
                "churn_probability",
                "risk_level",
                "retention_priority",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    plot_ltv_figures(ltv_df)

    retention = ltv_df[
        [
            "customerID",
            "churn_prediction",
            "churn_probability",
            "risk_level",
            "estimated_ltv",
            "ltv_segment",
            "retention_priority",
        ]
    ].copy()
    retention_path = REPORTS_DIR / "retention_priority.csv"
    retention.to_csv(retention_path, index=False)
    print(f"\nSaved: {retention_path} ({len(retention)} rows)")

    # Also save full LTV table
    ltv_path = PROCESSED_DIR / "customer_ltv.csv"
    ltv_df.to_csv(ltv_path, index=False)
    print(f"Saved: {ltv_path}")

    hrhl = high_risk_high_ltv(ltv_df)
    hrhl_path = REPORTS_DIR / "high_risk_high_ltv_customers.csv"
    hrhl.to_csv(hrhl_path, index=False)
    print(f"\nHigh-risk + High-LTV customers: {len(hrhl)}")
    print(hrhl.head(15).to_string(index=False))
    print(f"Saved: {hrhl_path}")
    print(
        "\nThese customers may deserve higher-priority retention attention because "
        "they have both higher estimated churn risk and higher estimated lifetime "
        "revenue. Retention actions are not guaranteed to prevent churn."
    )

    # Priority counts
    print("\nRetention priority counts:")
    print(ltv_df["retention_priority"].value_counts().to_string())

    # --- SHAP ---
    importance = run_shap(model, X_proc_df, ltv_df, sample_size=500)

    # Business insights from actual numbers
    print("\n" + "=" * 60)
    print("BUSINESS INSIGHTS (from actual Day 4 outputs)")
    print("=" * 60)
    top_feats = importance.head(5)["feature"].tolist()
    print("1. Top features influencing churn predictions (SHAP):")
    for i, f in enumerate(top_feats, 1):
        print(f"   {i}. {f}")

    print("\n2. Mean estimated_ltv by segment:")
    print(ltv_df.groupby("ltv_segment")["estimated_ltv"].mean().round(2).to_string())

    print("\n3. High-risk + High-LTV count:", len(hrhl))
    print("   Mean estimated_ltv in that group:", round(hrhl["estimated_ltv"].mean(), 2) if len(hrhl) else "n/a")

    print("\n4. Retention opportunity to investigate:")
    print(
        f"   Focus first on Critical priority customers "
        f"({(ltv_df['retention_priority']=='Critical').sum()} customers) "
        "and High-risk/High-LTV accounts for personalized retention offers."
    )

    results_path = REPORTS_DIR / "day4_results.txt"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("Day 4 Results\n")
        f.write("=============\n\n")
        f.write(LTV_METHODOLOGY + "\n")
        f.write(f"LTV terciles: q33={q33:.2f}, q66={q66:.2f}\n")
        f.write(f"High-risk High-LTV customers: {len(hrhl)}\n")
        f.write(f"Critical priority customers: {(ltv_df['retention_priority']=='Critical').sum()}\n\n")
        f.write("Top SHAP features:\n")
        f.write(importance.head(15).to_string(index=False))
        f.write("\n")
    print(f"\nSaved: {results_path}")
    print("\nDAY 4 RUNNER COMPLETE")


if __name__ == "__main__":
    main()
