"""
Day 3 runner — train, evaluate, compare, save churn models.

Run from project root:
    python src/run_day3.py

Uses Day 2 processed X/y splits (preprocessor already fitted on train only).
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from churn_model import (  # noqa: E402
    FIGURES_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    build_sample_predictions_table,
    evaluate_all_models,
    get_test_customer_ids,
    load_churn_model,
    load_preprocessor,
    load_processed_splits,
    predict_churn,
    predict_from_raw_customer,
    risk_level,
    save_model,
    select_best_model,
    train_models,
)


def plot_confusion_matrix(cm: list, title: str, save_path: Path) -> None:
    labels = np.array([["TN", "FP"], ["FN", "TP"]])
    annot = np.array(
        [[f"{labels[i, j]}\n{cm[i][j]}" for j in range(2)] for i in range(2)]
    )
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        np.array(cm),
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=["Pred No", "Pred Yes"],
        yticklabels=["Actual No", "Actual Yes"],
        cbar=False,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main() -> None:
    print("=" * 60)
    print("DAY 3 — CHURN MACHINE LEARNING")
    print("=" * 60)

    # --- Load Day 2 processed data (already split + preprocessed on train only) ---
    X_train, X_test, y_train, y_test = load_processed_splits()
    print("\nDay 2 processed splits:")
    print(f"  Training features: {X_train.shape}")
    print(f"  Testing features:  {X_test.shape}")
    print(f"  Training target:   {y_train.shape}")
    print(f"  Testing target:    {y_test.shape}")
    print(f"  Train churn %: {y_train.mean() * 100:.2f}")
    print(f"  Test churn %:  {y_test.mean() * 100:.2f}")

    preprocessor = load_preprocessor()
    print(f"\nLoaded Day 2 preprocessor: {type(preprocessor).__name__}")

    # --- Train models ---
    print("\nTraining Logistic Regression, Random Forest, XGBoost...")
    print("  Logistic Regression: simple linear baseline for binary classification")
    print("  Random Forest: many decision trees; handles non-linear patterns")
    print("  XGBoost: gradient-boosted trees; often strong on tabular data")
    trained = train_models(X_train, y_train)

    # --- Evaluate ---
    comparison, details = evaluate_all_models(trained, X_test, y_test)
    print("\nModel comparison (churn class = Yes/1):")
    print(comparison.to_string(index=False))
    print(
        "\nNote: Accuracy alone is not enough because ~73.5% of customers do not churn."
    )
    print("A model that always predicts 'No' would look accurate but miss churners.")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    cm_files = {
        "Logistic Regression": "logistic_confusion_matrix.png",
        "Random Forest": "random_forest_confusion_matrix.png",
        "XGBoost": "xgboost_confusion_matrix.png",
    }
    for name, fname in cm_files.items():
        plot_confusion_matrix(
            details[name]["Confusion Matrix"],
            f"{name} Confusion Matrix",
            FIGURES_DIR / fname,
        )

    comparison_path = REPORTS_DIR / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    print(f"Saved comparison: {comparison_path}")

    # --- Select + save best ---
    best_name, explanation = select_best_model(comparison)
    best_model = trained[best_name]
    best_metrics = details[best_name]
    print(f"\nBest model: {best_name}")
    print(explanation)

    metadata = {
        "best_model_name": best_name,
        "selection_explanation": explanation,
        "metrics": {
            k: best_metrics[k]
            for k in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
        },
        "comparison": comparison.to_dict(orient="records"),
        "risk_thresholds": {
            "low": "< 0.30",
            "medium": "0.30 to < 0.60",
            "high": ">= 0.60",
            "note": "Initial business thresholds; can be tuned later.",
        },
    }
    save_model(best_model, metadata)
    print(f"Saved: {MODELS_DIR / 'churn_model.pkl'}")
    print(f"Saved: {MODELS_DIR / 'churn_model_metadata.pkl'}")

    # --- Probabilities + risk distribution ---
    all_preds = predict_churn(best_model, X_test)
    proba = np.array(all_preds["probability"])
    risks = pd.Series(all_preds["risk_level"])
    risk_counts = risks.value_counts().reindex(["Low", "Medium", "High"]).fillna(0).astype(int)
    print("\nRisk level distribution on test set:")
    print(risk_counts.to_string())
    high_risk_n = int(risk_counts.get("High", 0))

    plt.figure(figsize=(8, 5))
    sns.histplot(proba, bins=30, kde=True, color="steelblue")
    plt.axvline(0.30, color="orange", linestyle="--", label="Medium threshold (0.30)")
    plt.axvline(0.60, color="red", linestyle="--", label="High threshold (0.60)")
    plt.title(f"Predicted Churn Probability ({best_name})")
    plt.xlabel("Churn probability")
    plt.ylabel("Number of test customers")
    plt.legend()
    plt.tight_layout()
    hist_path = FIGURES_DIR / "churn_probability_histogram.png"
    plt.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {hist_path}")

    plt.figure(figsize=(6, 4))
    sns.barplot(
        x=risk_counts.index,
        y=risk_counts.values,
        hue=risk_counts.index,
        palette=["#2ca02c", "#ff7f0e", "#d62728"],
        legend=False,
    )
    plt.title("Risk Level Distribution (Test Set)")
    plt.ylabel("Customers")
    plt.tight_layout()
    risk_path = FIGURES_DIR / "risk_level_distribution.png"
    plt.savefig(risk_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {risk_path}")

    # --- Sample predictions with real customerIDs ---
    id_test = get_test_customer_ids()
    sample = build_sample_predictions_table(
        best_model, X_test, y_test, id_test, n=10
    )
    sample_path = REPORTS_DIR / "sample_predictions.csv"
    sample.to_csv(sample_path, index=False)
    print("\nSample predictions (first 10 test customers):")
    print(sample.to_string(index=False))
    print(f"Saved: {sample_path}")

    # --- Reload test (FastAPI will do this later) ---
    print("\n--- Reload verification ---")
    loaded_model = load_churn_model()
    loaded_pre = load_preprocessor()
    reload_preds = predict_churn(loaded_model, X_test.head(3))
    print("Reloaded model predictions for 3 rows:", reload_preds)

    cleaned = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "cleaned_telco.csv")
    one = cleaned.iloc[[0]]
    end_to_end = predict_from_raw_customer(one, loaded_model, loaded_pre)
    print("End-to-end raw customer prediction:", end_to_end)

    # --- Business interpretation (actual numbers only) ---
    print("\n" + "=" * 60)
    print("BUSINESS INTERPRETATION (from actual results)")
    print("=" * 60)
    print(f"Best model: {best_name}")
    print(f"F1 Score:  {best_metrics['F1 Score']:.4f}")
    print(f"Recall:    {best_metrics['Recall']:.4f}")
    print(f"ROC-AUC:   {best_metrics['ROC-AUC']:.4f}")
    print(f"Precision: {best_metrics['Precision']:.4f}")
    print(f"Accuracy:  {best_metrics['Accuracy']:.4f}")
    print(f"High-risk test customers (prob >= 0.60): {high_risk_n} of {len(X_test)}")
    print(
        "For a telecom company, these high-risk customers are the first group "
        "to review for retention offers (discounts, contract upgrades, support outreach)."
    )

    # Save a short results text for README / docs
    results_txt = REPORTS_DIR / "day3_results.txt"
    with open(results_txt, "w", encoding="utf-8") as f:
        f.write("Day 3 Churn ML Results\n")
        f.write("======================\n\n")
        f.write(comparison.to_string(index=False) + "\n\n")
        f.write(f"Best model: {best_name}\n")
        f.write(explanation + "\n\n")
        f.write(f"High-risk test customers: {high_risk_n} / {len(X_test)}\n")
        f.write("Risk thresholds: Low <0.30, Medium <0.60, High >=0.60\n")
    print(f"\nSaved: {results_txt}")
    print("\nDAY 3 RUNNER COMPLETE")


if __name__ == "__main__":
    main()
