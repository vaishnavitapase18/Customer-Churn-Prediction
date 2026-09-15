"""
Day 3 — Churn classification models, evaluation, and prediction helpers.

Uses Day 2 processed train/test matrices and the saved preprocessor.
Does NOT train LTV / SHAP / API (later days).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Initial business thresholds (can be tuned later — not claimed as optimal)
LOW_RISK_MAX = 0.30
MEDIUM_RISK_MAX = 0.60


def load_processed_splits(
    processed_dir: Path = PROCESSED_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load Day 2 ML-ready train/test matrices."""
    X_train = pd.read_csv(processed_dir / "X_train.csv")
    X_test = pd.read_csv(processed_dir / "X_test.csv")
    y_train = pd.read_csv(processed_dir / "y_train.csv")["Churn"]
    y_test = pd.read_csv(processed_dir / "y_test.csv")["Churn"]
    return X_train, X_test, y_train, y_test


def load_preprocessor(path: Path = MODELS_DIR / "preprocessor.pkl"):
    """Load the Day 2 fitted preprocessor."""
    return joblib.load(path)


def get_test_customer_ids(
    cleaned_path: Path = PROCESSED_DIR / "cleaned_telco.csv",
) -> pd.Series:
    """
    Recreate the Day 2 stratified split to recover test-set customerIDs.
    Uses the same random_state=42 and stratify on Churn.
    """
    import sys

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from preprocessing import create_features, get_feature_matrix
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(cleaned_path)
    df_feat = create_features(df)
    X, y = get_feature_matrix(df_feat)
    ids = df_feat["customerID"]

    _, _, _, _, _, id_test = train_test_split(
        X,
        y,
        ids,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    return id_test.reset_index(drop=True)


def build_models() -> Dict[str, Any]:
    """Return unfitted candidate classifiers."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        ),
    }


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    models: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Fit each model on the processed training matrix."""
    if models is None:
        models = build_models()
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
    return trained


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """
    Compute classification metrics for the churn class (label=1).
    """
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # Fallback: decision_function scaled to [0,1] if needed
        scores = model.decision_function(X_test)
        y_proba = 1 / (1 + np.exp(-scores))

    metrics = {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "Recall": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "F1 Score": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_test, y_proba)),
        "Confusion Matrix": confusion_matrix(y_test, y_pred).tolist(),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }
    return metrics


def evaluate_all_models(
    trained_models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Evaluate every model and return a comparison table + full metric dicts."""
    rows = []
    details: Dict[str, Dict[str, Any]] = {}
    for name, model in trained_models.items():
        m = evaluate_model(model, X_test, y_test)
        details[name] = m
        rows.append(
            {
                "Model": name,
                "Accuracy": round(m["Accuracy"], 4),
                "Precision": round(m["Precision"], 4),
                "Recall": round(m["Recall"], 4),
                "F1 Score": round(m["F1 Score"], 4),
                "ROC-AUC": round(m["ROC-AUC"], 4),
            }
        )
    comparison = pd.DataFrame(rows)
    return comparison, details


def select_best_model(
    comparison: pd.DataFrame,
) -> Tuple[str, str]:
    """
    Select best model for churn using business-aware ranking.

    Priority for churn (finding customers likely to leave):
    1) F1 Score (balance of precision & recall)
    2) Recall (catch churners)
    3) ROC-AUC
    4) Accuracy (tie-breaker only)

    Returns (model_name, explanation).
    """
    ranked = comparison.sort_values(
        by=["F1 Score", "Recall", "ROC-AUC", "Accuracy"],
        ascending=False,
    ).reset_index(drop=True)
    best_name = ranked.loc[0, "Model"]
    best_row = ranked.loc[0]
    explanation = (
        f"Selected '{best_name}' because it has the strongest churn-class "
        f"F1 Score ({best_row['F1 Score']}), with Recall={best_row['Recall']} "
        f"and ROC-AUC={best_row['ROC-AUC']}. "
        f"Accuracy alone was not used as the primary criterion because the "
        f"dataset is imbalanced (~26.5% churn)."
    )
    return best_name, explanation


def risk_level(probability: float) -> str:
    """
    Map churn probability to an initial business risk band.

    Thresholds (documented as tunable, not scientifically optimal):
    - 0.00–0.30 → Low Risk
    - 0.30–0.60 → Medium Risk
    - 0.60–1.00 → High Risk
    """
    p = float(probability)
    if p < LOW_RISK_MAX:
        return "Low"
    if p < MEDIUM_RISK_MAX:
        return "Medium"
    return "High"


def save_model(
    model: Any,
    metadata: Dict[str, Any],
    model_path: Path = MODELS_DIR / "churn_model.pkl",
    metadata_path: Path = MODELS_DIR / "churn_model_metadata.pkl",
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(metadata, metadata_path)


def load_churn_model(
    model_path: Path = MODELS_DIR / "churn_model.pkl",
):
    return joblib.load(model_path)


def predict_churn(
    model: Any,
    features_processed: Union[pd.DataFrame, np.ndarray],
) -> Dict[str, Any]:
    """
    Predict for one or more already-preprocessed feature rows.

    Returns lists aligned to input rows:
    prediction (Yes/No), probability (0-1), risk_level.
    """
    proba = model.predict_proba(features_processed)[:, 1]
    pred = (proba >= 0.5).astype(int)
    results = {
        "prediction": ["Yes" if p == 1 else "No" for p in pred],
        "probability": [float(p) for p in proba],
        "risk_level": [risk_level(p) for p in proba],
    }
    return results


def predict_from_raw_customer(
    customer_row: pd.DataFrame,
    model: Any,
    preprocessor: Any,
) -> Dict[str, Any]:
    """
    End-to-end prediction for raw/cleaned customer row(s).
    Rows must include the original Telco columns used in Day 2.
    Churn is optional for inference (placeholder added if missing).
    """
    import sys

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from preprocessing import create_features, get_feature_matrix

    row = customer_row.copy()
    if "Churn" not in row.columns:
        row["Churn"] = "No"  # placeholder so get_feature_matrix can run; not used for X

    featured = create_features(row)
    X, _ = get_feature_matrix(featured)
    X_processed = preprocessor.transform(X)
    out = predict_churn(model, X_processed)

    if len(out["prediction"]) == 1:
        return {
            "customerID": row["customerID"].iloc[0] if "customerID" in row.columns else None,
            "churn_prediction": out["prediction"][0],
            "churn_probability": out["probability"][0],
            "risk_level": out["risk_level"][0],
        }
    return out


def build_sample_predictions_table(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    customer_ids: pd.Series,
    n: int = 10,
) -> pd.DataFrame:
    """Build a sample prediction table for the first n test customers."""
    n = min(n, len(X_test))
    subset = X_test.iloc[:n]
    preds = predict_churn(model, subset)
    actual = y_test.iloc[:n].map({0: "No", 1: "Yes"}).reset_index(drop=True)
    ids = customer_ids.iloc[:n].reset_index(drop=True)

    table = pd.DataFrame(
        {
            "customerID": ids,
            "actual_churn": actual,
            "predicted_churn": preds["prediction"],
            "churn_probability": [round(p, 4) for p in preds["probability"]],
            "risk_level": preds["risk_level"],
        }
    )
    return table
