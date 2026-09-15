"""
One-shot runner that executes the same cleaning + EDA logic as notebooks/01_eda.ipynb.
Useful to verify Day 1 outputs without opening Jupyter.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "telco_churn.csv"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_telco.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (8, 5)


def churn_rate_table(data: pd.DataFrame, column: str) -> pd.DataFrame:
    table = (
        data.groupby(column)["Churn"]
        .agg(
            total_customers="count",
            churned=lambda s: (s == "Yes").sum(),
        )
        .reset_index()
    )
    table["churn_rate_pct"] = (table["churned"] / table["total_customers"] * 100).round(2)
    return table.sort_values("churn_rate_pct", ascending=False)


def save_fig(name: str) -> None:
    path = FIGURES_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def main() -> None:
    df_raw = pd.read_csv(RAW_PATH)
    df = df_raw.copy()

    for col in df.select_dtypes(include=["object", "string", "str"]).columns:
        df[col] = df[col].astype(str).str.strip()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    invalid_total = df["TotalCharges"].isna().sum()
    print(f"Invalid/blank TotalCharges: {invalid_total}")
    print(df.loc[df["TotalCharges"].isna(), ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]])

    df.loc[df["TotalCharges"].isna() & (df["tenure"] == 0), "TotalCharges"] = 0.0
    remaining_invalid = df["TotalCharges"].isna().sum()
    if remaining_invalid > 0:
        df = df.dropna(subset=["TotalCharges"]).copy()

    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    print(f"Cleaned shape: {df.shape}")

    overall_churn = (df["Churn"] == "Yes").mean() * 100
    print(f"\nOverall churn rate: {overall_churn:.2f}%")
    print(f"Churned: {(df['Churn'] == 'Yes').sum()} | Non-churned: {(df['Churn'] == 'No').sum()}")
    print("\nChurn by Contract:\n", churn_rate_table(df, "Contract"))
    print("\nChurn by InternetService:\n", churn_rate_table(df, "InternetService"))
    print("\nChurn by PaymentMethod:\n", churn_rate_table(df, "PaymentMethod"))
    print("\nAvg tenure by Churn:\n", df.groupby("Churn")["tenure"].mean().round(2))
    print("\nAvg MonthlyCharges by Churn:\n", df.groupby("Churn")["MonthlyCharges"].mean().round(2))

    # Charts
    ax = sns.countplot(data=df, x="Churn", hue="Churn", palette="Set2", legend=False)
    ax.set_title("1. Churn Distribution")
    save_fig("01_churn_distribution.png")

    plt.figure(figsize=(9, 5))
    ax = sns.countplot(data=df, x="Contract", hue="Churn", palette="Set2")
    ax.set_title("2. Contract vs Churn")
    save_fig("02_contract_vs_churn.png")

    plt.figure()
    ax = sns.histplot(data=df, x="tenure", hue="Churn", bins=30, multiple="stack", palette="Set2")
    ax.set_title("3. Tenure vs Churn")
    save_fig("03_tenure_vs_churn.png")

    plt.figure()
    ax = sns.boxplot(data=df, x="Churn", y="MonthlyCharges", hue="Churn", palette="Set2", legend=False)
    ax.set_title("4. MonthlyCharges vs Churn")
    save_fig("04_monthlycharges_vs_churn.png")

    plt.figure(figsize=(9, 5))
    ax = sns.countplot(data=df, x="InternetService", hue="Churn", palette="Set2")
    ax.set_title("5. InternetService vs Churn")
    save_fig("05_internetservice_vs_churn.png")

    plt.figure(figsize=(11, 5))
    ax = sns.countplot(data=df, x="PaymentMethod", hue="Churn", palette="Set2")
    ax.set_title("6. PaymentMethod vs Churn")
    plt.xticks(rotation=20, ha="right")
    save_fig("06_paymentmethod_vs_churn.png")

    plt.figure()
    ax = sns.countplot(data=df, x="PaperlessBilling", hue="Churn", palette="Set2")
    ax.set_title("7. PaperlessBilling vs Churn")
    save_fig("07_paperlessbilling_vs_churn.png")

    plot_df = df.copy()
    plot_df["SeniorCitizen_label"] = plot_df["SeniorCitizen"].map({0: "Not Senior", 1: "Senior"})
    plt.figure()
    ax = sns.countplot(data=plot_df, x="SeniorCitizen_label", hue="Churn", palette="Set2")
    ax.set_title("8. SeniorCitizen vs Churn")
    save_fig("08_seniorcitizen_vs_churn.png")

    plt.figure()
    ax = sns.countplot(data=df, x="Partner", hue="Churn", palette="Set2")
    ax.set_title("9. Partner vs Churn")
    save_fig("09_partner_vs_churn.png")

    plt.figure()
    ax = sns.countplot(data=df, x="Dependents", hue="Churn", palette="Set2")
    ax.set_title("10. Dependents vs Churn")
    save_fig("10_dependents_vs_churn.png")

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    df_check = pd.read_csv(PROCESSED_PATH)
    assert df_check.shape == df.shape
    print(f"\nSaved and verified: {PROCESSED_PATH}")
    print("EDA runner complete.")


if __name__ == "__main__":
    main()
