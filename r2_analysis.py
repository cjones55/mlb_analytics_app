#!/usr/bin/env python3
import os, pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import argparse

def run_r2_analysis(csv_path, proj_csv_path):
    # Load and clean historical finance data
    df = pd.read_csv(csv_path)
    df.rename(columns={
        "Team Total WAR":"WAR",
        "Position Players WAR":"P_WAR",
        "Pitching WAR":"PP_WAR"
    }, inplace=True)
    df["Total_Payroll"] = df["Total_Payroll"].replace("[\$,]","",regex=True).astype(float)
    df = df[df["Season"]!=2020]

    # Restrict span
    df_span = df[(df["Season"]>=2015)&(df["Season"]<=2025)].copy()
    y_win = df_span["Win_%"]/100 if df_span["Win_%"].max()>1 else df_span["Win_%"]
    y_playoff = df_span["playoff_apperance"].astype(int)

    def fit_r2(features,target):
        m = LinearRegression().fit(features,target)
        return r2_score(target,m.predict(features))

    # Compute R² values
    r2_war_win        = fit_r2(df_span[["WAR"]], y_win)
    r2_war_playoff    = fit_r2(df_span[["WAR"]], y_playoff)
    r2_payroll_win    = fit_r2(df_span[["Total_Payroll"]], y_win)
    r2_payroll_playoff= fit_r2(df_span[["Total_Payroll"]], y_playoff)
    r2_pitching_win   = fit_r2(df_span[["P_WAR"]], y_win)
    r2_pitching_playoff=fit_r2(df_span[["P_WAR"]], y_playoff)
    r2_position_win   = fit_r2(df_span[["PP_WAR"]], y_win)
    r2_position_playoff=fit_r2(df_span[["PP_WAR"]], y_playoff)
    r2_combo_win      = fit_r2(df_span[["WAR","Total_Payroll"]], y_win)

    # Print results
    print("R² of WAR → Win% (2015–2025) =", round(r2_war_win, 3))
    print("R² of WAR → Playoff Flag (2015–2025) =", round(r2_war_playoff, 3))
    print("R² of Payroll → Win% (2015–2025) =", round(r2_payroll_win, 3))
    print("R² of Payroll → Playoff Flag (2015–2025) =", round(r2_payroll_playoff, 3))
    print("R² of Pitching WAR → Win% (2015–2025) =", round(r2_pitching_win, 3))
    print("R² of Pitching WAR → Playoff Flag (2015–2025) =", round(r2_pitching_playoff, 3))
    print("R² of Position Player WAR → Win% (2015–2025) =", round(r2_position_win, 3))
    print("R² of Position Player WAR → Playoff Flag (2015–2025) =", round(r2_position_playoff, 3))
    print("R² of WAR + Payroll → Win% (2015–2025) =", round(r2_combo_win, 3))

    # Graph 1: Payroll vs Win%
    payroll_range = np.linspace(65_000_000,180_000_000,500)
    payroll_df = pd.DataFrame(payroll_range, columns=["Total_Payroll"])
    payroll_model = LinearRegression().fit(df_span[["Total_Payroll"]], y_win)
    win_preds = payroll_model.predict(payroll_df)

    plt.figure(figsize=(10,6))
    plt.plot(payroll_range/1e6, win_preds, label="Predicted Win%", color="blue")
    plt.title("Payroll vs Win% (2015–2025)")
    plt.xlabel("Payroll ($M)")
    plt.ylabel("Win%")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Graph 2: Projected Salary vs Projected WAR
    proj_df = pd.read_csv(proj_csv_path)
    proj_df["projected_salary"] = proj_df["projected_salary"].astype(str)\
        .str.replace(r"[\$,]", "", regex=True).astype(float)

    X = proj_df[["projected_salary"]]
    y = proj_df["projected_war"]

    model = LinearRegression().fit(X, y)
    salary_range = np.linspace(X.min()[0], X.max()[0], 500)
    salary_df = pd.DataFrame(salary_range, columns=["projected_salary"])
    war_preds = model.predict(salary_df)

    r2_salary_war = r2_score(y, model.predict(X))

    plt.figure(figsize=(10,6))
    plt.scatter(X/1e6, y, color="gray", alpha=0.6, label="Teams")
    plt.plot(salary_range/1e6, war_preds, color="red", linewidth=2, label="Best Fit Line")
    plt.title("Projected Salary vs Projected WAR")
    plt.xlabel("Projected Salary ($M)")
    plt.ylabel("Projected WAR")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.text(
        0.05, 0.95,
        f"R² = {r2_salary_war:.3f}",
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray")
    )

    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Run R² analysis with visualizations")
    parser.add_argument("--finance_csv", required=True, help="Path to finance CSV")
    parser.add_argument("--proj_csv", required=True, help="Path to projected team CSV")
    args = parser.parse_args()

    run_r2_analysis(args.finance_csv, args.proj_csv)
