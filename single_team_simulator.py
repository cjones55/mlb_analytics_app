#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R² Analysis and Projections Module
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser(description="Single Team Simulator")
parser.add_argument("--payroll", type=float, required=True, help="Team payroll in dollars")
parser.add_argument("--war", type=float, required=True, help="Projected team WAR")
args = parser.parse_args()

user_payroll = args.payroll
user_war = args.war

# -------------------------------
# Load and clean dataset
# -------------------------------
desktop_path = os.path.expanduser("~/Desktop/mlbapp/mlb_finance_data_1.csv")
df = pd.read_csv(desktop_path, engine="python")

# Normalize columns
df.rename(columns={
    "Team Total WAR": "WAR",
    "Position Players WAR": "P_WAR",
    "Pitching WAR": "PP_WAR"
}, inplace=True)
df["Total_Payroll"] = df["Total_Payroll"].replace("[\$,]", "", regex=True).astype(float)
df["playoff_apperance"] = df["playoff_apperance"].astype(str).str.strip().str.lower().map({"true": True, "false": False})
df["Win_%"] = df["Win_%"] / 100.0 if df["Win_%"].max() > 1.0 else df["Win_%"]
df = df[df["Season"] != 2020]

# -------------------------------
# Restrict to modern playoff era: 2022+
# -------------------------------
df_modern = df[df["Season"] >= 2022].copy()
df_modern["playoff_flag"] = df_modern["playoff_apperance"].astype(int)

# -------------------------------
# Regression Analysis: Payroll → Win%
# -------------------------------
X = df[["Total_Payroll"]]
y = df["Win_%"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lin_model = LinearRegression().fit(X_train, y_train)
lin_r2 = r2_score(y_test, lin_model.predict(X_test))

poly = PolynomialFeatures(degree=2)
poly_model = LinearRegression().fit(poly.fit_transform(X_train), y_train)
poly_r2 = r2_score(y_test, poly_model.predict(poly.transform(X_test)))

best_model = poly_model if poly_r2 > lin_r2 else lin_model
transformer = poly if poly_r2 > lin_r2 else None

# -------------------------------
# Kernel-smoothed playoff probability
# -------------------------------
win_hist = df_modern["Win_%"].values
playoff_hist = df_modern["playoff_flag"].values

def kernel_gaussian(u): return np.exp(-0.5 * u**2)

def smoothed_playoff_prob(win_pct_query, bandwidth=0.02, floor=0.05):
    weights = kernel_gaussian((win_pct_query - win_hist) / bandwidth)
    prob = np.sum(weights * playoff_hist) / (np.sum(weights) + 1e-12)
    return max(prob, floor)

# -------------------------------
# Regression Analysis: WAR → Win%
# -------------------------------
X_war = df[["WAR", "P_WAR", "PP_WAR"]]
y_win = df["Win_%"]
war_model = LinearRegression().fit(X_war, y_win)

def war_needed_for_threshold(threshold, bandwidth=0.02, floor=0.05):
    for war in np.linspace(10, 70, 300):
        war_input = pd.DataFrame([[war, war*0.6, war*0.4]], columns=["WAR","P_WAR","PP_WAR"])
        win_pct = war_model.predict(war_input)[0]
        prob = smoothed_playoff_prob(win_pct, bandwidth, floor)
        if prob >= threshold:
            return war, win_pct, prob
    return None

# -------------------------------
# User Input Predictions
# -------------------------------

features = transformer.transform(pd.DataFrame([[user_payroll]], columns=["Total_Payroll"])) if transformer else pd.DataFrame([[user_payroll]], columns=["Total_Payroll"])
predicted_win_pct = best_model.predict(features)[0]
playoff_prob = smoothed_playoff_prob(predicted_win_pct)

print(f"Predicted Win_% for Payroll ${user_payroll:,.2f}: {predicted_win_pct:.3f}")
print(f"Playoff Probability (empirical, 2022+ with floor): {playoff_prob*100:.1f}%")


war_input = pd.DataFrame([[user_war, user_war*0.6, user_war*0.4]], columns=["WAR","P_WAR","PP_WAR"])
predicted_win_pct_war = war_model.predict(war_input)[0]
playoff_prob_war = smoothed_playoff_prob(predicted_win_pct_war)

print(f"Predicted Win_% for Projected WAR {user_war:.1f}: {predicted_win_pct_war:.3f}")
print(f"Playoff Probability (empirical, 2022+ with floor): {playoff_prob_war*100:.1f}%")

# -------------------------------
# WAR Additions Needed
# -------------------------------
print("\nWAR Additions Needed for Playoff Odds Targets:")
for t in [0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75]:
    if t <= playoff_prob_war: continue
    result = war_needed_for_threshold(t)
    if result:
        war_target, win_target, prob_target = result
        war_add = war_target - user_war
        print(f"- {int(t*100)}% odds: need ~{war_target:.1f} WAR "
              f"(add {war_add:.1f} WAR from current {user_war:.1f}) → "
              f"Win% ≈ {win_target:.3f}, Playoff ≈ {prob_target*100:.1f}%")

# -------------------------------
# Hybrid Dollar Cost of Adding WAR
# -------------------------------
print("\nEstimated Dollar Cost of Adding WAR (Hybrid Model):")
base_win_pct = war_model.predict(war_input)[0]
payroll_range = np.linspace(df["Total_Payroll"].min(), df["Total_Payroll"].max(), 1000)
payroll_df = pd.DataFrame(payroll_range, columns=["Total_Payroll"])
win_preds = best_model.predict(transformer.transform(payroll_df)) if transformer else best_model.predict(payroll_df)
base_payroll = payroll_range[np.argmin(np.abs(win_preds - base_win_pct))]
market_cost_per_war = 8_500_000

for inc in [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]:
    new_input = pd.DataFrame([[user_war+inc,(user_war+inc)*0.6,(user_war+inc)*0.4]], columns=["WAR","P_WAR","PP_WAR"])
    new_win_pct = war_model.predict(new_input)[0]
    new_payroll = payroll_range[np.argmin(np.abs(win_preds - new_win_pct))]
    reg_cost = new_payroll - base_payroll
    hybrid_cost = max(reg_cost, inc*market_cost_per_war)
    print(f"- Adding {inc:.1f} WAR costs ≈ ${hybrid_cost:,.0f}")

# -------------------------------
# Salary Floor Estimation
# -------------------------------
target_prob = 0.20
prob_preds = [smoothed_playoff_prob(win) for win in win_preds]
valid_indices = [i for i,p in enumerate(prob_preds) if p >= target_prob]
if valid_indices:
    salary_floor = payroll_range[min(valid_indices)]
    print("\nEstimated MLB Salary Floor for Competitiveness:")
    print(f"- To reach ≥{int(target_prob*100)}% playoff probability, teams need payroll ≈ ${salary_floor:,.0f}")

# -------------------------------
# Direct Payroll → WAR
# -------------------------------
df_span = df[(df["Season"] >= 2015) & (df["Season"] <= 2025)].copy()
df_span["log_payroll"] = np.log(df_span["Total_Payroll"])
war_from_payroll_model = LinearRegression().fit(df_span[["log_payroll"]], df_span["WAR"])

def estimate_metrics_from_payroll(payroll_value):
    est_war = war_from_payroll_model.predict(pd.DataFrame([[np.log(payroll_value)]], columns=["log_payroll"]))[0]
    war_input = pd.DataFrame([[est_war, est_war*0.6, est_war*0.4]], columns=["WAR","P_WAR","PP_WAR"])
    est_win = war_model.predict(war_input)[0]
    est_prob = smoothed_playoff_prob(est_win, floor=0.15)
    return est_win, est_war, est_prob

