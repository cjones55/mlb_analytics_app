#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Free Agent Addition + Contract Evaluation
Self-contained script: loads historical data, builds WAR→Win% model,
loads projected team + free agent CSVs, cleans salary fields,
and interactively evaluates adding a free agent.
"""

import argparse
import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LinearRegression

# -------------------------------
# Argument parser
# -------------------------------
parser = argparse.ArgumentParser(description="Free Agent Evaluator")
parser.add_argument("--team", type=str, required=True, help="Team name (e.g. Mets)")
parser.add_argument("--free_agent", type=str, required=True, help="Free agent name (e.g. Alex Bregman)")
parser.add_argument("--proj_csv", type=str, default="~/Desktop/mlbapp/projected_mlb_team_data.csv",
                    help="Path to projected team CSV")
parser.add_argument("--fa_csv", type=str, default="~/Desktop/mlbapp/2026_freeagents.csv",
                    help="Path to free agent CSV")
args = parser.parse_args()

team_name = args.team.strip()
player_name = args.free_agent.strip()
proj_csv = os.path.expanduser(args.proj_csv)
fa_csv = os.path.expanduser(args.fa_csv)

# -------------------------------
# Load historical dataset for WAR → Win% model
# -------------------------------
hist_path = os.path.expanduser("~/Desktop/mlbapp/mlb_finance_data_1.csv")
df = pd.read_csv(hist_path)

# Normalize columns
df.rename(columns={
    "Team Total WAR": "WAR",
    "Position Players WAR": "P_WAR",
    "Pitching WAR": "PP_WAR"
}, inplace=True)

# Normalize Win_% scale
if df["Win_%"].max() > 1.0:
    df["Win_%"] = df["Win_%"] / 100.0

# Exclude 2020 season
if "Season" in df.columns:
    df = df[df["Season"] != 2020]

# Train WAR → Win% regression
X_war = df[["WAR", "P_WAR", "PP_WAR"]]
y_win = df["Win_%"]
war_model = LinearRegression().fit(X_war, y_win)

# Kernel-smoothed playoff probability calibration (2022+)
df_modern = df[df["Season"] >= 2022].copy()
df_modern["playoff_flag"] = df_modern["playoff_apperance"].astype(int)
win_hist = df_modern["Win_%"].values
playoff_hist = df_modern["playoff_flag"].values

def kernel_gaussian(u):
    return np.exp(-0.5 * u**2)

def smoothed_playoff_prob(win_pct_query, bandwidth=0.02, floor=0.05):
    weights = kernel_gaussian((win_pct_query - win_hist) / bandwidth)
    num = np.sum(weights * playoff_hist)
    den = np.sum(weights) + 1e-12
    prob = num / den
    return max(prob, floor)

# -------------------------------
# Load projected team and free agent CSVs
# -------------------------------
proj_df = pd.read_csv(os.path.expanduser("~/Desktop/mlbapp/projected_mlb_team_data.csv"))
fa_df   = pd.read_csv(os.path.expanduser("~/Desktop/mlbapp/2026_freeagents.csv"))

# Normalize headers
proj_df.columns = proj_df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
fa_df.columns   = fa_df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)

# Clean numeric fields
if "projected_salary" in proj_df.columns:
    proj_df["projected_salary"] = (
        proj_df["projected_salary"].astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .astype(float)
    )
for col in ["med_total", "med_aav"]:
    if col in fa_df.columns:
        fa_df[col] = (
            fa_df[col].astype(str)
            .str.replace(r"[\$,]", "", regex=True)
            .astype(float)
        )

# -------------------------------
# Explicit maps
# -------------------------------
team_map = {
    "team": "team",
    "projected_war": "projected_war",
    "projected_salary": "projected_salary"
}
fa_map = {
    "name": "name",
    "projected_war": "proj_war",
    "contract_total_value": "med_total",
    "contract_years": "med_years",
    "contract_aav": "med_aav"
}

# -------------------------------
# Free Agent Addition + Contract Evaluation
# -------------------------------
def add_free_agent(team_name, player_name, proj_df, fa_df, team_map, fa_map,
                   war_model, smoothed_playoff_prob, CBA_threshold=75_000_000,
                   cost_per_war=8_500_000):
    team_row = proj_df[proj_df[team_map["team"]].str.lower() == team_name.lower()]
    player_row = fa_df[fa_df[fa_map["name"]].str.lower() == player_name.lower()]

    if team_row.empty:
        print(f"Team '{team_name}' not found.")
        return
    if player_row.empty:
        print(f"Free agent '{player_name}' not found.")
        return

    # Current team values
    team_war = float(team_row[team_map["projected_war"]].values[0])
    team_salary = float(team_row[team_map["projected_salary"]].values[0])

    # Current odds
    war_input = pd.DataFrame([[team_war, team_war*0.6, team_war*0.4]],
                             columns=["WAR","P_WAR","PP_WAR"])
    old_win_pct = float(war_model.predict(war_input)[0])
    old_playoff_prob = float(smoothed_playoff_prob(old_win_pct))

    # Free agent values
    player_war = float(player_row[fa_map["projected_war"]].values[0])
    contract_value = float(player_row[fa_map["contract_total_value"]].values[0])
    contract_years = int(player_row[fa_map["contract_years"]].values[0])
    aav = float(player_row[fa_map["contract_aav"]].values[0])

    # Net WAR gain
    new_war = team_war + player_war
    new_salary = team_salary + aav

    # Updated odds
    war_input_new = pd.DataFrame([[new_war, new_war*0.6, new_war*0.4]],
                                 columns=["WAR","P_WAR","PP_WAR"])
    new_win_pct = float(war_model.predict(war_input_new)[0])
    new_playoff_prob = float(smoothed_playoff_prob(new_win_pct))

    # Print results
    print(f"\nRoster-aware Results for {team_name} after signing {player_name}:")
    print(f"- Net WAR Gain = {player_war:.1f}")
    print("\n--- Team BEFORE signing ---")
    print(f"  WAR = {team_war:.1f}")
    print(f"  Salary = ${team_salary:,.0f}")
    print(f"  Win% ≈ {old_win_pct*100:.1f}%")
    print(f"  Playoff ≈ {old_playoff_prob*100:.1f}%")
    print("\n--- Team AFTER signing ---")
    print(f"  WAR = {new_war:.1f}")
    print(f"  Salary = ${new_salary:,.0f} (+AAV ${aav:,.0f})")
    print(f"  Win% ≈ {new_win_pct*100:.1f}% "
          f"(Change: {(new_win_pct-old_win_pct)*100:+.2f} pts)")
    print(f"  Playoff ≈ {new_playoff_prob*100:.1f}% "
          f"(Change: {(new_playoff_prob-old_playoff_prob)*100:+.2f} pts)")
    print(f"- Contract Terms: {contract_years} years, ${contract_value:,.0f} total, AAV = ${aav:,.0f}")
    print(f"- Below CBA Threshold? {new_salary < CBA_threshold}")

    # Contract evaluation
    market_value = player_war * cost_per_war
    diff = aav - market_value
    print(f"\nContract Evaluation for {player_name}:")
    print(f"- Projected WAR: {player_war:.1f}")
    print(f"- Market Value (WAR × ${cost_per_war:,.0f}/WAR): ${market_value:,.0f}")
    if diff > 0:
        print(f"→ Verdict: OVERPAY by ${diff:,.0f} per year")
    elif diff < 0:
        print(f"→ Verdict: UNDERPAY by ${-diff:,.0f} per year")
    else:
        print("→ Verdict: FAIR VALUE deal")


# -------------------------------
# Run evaluation using argparse values
# -------------------------------
add_free_agent(team_name, player_name,
               proj_df, fa_df, team_map, fa_map,
               war_model, smoothed_playoff_prob)