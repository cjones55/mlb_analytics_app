#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Projected Standings + Playoff Odds (WAR-only, CSV-driven → CSV output)
@author: chrisjones
"""
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Always point to mlbapp folder on Desktop
BASE_DIR = os.path.expanduser("~/Desktop/mlbapp")

# --- Load historical finance/performance data for training ---
finance_csv = os.path.join(BASE_DIR, "mlb_finance_data_1.csv")
df = pd.read_csv(finance_csv)

# Normalize WAR column names
df.rename(columns={
    "Team Total WAR": "WAR",
    "Position Players WAR": "P_WAR",
    "Pitching WAR": "PP_WAR"
}, inplace=True)

# Detect win% column
win_col_candidates = [c for c in df.columns if "Win" in c or "Pct" in c]
if not win_col_candidates:
    raise KeyError("No Win% column found. Please check df.columns.")
win_col = win_col_candidates[0]

# Normalize scale if needed
if df[win_col].max() > 1.0:
    df[win_col] = df[win_col] / 100.0

# Exclude 2020 season if present
if "Season" in df.columns:
    df = df[df["Season"] != 2020]

# -------------------------------
# Retrain WAR → Win% model (single feature only)
# -------------------------------
X_war = df[["WAR"]]
y_win = df[win_col]

war_model = LinearRegression()
war_model.fit(X_war, y_win)

# -------------------------------
# Kernel-smoothed playoff probability (2022+ calibration)
# -------------------------------
df_modern = df[df["Season"] >= 2022].copy()
df_modern["playoff_flag"] = df_modern["playoff_apperance"].astype(int)

win_hist = df_modern[win_col].values
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
# Load projected WAR data for standings
# -------------------------------
proj_csv = os.path.join(BASE_DIR, "projected_mlb_team_data.csv")
proj_df = pd.read_csv(proj_csv)

# Normalize WAR column name if needed
if "WAR" not in proj_df.columns:
    if "projected_war" in proj_df.columns:
        proj_df.rename(columns={"projected_war": "WAR"}, inplace=True)
    elif "Projected WAR" in proj_df.columns:
        proj_df.rename(columns={"Projected WAR": "WAR"}, inplace=True)
    else:
        raise KeyError(f"No WAR column found in {proj_csv}. Columns are: {proj_df.columns.tolist()}")

# Ensure WAR is numeric and handle NaNs
proj_df["WAR"] = pd.to_numeric(proj_df["WAR"], errors="coerce").fillna(0)

# Ensure White Sox are present
if "White Sox" not in proj_df["Team"].values:
    proj_df = pd.concat([proj_df, pd.DataFrame([{"Team":"White Sox","WAR":0}])], ignore_index=True)

# Division mapping
team_divisions = {
    "Yankees":("AL","East"),"Red Sox":("AL","East"),"Blue Jays":("AL","East"),
    "Rays":("AL","East"),"Orioles":("AL","East"),
    "Guardians":("AL","Central"),"White Sox":("AL","Central"),"Tigers":("AL","Central"),
    "Royals":("AL","Central"),"Twins":("AL","Central"),
    "Astros":("AL","West"),"Mariners":("AL","West"),"Rangers":("AL","West"),
    "Athletics":("AL","West"),"Angels":("AL","West"),
    "Braves":("NL","East"),"Mets":("NL","East"),"Phillies":("NL","East"),
    "Nationals":("NL","East"),"Marlins":("NL","East"),
    "Cubs":("NL","Central"),"Cardinals":("NL","Central"),"Brewers":("NL","Central"),
    "Pirates":("NL","Central"),"Reds":("NL","Central"),
    "Dodgers":("NL","West"),"Giants":("NL","West"),"Padres":("NL","West"),
    "Rockies":("NL","West"),"Diamondbacks":("NL","West"),
}

proj_df["league"] = proj_df["Team"].map(lambda t: team_divisions.get(t,("Unknown","Unknown"))[0])
proj_df["division"] = proj_df["Team"].map(lambda t: team_divisions.get(t,("Unknown","Unknown"))[1])

# -------------------------------
# Predict standings
# -------------------------------
proj_df["win_pct"] = war_model.predict(proj_df[["WAR"]])
proj_df["wins"] = (proj_df["win_pct"]*162).round().astype(int)
proj_df["losses"] = 162 - proj_df["wins"]
proj_df["playoff_odds"] = proj_df["win_pct"].apply(lambda p: smoothed_playoff_prob(p, bandwidth=0.02, floor=0.05))

# -------------------------------
# Save to CSV in mlbapp folder
# -------------------------------
output_path = os.path.join(BASE_DIR, "jonesy_mlb_projected_standings.csv")
proj_df.to_csv(output_path, index=False)

# -------------------------------
# Print grouped standings
# -------------------------------
for lg in ["AL","NL"]:
    print(f"\n{lg} Standings:")
    for div in ["East","Central","West"]:
        block = proj_df[(proj_df["league"]==lg)&(proj_df["division"]==div)].sort_values("wins",ascending=False)
        print(f" {lg} {div}")
        for _, r in block.iterrows():
            print(f" - {r['Team']}: {r['wins']}-{r['losses']} ({r['win_pct']:.3f}), "
                  f"Playoff Odds {r['playoff_odds']*100:.1f}%")

# -------------------------------
# Save grouped standings to Desktop (optional)
# -------------------------------
sorted_df = proj_df.sort_values(["league", "division", "wins"], ascending=[True, True, False])
cols_to_keep = ["Team", "league", "division", "win_pct", "wins", "losses", "playoff_odds"]
clean_df = sorted_df[cols_to_keep].copy()

save_to_desktop = False  # set True if you want a copy on Desktop

if save_to_desktop:
    desktop_path = os.path.expanduser("~/Desktop/jonesy_mlb_projected_standings.csv")
    clean_df.to_csv(desktop_path, index=False)
    print(f"Grouped standings saved to: {desktop_path}")
else:
    print("Grouped standings not saved (set save_to_desktop=True to enable).")

