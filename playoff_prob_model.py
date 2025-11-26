import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# -------------------------------
# Load and Clean Data ***put file on desktop, make sure name matches***
# -------------------------------
desktop_path = os.path.expanduser("~/Desktop/mlb_finance_data_1.csv")
df = pd.read_csv(desktop_path, engine="python")

# Rename WAR columns for consistency
df.rename(columns={
    "Team Total WAR": "WAR",
    "Position Players WAR": "P_WAR",
    "Pitching WAR": "PP_WAR"
}, inplace=True)

# Clean payroll column
df["Total_Payroll"] = df["Total_Payroll"].replace("[\$,]", "", regex=True).astype(float)

# Normalize boolean flags
for col in ["division_winner", "playoff_apperance", "world_series", "world_series_winner"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.lower().map({"true": True, "false": False})

# Normalize Win_% scale
if df["Win_%"].max() > 1.0:
    df["Win_%"] = df["Win_%"] / 100.0

# Exclude 2020 season
if "Season" in df.columns:
    df = df[df["Season"] != 2020]

# -------------------------------
# Span dataset: 2015–2025
# -------------------------------
df_span = df[(df["Season"] >= 2015) & (df["Season"] <= 2025)].copy()

# -------------------------------
# R² Analysis Block (2015–2025) PART 1
# -------------------------------

# Restrict to span years
df_span = df[(df["Season"] >= 2015) & (df["Season"] <= 2025)].copy()

# Auto-detect win% column (any column containing 'Win')
win_col_candidates = [c for c in df_span.columns if "Win" in c or "W_pct" in c or "Pct" in c]
if not win_col_candidates:
    raise KeyError("No Win% column found. Please check df_span.columns.")
win_col = win_col_candidates[0]   # pick the first match

# Targets
y_win = df_span[win_col]
y_playoff = df_span["playoff_apperance"].astype(int)

# Helper function
def fit_r2(features, target):
    model = LinearRegression().fit(features, target)
    preds = model.predict(features)
    return r2_score(target, preds)

# R² calculations
r2_war_win = fit_r2(df_span[["WAR"]], y_win)
r2_war_playoff = fit_r2(df_span[["WAR"]], y_playoff)

r2_payroll_win = fit_r2(df_span[["Total_Payroll"]], y_win)
r2_payroll_playoff = fit_r2(df_span[["Total_Payroll"]], y_playoff)

r2_pitching_win = fit_r2(df_span[["P_WAR"]], y_win)
r2_pitching_playoff = fit_r2(df_span[["P_WAR"]], y_playoff)

r2_position_win = fit_r2(df_span[["PP_WAR"]], y_win)
r2_position_playoff = fit_r2(df_span[["PP_WAR"]], y_playoff)

# NEW: Combined WAR + Payroll → Win%
r2_combo_win = fit_r2(df_span[["WAR","Total_Payroll"]], y_win)

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

# -------------------------------
# Restrict to modern playoff era: 2022+ PART 2 INPUT PROJECTIONs
# -------------------------------
if "Season" in df.columns:
    df_modern = df[df["Season"] >= 2022].copy()
else:
    df_modern = df.copy()

df_modern["playoff_flag"] = df_modern["playoff_apperance"].astype(int)

# -------------------------------
# Regression Analysis: Payroll → Win%
# -------------------------------
X = df[["Total_Payroll"]]
y = df["Win_%"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
lin_pred = lin_model.predict(X_test)
lin_r2 = r2_score(y_test, lin_pred)

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
poly_pred = poly_model.predict(X_test_poly)
poly_r2 = r2_score(y_test, poly_pred)

if poly_r2 > lin_r2:
    best_model = poly_model
    transformer = poly
    best_r2 = poly_r2
else:
    best_model = lin_model
    transformer = None
    best_r2 = lin_r2

# -------------------------------
# Empirical calibration (2022+): Kernel-smoothed playoff probability
# -------------------------------
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
# Regression Analysis: WAR → Win% (full df)
# -------------------------------
X_war = df[["WAR", "P_WAR", "PP_WAR"]]
y_win = df["Win_%"]

war_model = LinearRegression()
war_model.fit(X_war, y_win)

# -------------------------------
# Function: WAR needed for arbitrary playoff probability thresholds
# -------------------------------
def war_needed_for_threshold(threshold, bandwidth=0.02, floor=0.05):
    war_range = np.linspace(10, 70, 300)
    for war in war_range:
        war_input = pd.DataFrame([[war, war*0.6, war*0.4]],
                                 columns=["WAR", "P_WAR", "PP_WAR"])
        win_pct = war_model.predict(war_input)[0]
        prob = smoothed_playoff_prob(win_pct, bandwidth=bandwidth, floor=floor)
        if prob >= threshold:
            return war, win_pct, prob
    return None

# -------------------------------
# User Input Prediction (Payroll)
# -------------------------------
user_payroll = float(input("\nEnter a team's payroll (in dollars): "))

if transformer:
    user_features = transformer.transform(pd.DataFrame([[user_payroll]], columns=["Total_Payroll"]))
else:
    user_features = pd.DataFrame([[user_payroll]], columns=["Total_Payroll"])

predicted_win_pct = best_model.predict(user_features)[0]
playoff_prob = smoothed_playoff_prob(predicted_win_pct, bandwidth=0.02, floor=0.05)

print("Predicted Win_% for Payroll $" + format(user_payroll, ",.2f") + ": " + format(predicted_win_pct, ".3f"))
print("Playoff Probability (empirical, 2022+ with floor): " + format(playoff_prob * 100, ".1f") + "%")

# -------------------------------
# User Input Prediction (Projected WAR) 
# -------------------------------
user_war = float(input("\nEnter a team's projected WAR: "))
war_input = pd.DataFrame([[user_war, user_war*0.6, user_war*0.4]],
                         columns=["WAR", "P_WAR", "PP_WAR"])
predicted_win_pct_war = war_model.predict(war_input)[0]
playoff_prob_war = smoothed_playoff_prob(predicted_win_pct_war, bandwidth=0.02, floor=0.05)

print("Predicted Win_% for Projected WAR " + format(user_war, ".1f") + ": " + format(predicted_win_pct_war, ".3f"))
print("Playoff Probability (empirical, 2022+ with floor): " + format(playoff_prob_war * 100, ".1f") + "%")

# -------------------------------
# WAR Additions Needed for Multiple Thresholds (skip exceeded) #part 3 COST OF ADDING WAR TO IMPROVE win% and playoffs odds
# -------------------------------
thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
print("\nWAR Additions Needed for Playoff Odds Targets:")

current_prob = playoff_prob_war  # current playoff probability from projected WAR

for t in thresholds:
    if t <= current_prob:  # skip thresholds already exceeded
        continue
    result = war_needed_for_threshold(t)
    if result:
        war_target, win_target, prob_target = result
        war_add = war_target - user_war
        print(f"- {int(t*100)}% odds: need ~{war_target:.1f} WAR "
              f"(add {war_add:.1f} WAR from current {user_war:.1f}) → "
              f"Win% ≈ {win_target:.3f}, Playoff ≈ {prob_target*100:.1f}%")
    else:
        print(f"- {int(t*100)}% odds: not achievable in search range")
# -------------------------------
# Hybrid Dollar Cost of Adding WAR
# -------------------------------

increments = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
print("\nEstimated Dollar Cost of Adding WAR (Hybrid Model):")

# Current WAR baseline
base_input = pd.DataFrame([[user_war, user_war*0.6, user_war*0.4]],
                          columns=["WAR", "P_WAR", "PP_WAR"])
base_win_pct = war_model.predict(base_input)[0]

# Payroll range for inversion
payroll_range = np.linspace(df["Total_Payroll"].min(), df["Total_Payroll"].max(), 1000)
payroll_df = pd.DataFrame(payroll_range, columns=["Total_Payroll"])
if transformer:
    win_preds = best_model.predict(transformer.transform(payroll_df))
else:
    win_preds = best_model.predict(payroll_df)

# Find payroll for baseline Win%
base_payroll = payroll_range[np.argmin(np.abs(win_preds - base_win_pct))]

# Market baseline $/WAR (empirical free agent cost)
market_cost_per_war = 8_500_000  # ~8.5M per WAR

for inc in increments:
    new_war = user_war + inc
    new_input = pd.DataFrame([[new_war, new_war*0.6, new_war*0.4]],
                             columns=["WAR", "P_WAR", "PP_WAR"])
    new_win_pct = war_model.predict(new_input)[0]

    # Regression inversion
    new_payroll = payroll_range[np.argmin(np.abs(win_preds - new_win_pct))]
    reg_cost = new_payroll - base_payroll

    # Hybrid: take max of regression cost and market baseline
    hybrid_cost = max(reg_cost, inc * market_cost_per_war)

    print(f"- Adding {inc:.1f} WAR costs ≈ ${hybrid_cost:,.0f}")

    
# -------------------------------
# Estimate Salary Floor for ≥20% Playoff Probability PART 4 EXPIREMENTAL FLOOR
# -------------------------------

target_prob = 0.20  # minimum playoff probability
payroll_range = np.linspace(df["Total_Payroll"].min(), df["Total_Payroll"].max(), 1000)
payroll_df = pd.DataFrame(payroll_range, columns=["Total_Payroll"])

# Predict Win% for each payroll in the range
if transformer:
    win_preds = best_model.predict(transformer.transform(payroll_df))
else:
    win_preds = best_model.predict(payroll_df)

# Convert Win% predictions into playoff probabilities (empirical kernel smoother)
prob_preds = [smoothed_playoff_prob(win, bandwidth=0.02, floor=0.05) for win in win_preds]

# Find the lowest payroll that achieves ≥20% playoff probability
valid_indices = [i for i, prob in enumerate(prob_preds) if prob >= target_prob]
if valid_indices:
    salary_floor = payroll_range[min(valid_indices)]
    print("\nEstimated MLB Salary Floor for Competitiveness:")
    print(f"- To reach ≥{int(target_prob*100)}% playoff probability, "
          f"teams need payroll ≈ ${salary_floor:,.0f}")
else:
    print("\nNo payroll level in the dataset achieves ≥20% playoff probability.")
   # -------------------------------
# Robust: Direct Payroll → WAR (2015–2025, log payroll)
# -------------------------------
df_span = df[(df["Season"] >= 2015) & (df["Season"] <= 2025)].copy()
df_span["log_payroll"] = np.log(df_span["Total_Payroll"])

# Fit WAR ~ log(payroll)
X_pw = df_span[["log_payroll"]]
y_war = df_span["WAR"]
war_from_payroll_model = LinearRegression().fit(X_pw, y_war)

def estimate_metrics_from_payroll(payroll_value):
    # Predict WAR directly from payroll
    log_p = np.log(payroll_value)
    est_war = war_from_payroll_model.predict(pd.DataFrame([[log_p]], columns=["log_payroll"]))[0]
    
    # Map WAR → Win%
    war_input = pd.DataFrame([[est_war, est_war*0.6, est_war*0.4]], columns=["WAR","P_WAR","PP_WAR"])
    est_win = war_model.predict(war_input)[0]
    
    # Empirical playoff probability from Win%
    est_prob = smoothed_playoff_prob(est_win, bandwidth=0.02, floor=0.15)
    return est_win, est_war, est_prob

# Evaluate $120M (can do any number)
payroll_value = 120_000_000
pred_win, est_war, prob = estimate_metrics_from_payroll(payroll_value)

print("\nEstimated Team Metrics (Direct Payroll → WAR) for $" + format(payroll_value, ",.0f"))
print(f"- Predicted Win% ≈ {pred_win:.3f}")
print(f"- Estimated WAR ≈ {est_war:.1f}")
print(f"- Playoff Probability ≈ {prob*100:.1f}%")

# Diagnostics: local payroll → Win% → Playoff% curve around $120M 


test_payrolls = [100_000_000, 110_000_000, 120_000_000, 128_819_706, 130_000_000, 135_000_000, 140_000_000]
for p in test_payrolls:
    if transformer:
        features = transformer.transform(pd.DataFrame([[p]], columns=["Total_Payroll"]))
    else:
        features = pd.DataFrame([[p]], columns=["Total_Payroll"])
    win = best_model.predict(features)[0]
    prob = smoothed_playoff_prob(win, bandwidth=0.02, floor=0.05)
    print(f"Payroll ${p:,.0f} → Win% {win:.3f} → Playoff {prob*100:.1f}%")
# -------------------------------
# Visualization: Payroll vs Win% and Playoff Probability graph related to part 6
# -------------------------------

# Generate payroll range
payroll_range = np.linspace(65_000_000, 180_000_000, 500)
payroll_df = pd.DataFrame(payroll_range, columns=["Total_Payroll"])

# Predict Win% for each payroll
if transformer:
    win_preds = best_model.predict(transformer.transform(payroll_df))
else:
    win_preds = best_model.predict(payroll_df)

# Convert Win% predictions into playoff probabilities
prob_preds = [smoothed_playoff_prob(win, bandwidth=0.02, floor=0.05) for win in win_preds]

# Plot
plt.figure(figsize=(10,6))
plt.plot(payroll_range/1e6, win_preds, label="Predicted Win%", color="blue", linewidth=2)
plt.plot(payroll_range/1e6, prob_preds, label="Playoff Probability", color="green", linewidth=2)

plt.title("Payroll vs Win% and Playoff Probability (2015–2025)")
plt.xlabel("Payroll ($ Millions)")
plt.ylabel("Win% / Playoff Probability")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

import os
import pandas as pd
import numpy as np

# -------------------------------
# Load projected team and free agent CSVs directly
# -------------------------------
proj_df = pd.read_csv(os.path.expanduser("~/Desktop/projected_mlb_team_data.csv"))
fa_df   = pd.read_csv(os.path.expanduser("~/Desktop/2026_freeagents.csv"))

# Normalize headers (lowercase, underscores)
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
# Explicit maps — match your actual CSV headers
# -------------------------------
team_map = {
    "team": "team",
    "projected_war": "projected_war",
    "projected_salary": "projected_salary"
}
fa_map = {
    "name": "name",              # free agent identifier
    "projected_war": "proj_war", # projected WAR
    "contract_total_value": "med_total",
    "contract_years": "med_years",
    "contract_aav": "med_aav"
}


# -------------------------------
# Visualization Example
# -------------------------------
def plot_payroll_vs_win(df, war_model):
    payroll_range = np.linspace(df["Total_Payroll"].min(), df["Total_Payroll"].max(), 200)
    payroll_df = pd.DataFrame(payroll_range, columns=["Total_Payroll"])
    if transformer:
        win_preds = best_model.predict(transformer.transform(payroll_df))
    else:
        win_preds = best_model.predict(payroll_df)
    prob_preds = [smoothed_playoff_prob(win) for win in win_preds]

    plt.figure(figsize=(10,6))
    plt.plot(payroll_range/1e6, win_preds, label="Predicted Win%", color="blue")
    plt.plot(payroll_range/1e6, prob_preds, label="Playoff Probability", color="green")
    plt.title("Payroll vs Win% and Playoff Probability")
    plt.xlabel("Payroll ($ Millions)")
    plt.ylabel("Win% / Playoff Probability")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()


