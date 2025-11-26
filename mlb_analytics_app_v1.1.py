#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLB Analytics App Launcher
"""
import subprocess
import os

# Always point to mlbapp folder on Desktop
DESKTOP_DIR = os.path.expanduser("~/Desktop")
BASE_DIR = os.path.join(DESKTOP_DIR, "mlbapp")

def main():
    print("MLB Analytics App")
    print("1. Team projected evaluation via WAR")
    print("2. Run full standings projections")
    print("3. Add free agent to team projections")
    print("4. Run R² analysis with visualizations")
    choice = input("Select option (1, 2, 3, or 4): ").strip()

    if choice == "1":
        try:
            payroll = float(input("Enter team payroll in dollars: "))
            war = float(input("Enter projected team WAR: "))
        except Exception:
            print("Invalid input. Using defaults.")
            payroll, war = 120_000_000, 35.0

        subprocess.run([
            "python", os.path.join(BASE_DIR, "single_team_simulator.py"),
            "--payroll", str(payroll),
            "--war", str(war)
        ])

    elif choice == "2":
        proj_csv = os.path.join(BASE_DIR, "projected_mlb_team_data.csv")
        finance_csv = os.path.join(BASE_DIR, "mlb_finance_data_1.csv")
        save_choice = input("Save grouped standings CSV to mlbapp folder? (y/n): ").strip().lower()
        cmd = [
            "python", os.path.join(BASE_DIR, "standings_projections.py"),
            "--csv", proj_csv,
            "--finance_csv", finance_csv
        ]
        if save_choice == "y":
            cmd.append("--save")
        subprocess.run(cmd)

    elif choice == "3":
        team_name = input("Which team is adding the free agent? ").strip()
        player_name = input("Which free agent is being added? ").strip()

        proj_csv = os.path.join(BASE_DIR, "projected_mlb_team_data.csv")
        fa_csv   = os.path.join(BASE_DIR, "2026_freeagents.csv")

        subprocess.run([
            "python", os.path.join(BASE_DIR, "free_agent_evaluator.py"),
            "--team", team_name,
            "--free_agent", player_name,
            "--proj_csv", proj_csv,
            "--fa_csv", fa_csv
        ])

    elif choice == "4":
        subprocess.run([
            "python", os.path.join(BASE_DIR, "r2_analysis.py"),
            "--finance_csv", os.path.join(BASE_DIR, "mlb_finance_data_1.csv"),
            "--proj_csv", os.path.join(BASE_DIR, "projected_mlb_team_data.csv")
        ])

    else:
        print("Invalid choice. Please run again.")

if __name__ == "__main__":
    main()
