# mlb_analytics_app
download zip file!!!!!! ALL CONTENTS INSIDE!!!!!

_____________________________________________________________________________________

A command‑line application for running MLB team analytics, standings projections, free agent evaluations, and R² analysis with visualizations based on data from 2015-present day.

_____________________________________________________________________________________

📂 Folder Contents
- `mlb_analytics_app.py` → The launcher script (menu interface).
- `standings_projections.py` → Runs full standings projections based on WAR and finance data.
- `single_team_simulator.py` → Evaluates a single team’s projected performance via WAR.
- `free_agent_evaluator.py` → Adds a free agent to team projections.
- `r2_analysis.py` → Runs R² analysis with visualizations.
- `projected_mlb_team_data.csv` → Projected WAR data for all teams.
- `mlb_finance_data_1.csv` → Historical finance/performance dataset.
- `2026_freeagents.csv` → Free agent list for 2026.
- Output files (e.g. `jonesy_mlb_projected_standings.csv`) will be generated here.

_____________________________________________________________________________________

⚙️ Requirements
- Python 3.9+ (tested with Python 3.11).
- Required packages:
  ```bash
  pip install pandas numpy scikit-learn

— IMPORTANT
Place the mlbapp folder on your Desktop.
	•	Path should be: ~/Desktop/mlbapp
Open a terminal and navigate to the folder:

cd ~/Desktop/mlbapp

Now we run the launcher...

type in: 

python mlb_analytics_app_v1.1.py

MLB Analytics App
1. Team projected evaluation via WAR
2. Run full standings projections
3. Add free agent to team projections
4. Run R² analysis with visualizations
Select option (1, 2, 3, or 4):

