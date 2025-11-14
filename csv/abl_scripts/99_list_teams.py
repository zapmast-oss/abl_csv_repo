import pandas as pd

from abl_config import csv_path

# 1. We'll resolve teams.csv relative to the shared OOTP export folder.
TEAMS_CSV = csv_path("teams.csv")

# 2. Read teams.csv into a table (DataFrame)
teams = pd.read_csv(TEAMS_CSV)

# 3. Pick the columns you care about (adjust names if needed)
cols = ["team_id", "name", "nickname", "abbr", "league_id", "sub_league_id", "division_id"]

# 4. Print the first 30 rows so you can see them
print(teams[cols].head(30).to_string(index=False))
