import pandas as pd

LEAGUE_ID = 200
LAST_N = 10

TEAMS_CSV = "teams.csv"
GAMES_CSV = "games.csv"
OUTPUT_CSV = "abl_last10.csv"

# 1) Load teams and get the 24 ABL teams
teams = pd.read_csv(TEAMS_CSV)
abl_teams = teams[teams["league_id"] == LEAGUE_ID][["team_id", "name", "nickname"]].copy()
abl_team_ids = set(abl_teams["team_id"].tolist())

# Build a display name (e.g. "Las Vegas Gamblers")
abl_teams["team_display"] = abl_teams["name"] + " " + abl_teams["nickname"]

# 2) Load games for this league
games = pd.read_csv(GAMES_CSV)

# Filter to ABL games only
games = games[games["league_id"] == LEAGUE_ID].copy()

# Keep only games that have actually been played
if "innings" in games.columns:
    games = games[games["innings"] > 0].copy()

# Convert date to proper datetime
games["date_dt"] = pd.to_datetime(games["date"], errors="coerce")

# 🛠️ Corrected score assignments: 0 = away, 1 = home
games["away_score"] = games["runs0"]
games["home_score"] = games["runs1"]

team_games_rows = []

# 3) Build a per-team game log: one row per team per game
for _, row in games.iterrows():
    date = row["date"]
    date_dt = row["date_dt"]
    time = row["time"] if "time" in games.columns else ""

    away_team = row["away_team"]
    home_team = row["home_team"]
    away_score = row["away_score"]
    home_score = row["home_score"]
    game_id = row["game_id"]

    # Skip non-ABL games
    if away_team not in abl_team_ids and home_team not in abl_team_ids:
        continue

    # ✅ Corrected win/loss logic based on away/home score
    if away_score > home_score:
        away_result = "W"
        home_result = "L"
    else:
        away_result = "L"
        home_result = "W"

    # Append row for away team
    team_games_rows.append({
        "team_id": away_team,
        "date": date,
        "date_dt": date_dt,
        "time": time,
        "game_id": game_id,
        "result": away_result,
    })

    # Append row for home team
    team_games_rows.append({
        "team_id": home_team,
        "date": date,
        "date_dt": date_dt,
        "time": time,
        "game_id": game_id,
        "result": home_result,
    })

# 4) Convert list to DataFrame
team_games = pd.DataFrame(team_games_rows)

# Filter to ABL teams only
team_games = team_games[team_games["team_id"].isin(abl_team_ids)].copy()

# 5) Sort games for each team by chronological order
sort_cols = ["team_id", "date_dt"]
if "time" in team_games.columns:
    sort_cols.append("time")
sort_cols.append("game_id")

team_games = team_games.sort_values(sort_cols)

# 6) Take LAST N games per team
last_n = team_games.groupby("team_id").tail(LAST_N).copy()

# 7) Count W/L in those games
counts = last_n.groupby(["team_id", "result"]).size().unstack(fill_value=0)

# Ensure both columns exist
counts["W"] = counts.get("W", 0)
counts["L"] = counts.get("L", 0)

# Rename columns and compute win/loss diff
counts = counts.rename(columns={"W": "last10_w", "L": "last10_l"})
counts["last10_diff"] = counts["last10_w"] - counts["last10_l"]

# 8) Attach team names
summary = counts.reset_index().merge(
    abl_teams[["team_id", "team_display"]], on="team_id", how="left"
)

# Reorder columns nicely
summary = summary[["team_id", "team_display", "last10_w", "last10_l", "last10_diff"]]

# Sort by team_id
summary = summary.sort_values("last10_diff",ascending=False).reset_index(drop=True)


# 9) Print
print("=== ABL Last 10 (per team) ===")
print(summary.to_string(index=False))
print("\nSaved last 10 data to", OUTPUT_CSV)

# 10) Save to CSV
summary.to_csv(OUTPUT_CSV, index=False)
