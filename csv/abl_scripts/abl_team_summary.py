import pandas as pd

# These files are in the SAME folder as this script
TEAMS_CSV = "teams.csv"
GAMES_CSV = "games.csv"
SCORES_CSV = "games.score.csv"  # default OOTP name for game scores

# --- LOAD TABLES ---

teams = pd.read_csv(TEAMS_CSV)
games = pd.read_csv(GAMES_CSV)
scores = pd.read_csv(SCORES_CSV)

# Keep only the main ABL league (league_id = 200)
abl_teams = teams[teams["league_id"] == 200].copy()

# Standardize column names we care about from games.csv
games = games.rename(
    columns={
        "GameID": "game_id",
        "Game Id": "game_id",
        "Date": "date",
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
    }
)

# Standardize column names from games.score.csv
scores = scores.rename(
    columns={
        "GameID": "game_id",
        "Team": "team",
        "Runs": "runs",
    }
)

# --- BUILD PER-TEAM / PER-GAME RESULTS TABLE ---

# Each row: one team in one game, with its runs
game_scores = games.merge(scores, on="game_id", how="inner")

# Copy to line up opponent runs
opp = game_scores.copy()
opp = opp.rename(columns={"team": "opp_team", "runs": "opp_runs"})

# Merge back so each row has team and opponent for the same game
team_games = game_scores.merge(
    opp[["game_id", "opp_team", "opp_runs"]],
    on="game_id",
    how="inner"
)

# Drop self-joins (where team == opp_team)
team_games = team_games[team_games["team"] != team_games["opp_team"]].copy()

# Win/loss and runs for each team in each game
team_games["win"] = (team_games["runs"] > team_games["opp_runs"]).astype(int)
team_games["loss"] = (team_games["runs"] < team_games["opp_runs"]).astype(int)
team_games["runs_scored"] = team_games["runs"]
team_games["runs_allowed"] = team_games["opp_runs"]

# --- AGGREGATE BY TEAM ---

totals = (
    team_games
    .groupby("team")
    .agg(
        games_played=("game_id", "count"),
        wins=("win", "sum"),
        losses=("loss", "sum"),
        runs_scored=("runs_scored", "sum"),
        runs_allowed=("runs_allowed", "sum"),
    )
    .reset_index()
)

totals["run_diff"] = totals["runs_scored"] - totals["runs_allowed"]

# --- JOIN BACK TO TEAM INFO (ABL ONLY) ---

summary = totals.merge(
    abl_teams,
    left_on="team",
    right_on="team_id",
    how="inner"
)

# Columns to show
summary = summary[
    ["team_id", "abbr", "name", "nickname",
     "wins", "losses", "runs_scored", "runs_allowed", "run_diff"]
].sort_values(["wins", "run_diff"], ascending=[False, False])

print(summary.to_string(index=False))
