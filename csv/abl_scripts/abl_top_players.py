import pandas as pd

LEAGUE_ID = 200
TOP_N = 5  # how many hitters/pitchers to show

PLAYERS_CSV = "players.csv"
TEAMS_CSV = "teams.csv"
PLAYERS_VALUE_CSV = "players_value.csv"

# -------- LOAD BASE TABLES --------

players = pd.read_csv(PLAYERS_CSV)
teams = pd.read_csv(TEAMS_CSV)
pval = pd.read_csv(PLAYERS_VALUE_CSV)

# ABL teams only
abl_teams = teams[teams["league_id"] == LEAGUE_ID][["team_id", "abbr", "nickname"]].copy()
abl_team_ids = set(abl_teams["team_id"].tolist())

# Filter player values to ABL teams
pval = pval[pval["team_id"].isin(abl_team_ids)].copy()

# Merge in player and team info
players_small = players[["player_id", "first_name", "last_name"]].copy()


pval = pval.merge(players_small, on="player_id", how="left")
pval = pval.merge(abl_teams, on="team_id", how="left", suffixes=("", "_team"))

pval["player_name"] = pval["first_name"] + " " + pval["last_name"]
pval["team_display"] = pval["abbr"].fillna("") + " " + pval["nickname"].fillna("")

# -------- TOP HITTERS (OFFENSIVE VALUE) --------

hitters = pval[pval["offensive_value"] > 0].copy()

if hitters.empty:
    print("No hitters found with offensive_value > 0.")
else:
    top_hitters = hitters.sort_values(
        ["offensive_value", "overall_value"],
        ascending=[False, False]
    ).head(TOP_N).copy()

    print(f"=== Top {TOP_N} ABL Hitters (by offensive value) ===")
    print(
        top_hitters[["player_name", "team_display", "offensive_value", "overall_value", "season_performance"]]
        .assign(
            offensive_value=lambda df: df["offensive_value"].round(1),
            overall_value=lambda df: df["overall_value"].round(1),
            season_performance=lambda df: df["season_performance"].round(1)
        )
        .to_string(index=False)
    )

    print("\n--- Hitters (Broadcast lines) ---")
rank = 1
for _, row in top_hitters.iterrows():
    print(
        f"#{rank} hitter: {row['player_name']} ({row['team_display']}), "
        f"season performance {row['season_performance']:.1f}."
    )
    rank += 1


# -------- TOP PITCHERS (PITCHING VALUE) --------

pitchers = pval[pval["pitching_value"] > 0].copy()

if pitchers.empty:
    print("\nNo pitchers found with pitching_value > 0.")
else:
    top_pitchers = pitchers.sort_values(
        ["pitching_value", "overall_value"],
        ascending=[False, False]
    ).head(TOP_N).copy()

    print(f"\n=== Top {TOP_N} ABL Pitchers (by pitching value) ===")
    print(
        top_pitchers[["player_name", "team_display", "pitching_value", "overall_value", "season_performance"]]
        .assign(
            pitching_value=lambda df: df["pitching_value"].round(1),
            overall_value=lambda df: df["overall_value"].round(1),
            season_performance=lambda df: df["season_performance"].round(1)
        )
        .to_string(index=False)
    )

    print("\n--- Pitchers (Broadcast lines) ---")
rank = 1
for _, row in top_pitchers.iterrows():
    print(
        f"#{rank} pitcher: {row['player_name']} ({row['team_display']}), "
        f"season performance {row['season_performance']:.1f}."
    )
    rank += 1

