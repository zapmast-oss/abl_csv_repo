import pandas as pd

# ---------------- CONFIG ----------------
LEAGUE_ID = 200
SEASON_YEAR = 1981
MIN_PA = 50      # minimum plate appearances for hitters
MIN_OUTS = 60    # minimum outs for pitchers (60 outs = 20 IP)

# ---------------- FILES -----------------
PLAYERS_CSV = "players.csv"
TEAMS_CSV = "teams.csv"
BAT_STATS_CSV = "players_career_batting_stats.csv"
PIT_STATS_CSV = "players_career_pitching_stats.csv"

# ---------------- LOAD BASE TABLES -----------------

players = pd.read_csv(PLAYERS_CSV)
teams = pd.read_csv(TEAMS_CSV)
bat_stats = pd.read_csv(BAT_STATS_CSV)
pit_stats = pd.read_csv(PIT_STATS_CSV)

# ABL team list (league_id = 200)
abl_teams = teams[teams["league_id"] == LEAGUE_ID][["team_id", "name", "nickname", "abbr"]].copy()
abl_team_ids = set(abl_teams["team_id"].tolist())

# Small lookup tables
players_small = players[["player_id", "first_name", "last_name", "team_id"]].copy()

# ---------------- HITTERS: TOP BATS -----------------

# Filter to ABL, correct year, overall split
bat = bat_stats[
    (bat_stats["league_id"] == LEAGUE_ID)
    & (bat_stats["year"] == SEASON_YEAR)
    & (bat_stats["split_id"] == 0)
    & (bat_stats["team_id"].isin(abl_team_ids))
].copy()

# Require a minimum PA
bat = bat[bat["pa"] >= MIN_PA].copy()

if bat.empty:
    print("No batting stats found for this league/year with the given filters.")
else:
    # Compute total bases: TB = 1B + 2*2B + 3*3B + 4*HR
    bat["singles"] = bat["h"] - bat["d"] - bat["t"] - bat["hr"]
    bat["tb"] = bat["singles"] + 2 * bat["d"] + 3 * bat["t"] + 4 * bat["hr"]

    # AVG, OBP, SLG, OPS (guard against division by zero)
    bat["avg"] = bat["h"] / bat["ab"].where(bat["ab"] > 0, 1)
    obp_denom = bat["ab"] + bat["bb"] + bat["hp"] + bat["sf"]
    bat["obp"] = (bat["h"] + bat["bb"] + bat["hp"]) / obp_denom.where(obp_denom > 0, 1)
    bat["slg"] = bat["tb"] / bat["ab"].where(bat["ab"] > 0, 1)
    bat["ops"] = bat["obp"] + bat["slg"]

    # Merge in player & team info
    bat = bat.merge(players_small, on="player_id", how="left")
    bat = bat.merge(abl_teams, on="team_id", how="left", suffixes=("", "_team"))

    # Nice display columns
    bat["player_name"] = bat["first_name"] + " " + bat["last_name"]
    bat["team_display"] = bat["abbr"].fillna("") + " " + bat["nickname"].fillna("")

    # Sort: HR desc, OPS desc
    top_hitters = bat.sort_values(
        ["hr", "ops"], ascending=[False, False]
    ).head(5).copy()

    print("=== Top ABL Hitters (Season " + str(SEASON_YEAR) + ") ===")
    print(
        top_hitters[
            ["player_name", "team_display", "pa", "avg", "hr", "rbi", "ops"]
        ]
        .assign(
            avg=lambda df: df["avg"].round(3),
            ops=lambda df: df["ops"].round(3)
        )
        .to_string(index=False)
    )

    print("\n--- Hitters (Broadcast lines) ---")
    for _, row in top_hitters.iterrows():
        print(
            f"{row['player_name']} ({row['team_display']}): "
            f"{int(row['hr'])} HR, {int(row['rbi'])} RBI, "
            f"AVG {row['avg']:.3f}, OPS {row['ops']:.3f}"
        )

# ---------------- PITCHERS: TOP ARMS -----------------

# Filter to ABL, correct year, overall split
pit = pit_stats[
    (pit_stats["league_id"] == LEAGUE_ID)
    & (pit_stats["year"] == SEASON_YEAR)
    & (pit_stats["split_id"] == 0)
    & (pit_stats["team_id"].isin(abl_team_ids))
].copy()

# Require a minimum number of outs (to avoid tiny samples)
pit = pit[pit["outs"] >= MIN_OUTS].copy()

if pit.empty:
    print("\nNo pitching stats found for this league/year with the given filters.")
else:
    # ERA = 27 * ER / outs  (since outs = IP * 3)
    pit["era"] = 27 * pit["er"] / pit["outs"].where(pit["outs"] > 0, 1)

    # Merge in player & team info
    pit = pit.merge(players_small, on="player_id", how="left")
    pit = pit.merge(abl_teams, on="team_id", how="left", suffixes=("", "_team"))

    pit["player_name"] = pit["first_name"] + " " + pit["last_name"]
    pit["team_display"] = pit["abbr"].fillna("") + " " + pit["nickname"].fillna("")

    # Compute IP from outs
    pit["ip"] = pit["outs"] / 3.0

    # Sort: WAR desc, ERA asc
    if "war" in pit.columns:
        sort_cols = ["war", "era"]
        ascending = [False, True]
    else:
        # Fallback: sort by ERA only
        sort_cols = ["era"]
        ascending = [True]

    top_pitchers = pit.sort_values(
        sort_cols,
        ascending=ascending
    ).head(5).copy()

    print("\n=== Top ABL Pitchers (Season " + str(SEASON_YEAR) + ") ===")
    cols_to_show = ["player_name", "team_display", "ip", "w", "l", "era"]
    if "war" in top_pitchers.columns:
        cols_to_show.append("war")

    print(
        top_pitchers[cols_to_show]
        .assign(
            ip=lambda df: df["ip"].round(1),
            era=lambda df: df["era"].round(2),
            war=lambda df: df["war"].round(1) if "war" in df.columns else df.get("war")
        )
        .to_string(index=False)
    )

    print("\n--- Pitchers (Broadcast lines) ---")
    for _, row in top_pitchers.iterrows():
        war_part = ""
        if "war" in top_pitchers.columns and not pd.isna(row.get("war", None)):
            war_part = f", WAR {row['war']:.1f}"
        print(
            f"{row['player_name']} ({row['team_display']}): "
            f"{int(row['w'])}-{int(row['l'])}, "
            f"{row['ip']:.1f} IP, ERA {row['era']:.2f}{war_part}"
        )
