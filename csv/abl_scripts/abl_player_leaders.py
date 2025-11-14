import pandas as pd

# ---------- CONFIG ----------
LEAGUE_ID = 200
SEASON_YEAR = 1981
MIN_PA = 50       # minimum PA for hitters
MIN_OUTS = 60     # minimum outs for pitchers (~20 IP)
TOP_N = 5

# ---------- FILES ----------
PLAYERS_CSV = "players.csv"
TEAMS_CSV = "teams.csv"
BAT_CSV = "players_career_batting_stats.csv"
PIT_CSV = "players_career_pitching_stats.csv"

# ---------- LOAD BASE TABLES ----------

players = pd.read_csv(PLAYERS_CSV)
teams = pd.read_csv(TEAMS_CSV)
bat_raw = pd.read_csv(BAT_CSV)
pit_raw = pd.read_csv(PIT_CSV)

# ABL team list
abl_teams = teams[teams["league_id"] == LEAGUE_ID][["team_id", "abbr", "nickname"]].copy()
abl_team_ids = set(abl_teams["team_id"].tolist())

players_small = players[["player_id", "first_name", "last_name"]].copy()

# ---------- HITTERS: TOP BATS ----------

# Filter for this season, this league, ABL teams
bat = bat_raw[
    (bat_raw["league_id"] == LEAGUE_ID)
    & (bat_raw["year"] == SEASON_YEAR)
    & (bat_raw["team_id"].isin(abl_team_ids))
].copy()

if bat.empty:
    print("No batting stats found for this league/year.")
else:
    # For each player+team, keep ONLY the row with the highest PA
    bat_sorted = bat.sort_values(["player_id", "team_id", "pa"], ascending=[True, True, False])
    b = bat_sorted.drop_duplicates(subset=["player_id", "team_id"], keep="first").copy()

    # Require minimum PA
    b = b[b["pa"] >= MIN_PA].copy()

    if b.empty:
        print("No hitters meet the minimum PA threshold.")
    else:
        # Compute AVG, OBP, SLG, OPS from that one row
        singles = b["h"] - b["d"] - b["t"] - b["hr"]
        b["tb"] = singles + 2 * b["d"] + 3 * b["t"] + 4 * b["hr"]

        # AVG
        b["avg"] = b["h"] / b["ab"].where(b["ab"] > 0, 1)

        # OBP = (H + BB) / (AB + BB + SF)   (we treat HBP as 0)
        obp_denom = b["ab"] + b["bb"] + b["sf"]
        b["obp"] = (b["h"] + b["bb"]) / obp_denom.where(obp_denom > 0, 1)

        # SLG & OPS
        b["slg"] = b["tb"] / b["ab"].where(b["ab"] > 0, 1)
        b["ops"] = b["obp"] + b["slg"]

        # Merge names & team info
        b = b.merge(players_small, on="player_id", how="left")
        b = b.merge(abl_teams, on="team_id", how="left")

        b["player_name"] = b["first_name"] + " " + b["last_name"]
        b["team_display"] = b["abbr"].fillna("") + " " + b["nickname"].fillna("")

        # Sort: HR desc, OPS desc
        top_hitters = b.sort_values(
            ["hr", "ops"], ascending=[False, False]
        ).head(TOP_N).copy()

        print(f"=== Top {TOP_N} ABL Hitters (Season {SEASON_YEAR}) ===")
        print(
            top_hitters[
                ["player_name", "team_display", "pa", "avg", "hr", "rbi", "ops"]
            ]
            .assign(
                avg=lambda df: df["avg"].round(3),
                ops=lambda df: df["ops"].round(3),
            )
            .to_string(index=False)
        )

        print("\n--- Hitters (Broadcast lines) ---")
        rank = 1
        for _, row in top_hitters.iterrows():
            print(
                f"#{rank} hitter: {row['player_name']} ({row['team_display']}): "
                f"AVG {row['avg']:.3f}, {int(row['hr'])} HR, {int(row['rbi'])} RBI, OPS {row['ops']:.3f}."
            )
            rank += 1


# ---------- PITCHERS: TOP ARMS ----------

pit = pit_raw[
    (pit_raw["league_id"] == LEAGUE_ID)
    & (pit_raw["year"] == SEASON_YEAR)
    & (pit_raw["team_id"].isin(abl_team_ids))
].copy()

if pit.empty:
    print("\nNo pitching stats found for this league/year.")
else:
    # For each player+team, keep ONLY the row with the highest outs
    pit_sorted = pit.sort_values(["player_id", "team_id", "outs"], ascending=[True, True, False])
    p = pit_sorted.drop_duplicates(subset=["player_id", "team_id"], keep="first").copy()

    # Minimum outs (~20 IP)
    p = p[p["outs"] >= MIN_OUTS].copy()

    # Require at least one decision
    p = p[(p["w"] + p["l"]) > 0].copy()

    if p.empty:
        print("\nNo pitchers meet the minimum IP/decision threshold.")
    else:
        # IP from outs
        p["ip"] = p["outs"] / 3.0

        # ERA = 27 * ER / outs
        p["era"] = 27 * p["er"] / p["outs"].where(p["outs"] > 0, 1)

        # WHIP = (BB + H) / IP
        p["whip"] = (p["bb"] + p["ha"]) / p["ip"].where(p["ip"] > 0, 1)

        # Merge names & team info
        p = p.merge(players_small, on="player_id", how="left")
        p = p.merge(abl_teams, on="team_id", how="left")


        p["player_name"] = p["first_name"] + " " + p["last_name"]
        p["team_display"] = p["abbr"].fillna("") + " " + p["nickname"].fillna("")

        # Sort: ERA asc, then K desc
        top_pitchers = p.sort_values(
            ["era", "k"], ascending=[True, False]
        ).head(TOP_N).copy()

        print(f"\n=== Top {TOP_N} ABL Pitchers (Season {SEASON_YEAR}) ===")
        print(
            top_pitchers[
                ["player_name", "team_display", "ip", "w", "l", "era", "k", "whip"]
            ]
            .assign(
                ip=lambda df: df["ip"].round(1),
                era=lambda df: df["era"].round(2),
                whip=lambda df: df["whip"].round(2),
            )
            .to_string(index=False)
        )

        print("\n--- Pitchers (Broadcast lines) ---")
        rank = 1
        for _, row in top_pitchers.iterrows():
            print(
                f"#{rank} pitcher: {row['player_name']} ({row['team_display']}): "
                f"{int(row['w'])}-{int(row['l'])}, {row['ip']:.1f} IP, "
                f"ERA {row['era']:.2f}, {int(row['k'])} K, WHIP {row['whip']:.2f}."
            )
            rank += 1


