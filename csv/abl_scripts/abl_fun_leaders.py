import pandas as pd

# ---------- CONFIG ----------
LEAGUE_ID = 200
SEASON_YEAR = 1981
MIN_PA = 50       # minimum PA for speed leaders
MIN_OUTS = 30     # minimum outs for K leaders (~10 IP)
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

# ---------- SPEED DEMONS (STEALS) ----------

bat = bat_raw[
    (bat_raw["league_id"] == LEAGUE_ID)
    & (bat_raw["year"] == SEASON_YEAR)
    & (bat_raw["team_id"].isin(abl_team_ids))
].copy()

if bat.empty:
    print("No batting stats found for this league/year.")
else:
    # One row per player+team: keep the one with the most PA (full season line)
    bat_sorted = bat.sort_values(["player_id", "team_id", "pa"], ascending=[True, True, False])
    b = bat_sorted.drop_duplicates(subset=["player_id", "team_id"], keep="first").copy()

    # Require minimum PA
    b = b[b["pa"] >= MIN_PA].copy()

    if b.empty:
        print("No hitters meet the minimum PA threshold for speed leaders.")
    else:
        # AVG for context
        b["avg"] = b["h"] / b["ab"].where(b["ab"] > 0, 1)

        # SB, CS, SB%
        b["sb_attempts"] = b["sb"] + b["cs"]
        b["sbp"] = b["sb"] / b["sb_attempts"].where(b["sb_attempts"] > 0, 1)

        # Merge names & team info
        b = b.merge(players_small, on="player_id", how="left")
        b = b.merge(abl_teams, on="team_id", how="left")

        b["player_name"] = b["first_name"] + " " + b["last_name"]
        b["team_display"] = b["abbr"].fillna("") + " " + b["nickname"].fillna("")

        # Sort: SB desc, then SB% desc
        top_runners = b.sort_values(
            ["sb", "sbp"], ascending=[False, False]
        ).head(TOP_N).copy()

        print(f"=== Top {TOP_N} ABL Base Stealers (Season {SEASON_YEAR}) ===")
        print(
            top_runners[
                ["player_name", "team_display", "pa", "sb", "cs", "sbp", "avg"]
            ]
            .assign(
                avg=lambda df: df["avg"].round(3),
                sbp=lambda df: df["sbp"].round(3),
            )
            .to_string(index=False)
        )

        print("\n--- Speed Demons (Broadcast lines) ---")
        rank = 1
        for _, row in top_runners.iterrows():
            print(
                f"#{rank} speedster: {row['player_name']} ({row['team_display']}): "
                f"{int(row['sb'])} SB, {int(row['cs'])} CS, SB% {row['sbp']:.3f}, AVG {row['avg']:.3f}."
            )
            rank += 1

# ---------- STRIKEOUT ARTISTS (K LEADERS) ----------

pit = pit_raw[
    (pit_raw["league_id"] == LEAGUE_ID)
    & (pit_raw["year"] == SEASON_YEAR)
    & (pit_raw["team_id"].isin(abl_team_ids))
].copy()

if pit.empty:
    print("\nNo pitching stats found for this league/year.")
else:
    # One row per player+team: keep the one with the most outs (full season line)
    pit_sorted = pit.sort_values(["player_id", "team_id", "outs"], ascending=[True, True, False])
    p = pit_sorted.drop_duplicates(subset=["player_id", "team_id"], keep="first").copy()

    # Minimum outs (~10 IP)
    p = p[p["outs"] >= MIN_OUTS].copy()

    if p.empty:
        print("\nNo pitchers meet the minimum IP threshold for K leaders.")
    else:
        # IP from outs
        p["ip"] = p["outs"] / 3.0

        # ERA = 27 * ER / outs
        p["era"] = 27 * p["er"] / p["outs"].where(p["outs"] > 0, 1)

        # K/9 = 27 * K / outs
        p["k9"] = 27 * p["k"] / p["outs"].where(p["outs"] > 0, 1)

        # Merge names & team info
        p = p.merge(players_small, on="player_id", how="left")
        p = p.merge(abl_teams, on="team_id", how="left")

        p["player_name"] = p["first_name"] + " " + p["last_name"]
        p["team_display"] = p["abbr"].fillna("") + " " + p["nickname"].fillna("")

        # Sort: K desc, then K/9 desc
        top_strikeouts = p.sort_values(
            ["k", "k9"], ascending=[False, False]
        ).head(TOP_N).copy()

        print(f"\n=== Top {TOP_N} ABL Strikeout Pitchers (Season {SEASON_YEAR}) ===")
        print(
            top_strikeouts[
                ["player_name", "team_display", "ip", "k", "k9", "era"]
            ]
            .assign(
                ip=lambda df: df["ip"].round(1),
                k9=lambda df: df["k9"].round(1),
                era=lambda df: df["era"].round(2),
            )
            .to_string(index=False)
        )

        print("\n--- Strikeout Artists (Broadcast lines) ---")
        rank = 1
        for _, row in top_strikeouts.iterrows():
            print(
                f"#{rank} strikeout arm: {row['player_name']} ({row['team_display']}): "
                f"{int(row['k'])} K in {row['ip']:.1f} IP, K/9 {row['k9']:.1f}, ERA {row['era']:.2f}."
            )
            rank += 1
