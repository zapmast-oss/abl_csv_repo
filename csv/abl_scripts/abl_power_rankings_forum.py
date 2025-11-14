import pandas as pd


def compute_power_rankings() -> pd.DataFrame:
    # Load core data
    games = pd.read_csv("team_game_log.csv")
    teams = pd.read_csv("teams.csv")
    last10 = pd.read_csv("abl_last10.csv")

    # Regular season, played games only
    games = games[
        (games.get("played", 1) == 1)
        & (games.get("game_type", 2) == 2)
    ]

    # Aggregate wins/losses and runs
    agg = (
        games.groupby("team_id")
        .agg(
            wins=("result", lambda x: (x == "W").sum()),
            losses=("result", lambda x: (x == "L").sum()),
            runs_scored=("team_score", "sum"),
            runs_allowed=("opp_score", "sum"),
        )
        .reset_index()
    )

    agg["games"] = agg["wins"] + agg["losses"]
    agg["win_pct"] = agg["wins"] / agg["games"]
    agg["run_diff"] = agg["runs_scored"] - agg["runs_allowed"]

    # Build team_display from teams.csv
    if "name" in teams.columns:
        name_col = "name"
    elif "team_name" in teams.columns:
        name_col = "team_name"
    else:
        name_col = None

    if name_col:
        teams["team_display"] = (
            teams[name_col].fillna("") + " " + teams["nickname"].fillna("")
        ).str.strip()
    else:
        teams["team_display"] = teams["nickname"].fillna("")

    teams_small = teams[["team_id", "team_display"]]

    # Merge in names
    df = agg.merge(teams_small, on="team_id", how="left")

    # Merge in last10 data
    df = df.merge(
        last10[["team_id", "last10_w", "last10_l", "last10_diff"]],
        on="team_id",
        how="left",
    )

    # Compute streak info from game log
    def compute_streak_for_team(tid: int):
        g = games[games["team_id"] == tid].sort_values(
            ["date", "time", "game_id"]
        )
        g = g[g["result"].isin(["W", "L"])]
        if g.empty:
            return "—", 0, 0
        last_res = g.iloc[-1]["result"]
        streak_len = 0
        for r in reversed(g["result"].tolist()):
            if r == last_res:
                streak_len += 1
            else:
                break
        sign = 1 if last_res == "W" else -1
        return f"{last_res}{streak_len}", streak_len, sign

    streak_display = []
    streak_len = []
    streak_sign = []

    for tid in df["team_id"]:
        disp, length, sign = compute_streak_for_team(tid)
        streak_display.append(disp)
        streak_len.append(length)
        streak_sign.append(sign)

    df["streak_display"] = streak_display
    df["streak_len"] = streak_len
    df["streak_sign"] = streak_sign

    # --- Forum formula: A + B + C + D ---

    # A) 90
    df["A"] = 90

    # B) ROUND( ( (Winning Pct) - 0.500 ) * 162 * 10/9 )
    # 162 * 10 / 9 = 180
    df["B"] = ((df["win_pct"] - 0.5) * 180).round().astype(int)

    # C) ( Wins in Last10 games - Losses in Last10 games )
    df["C"] = df["last10_diff"].fillna(0).astype(int)

    # D) (+1 if on winning streak OR -1 if on losing streak) *
    #    ROUND( ( (Streak length) - 1 ) / 2 )
    df["D"] = (
        df["streak_sign"] * ((df["streak_len"] - 1) / 2).round()
    ).astype(int)

    df["power_total"] = df["A"] + df["B"] + df["C"] + df["D"]

    # Keep only the 24 league teams (no All-Star squads, etc.)
    df = df[df["team_id"] <= 24].copy()

    # Sort by power_total desc, then win_pct, then run_diff
    df = df.sort_values(
        ["power_total", "win_pct", "run_diff"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    # Add rank
    df.index = df.index + 1
    df["rank"] = df.index

    return df


def main():
    try:
        df = compute_power_rankings()
    except FileNotFoundError as e:
        print(f"[Error computing power rankings: {e}]")
        return

    cols = [
        "rank",
        "team_display",
        "wins",
        "losses",
        "win_pct",
        "run_diff",
        "last10_w",
        "last10_l",
        "streak_display",
        "A",
        "B",
        "C",
        "D",
        "power_total",
    ]

    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"[Missing column(s) in power rankings data: {missing}]")
        return

    print("=== ABL Power Rankings (Forum A+B+C+D Formula) ===")
    print(df[cols].to_string(index=False))

    print("\n--- Power Rankings (Broadcast lines) ---")
    for _, row in df.head(10).iterrows():
        print(
            f"#{row['rank']} power: {row['team_display']}: "
            f"{int(row['wins'])}-{int(row['losses'])} ({row['win_pct']:.3f}), "
            f"last 10 {int(row['last10_w'])}-{int(row['last10_l'])}, "
            f"streak {row['streak_display']}, "
            f"A={int(row['A'])}, B={int(row['B'])}, C={int(row['C'])}, D={int(row['D'])}, "
            f"total {int(row['power_total'])}."
        )


if __name__ == "__main__":
    main()
