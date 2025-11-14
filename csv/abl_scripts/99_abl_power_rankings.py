import pandas as pd

LEAGUE_ID = 200
STANDINGS_CSV = "standings_snapshot.csv"
TEAMS_CSV = "teams.csv"
TOP_N = 24   # you can change to 10 if you only want the top 10

# Load data
standings = pd.read_csv(STANDINGS_CSV)
teams = pd.read_csv(TEAMS_CSV)

# ABL team ids (league_id = 200)
abl_team_ids = set(
    teams[teams["league_id"] == LEAGUE_ID]["team_id"].tolist()
)

# Filter standings to ABL only
abl = standings[standings["team_id"].isin(abl_team_ids)].copy()


if abl.empty:
    print("No standings data found for ABL in standings_snapshot.csv.")
else:
    # Ensure we have win_pct; standings_snapshot already has it, but we recompute just in case
    abl["games"] = abl["wins"] + abl["losses"]
    abl["win_pct"] = abl["wins"] / abl["games"].where(abl["games"] > 0, 1)

    # Simple power score: mostly win%, with a run differential kicker
    abl["power_score"] = abl["win_pct"] * 100 + abl["run_diff"] / 5.0

    # Sort: higher power_score is better
    abl = abl.sort_values("power_score", ascending=False).reset_index(drop=True)

    # Add rank
    abl["rank"] = abl.index + 1

    # Limit to top N
    top = abl.head(TOP_N).copy()

    print("=== ABL Power Rankings ===")
    print(
        top[
            [
                "rank",
                "team_display",
                "wins",
                "losses",
                "win_pct",
                "run_diff",
                "power_score",
            ]
        ]
        .assign(
            win_pct=lambda df: df["win_pct"].round(3),
            power_score=lambda df: df["power_score"].round(1),
        )
        .to_string(index=False)
    )

    print("\n--- Power Rankings (Broadcast lines) ---")
    for _, row in top.iterrows():
        print(
            f"#{int(row['rank'])} in ABL Power Rankings: "
            f"{row['team_display']}: {int(row['wins'])}-{int(row['losses'])} "
            f"({row['win_pct']:.3f}), run diff {int(row['run_diff'])}, "
            f"power score {row['power_score']:.1f}."
        )
