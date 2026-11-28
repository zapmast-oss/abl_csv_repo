import pandas as pd
from pathlib import Path


def main() -> int:
    # Path to the league season summary we already trust for W/L and RS/RA
    summary_path = Path("csv/out/almanac/1972/league_season_summary_1972_league200.csv")

    if not summary_path.exists():
        raise SystemExit(f"ERROR: league_season_summary not found at {summary_path}")

    print(f"[INFO] Loading league_season_summary from {summary_path}")
    df = pd.read_csv(summary_path)

    expected_cols = [
        "season",
        "league_id",
        "team_id",
        "team_abbr",
        "team_name",
        "conf",
        "division",
        "wins",
        "losses",
        "pct",
        "runs_for",
        "runs_against",
        "run_diff",
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"ERROR: league_season_summary is missing columns: {missing}")

    # Make sure run_diff is numeric (in case of stray strings)
    df["run_diff"] = pd.to_numeric(df["run_diff"], errors="coerce")
    if df["run_diff"].isna().any():
        bad = df[df["run_diff"].isna()][
            ["team_id", "team_abbr", "team_name", "conf", "division"]
        ]
        raise SystemExit(
            "ERROR: NaN run_diff values found for these teams:\n"
            + bad.to_string(index=False)
        )

    # 1) Whole league sanity check
    league_total_rd = df["run_diff"].sum()
    league_total_wins = df["wins"].sum()
    league_total_losses = df["losses"].sum()

    print("\n=== LEAGUE-LEVEL CHECK ===")
    print(f"Total wins   : {league_total_wins}")
    print(f"Total losses : {league_total_losses}")
    print(f"Total run_diff (sum of all teams): {league_total_rd}")

    # 2) Conference-level check
    print("\n=== CONFERENCE-LEVEL RUN_DIFF TOTALS ===")
    conf_grp = df.groupby("conf", as_index=False).agg(
        total_wins=("wins", "sum"),
        total_losses=("losses", "sum"),
        total_run_diff=("run_diff", "sum"),
    )
    for _, row in conf_grp.iterrows():
        print(
            f"- {row['conf']}: wins={row['total_wins']}, "
            f"losses={row['total_losses']}, "
            f"run_diff_total={row['total_run_diff']}"
        )

    # 3) Division-level detailed breakdown
    print("\n=== DIVISION-LEVEL DETAIL (TEAM BREAKDOWN + TOTALS) ===")

    div_grp = df.groupby(["conf", "division"])

    for (conf, division), sub in div_grp:
        print(f"\n--- {conf} / {division} ---")
        print("Teams:")
        for _, row in sub.sort_values("team_name").iterrows():
            print(
                f"  {row['team_name']} ({row['team_abbr']}): "
                f"{int(row['wins'])}-{int(row['losses'])}, run_diff={int(row['run_diff'])}"
            )
        div_total_rd = sub["run_diff"].sum()
        div_total_wins = sub["wins"].sum()
        div_total_losses = sub["losses"].sum()
        print(
            f"Division totals -> wins={int(div_total_wins)}, "
            f"losses={int(div_total_losses)}, run_diff_total={int(div_total_rd)}"
        )

    # 4) Quick summary of division run_diff totals
    print("\n=== SUMMARY: DIVISION RUN_DIFF TOTALS ===")
    div_summary = (
        df.groupby(["conf", "division"], as_index=False)["run_diff"]
        .sum()
        .rename(columns={"run_diff": "run_diff_total"})
    )
    for _, row in div_summary.iterrows():
        print(
            f"- {row['conf']} / {row['division']}: "
            f"run_diff_total={int(row['run_diff_total'])}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
