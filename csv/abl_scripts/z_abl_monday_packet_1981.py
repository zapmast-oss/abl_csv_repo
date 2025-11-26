from pathlib import Path
import argparse
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
CSV_ROOT = SCRIPT_PATH.parents[1]
STAR_DIR = CSV_ROOT / "out" / "star_schema"


def main(dry_run: bool = False) -> None:
    curr_path = STAR_DIR / "fact_team_reporting_1981_current.csv"
    if not curr_path.exists():
        print(f"ERROR: Current snapshot not found: {curr_path}")
        print("Run z_abl_current_team_snapshot.py before building the Monday packet.")
        return

    curr = pd.read_csv(curr_path)
    print("CURRENT snapshot columns:", list(curr.columns))

    required_cols = ["team_abbr", "team_name", "sub_league", "division", "wins", "losses", "win_pct"]
    for col in required_cols:
        if col not in curr.columns:
            raise SystemExit(f"Current snapshot missing column: {col}")

    if "run_diff" not in curr.columns:
        if {"runs_scored", "runs_allowed"}.issubset(curr.columns):
            curr["run_diff"] = curr["runs_scored"] - curr["runs_allowed"]
        else:
            curr["run_diff"] = 0
    curr["games"] = curr["wins"] + curr["losses"]

    standings_cols = ["sub_league", "division", "team_abbr", "team_name", "games", "wins", "losses", "win_pct", "run_diff"]
    standings = curr[standings_cols].copy().sort_values(
        by=["sub_league", "division", "win_pct", "run_diff"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)
    out_standings = STAR_DIR / "monday_1981_standings_by_division.csv"
    if not dry_run:
        standings.to_csv(out_standings, index=False)
    print("MONDAY PACKET: wrote division standings to", out_standings)

    out_power = STAR_DIR / "monday_1981_power_ranking.csv"
    if out_power.exists():
        print("MONDAY PACKET: power ranking already present; leaving as-is:", out_power)
    else:
        power_cols = ["team_abbr", "team_name", "sub_league", "division", "games", "wins", "losses", "win_pct", "run_diff"]
        power = curr[power_cols].copy().sort_values(
            by=["win_pct", "run_diff"],
            ascending=[False, False],
        ).reset_index(drop=True)
        power["power_rank"] = range(1, len(power) + 1)
        if not dry_run:
            power.to_csv(out_power, index=False)
        print("MONDAY PACKET: wrote fallback power ranking to", out_power)

    change_path = STAR_DIR / "fact_team_reporting_1981_weekly_change.csv"
    if not change_path.exists():
        print(f"No weekly change file found at: {change_path}")
        print("MONDAY PACKET: weekly change missing; risers/fallers skipped.")
        return

    change = pd.read_csv(change_path)
    print("WEEKLY CHANGE columns:", list(change.columns))

    team_col = None
    for candidate in ["team_abbr_curr", "team_abbr"]:
        if candidate in change.columns:
            team_col = candidate
            break
    if team_col is None:
        print("MONDAY PACKET: no recognizable team column in weekly change; skipping risers/fallers.")
        return

    base_cols = [team_col]
    for col in ["delta_wins", "delta_run_diff", "delta_win_pct", "games_this_week", "wins_curr", "losses_curr"]:
        if col in change.columns:
            base_cols.append(col)

    change_sel = change[base_cols].copy().rename(columns={team_col: "team_abbr"})

    if "delta_win_pct" not in change_sel.columns:
        print("MONDAY PACKET: weekly change missing delta_win_pct; skipping risers/fallers.")
        return

    risers = change_sel.sort_values(by="delta_win_pct", ascending=False).head(3)
    fallers = change_sel.sort_values(by="delta_win_pct", ascending=True).head(3)

    out_risers = STAR_DIR / "monday_1981_risers.csv"
    out_fallers = STAR_DIR / "monday_1981_fallers.csv"
    if not dry_run:
        risers.to_csv(out_risers, index=False)
        fallers.to_csv(out_fallers, index=False)
    print("MONDAY PACKET: wrote risers to", out_risers)
    print("MONDAY PACKET: wrote fallers to", out_fallers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run without writing output CSVs")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
