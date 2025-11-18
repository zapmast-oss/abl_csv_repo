from pathlib import Path
import shutil
import argparse
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
CSV_ROOT = SCRIPT_PATH.parents[1]
STAR_DIR = CSV_ROOT / "out" / "star_schema"
TEAM_CANON = set(range(1, 25))


def main(dry_run: bool = False) -> None:
    prev_path = STAR_DIR / "fact_team_reporting_1981_prev.csv"
    curr_path = STAR_DIR / "fact_team_reporting_1981_current.csv"

    if not curr_path.exists():
        print(f"ERROR: Current snapshot not found: {curr_path}")
        print("Run z_abl_current_team_snapshot.py first to create fact_team_reporting_1981_current.csv.")
        return

    if not prev_path.exists():
        print(f"No previous snapshot found at: {prev_path}")
        print("Bootstrapping weekly change by copying current snapshot to prev.")
        shutil.copy2(curr_path, prev_path)

    prev = pd.read_csv(prev_path)
    curr = pd.read_csv(curr_path)

    print("PREV snapshot columns:", list(prev.columns))
    print("CURR snapshot columns:", list(curr.columns))

    team_id_col = None
    for candidate in ["ID", "team_id", "Team ID"]:
        if candidate in prev.columns and candidate in curr.columns:
            team_id_col = candidate
            break
    if not team_id_col:
        raise SystemExit("Unable to find common team ID column")

    prev = prev.copy()
    curr = curr.copy()
    prev[team_id_col] = prev[team_id_col].astype(int)
    curr[team_id_col] = curr[team_id_col].astype(int)

    if len(prev) != 24 or len(curr) != 24:
        raise SystemExit("Snapshots must each have 24 rows")
    if set(prev[team_id_col]) != set(curr[team_id_col]) or set(prev[team_id_col]) != TEAM_CANON:
        raise SystemExit("Team IDs do not match ABL canon or snapshots misaligned")

    if "run_diff" not in prev.columns:
        if {"runs_scored", "runs_allowed"}.issubset(prev.columns):
            prev["run_diff"] = prev["runs_scored"] - prev["runs_allowed"]
        else:
            prev["run_diff"] = 0
    if "run_diff" not in curr.columns:
        if {"runs_scored", "runs_allowed"}.issubset(curr.columns):
            curr["run_diff"] = curr["runs_scored"] - curr["runs_allowed"]
        else:
            curr["run_diff"] = 0

    base_cols = [team_id_col]
    for col in ["team_abbr", "team_name", "wins", "losses", "win_pct", "run_diff"]:
        if col not in prev.columns or col not in curr.columns:
            raise SystemExit(f"Column {col} missing from snapshots")
        base_cols.append(col)

    prev_sel = prev[base_cols].copy()
    curr_sel = curr[base_cols].copy()

    prev_renamed = prev_sel.rename(
        columns={
            "team_abbr": "team_abbr_prev",
            "team_name": "team_name_prev",
            "wins": "wins_prev",
            "losses": "losses_prev",
            "win_pct": "win_pct_prev",
            "run_diff": "run_diff_prev",
        }
    )
    curr_renamed = curr_sel.rename(
        columns={
            "team_abbr": "team_abbr_curr",
            "team_name": "team_name_curr",
            "wins": "wins_curr",
            "losses": "losses_curr",
            "win_pct": "win_pct_curr",
            "run_diff": "run_diff_curr",
        }
    )

    merged = prev_renamed.merge(curr_renamed, on=team_id_col, how="inner", validate="one_to_one")
    if len(merged) != 24:
        raise SystemExit("Merged snapshot mismatch")

    merged["delta_wins"] = merged["wins_curr"] - merged["wins_prev"]
    merged["delta_losses"] = merged["losses_curr"] - merged["losses_prev"]
    merged["delta_win_pct"] = merged["win_pct_curr"] - merged["win_pct_prev"]
    merged["delta_run_diff"] = merged["run_diff_curr"] - merged["run_diff_prev"]
    merged["games_prev"] = merged["wins_prev"] + merged["losses_prev"]
    merged["games_curr"] = merged["wins_curr"] + merged["losses_curr"]
    merged["games_this_week"] = merged["games_curr"] - merged["games_prev"]

    cols = [
        team_id_col,
        "team_abbr_curr",
        "team_name_curr",
        "wins_prev",
        "losses_prev",
        "win_pct_prev",
        "run_diff_prev",
        "wins_curr",
        "losses_curr",
        "win_pct_curr",
        "run_diff_curr",
        "games_this_week",
        "delta_wins",
        "delta_losses",
        "delta_win_pct",
        "delta_run_diff",
    ]
    movement = merged[cols].copy()
    movement = movement.rename(columns={"team_abbr_curr": "team_abbr", "team_name_curr": "team_name"})

    movement.sort_values(
        by=["delta_wins", "delta_run_diff", "win_pct_curr"],
        ascending=[False, False, False],
        inplace=True,
    )

    out_path = STAR_DIR / "fact_team_reporting_1981_weekly_change.csv"
    if not dry_run:
        movement.to_csv(out_path, index=False)

    print("WEEKLY_CHANGE_1981: built 24 rows")
    print("Output:", out_path)
    preview_cols = [
        "team_abbr",
        "wins_prev",
        "wins_curr",
        "games_this_week",
        "delta_wins",
        "delta_run_diff",
        "delta_win_pct",
    ]
    print(movement[preview_cols].head(10))


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run without writing output CSV")
    args = parser.parse_args()
    main(dry_run=args.dry_run)


if __name__ == "__main__":
    cli()





