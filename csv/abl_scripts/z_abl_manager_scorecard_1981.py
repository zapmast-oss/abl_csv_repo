from pathlib import Path
import argparse
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
CSV_ROOT = SCRIPT_PATH.parents[1]
STAR_DIR = CSV_ROOT / "out" / "star_schema"
CSV_OUT_DIR = CSV_ROOT / "out" / "csv_out"
TEAM_CANON = set(range(1, 25))


def main(dry_run: bool = False) -> None:
    snapshot_path = STAR_DIR / "fact_team_reporting_1981_current.csv"
    if not snapshot_path.exists():
        print(f"ERROR: Current snapshot not found: {snapshot_path}")
        print("Run z_abl_current_team_snapshot.py first.")
        return

    dim_mgr_path = CSV_OUT_DIR / "z_ABL_DIM_Managers.csv"
    if not dim_mgr_path.exists():
        print(f"ERROR: Manager dimension not found: {dim_mgr_path}")
        print("Run z_abl_dim_managers.py first.")
        return

    snap = pd.read_csv(snapshot_path)
    dim_mgr = pd.read_csv(dim_mgr_path)

    print("SNAPSHOT columns:", list(snap.columns))
    print("DIM_MANAGERS columns:", list(dim_mgr.columns))

    team_id_col = None
    for candidate in ["ID", "team_id", "Team ID"]:
        if candidate in snap.columns:
            team_id_col = candidate
            break
    if not team_id_col:
        raise SystemExit("Unable to find team ID column in snapshot")

    snap = snap.copy()
    snap[team_id_col] = snap[team_id_col].astype(int)
    if len(snap) != 24 or set(snap[team_id_col]) != TEAM_CANON:
        raise SystemExit("Snapshot does not contain canonical 24 teams")

    required_snap_cols = [
        "team_abbr",
        "team_name",
        "sub_league",
        "division",
        "wins",
        "losses",
        "win_pct",
    ]
    for col in required_snap_cols:
        if col not in snap.columns:
            raise SystemExit(f"Snapshot missing column: {col}")

    if "run_diff" not in snap.columns:
        if {"runs_scored", "runs_allowed"}.issubset(snap.columns):
            snap["run_diff"] = snap["runs_scored"] - snap["runs_allowed"]
        else:
            snap["run_diff"] = 0

    snap_sel = snap[[
        team_id_col,
        "team_abbr",
        "team_name",
        "sub_league",
        "division",
        "wins",
        "losses",
        "win_pct",
        "run_diff",
    ]].copy()

    mgr_team_col = None
    for candidate in ["current_team_id", "team_id", "Team ID"]:
        if candidate in dim_mgr.columns:
            mgr_team_col = candidate
            break
    if not mgr_team_col:
        raise SystemExit("Manager dimension missing current team column")

    mgr = dim_mgr[dim_mgr[mgr_team_col].isin(TEAM_CANON)].copy()
    mgr[mgr_team_col] = mgr[mgr_team_col].astype(int)
    sort_cols = []
    if "last_year" in mgr.columns:
        sort_cols.append("last_year")
    for col in ["total_seasons", "career_wins"]:
        if col in mgr.columns:
            sort_cols.append(col)
    if sort_cols:
        mgr = mgr.sort_values(by=sort_cols, ascending=[False] * len(sort_cols))
    mgr = mgr.drop_duplicates(subset=[mgr_team_col], keep="first")
    if mgr[mgr_team_col].nunique() != 24:
        raise SystemExit("Manager dimension does not provide 24 unique assignments")

    if "full_name" in mgr.columns:
        mgr["manager_name"] = mgr["full_name"]
    elif {"first_name", "last_name"}.issubset(mgr.columns):
        mgr["manager_name"] = (
            mgr["first_name"].str.strip() + " " + mgr["last_name"].str.strip()
        ).str.strip()
    else:
        raise SystemExit("Manager dimension missing name fields")

    rename_mgr = {mgr_team_col: team_id_col, "manager_name": "manager_name"}
    if "manager_id" in mgr.columns:
        rename_mgr["manager_id"] = "manager_id"
    if "career_wins" in mgr.columns:
        rename_mgr["career_wins"] = "manager_career_wins"
    if "career_losses" in mgr.columns:
        rename_mgr["career_losses"] = "manager_career_losses"
    if "career_win_pct" in mgr.columns:
        rename_mgr["career_win_pct"] = "manager_career_win_pct"
    if "total_seasons" in mgr.columns:
        rename_mgr["total_seasons"] = "manager_total_seasons"
    if "titles_won" in mgr.columns:
        rename_mgr["titles_won"] = "manager_titles"

    manager_subset = mgr[list(rename_mgr.keys())].rename(columns=rename_mgr)

    scorecard = snap_sel.merge(
        manager_subset,
        on=team_id_col,
        how="left",
        validate="one_to_one",
    )

    if scorecard.shape[0] != 24:
        raise SystemExit("Scorecard join did not produce 24 rows")
    if scorecard["manager_name"].isna().any():
        raise SystemExit("Some teams missing manager_name after join")

    output_cols = [
        team_id_col,
        "team_abbr",
        "team_name",
        "sub_league",
        "division",
        "wins",
        "losses",
        "win_pct",
        "run_diff",
        "manager_name",
    ]
    for col in [
        "manager_id",
        "manager_career_wins",
        "manager_career_losses",
        "manager_career_win_pct",
        "manager_total_seasons",
        "manager_titles",
    ]:
        if col in scorecard.columns:
            output_cols.append(col)

    scorecard = scorecard[output_cols].copy()
    scorecard = scorecard.sort_values(
        by=["win_pct", "run_diff", "team_abbr"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    out_path = STAR_DIR / "fact_manager_scorecard_1981_current.csv"
    if not dry_run:
        scorecard.to_csv(out_path, index=False)
    print("MANAGER_SCORECARD_1981: built 24 rows")
    print("Output:", out_path)
    print(scorecard.head(10))


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run without writing output CSV")
    args = parser.parse_args()
    main(dry_run=args.dry_run)


if __name__ == "__main__":
    cli()
