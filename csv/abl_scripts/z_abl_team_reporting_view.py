from pathlib import Path
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
CSV_ROOT = SCRIPT_PATH.parents[1]
STAR_DIR = CSV_ROOT / "out" / "star_schema"

TEAM_CANON = set(range(1, 25))

def main(dry_run: bool = False) -> None:
    dim_team_park = pd.read_csv(STAR_DIR / "dim_team_park.csv")
    fact_team_mgr = pd.read_csv(STAR_DIR / "fact_team_standings_with_managers.csv")

    print("dim_team_park columns:", list(dim_team_park.columns))
    print("fact_team_standings_with_managers columns:", list(fact_team_mgr.columns))

    team_id_col = None
    for candidate in ["ID", "team_id", "Team ID"]:
        if candidate in dim_team_park.columns:
            team_id_col = candidate
            break
    if not team_id_col:
        raise SystemExit("Unable to find team ID column in dim_team_park")

    dim_team_filtered = dim_team_park[dim_team_park[team_id_col].isin(TEAM_CANON)].copy()
    dim_team_filtered[team_id_col] = dim_team_filtered[team_id_col].astype(int)
    if set(dim_team_filtered[team_id_col]) != TEAM_CANON:
        raise SystemExit("dim_team_park filtered set not equal to canonical team IDs")

    standings_id_col = None
    for candidate in [team_id_col, "ID", "team_id"]:
        if candidate in fact_team_mgr.columns:
            standings_id_col = candidate
            break
    if not standings_id_col:
        raise SystemExit("Unable to find team ID column in fact_team_standings_with_managers")

    fact_team_mgr = fact_team_mgr[fact_team_mgr[standings_id_col].isin(TEAM_CANON)].copy()
    if set(fact_team_mgr[standings_id_col]) != TEAM_CANON:
        raise SystemExit("fact_team_standings_with_managers does not cover canonical team set")

    rename_park = {}
    if "Abbr" in dim_team_filtered.columns:
        rename_park["Abbr"] = "team_abbr"
    if "Team Name" in dim_team_filtered.columns:
        rename_park["Team Name"] = "team_name"
    elif "Name" in dim_team_filtered.columns:
        rename_park["Name"] = "team_name"
    if "SL" in dim_team_filtered.columns:
        rename_park["SL"] = "sub_league"
    if "DIV" in dim_team_filtered.columns:
        rename_park["DIV"] = "division"
    if "Park" in dim_team_filtered.columns:
        rename_park["Park"] = "ballpark_name"

    team_select_cols = [team_id_col] + list(rename_park.keys())
    teams = dim_team_filtered[team_select_cols].rename(columns=rename_park)

    standings_rename = {}
    for src, dest in [
        ("Abbr", "team_abbr"),
        ("SL", "sub_league"),
        ("DIV", "division"),
        ("G", "games"),
        ("W", "wins"),
        ("L", "losses"),
        ("T", "ties"),
        ("%", "win_pct"),
        ("GB", "games_back"),
        ("POS", "division_rank"),
    ]:
        if src in fact_team_mgr.columns:
            standings_rename[src] = dest

    manager_cols = [
        col
        for col in [
            "manager_name",
            "manager_career_win_pct",
            "manager_total_seasons",
            "manager_titles",
            "manager_playoff_apps",
        ]
        if col in fact_team_mgr.columns
    ]

    standings_cols = [standings_id_col] + list(standings_rename.keys()) + manager_cols
    standings_mgr = fact_team_mgr[standings_cols].rename(columns=standings_rename)

    if {"RS", "RA"}.issubset(fact_team_mgr.columns):
        standings_mgr["run_diff"] = fact_team_mgr["RS"] - fact_team_mgr["RA"]

    merged = teams.merge(
        standings_mgr,
        left_on=team_id_col,
        right_on=standings_id_col,
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != 24:
        raise SystemExit("Merged reporting view is not 24 rows")
    if "manager_name" in merged.columns and merged["manager_name"].isna().any():
        raise SystemExit("Some manager_name values are missing in reporting view")

    if {"sub_league", "division", "team_abbr"}.issubset(merged.columns):
        merged = merged.sort_values(["sub_league", "division", "team_abbr"])
    elif "team_abbr" in merged.columns:
        merged = merged.sort_values("team_abbr")

    output_path = STAR_DIR / "fact_team_reporting_view.csv"
    if not dry_run:
        merged.to_csv(output_path, index=False)

    print("TEAM_REPORTING_VIEW: built 24 rows")
    print(f"Output: {output_path.relative_to(CSV_ROOT.parent)}")
    preview_cols = [
        col
        for col in [
            "team_abbr",
            "team_name",
            "sub_league",
            "division",
            "wins",
            "losses",
            "win_pct",
            "manager_name",
            "manager_career_win_pct",
        ]
        if col in merged.columns
    ]
    print(merged[preview_cols].head(10))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without writing output CSV",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
