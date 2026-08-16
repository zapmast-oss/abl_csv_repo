from pathlib import Path
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
CSV_ROOT = SCRIPT_PATH.parents[1]
STAR_DIR = CSV_ROOT / "out" / "star_schema"
CSV_OUT_DIR = CSV_ROOT / "out" / "csv_out"

TEAM_CANON = set(range(1, 25))


def main(dry_run: bool = False) -> None:
    dim_team_park = pd.read_csv(STAR_DIR / "dim_team_park.csv")
    fact_team_standings = pd.read_csv(STAR_DIR / "fact_team_standings.csv")
    dim_managers = pd.read_csv(CSV_OUT_DIR / "z_ABL_DIM_Managers.csv")

    print("dim_team_park columns:", dim_team_park.columns.tolist())
    print("fact_team_standings columns:", fact_team_standings.columns.tolist())
    print("dim_managers columns:", dim_managers.columns.tolist())

    team_id_col = None
    for candidate in ["ID", "team_id", "Team ID"]:
        if candidate in dim_team_park.columns:
            team_id_col = candidate
            break
    if not team_id_col:
        raise SystemExit("Unable to locate team ID column in dim_team_park")

    dim_team_park = dim_team_park[dim_team_park[team_id_col].isin(TEAM_CANON)].copy()
    dim_team_park[team_id_col] = dim_team_park[team_id_col].astype(int)

    standings_id_col = None
    for candidate in ["ID", "team_id", team_id_col]:
        if candidate in fact_team_standings.columns:
            standings_id_col = candidate
            break
    if not standings_id_col:
        raise SystemExit("Unable to locate team ID column in fact_team_standings")

    if set(fact_team_standings[standings_id_col]) != TEAM_CANON:
        raise SystemExit("fact_team_standings does not cover canonical ABL teams")

    manager_team_col = None
    for candidate in ["current_team_id", "team_id", "Team ID"]:
        if candidate in dim_managers.columns:
            manager_team_col = candidate
            break
    if not manager_team_col:
        raise SystemExit("Unable to locate team assignment column in dim_managers")

    season_col = None
    for candidate in ["last_year", "latest_year", "year", "season"]:
        if candidate in dim_managers.columns:
            season_col = candidate
            break
    if not season_col:
        raise SystemExit("Unable to locate season indicator in dim_managers")

    manager_map = dim_managers.copy()
    manager_map = manager_map[manager_map[manager_team_col].isin(TEAM_CANON)]
    manager_map = manager_map[manager_map[season_col] == manager_map[season_col].max()].copy()

    sort_columns = []
    for candidate in ["total_seasons", "career_wins"]:
        if candidate in manager_map.columns:
            sort_columns.append(candidate)
    if sort_columns:
        manager_map = manager_map.sort_values(sort_columns, ascending=False)
    manager_map = manager_map.drop_duplicates(manager_team_col, keep="first")

    if len(manager_map) != len(TEAM_CANON):
        raise SystemExit("Expected 24 manager rows after filtering; got %d" % len(manager_map))

    rename_map = {}
    if "full_name" in manager_map.columns:
        rename_map["full_name"] = "manager_name"
    elif "first_name" in manager_map.columns and "last_name" in manager_map.columns:
        manager_map["manager_name"] = (
            manager_map["first_name"].str.strip() + " " + manager_map["last_name"].str.strip()
        ).str.strip()
        rename_map["manager_name"] = "manager_name"
    if "career_win_pct" in manager_map.columns:
        rename_map["career_win_pct"] = "manager_career_win_pct"
    if "total_seasons" in manager_map.columns:
        rename_map["total_seasons"] = "manager_total_seasons"
    if "titles_won" in manager_map.columns:
        rename_map["titles_won"] = "manager_titles"
    if "playoff_appearances" in manager_map.columns:
        rename_map["playoff_appearances"] = "manager_playoff_apps"

    keep_cols = [manager_team_col] + [col for col in rename_map.keys() if col != "manager_name"]
    if "manager_name" not in rename_map:
        keep_cols.append("manager_name")
    manager_subset = manager_map[[c for c in manager_map.columns if c in keep_cols]].rename(columns=rename_map)
    if "manager_name" not in manager_subset.columns:
        raise SystemExit("manager_name column missing after renaming")
    manager_subset = manager_subset.rename(columns={manager_team_col: standings_id_col})

    merged = fact_team_standings.merge(
        manager_subset,
        on=standings_id_col,
        how="left",
        validate="one_to_one",
    )

    if len(merged) != len(TEAM_CANON):
        raise SystemExit("Merged standings shape mismatch")
    if merged["manager_name"].isna().any():
        raise SystemExit("Some teams missing manager_name after merge")

    output_path = STAR_DIR / "fact_team_standings_with_managers.csv"
    if not dry_run:
        merged.to_csv(output_path, index=False)

    print("DIM_MANAGERS_ATTACH: attached managers to 24 ABL teams")
    print(f"Output: {output_path.relative_to(CSV_ROOT.parent)}")
    preview_cols = [
        col
        for col in [
            "Abbr",
            "manager_name",
            "manager_career_win_pct",
            "manager_total_seasons",
        ]
        if col in merged.columns
    ]
    print(merged[preview_cols].head(10))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run without writing output CSV")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
