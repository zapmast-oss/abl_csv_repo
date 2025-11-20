from pathlib import Path
import argparse
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

    print("dim_team_park columns:", list(dim_team_park.columns))
    print("fact_team_standings columns:", list(fact_team_standings.columns))
    print("z_ABL_DIM_Managers columns:", list(dim_managers.columns))

    team_id_col = None
    for candidate in ["ID", "team_id", "Team ID"]:
        if candidate in dim_team_park.columns:
            team_id_col = candidate
            break
    if not team_id_col:
        raise SystemExit("Unable to find team ID column in dim_team_park")

    teams = dim_team_park[dim_team_park[team_id_col].isin(TEAM_CANON)].copy()
    teams[team_id_col] = teams[team_id_col].astype(int)
    if set(teams[team_id_col]) != TEAM_CANON:
        raise SystemExit("dim_team_park filtered set does not match ABL canon")

    standings_id_col = None
    for candidate in [team_id_col, "ID", "team_id"]:
        if candidate in fact_team_standings.columns:
            standings_id_col = candidate
            break
    if not standings_id_col:
        raise SystemExit("Unable to find team ID column in fact_team_standings")

    fact_team_standings = fact_team_standings[
        fact_team_standings[standings_id_col].isin(TEAM_CANON)
    ].copy()
    fact_team_standings[standings_id_col] = fact_team_standings[standings_id_col].astype(int)
    if set(fact_team_standings[standings_id_col]) != TEAM_CANON:
        raise SystemExit("fact_team_standings does not cover all ABL teams")

    rename_park = {}
    if "Abbr" in teams.columns:
        rename_park["Abbr"] = "team_abbr"
    if "Team Name" in teams.columns:
        rename_park["Team Name"] = "team_name"
    elif "Name" in teams.columns:
        rename_park["Name"] = "team_name"
    if "SL" in teams.columns:
        rename_park["SL"] = "sub_league"
    if "DIV" in teams.columns:
        rename_park["DIV"] = "division"
    if "Park" in teams.columns:
        rename_park["Park"] = "ballpark_name"
    team_cols = [team_id_col] + list(rename_park.keys())
    teams = teams[team_cols].rename(columns=rename_park)

    standings_rename = {}
    for src, dest in [
        ("G", "games"),
        ("W", "wins"),
        ("L", "losses"),
        ("T", "ties"),
        ("%", "win_pct"),
        ("GB", "games_back"),
        ("POS", "division_rank"),
    ]:
        if src in fact_team_standings.columns:
            standings_rename[src] = dest
    standings_select = [standings_id_col] + list(standings_rename.keys())
    standings = fact_team_standings[standings_select].rename(columns=standings_rename)

    manager_team_col = None
    for candidate in ["current_team_id", "team_id", "Team ID"]:
        if candidate in dim_managers.columns:
            manager_team_col = candidate
            break
    if not manager_team_col:
        raise SystemExit("Unable to find team assignment column in manager dimension")

    manager_map = dim_managers[dim_managers[manager_team_col].isin(TEAM_CANON)].copy()
    manager_map[manager_team_col] = manager_map[manager_team_col].astype(int)
    sort_cols = []
    if "last_year" in manager_map.columns:
        sort_cols.append("last_year")
    for col in ["total_seasons", "career_wins"]:
        if col in manager_map.columns:
            sort_cols.append(col)
    if sort_cols:
        manager_map = manager_map.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    manager_map = manager_map.drop_duplicates(manager_team_col, keep="first")
    if len(manager_map) != 24:
        raise SystemExit("Manager dimension does not yield exactly 24 current assignments")

    manager_rename = {}
    if "manager_id" in manager_map.columns:
        manager_rename["manager_id"] = "manager_id"
    if "full_name" in manager_map.columns:
        manager_rename["full_name"] = "manager_name"
    elif {"first_name", "last_name"}.issubset(manager_map.columns):
        manager_map["manager_name"] = (
            manager_map["first_name"].str.strip() + " " + manager_map["last_name"].str.strip()
        ).str.strip()
        manager_rename["manager_name"] = "manager_name"
    if "career_wins" in manager_map.columns:
        manager_rename["career_wins"] = "manager_career_wins"
    if "career_losses" in manager_map.columns:
        manager_rename["career_losses"] = "manager_career_losses"
    if "career_win_pct" in manager_map.columns:
        manager_rename["career_win_pct"] = "manager_career_win_pct"
    if "total_seasons" in manager_map.columns:
        manager_rename["total_seasons"] = "manager_total_seasons"
    if "titles_won" in manager_map.columns:
        manager_rename["titles_won"] = "manager_titles"
    manager_cols = [manager_team_col] + list(manager_rename.keys())
    manager_subset = manager_map[manager_cols].rename(columns=manager_rename)
    manager_subset = manager_subset.rename(columns={manager_team_col: team_id_col})

    teams_standings = teams.merge(
        standings, on=team_id_col, how="inner", validate="one_to_one"
    )
    snapshot = teams_standings.merge(
        manager_subset, on=team_id_col, how="left", validate="one_to_one"
    )

    if len(snapshot) != 24:
        raise SystemExit("Snapshot does not contain 24 teams")
    if "manager_name" in snapshot.columns and snapshot["manager_name"].isna().any():
        raise SystemExit("One or more teams missing manager_name after merge")

    if {"sub_league", "division", "team_abbr"}.issubset(snapshot.columns):
        snapshot = snapshot.sort_values(["sub_league", "division", "team_abbr"])
    elif "team_abbr" in snapshot.columns:
        snapshot = snapshot.sort_values("team_abbr")

    output_path = STAR_DIR / "fact_team_reporting_1981_current.csv"
    if not dry_run:
        snapshot.to_csv(output_path, index=False)

    print("CURRENT_TEAM_SNAPSHOT: built 24 rows")
    print("Output:", output_path)
    preview_cols = [
        col
        for col in [
            "team_abbr",
            "team_name",
            "wins",
            "losses",
            "win_pct",
            "manager_name",
        ]
        if col in snapshot.columns
    ]
    print(snapshot[preview_cols].head(10))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run without writing output CSV")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
