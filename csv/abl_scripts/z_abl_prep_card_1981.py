from pathlib import Path
import argparse
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
CSV_ROOT = SCRIPT_PATH.parents[1]
STAR_DIR = CSV_ROOT / "out" / "star_schema"


def load_csv(path: Path, label: str):
    if not path.exists():
        print(f"ERROR: required file missing: {path}")
        return None
    df = pd.read_csv(path)
    print(f"{label} columns:", list(df.columns))
    return df


def main(away_abbr: str, home_abbr: str, dry_run: bool = False):
    away_abbr = away_abbr.upper()
    home_abbr = home_abbr.upper()

    snapshot_path = STAR_DIR / "fact_team_reporting_1981_current.csv"
    manager_path = STAR_DIR / "fact_manager_scorecard_1981_current.csv"
    backbone_path = STAR_DIR / "fact_team_season_1981_backbone.csv"
    change_path = STAR_DIR / "fact_team_reporting_1981_weekly_change.csv"

    snapshot = load_csv(snapshot_path, "SNAPSHOT")
    if snapshot is None:
        return
    managers = load_csv(manager_path, "MANAGER")
    if managers is None:
        return
    backbone = load_csv(backbone_path, "BACKBONE")
    if backbone is None:
        return

    change = None
    if change_path.exists():
        change = pd.read_csv(change_path)
        print("WEEKLY CHANGE columns:", list(change.columns))
    else:
        print(f"No weekly change file found at: {change_path}")

    required_snapshot_cols = {"team_abbr", "team_name", "sub_league", "division", "wins", "losses", "win_pct"}
    missing_snapshot = required_snapshot_cols - set(snapshot.columns)
    if missing_snapshot:
        print(f"ERROR: Snapshot missing columns: {missing_snapshot}")
        return

    unique_teams = snapshot["team_abbr"].unique()
    if len(unique_teams) != 24:
        print("WARNING: Snapshot does not contain 24 teams.")

    for abbr in [away_abbr, home_abbr]:
        if abbr not in unique_teams:
            print(f"ERROR: team_abbr '{abbr}' not found in 1981 snapshot.")
            return

    base = snapshot[
        ["team_abbr", "team_name", "sub_league", "division", "wins", "losses", "win_pct"]
        + (["run_diff"] if "run_diff" in snapshot.columns else [])
    ].copy()
    base["games"] = base["wins"] + base["losses"]
    base = base[base["team_abbr"].isin([away_abbr, home_abbr])].copy()

    backbone_cols = {
        "runs_scored": "season_rs",
        "runs_allowed": "season_ra",
        "run_diff": "season_run_diff",
    }
    missing_backbone = [col for col in ["team_abbr"] + list(backbone_cols.keys()) if col not in backbone.columns]
    if missing_backbone:
        print(f"ERROR: Backbone missing columns: {missing_backbone}")
        return

    backbone_renamed = backbone.rename(columns=backbone_cols)
    base = base.merge(
        backbone_renamed[
            [
                "team_abbr",
                "season_rs",
                "season_ra",
                "season_run_diff",
                "pythag_win_pct",
                "pythag_expected_wins",
                "pythag_diff",
            ]
        ],
        on="team_abbr",
        how="left",
        validate="one_to_one",
    )

    mgr_cols = [
        "team_abbr",
        "manager_name",
        "manager_career_wins",
        "manager_career_losses",
        "manager_career_win_pct",
        "manager_total_seasons",
    ]
    missing_mgr = [col for col in mgr_cols if col not in managers.columns]
    if missing_mgr:
        print(f"ERROR: Manager scorecard missing columns: {missing_mgr}")
        return
    base = base.merge(
        managers[mgr_cols],
        on="team_abbr",
        how="left",
        validate="one_to_one",
    )

    delta_fields = ["delta_wins", "delta_run_diff", "delta_win_pct", "games_this_week"]
    if change is not None and "team_abbr" in change.columns:
        change_sel_cols = ["team_abbr"] + [c for c in delta_fields if c in change.columns]
        change_sel = change[change_sel_cols].copy()
        base = base.merge(
            change_sel,
            on="team_abbr",
            how="left",
            validate="one_to_one",
        )
        for col in delta_fields:
            if col not in base.columns:
                base[col] = 0
    else:
        for col in delta_fields:
            if col not in base.columns:
                base[col] = 0

    def assign_role(abbr):
        if abbr == away_abbr:
            return "away"
        if abbr == home_abbr:
            return "home"
        return ""

    base["role"] = base["team_abbr"].apply(assign_role)
    base = base.sort_values(
        by="role",
        key=lambda s: s.map({"away": 0, "home": 1}),
    ).reset_index(drop=True)

    ordered_cols = [
        "role",
        "team_abbr",
        "team_name",
        "sub_league",
        "division",
        "games",
        "wins",
        "losses",
        "win_pct",
        "run_diff",
        "season_rs",
        "season_ra",
        "season_run_diff",
        "pythag_win_pct",
        "pythag_expected_wins",
        "pythag_diff",
        "manager_name",
        "manager_career_wins",
        "manager_career_losses",
        "manager_career_win_pct",
        "manager_total_seasons",
        "delta_wins",
        "delta_run_diff",
        "delta_win_pct",
        "games_this_week",
    ]
    ordered_cols = [col for col in ordered_cols if col in base.columns]
    prep = base[ordered_cols].copy()

    out_path = STAR_DIR / f"prep_1981_{away_abbr}_at_{home_abbr}.csv"
    if not dry_run:
        prep.to_csv(out_path, index=False)
        print("PREP_CARD_1981: wrote", out_path)

    print(prep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--away", required=True, help="Away team abbreviation (e.g., CHI)")
    parser.add_argument("--home", required=True, help="Home team abbreviation (e.g., MIA)")
    parser.add_argument("--dry-run", action="store_true", help="Run without writing output CSV")
    args = parser.parse_args()
    main(away_abbr=args.away, home_abbr=args.home, dry_run=args.dry_run)
