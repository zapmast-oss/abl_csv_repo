from pathlib import Path
import argparse
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
CSV_ROOT = SCRIPT_PATH.parents[1]
STAR_DIR = CSV_ROOT / "out" / "star_schema"


def main(dry_run: bool = False):
    standings_path = STAR_DIR / "fact_team_standings.csv"
    batting_path = STAR_DIR / "fact_team_batting.csv"
    pitching_path = STAR_DIR / "fact_team_pitching.csv"
    team_dim_path = STAR_DIR / "dim_team_park.csv"

    if not standings_path.exists():
        print(f"ERROR: fact_team_standings.csv not found at: {standings_path}")
        return

    df = pd.read_csv(standings_path)
    print("fact_team_standings columns:", list(df.columns))

    if not batting_path.exists() or not pitching_path.exists():
        print("ERROR: Required batting or pitching files missing for RS/RA computation.")
        print(f"Battings exists: {batting_path.exists()}, pitching exists: {pitching_path.exists()}")
        return

    batting = pd.read_csv(batting_path)
    print("fact_team_batting columns:", list(batting.columns))
    pitching = pd.read_csv(pitching_path)
    print("fact_team_pitching columns:", list(pitching.columns))

    if not team_dim_path.exists():
        print(f"ERROR: dim_team_park.csv not found at: {team_dim_path}")
        return
    teams = pd.read_csv(team_dim_path)
    print("dim_team_park columns:", list(teams.columns))

    league_col_candidates = [c for c in df.columns if c.lower() in {"league_id", "league"}]
    season_col_candidates = [c for c in df.columns if c.lower() in {"season", "year"}]

    if league_col_candidates:
        league_col = league_col_candidates[0]
        df = df[df[league_col] == 200].copy()
    else:
        df = df.copy()
        df["league_id"] = 200
        league_col = "league_id"
        print("WARNING: No explicit league_id column. Assuming entire file is league 200.")

    if season_col_candidates:
        season_col = season_col_candidates[0]
        df = df[df[season_col] == 1981].copy()
    else:
        df["season"] = 1981
        season_col = "season"
        print("WARNING: No explicit season column. Assuming data reflects season 1981.")

    if df.empty:
        print("ERROR: No rows found for league_id=200, season=1981 in fact_team_standings.")
        return

    df = df.rename(
        columns={
            "ID": "team_id",
            "Abbr": "team_abbr",
            "W": "wins",
            "L": "losses",
        }
    )
    required_columns = {"team_id", "team_abbr", "wins", "losses"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        print(f"ERROR: Missing required columns in standings: {missing_columns}")
        return

    df["team_id"] = df["team_id"].astype(int)

    expected_ids = set(range(1, 25))
    ids = set(df["team_id"])
    if ids != expected_ids:
        print("ERROR: 1981 standings do not cover the 24-team canon.")
        print("Found IDs:", sorted(ids))
        return

    teams = teams.rename(columns={"ID": "team_id", "Team Name": "team_name"})
    if "team_name" not in teams.columns:
        print("ERROR: dim_team_park is missing 'Team Name' column.")
        return
    df = df.merge(
        teams[["team_id", "team_name"]],
        on="team_id",
        how="left",
        validate="one_to_one",
    )

    batting = batting.rename(columns={"ID": "team_id", "R": "runs_scored"})
    if "runs_scored" not in batting.columns:
        print("ERROR: fact_team_batting missing runs scored column.")
        return
    df = df.merge(
        batting[["team_id", "runs_scored"]],
        on="team_id",
        how="left",
        validate="one_to_one",
    )

    pitching = pitching.rename(columns={"ID": "team_id"})
    if "RA" not in pitching.columns:
        print("ERROR: fact_team_pitching missing 'RA' column.")
        return
    df = df.merge(
        pitching[["team_id", "RA"]],
        on="team_id",
        how="left",
        validate="one_to_one",
    )
    df = df.rename(columns={"RA": "runs_allowed"})

    if df["runs_scored"].isna().any() or df["runs_allowed"].isna().any():
        print("ERROR: Missing runs data for some teams.")
        return

    df["games_played"] = df["wins"] + df["losses"]

    rs_sq = df["runs_scored"] ** 2
    ra_sq = df["runs_allowed"] ** 2
    denom = rs_sq + ra_sq

    df["pythag_win_pct"] = 0.0
    mask = denom > 0
    df.loc[mask, "pythag_win_pct"] = rs_sq[mask] / denom[mask]

    df["pythag_expected_wins"] = df["pythag_win_pct"] * df["games_played"]
    df["pythag_expected_losses"] = df["games_played"] - df["pythag_expected_wins"]
    df["run_diff"] = df["runs_scored"] - df["runs_allowed"]
    df["pythag_diff"] = df["wins"] - df["pythag_expected_wins"]

    backbone_cols = [
        "team_id",
        "team_abbr",
        "team_name",
        season_col,
        league_col,
        "wins",
        "losses",
        "games_played",
        "runs_scored",
        "runs_allowed",
        "run_diff",
        "pythag_win_pct",
        "pythag_expected_wins",
        "pythag_expected_losses",
        "pythag_diff",
    ]
    backbone_cols = [c for c in backbone_cols if c in df.columns]
    backbone = df[backbone_cols].copy()

    if not dry_run:
        out_backbone = STAR_DIR / "fact_team_season_1981_backbone.csv"
        backbone.to_csv(out_backbone, index=False)
        print("TEAM_SEASON_1981_BACKBONE: wrote", out_backbone)

    report = backbone.sort_values(
        by=["pythag_diff", "run_diff"],
        ascending=[False, False],
    ).reset_index(drop=True)

    if not dry_run:
        out_report = STAR_DIR / "fact_team_season_1981_pythag_report.csv"
        report.to_csv(out_report, index=False)
        print("TEAM_SEASON_1981_PYTHAG_REPORT: wrote", out_report)

    print(report.head(10))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run without writing outputs")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
