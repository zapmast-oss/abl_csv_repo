import argparse
from pathlib import Path

import pandas as pd


def load_standings(standings_path: Path) -> pd.DataFrame:
    print(f"[INFO] Loading standings from {standings_path}")
    if not standings_path.exists():
        raise FileNotFoundError(f"Standings file not found: {standings_path}")

    df = pd.read_csv(standings_path)

    if "team_name" not in df.columns:
        # Try to recover if the column is still called "Team"
        if "Team" in df.columns:
            df = df.rename(columns={"Team": "team_name"})
        else:
            raise RuntimeError(
                "Standings file must contain 'team_name' (or 'Team') column."
            )

    # Known name fixes between almanac and dim_team_park
    name_fixes = {
        "Seatlle Comets": "Seattle Comets",  # typo fix
    }
    df["team_name"] = df["team_name"].replace(name_fixes)

    required_cols = ["season", "league_id", "team_name"]
    for col in required_cols:
        if col not in df.columns:
            raise RuntimeError(
                f"Standings file is missing required column '{col}'. "
                f"Found columns: {list(df.columns)}"
            )

    return df


def load_dim_team_park(dim_path: Path) -> pd.DataFrame:
    print(f"[INFO] Loading dim_team_park from {dim_path}")
    if not dim_path.exists():
        raise FileNotFoundError(f"dim_team_park file not found: {dim_path}")

    dim = pd.read_csv(dim_path)

    needed = ["ID", "Team Name", "Abbr", "SL", "DIV"]
    for col in needed:
        if col not in dim.columns:
            raise RuntimeError(
                f"dim_team_park is missing required column '{col}'. "
                f"Found columns: {list(dim.columns)}"
            )

    dim_small = dim[needed].rename(
        columns={
            "ID": "team_id",
            "Team Name": "team_name",
            "Abbr": "team_abbr",
            "SL": "conf",
            "DIV": "division",
        }
    )

    return dim_small


def enrich_standings_with_dim(
    standings: pd.DataFrame, dim_team: pd.DataFrame
) -> pd.DataFrame:
    print("[INFO] Joining standings with dim_team_park on 'team_name'")
    merged = standings.merge(dim_team, on="team_name", how="left", validate="one_to_one")

    missing = merged[merged["team_id"].isna()]
    if not missing.empty:
        print("[WARN] The following team_name values did not match any dim_team_park row:")
        for name in sorted(missing["team_name"].unique()):
            print(f"       - {name}")
    else:
        print("[INFO] All teams matched dim_team_park successfully.")

    # Sort for sanity
    sort_cols = [c for c in ["season", "league_id", "team_id", "team_name"] if c in merged.columns]
    merged = merged.sort_values(sort_cols).reset_index(drop=True)

    return merged


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Enrich almanac standings with team_id, conference, and division from dim_team_park."
    )
    parser.add_argument(
        "--season",
        type=int,
        default=1972,
        help="Season year to process (default: 1972).",
    )
    parser.add_argument(
        "--league-id",
        type=int,
        default=200,
        help="League ID (ABL = 200). Default: 200.",
    )
    parser.add_argument(
        "--standings-root",
        default="csv/out/almanac",
        help="Root folder where standings_{season}_league{league_id}.csv lives (default: csv/out/almanac).",
    )
    parser.add_argument(
        "--dim-team-park",
        default="csv/out/star_schema/dim_team_park.csv",
        help="Path to dim_team_park.csv (default: csv/out/star_schema/dim_team_park.csv).",
    )
    parser.add_argument(
        "--out-root",
        default="csv/out/almanac",
        help="Root folder for enriched output (default: csv/out/almanac).",
    )

    args = parser.parse_args(argv)

    standings_path = (
        Path(args.standings_root)
        / str(args.season)
        / f"standings_{args.season}_league{args.league_id}.csv"
    )
    dim_path = Path(args.dim_team_park)

    out_dir = Path(args.out_root) / str(args.season)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"standings_{args.season}_league{args.league_id}_enriched.csv"

    print(f"[DEBUG] season={args.season}, league_id={args.league_id}")
    print(f"[DEBUG] standings_path={standings_path}")
    print(f"[DEBUG] dim_team_park={dim_path}")
    print(f"[DEBUG] out_path={out_path}")

    standings = load_standings(standings_path)
    dim_team = load_dim_team_park(dim_path)
    enriched = enrich_standings_with_dim(standings, dim_team)

    enriched.to_csv(out_path, index=False)
    print(f"[OK] Wrote enriched standings to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
