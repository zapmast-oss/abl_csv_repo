import argparse
import re
import zipfile
from pathlib import Path

import pandas as pd


def parse_standings_from_html_bytes(html_bytes: bytes, season: int, league_id: int) -> pd.DataFrame:
    """
    Parse league standings from an OOTP almanac standings HTML page.
    Returns one row per team for the given season/league.
    """
    print(f"[DEBUG] Parsing standings for season={season}, league_id={league_id}")
    html_text = html_bytes.decode("utf-8", errors="ignore")

    # Parse all tables on the page
    tables = pd.read_html(html_text)

    standings_tables = []
    for df in tables:
        cols = [str(c) for c in df.columns]
        if {"Team", "W", "L", "PCT"}.issubset(set(cols)):
            standings_tables.append(df)

    if not standings_tables:
        raise RuntimeError(
            f"No standings tables with Team/W/L/PCT found for season={season}, league_id={league_id}"
        )

    standings = pd.concat(standings_tables, ignore_index=True)

    # Keep only rows that look like actual teams
    if "Team" not in standings.columns:
        raise RuntimeError("Parsed standings do not contain a 'Team' column.")

    standings = standings[standings["Team"].notna()]
    standings = standings[standings["Team"].astype(str).str.strip().ne("")]
    standings = standings[
        ~standings["Team"].astype(str).str.contains("total", case=False, na=False)
    ]

    # Add season and league_id
    standings["season"] = int(season)
    standings["league_id"] = int(league_id)

    # Rename columns to snake_case where possible
    rename_map = {
        "Team": "team_name",
        "W": "wins",
        "L": "losses",
        "PCT": "pct",
        "GB": "gb",
        "Pyt.Rec": "pyt_rec",
        "Diff": "pyt_diff",
        "Home": "home_rec",
        "Away": "away_rec",
        "XInn": "xinn_rec",
        "1Run": "one_run_rec",
        "M#": "magic_num",
        "Streak": "streak",
        "Last10": "last10",
    }
    existing_rename_map = {k: v for k, v in rename_map.items() if k in standings.columns}
    standings = standings.rename(columns=existing_rename_map)

    # Deduplicate: some teams appear twice (division + wildcard).
    # Rule: if any row has magic_num == 'Clinched', keep that one; otherwise keep the first.
    if "team_name" not in standings.columns:
        raise RuntimeError("Expected 'team_name' after rename, but it was not found.")

    def pick_group(group: pd.DataFrame) -> pd.Series:
        if "magic_num" in group.columns:
            clinched_mask = group["magic_num"].astype(str).str.contains("Clinched", na=False)
            if clinched_mask.any():
                return group[clinched_mask].iloc[0]
        return group.iloc[0]

    standings = standings.groupby("team_name", as_index=False).apply(pick_group)
    standings = standings.reset_index(drop=True)

    # Ensure minimal required columns
    required_cols = ["season", "league_id", "team_name", "wins", "losses", "pct"]
    for col in required_cols:
        if col not in standings.columns:
            raise RuntimeError(f"Required column '{col}' missing after parsing/renaming.")

    standings = standings.sort_values(["season", "league_id", "team_name"]).reset_index(drop=True)
    return standings


def process_almanac_zip(zip_path: Path, league_id: int, out_root: Path) -> Path:
    """
    Process a single almanac_YYYY.zip file and write a standings CSV for league_id.
    Returns the output path.
    """
    match = re.search(r"almanac_(\d{4})\.zip$", zip_path.name, re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not infer season year from file name '{zip_path.name}'")

    season = int(match.group(1))
    internal_html = f"almanac_{season}/leagues/league_{league_id}_standings.html"
    print(f"[INFO] Processing {zip_path.name} (season={season})")
    print(f"[DEBUG] Looking for '{internal_html}' inside zip")

    with zipfile.ZipFile(zip_path, "r") as zf:
        try:
            html_bytes = zf.read(internal_html)
        except KeyError as exc:
            raise FileNotFoundError(
                f"Could not find '{internal_html}' inside '{zip_path}'"
            ) from exc

    df = parse_standings_from_html_bytes(html_bytes, season, league_id)

    out_dir = out_root / str(season)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"standings_{season}_league{league_id}.csv"
    df.to_csv(out_path, index=False)
    print(f"[OK] Wrote standings to {out_path}")
    return out_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Batch-parse ABL standings from OOTP almanac_YYYY.zip files."
    )
    parser.add_argument(
        "--almanac-dir",
        default=".",
        help="Directory containing almanac_YYYY.zip files (default: current directory).",
    )
    parser.add_argument(
        "--pattern",
        default="almanac_*.zip",
        help="Glob pattern to match almanac zip files (default: almanac_*.zip).",
    )
    parser.add_argument(
        "--league-id",
        type=int,
        default=200,
        help="League ID to parse (ABL = 200). Default: 200.",
    )
    parser.add_argument(
        "--out-root",
        default="csv/out/almanac",
        help="Root output folder for parsed standings (default: csv/out/almanac).",
    )

    args = parser.parse_args(argv)

    almanac_dir = Path(args.almanac_dir)
    out_root = Path(args.out_root)

    matches = sorted(almanac_dir.glob(args.pattern))
    if not matches:
        print(f"[WARN] No files matching {args.pattern} in {almanac_dir}")
        return 1

    print(f"[INFO] Found {len(matches)} almanac zip(s) in {almanac_dir}")

    for zip_path in matches:
        try:
            process_almanac_zip(zip_path, args.league_id, out_root)
        except Exception as exc:
            print(f"[ERROR] Failed for {zip_path.name}: {exc}")

    return 0


if __name__ == "__main__":
    print("[DEBUG] z_abl_almanac_standings_pipeline.py starting up")
    raise SystemExit(main())
