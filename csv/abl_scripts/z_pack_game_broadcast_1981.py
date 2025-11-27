"""
z_pack_game_broadcast_1981.py

Build a single-row broadcast pack for the May 11, 1981 game:
    Chicago Fire (CHI) at Miami Hurricanes (MIA)

Input (must already exist, produced by your star-schema pipeline):
    csv/out/star_schema/fact_team_reporting_1981_current.csv

Output:
    csv/out/csv_out/season_1981/eb_game_pack_1981-05-11_CHI_at_MIA.csv

Design:
    - We trust fact_team_reporting_1981_current.csv as the canonical
      "team snapshot" table: standings, run diff, splits, etc.
    - We DO NOT invent any numbers. We only copy what is already there.
    - We DO NOT depend on face_of_franchise, team_aces, or any other
      star-schema tables for this first broadcast pack.
    - We fail loudly if the required file or teams are missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Constants: adjust here if you ever need a different game/date
# ---------------------------------------------------------------------------

GAME_DATE = "1981-05-11"
AWAY_ABBR = "CHI"
HOME_ABBR = "MIA"

# Output file name pattern (keep consistent going forward)
OUTPUT_FILE_NAME = f"eb_game_pack_{GAME_DATE}_CHI_at_MIA.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def repo_root_from_this_file() -> Path:
    """
    Resolve the repo root assuming this file lives at:
        <root>/csv/abl_scripts/z_pack_game_broadcast_1981.py
    """
    this_file = Path(__file__).resolve()
    # .../csv/abl_scripts/z_pack_game_broadcast_1981.py
    # parents[0] = abl_scripts
    # parents[1] = csv
    # parents[2] = repo root (abl_csv_repo)
    return this_file.parents[2]


def load_fact_team_reporting_current(root: Path) -> pd.DataFrame:
    """
    Load the canonical per-team snapshot table for 1981.

    Required file:
        csv/out/star_schema/fact_team_reporting_1981_current.csv

    Hard requirement:
        - must contain a 'team_abbr' column, used for CHI/MIA lookup.

    We do NOT enforce any other column names here. Whatever other columns
    are present will simply be mirrored into the broadcast pack with
    away_/home_ prefixes.
    """
    path = root / "csv" / "out" / "star_schema" / "fact_team_reporting_1981_current.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Make sure your star-schema pipeline has produced "
            "fact_team_reporting_1981_current.csv before running "
            "z_pack_game_broadcast_1981.py."
        )

    df = pd.read_csv(path)

    if "team_abbr" not in df.columns:
        raise ValueError(
            f"'team_abbr' column not found in {path}\n"
            f"Columns present: {list(df.columns)}\n"
            "This script expects a team abbreviation column named 'team_abbr'."
        )

    return df


def fetch_team_row(df: pd.DataFrame, team_abbr: str) -> pd.Series:
    """
    Pull exactly one row for the given team_abbr from the provided
    fact_team_reporting_1981_current DataFrame.

    We fail if there is zero or more than one matching row.
    """
    mask = df["team_abbr"] == team_abbr
    subset = df[mask]

    if subset.empty:
        raise ValueError(
            f"No row found for team_abbr='{team_abbr}' in fact_team_reporting_1981_current.\n"
            f"Available team_abbr values: {sorted(df['team_abbr'].unique())}"
        )

    if len(subset) > 1:
        raise ValueError(
            f"Multiple rows found for team_abbr='{team_abbr}' in "
            "fact_team_reporting_1981_current. Expected exactly one."
        )

    return subset.iloc[0]


def safe_column_name(col: str) -> str:
    """
    Normalize a source column name into a safe suffix:
    - lowercased
    - spaces replaced with underscores

    We don't strip other characters; we want a reversible mapping.
    """
    return col.strip().lower().replace(" ", "_")


# ---------------------------------------------------------------------------
# Core packing logic
# ---------------------------------------------------------------------------

def build_game_pack_row(df_reporting: pd.DataFrame) -> pd.DataFrame:
    """
    Build a one-row DataFrame representing the broadcast pack for the
    CHI at MIA game on GAME_DATE.

    Structure:
        - game_date
        - matchup         (e.g., "CHI at MIA")
        - away_abbr
        - home_abbr
        - away_<col>      for every non-'team_abbr' column in df_reporting
        - home_<col>      same as above
    """
    away_row = fetch_team_row(df_reporting, AWAY_ABBR)
    home_row = fetch_team_row(df_reporting, HOME_ABBR)

    out_dict = {
        "game_date": GAME_DATE,
        "matchup": f"{AWAY_ABBR} at {HOME_ABBR}",
        "away_abbr": AWAY_ABBR,
        "home_abbr": HOME_ABBR,
    }

    for col in df_reporting.columns:
        if col == "team_abbr":
            # Already broken out as away_abbr / home_abbr
            continue

        suffix = safe_column_name(col)
        away_key = f"away_{suffix}"
        home_key = f"home_{suffix}"

        out_dict[away_key] = away_row[col]
        out_dict[home_key] = home_row[col]

    return pd.DataFrame([out_dict])


def main() -> None:
    """
    Entry point:
        - Resolve repo root.
        - Load fact_team_reporting_1981_current.
        - Build the broadcast pack row.
        - Write CSV to csv/out/csv_out/season_1981/.
    """
    root = repo_root_from_this_file()

    # 1) Load canonical per-team snapshot
    df_reporting = load_fact_team_reporting_current(root)

    # 2) Build single-row game pack
    df_pack = build_game_pack_row(df_reporting)

    # 3) Output path
    out_dir = root / "csv" / "out" / "csv_out" / "season_1981"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / OUTPUT_FILE_NAME
    df_pack.to_csv(out_path, index=False)

    print(f"[z_pack_game_broadcast_1981] Wrote broadcast pack to: {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # fail loudly, but with a clear message
        print("\n[z_pack_game_broadcast_1981] ERROR:", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        sys.exit(1)
