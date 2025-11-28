from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


REQUIRED_STANDINGS_COLS = ["team_name", "wins", "losses", "pct"]
OPTIONAL_COLS_MAP = {
    "gb": "gb",
    "pyt_rec": "pyt_rec",
    "pyt_diff": "pyt_diff",
    "home_rec": "home_rec",
    "away_rec": "away_rec",
    "xinn_rec": "xinn_rec",
    "one_run_rec": "one_run_rec",
    "magic_num": "magic_num",
    "streak": "streak",
    "last10": "last10",
}


def load_standings(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"Error: standings file not found at {path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {len(df)} rows from standings_enriched: {path}")
    missing = [c for c in REQUIRED_STANDINGS_COLS if c not in df.columns]
    if missing:
        print(f"Error: standings missing required columns: {missing}", file=sys.stderr)
        print(f"Columns present: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)
    return df


def build_summary(df: pd.DataFrame, season: int, league_id: int) -> pd.DataFrame:
    df = df.copy()

    # Drop non-team rows (e.g., conference summary lines) if present
    if "team_id" in df.columns:
        df = df[df["team_id"].notna()].copy()
    else:
        banned = {"American Baseball Conference", "National Baseball Conference"}
        df = df[~df["team_name"].astype(str).isin(banned)].copy()

    # Ensure core columns
    for col in ["season", "league_id"]:
        if col not in df.columns:
            df[col] = season if col == "season" else league_id

    # Map optional columns if present
    extras = {dest: src for src, dest in OPTIONAL_COLS_MAP.items() if src in df.columns}

    base_cols = [
        "season",
        "league_id",
        "team_id",
        "team_abbr",
        "team_name",
        "conf",
        "division",
        "wins",
        "losses",
        "pct",
    ]

    present_base = [c for c in base_cols if c in df.columns]
    missing_base = [c for c in base_cols if c not in df.columns]
    if missing_base:
        print(f"[WARN] Missing expected columns (will fill with NA): {missing_base}")
        for col in missing_base:
            df[col] = pd.NA
        present_base = base_cols

    cols_out = present_base + [extras[k] for k in extras]
    summary = df[cols_out].copy()

    # Compute run_diff if possible
    if "run_diff" in df.columns:
        summary["run_diff"] = df["run_diff"]
    elif {"runs_for", "runs_against"}.issubset(df.columns):
        summary["run_diff"] = df["runs_for"] - df["runs_against"]
    else:
        summary["run_diff"] = pd.NA

    # Ordering
    sort_keys = []
    if "conf" in summary.columns:
        sort_keys.append("conf")
    if "division" in summary.columns:
        sort_keys.append("division")
    sort_keys.extend(["wins", "pct"])
    summary = summary.sort_values(sort_keys, ascending=[True, True, False, False][: len(sort_keys)])

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build league-season summary from standings_enriched.")
    parser.add_argument("--season", type=int, required=True, help="Season year (e.g., 1972)")
    parser.add_argument("--league-id", type=int, required=True, help="League ID (ABL=200)")
    parser.add_argument(
        "--almanac-root",
        default=Path("csv/out/almanac"),
        type=Path,
        help="Root of almanac outputs (default: csv/out/almanac)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    season_dir = args.almanac_root / str(args.season)
    standings_path = season_dir / f"standings_{args.season}_league{args.league_id}_enriched.csv"
    out_path = season_dir / f"league_season_summary_{args.season}_league{args.league_id}.csv"

    print(f"[DEBUG] season={args.season}, league_id={args.league_id}")
    print(f"[DEBUG] standings_path={standings_path}")

    df = load_standings(standings_path)
    summary = build_summary(df, args.season, args.league_id)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(f"[INFO] Wrote league season summary to {out_path} ({len(summary)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
