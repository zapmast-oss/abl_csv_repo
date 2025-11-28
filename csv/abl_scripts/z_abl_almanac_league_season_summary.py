from __future__ import annotations

import argparse
import sys
import re
import zipfile
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


def normalize_name(val: str) -> str:
    """Lowercase alnum key for joining."""
    return re.sub(r"[^a-z0-9]", "", str(val).lower())


def assert_no_nan(df: pd.DataFrame, cols: list[str], context: str) -> None:
    bad = df[cols].isna().any()
    if bad.any():
        missing = [c for c in cols if bad[c]]
        raise ValueError(f"{context}: NaN in columns {missing}")


def load_team_runs_from_games(season_dir: Path, season: int, league_id: int, dim_path: Path) -> pd.DataFrame:
    games_path = season_dir / f"games_{season}_league{league_id}.csv"
    if not games_path.exists():
        raise FileNotFoundError(f"games file not found at {games_path}")
    dim = pd.read_csv(dim_path)
    dim["team_name_norm"] = dim["Team Name"].apply(normalize_name)
    dim["city_norm"] = dim["City"].str.split("(").str[0].apply(normalize_name)
    dim["abbr_norm"] = dim["Abbr"].apply(normalize_name)

    override = {
        "seatlle": "seattle",
        "tampabay": "tampa",
    }

    key_map = {}
    for _, row in dim.iterrows():
        for raw in {row.team_name_norm, row.city_norm, row.abbr_norm}:
            key = override.get(raw, raw)
            key_map[key] = {
                "team_id": row["ID"],
                "team_abbr": row["Abbr"],
                "team_name": row["Team Name"],
                "conf": row.get("SL"),
                "division": row.get("DIV"),
            }

    games = pd.read_csv(games_path)
    home = games.rename(
        columns={
            "home_team_name": "team_name",
            "home_runs": "runs_for",
            "away_runs": "runs_against",
        }
    )[["team_name", "runs_for", "runs_against"]]
    away = games.rename(
        columns={
            "away_team_name": "team_name",
            "away_runs": "runs_for",
            "home_runs": "runs_against",
        }
    )[["team_name", "runs_for", "runs_against"]]
    tall = pd.concat([home, away], ignore_index=True)
    tall["key"] = tall["team_name"].apply(normalize_name).apply(lambda k: override.get(k, k))

    agg = tall.groupby("key")[["runs_for", "runs_against"]].sum().reset_index()

    records = []
    for _, row in agg.iterrows():
        key = row["key"]
        if key not in key_map:
            # Ignore non-team aggregates (e.g., conference rows if any)
            continue
        meta = key_map[key]
        records.append(
            {
                "season": season,
                "league_id": league_id,
                "team_id": meta["team_id"],
                "team_abbr": meta["team_abbr"],
                "team_name": meta["team_name"],
                "conf": meta.get("conf"),
                "division": meta.get("division"),
                "runs_for": row["runs_for"],
                "runs_against": row["runs_against"],
            }
        )

    stats = pd.DataFrame.from_records(records)
    stats["runs_for"] = pd.to_numeric(stats["runs_for"], errors="raise")
    stats["runs_against"] = pd.to_numeric(stats["runs_against"], errors="raise")
    stats["run_diff"] = stats["runs_for"] - stats["runs_against"]
    assert_no_nan(stats, ["runs_for", "runs_against", "run_diff"], "team season runs")
    return stats


def load_team_runs_from_stats_html(season: int, league_id: int, dim_path: Path) -> pd.DataFrame:
    """Read runs for/against from league stats HTML inside the almanac zip."""
    zip_path = Path("data_raw/ootp_html") / f"almanac_{season}.zip"
    internal = f"almanac_{season}/leagues/league_{league_id}_stats.html"
    if not zip_path.exists():
        raise FileNotFoundError(f"Almanac zip not found at {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        try:
            html_bytes = zf.read(internal)
        except KeyError as exc:
            raise FileNotFoundError(f"Stats HTML {internal} not found in {zip_path}") from exc

    tables = pd.read_html(html_bytes)
    bat_tables = []
    pit_tables = []
    for tbl in tables:
        cols_norm = [normalize_name(c) for c in tbl.columns]
        if "team" in cols_norm and "r" in cols_norm:
            if any(c in cols_norm for c in ["avg", "obp", "slg", "ab"]):
                bat_tables.append(tbl)
            elif any(c in cols_norm for c in ["era", "whip", "ha", "ip", "oavg", "h"]):
                pit_tables.append(tbl)
    if not bat_tables or not pit_tables:
        raise ValueError("Could not locate batting/pitching tables with Team/R columns in stats HTML.")

    def collect_runs(tables_list: list[pd.DataFrame]) -> pd.DataFrame:
        frames = []
        for df in tables_list:
            df = df.rename(columns={df.columns[0]: "team"})
            r_col = None
            for c in df.columns:
                if normalize_name(c) == "r":
                    r_col = c
                    break
            if r_col is None:
                continue
            df = df[["team", r_col]].copy()
            df = df[
                ~df["team"].astype(str).str.contains("Totals", case=False, na=False)
                & ~df["team"].astype(str).str.contains("Average", case=False, na=False)
            ].copy()
            df["key"] = df["team"].apply(normalize_name).apply(lambda k: override.get(k, k))
            df = df.rename(columns={r_col: "runs"})
            frames.append(df)
        if not frames:
            raise ValueError("No usable runs tables found in stats HTML.")
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(subset="key", keep="first")
        return combined

    # Build dim map
    dim = pd.read_csv(dim_path)
    dim["team_name_norm"] = dim["Team Name"].apply(normalize_name)
    dim["city_norm"] = dim["City"].str.split("(").str[0].apply(normalize_name)
    dim["abbr_norm"] = dim["Abbr"].apply(normalize_name)
    override = {"seatlle": "seattle", "tampabay": "tampa"}
    key_map = {}
    for _, row in dim.iterrows():
        for raw in {row.team_name_norm, row.city_norm, row.abbr_norm}:
            key = override.get(raw, raw)
            key_map[key] = {
                "team_id": row["ID"],
                "team_abbr": row["Abbr"],
                "team_name": row["Team Name"],
                "conf": row.get("SL"),
                "division": row.get("DIV"),
            }

    # Extract runs
    bat_runs = collect_runs(bat_tables)
    pit_runs = collect_runs(pit_tables)
    bat_runs["runs"] = pd.to_numeric(bat_runs["runs"], errors="raise")
    pit_runs["runs"] = pd.to_numeric(pit_runs["runs"], errors="raise")
    merged = bat_runs.merge(pit_runs, on="key", suffixes=("_for", "_against"), validate="1:1")

    records = []
    for _, row in merged.iterrows():
        key = row["key"]
        if key not in key_map:
            continue
        meta = key_map[key]
        records.append(
            {
                "season": season,
                "league_id": league_id,
                "team_id": meta["team_id"],
                "team_abbr": meta["team_abbr"],
                "team_name": meta["team_name"],
                "conf": meta.get("conf"),
                "division": meta.get("division"),
                "runs_for": row["runs_for"],
                "runs_against": row["runs_against"],
            }
        )
    stats = pd.DataFrame.from_records(records)
    stats["run_diff"] = stats["runs_for"] - stats["runs_against"]
    assert_no_nan(stats, ["runs_for", "runs_against", "run_diff"], "team season runs (stats html)")
    return stats


def build_summary(df: pd.DataFrame, season: int, league_id: int, season_dir: Path) -> pd.DataFrame:
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

    # Replace run_diff using canonical season runs from games
    dim_path = Path("csv/out/star_schema/dim_team_park.csv")
    try:
        stats = load_team_runs_from_stats_html(season, league_id, dim_path)
    except Exception as exc:
        print(f"[WARN] Falling back to games aggregation for runs due to: {exc}")
        stats = load_team_runs_from_games(season_dir, season, league_id, dim_path)

    summary["join_key"] = summary["team_id"]
    stats["join_key"] = stats["team_id"]
    merged = summary.merge(
        stats[
            [
                "join_key",
                "runs_for",
                "runs_against",
                "run_diff",
            ]
        ],
        on="join_key",
        how="left",
        validate="1:1",
    )
    merged = merged.drop(columns=["join_key"])
    assert_no_nan(merged, ["runs_for", "runs_against", "run_diff"], "league_season_summary")
    summary = merged

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
    summary = build_summary(df, args.season, args.league_id, season_dir)

    # Debug trace for ATL in 1972/200
    if args.season == 1972 and args.league_id == 200:
        atl_row = summary[summary["team_abbr"] == "ATL"]
        print("[DEBUG] ATL row after recompute:")
        print(atl_row[["team_name", "team_abbr", "runs_for", "runs_against", "run_diff", "wins", "losses"]])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(f"[INFO] Wrote league season summary to {out_path} ({len(summary)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
