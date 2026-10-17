#!/usr/bin/env python
"""Build an EB All-Star data brief for any season/league."""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd


def log(msg: str) -> None:
    print(msg, flush=True)


def load_config(config_path: Path, season: int) -> pd.Series:
    if not config_path.exists():
        log(f"[ERROR] All-Star config not found: {config_path}")
        sys.exit(1)
    cfg = pd.read_csv(config_path)
    row = cfg[cfg["season"] == season]
    if row.empty:
        log(f"[ERROR] No All-Star config row for season {season} in {config_path}")
        sys.exit(1)
    row = row.iloc[0]
    log(f"[INFO] Loaded All-Star config for season {season}: host={row['host_team']}, winning_conf={row.get('winning_conf')}")
    return row


def infer_host_conf(season: int) -> str:
    return "ABC" if (season - 1972) % 2 == 0 else "NBC"


def find_date_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if "date" in str(col).lower():
            return col
    return ""


def detect_all_star_break(games_path: Path, season: int) -> tuple[list[date], list[date]]:
    if not games_path.exists():
        log(f"[ERROR] Games log not found: {games_path}")
        sys.exit(1)
    df = pd.read_csv(games_path)
    date_col = find_date_column(df)
    if not date_col:
        log(f"[ERROR] No date column found in games file {games_path}")
        sys.exit(1)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df_dates = df[date_col].dt.date
    if df_dates.empty:
        return [], []
    min_date = df_dates.min()
    max_date = df_dates.max()
    full_range = pd.date_range(min_date, max_date, freq="D").date
    game_dates = set(df_dates.unique())
    idle_dates = sorted(d for d in full_range if d not in game_dates)
    july_idle = [d for d in idle_dates if d.month == 7 and d.year == season]
    runs: list[list[date]] = []
    current: list[date] = []
    for d in july_idle:
        if not current:
            current = [d]
        elif (d - current[-1]).days == 1:
            current.append(d)
        else:
            runs.append(current)
            current = [d]
    if current:
        runs.append(current)
    if not runs:
        log("[WARN] No July idle window detected.")
        return [], []
    runs = sorted(runs, key=len, reverse=True)
    best = runs[0]
    log(f"[INFO] Longest July idle run: {best[0]} to {best[-1]} (len={len(best)})")
    return best, july_idle


def attach_host_abbr(host_team: str, dim_path: Path) -> str | None:
    if not dim_path.exists():
        log(f"[WARN] dim_team_park missing at {dim_path}; skipping host abbr lookup.")
        return None
    dim = pd.read_csv(dim_path)
    name_cols = [c for c in dim.columns if "name" in c.lower() or "city" in c.lower()]
    abbr_col = "team_abbr" if "team_abbr" in dim.columns else ("Abbr" if "Abbr" in dim.columns else None)
    if not name_cols or not abbr_col:
        log("[WARN] dim_team_park lacks expected name/abbr columns; skipping host abbr lookup.")
        return None
    target = host_team.lower()
    for col in name_cols:
        match = dim[dim[col].str.lower() == target] if pd.api.types.is_string_dtype(dim[col]) else pd.DataFrame()
        if not match.empty:
            abbr = match.iloc[0][abbr_col]
            log(f"[INFO] Host abbr resolved: {abbr} via column {col}")
            return abbr
    log(f"[WARN] Could not resolve host abbr for {host_team}")
    return None


def historical_ledger(config_path: Path, season: int) -> tuple[int, int]:
    cfg = pd.read_csv(config_path)
    subset = cfg[(cfg["season"] <= season) & (cfg["winning_conf"].notna()) & (cfg["winning_conf"] != "")]
    nbc_wins = (subset["winning_conf"].str.upper() == "NBC").sum()
    abc_wins = (subset["winning_conf"].str.upper() == "ABC").sum()
    log(f"[INFO] Historical ledger through {season}: NBC={nbc_wins}, ABC={abc_wins}")
    return nbc_wins, abc_wins


def main() -> int:
    parser = argparse.ArgumentParser(description="EB All-Star brief generator for any season/league.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--league-id", type=int, default=200)
    args = parser.parse_args()
    season = args.season
    league_id = args.league_id

    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "csv" / "config" / "abl_all_star_venues.csv"
    games_path = repo_root / "csv" / "out" / "almanac" / str(season) / f"games_{season}_league{league_id}.csv"
    dim_team_path = repo_root / "csv" / "out" / "star_schema" / "dim_team_park.csv"
    out_dir = repo_root / "csv" / "out" / "almanac" / str(season)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_md = out_dir / f"eb_all_star_{season}_league{league_id}.md"

    log(f"[INFO] Season={season}, league={league_id}")
    cfg_row = load_config(config_path, season)
    host_team = cfg_row["host_team"]
    winning_conf_val = cfg_row.get("winning_conf")
    winning_conf = winning_conf_val if isinstance(winning_conf_val, str) and winning_conf_val.strip() else "Unknown (future or not recorded)"
    host_conf = infer_host_conf(season)
    host_abbr = attach_host_abbr(host_team, dim_team_path)

    best_run, all_july_idle = detect_all_star_break(games_path, season)
    if best_run:
        hr_derby = best_run[0]
        asg = best_run[1] if len(best_run) > 1 else None
        travel = best_run[2] if len(best_run) > 2 else None
        idle_summary = f"{best_run[0]} to {best_run[-1]}"
        run_len = len(best_run)
    else:
        hr_derby = asg = travel = None
        idle_summary = "No July idle window detected"
        run_len = 0

    nbc_wins, abc_wins = historical_ledger(config_path, season)
    if nbc_wins + abc_wins > 0:
        if nbc_wins > abc_wins:
            narrative = f"Through {season}, the NBC has taken {nbc_wins} of the first {nbc_wins + abc_wins} All-Star Games."
        elif abc_wins > nbc_wins:
            narrative = f"Through {season}, the ABC leads the series {abc_wins}-{nbc_wins}."
        else:
            narrative = f"Through {season}, the All-Star series is level at {nbc_wins}-{abc_wins}."
    else:
        narrative = "No completed All-Star results recorded yet."

    md_lines: list[str] = []
    md_lines.append(f"# EB All-Star {season} – Data Brief (DO NOT PUBLISH)")
    md_lines.append(f"_League ID {league_id}_")
    md_lines.append("")
    md_lines.append("## Host & Venue")
    abbr_suffix = f" ({host_abbr})" if host_abbr else ""
    md_lines.append(f"- Season: {season}")
    md_lines.append(f"- Host club: {host_team}{abbr_suffix}")
    md_lines.append(f"- Host conference (by rotation): {host_conf}")
    md_lines.append(f"- All-Star Game winner (conference): {winning_conf}")
    md_lines.append(f"- Canon source: csv/config/abl_all_star_venues.csv")
    md_lines.append("")
    md_lines.append(f"## All-Star break window (detected from games_{season}_league{league_id}.csv)")
    md_lines.append(f"- Home Run Derby: {hr_derby if hr_derby else 'Unknown'}")
    md_lines.append(f"- All-Star Game: {asg if asg else 'Unknown'}")
    md_lines.append(f"- Travel/off day: {travel if travel else 'Unknown'}")
    md_lines.append(f"- July idle run used: {idle_summary} (length: {run_len})")
    md_lines.append("")
    md_lines.append(f"## Historical ledger (through {season})")
    md_lines.append(f"- NBC All-Star wins: {nbc_wins}")
    md_lines.append(f"- ABC All-Star wins: {abc_wins}")
    md_lines.append(f"- Narrative hook: {narrative}")
    md_lines.append("")

    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    log(f"[OK] Wrote All-Star brief to {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
