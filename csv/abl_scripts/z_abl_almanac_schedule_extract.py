#!/usr/bin/env python
"""Extract schedule grid/evaluator from almanac HTML into tidy CSVs."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd


def log(msg: str) -> None:
    print(msg)


def read_tables(path: Path) -> List[pd.DataFrame]:
    try:
        tables = pd.read_html(path, flavor="lxml")
    except ValueError:
        tables = []
    log(f"[DEBUG] {path.name}: found {len(tables)} tables")
    for i, t in enumerate(tables):
        log(f"[DEBUG] table {i} shape {t.shape} cols={t.columns.tolist()[:8]}")
    return tables


def pick_evaluator_table(tables: List[pd.DataFrame]) -> pd.DataFrame:
    candidates = [t for t in tables if t.shape[0] >= 8 and t.shape[1] >= 6]
    if not candidates:
        cols = [t.columns.tolist() for t in tables]
        raise RuntimeError(f"Could not identify schedule evaluator table. Available columns: {cols}")
    target = max(candidates, key=lambda x: x.shape[0] * x.shape[1])
    return target


def extract_schedule_evaluator(html_path: Path, season: int, league_id: int, out_path: Path) -> pd.DataFrame:
    tables = read_tables(html_path)
    target = pick_evaluator_table(tables)
    target = target.dropna(how="all")
    target.insert(0, "season", season)
    target.insert(1, "league_id", league_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    target.to_csv(out_path, index=False)
    log(f"[OK] Wrote schedule evaluator to {out_path} ({len(target)} rows)")
    return target


def parse_cell(cell: str):
    if not isinstance(cell, str):
        cell = str(cell)
    cell = cell.strip()
    if not cell or cell.lower() in {"nan", "none"}:
        return None
    result = None
    rf = ra = None
    opp = None
    home_away = ""
    m = re.search(r"([WL])\s*(\d+)[-–](\d+)", cell)
    if m:
        result, rf, ra = m.group(1), int(m.group(2)), int(m.group(3))
    if cell.startswith("@"):
        home_away = "A"
        rest = cell[1:]
    else:
        rest = cell
    m2 = re.match(r"@?([A-Z]{2,3})", rest)
    if m2:
        opp = m2.group(1)
    return {"home_away": home_away, "opp_team_abbr": opp, "result": result, "runs_for": rf, "runs_against": ra, "raw_cell": cell}


def detect_allstar(dates: List[str], filled_counts: dict) -> set:
    # dates already in order
    allstar_dates = set()
    for i in range(len(dates) - 2):
        if filled_counts.get(dates[i], 0) == 0 and filled_counts.get(dates[i + 1], 0) == 0 and filled_counts.get(dates[i + 2], 0) == 0:
            allstar_dates.update({dates[i], dates[i + 1], dates[i + 2]})
            break
    return allstar_dates


def extract_schedule_grid(html_path: Path, season: int, league_id: int, out_path: Path) -> pd.DataFrame:
    tables = read_tables(html_path)
    if not tables:
        raise RuntimeError(f"No tables found in {html_path}")
    # pick largest table assuming grid
    target = max(tables, key=lambda t: t.shape[0] * t.shape[1])
    target = target.dropna(how="all")
    if target.shape[1] < 3:
        raise RuntimeError(f"Schedule grid table seems too small: {target.shape}")
    cols = list(target.columns)
    team_col = cols[0]
    date_cols = cols[1:]
    records = []
    filled_counts = {}
    for _, row in target.iterrows():
        team = row[team_col]
        for dc in date_cols:
            cell = row[dc]
            if isinstance(cell, float) and pd.isna(cell):
                continue
            parsed = parse_cell(str(cell))
            if parsed is None:
                continue
            filled_counts[dc] = filled_counts.get(dc, 0) + 1
            rec = {
                "season": season,
                "league_id": league_id,
                "team_abbr": team,
                "date": dc,
                "game_type": "regular",
            }
            rec.update(parsed)
            records.append(rec)
    dates = list(date_cols)
    allstar_dates = detect_allstar(dates, filled_counts)
    for rec in records:
        rec["is_allstar_break"] = rec["date"] in allstar_dates
        if rec["is_allstar_break"]:
            rec["game_type"] = "allstar_break"
    df = pd.DataFrame(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log(f"[OK] Wrote team schedule to {out_path} ({len(df)} rows, {df['team_abbr'].nunique()} teams)")
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract schedule grid/evaluator into CSV.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--league-id", type=int, required=True)
    args = parser.parse_args()
    season = args.season
    league_id = args.league_id

    base_in = Path("csv/in/almanac_core") / str(season) / "leagues"
    base_out = Path("csv/out/almanac") / str(season)
    grid_html = base_in / f"league_{league_id}_schedule_grid.html"
    eval_html = base_in / f"league_{league_id}_schedule_evaluator.html"
    if not grid_html.exists():
        # fallback search
        matches = list(base_in.glob(f"league_{league_id}_schedule_*grid*.html"))
        if matches:
            grid_html = matches[0]
    if not eval_html.exists():
        matches = list(base_in.glob(f"league_{league_id}_schedule_*eval*.html"))
        if matches:
            eval_html = matches[0]
    if not grid_html.exists():
        raise FileNotFoundError(f"Missing schedule grid HTML: {grid_html}")
    if not eval_html.exists():
        raise FileNotFoundError(f"Missing schedule evaluator HTML: {eval_html}")

    team_sched_out = base_out / f"team_schedule_{season}_league{league_id}.csv"
    eval_out = base_out / f"schedule_evaluator_{season}_league{league_id}.csv"

    extract_schedule_grid(grid_html, season, league_id, team_sched_out)
    extract_schedule_evaluator(eval_html, season, league_id, eval_out)
    log("[INFO] Schedule extraction completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
