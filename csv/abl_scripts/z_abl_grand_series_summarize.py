#!/usr/bin/env python
"""
Summarize Grand Series box scores into a simple markdown recap.

Inputs (defaults assume prior extractor run):
  - --season (int, required), --league-id (int, default 200)
  - --gs-dir (path, default csv/out/eb/gs_<season>)
    containing game_box_*.html (and optionally game_log_*.html)

Output:
  - csv/out/eb/gs_<season>_summary.md with a generated-on timestamp and per-game bullets.

We keep parsing lightweight: use pandas.read_html to grab the primary line-score
table, pull team names and the R column, and derive a simple series tally.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import logging
import re
from io import StringIO
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from z_abl_almanac_html_helpers import parse_schedule_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Grand Series box scores.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--league-id", type=int, default=200)
    parser.add_argument(
        "--gs-dir",
        type=Path,
        help="Directory with game_box_*.html; defaults to csv/out/eb/gs_<season>.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        help="Optional explicit output path for summary markdown.",
    )
    return parser.parse_args()


def _extract_date_str(html_text: str, season: Optional[int] = None) -> Optional[str]:
    """
    Try several patterns to pull a human-friendly date from the box HTML.
    Handles forms like:
      - Tuesday, October 21st , 1980
      - October 21, 1980
      - 10/21/1980 or 1980-10-21
    Returns the first match as string or None.
    """
    # Long-form with weekday and ordinal
    m = re.search(
        r"([A-Z][a-z]+,\s+)?([A-Z][a-z]+)\s+\d{1,2}(st|nd|rd|th)?\s*,\s*\d{4}",
        html_text,
    )
    if m:
        candidate = m.group(0).replace("  ", " ").replace(" ,", ",").strip()
        if season is None or str(season) in candidate:
            return candidate

    # Simple Month Day, Year
    m = re.search(r"[A-Z][a-z]+\s+\d{1,2},\s*\d{4}", html_text)
    if m:
        candidate = m.group(0).replace(" ,", ",").strip()
        if season is None or str(season) in candidate:
            return candidate

    # Numeric variants
    m = re.search(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", html_text)
    if m:
        y, mo, dy = m.group(1), m.group(2).zfill(2), m.group(3).zfill(2)
        candidate = f"{y}-{mo}-{dy}"
        if season is None or str(season) in candidate:
            return candidate
    m = re.search(r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b", html_text)
    if m:
        mo, dy, y = m.group(1).zfill(2), m.group(2).zfill(2), m.group(3)
        candidate = f"{y}-{mo}-{dy}"
        if season is None or str(season) in candidate:
            return candidate
    return None


def _normalize_date_str(date_str: Optional[str], season: Optional[int]) -> Optional[str]:
    """
    Normalize any matched date string to ISO YYYY-MM-DD when possible.
    Only accept the normalization if the year matches the target season (if provided).
    """
    if not date_str:
        return None
    dt = pd.to_datetime(date_str, errors="coerce")
    if dt is not None and not pd.isna(dt):
        if season is None or dt.year == season:
            return dt.strftime("%Y-%m-%d")
    return None


def load_box_table(html_text: str, season: Optional[int] = None) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Load the primary line-score table (teams as rows, innings + R/H/E as columns).
    Returns the table and the date string if present in the title/subtitle.
    """
    tables = pd.read_html(StringIO(html_text))
    table = None
    for t in tables:
        cols = [str(c).strip().lower() for c in t.columns]
        if any(c == "r" for c in cols) and len(t) >= 2:
            table = t
            break
    if table is None:
        raise ValueError("No line-score table with an R column found in box HTML.")

    date_str = _extract_date_str(html_text, season=season)
    return table, date_str


def parse_game(box_path: Path, season: Optional[int]) -> dict:
    html_text = box_path.read_text(encoding="utf-8", errors="ignore")
    table, date_str = load_box_table(html_text, season=season)
    date_display = _normalize_date_str(date_str, season) or date_str

    # Normalize column names and find R column
    table.columns = [str(c).strip() for c in table.columns]
    r_col = next((c for c in table.columns if str(c).strip().lower() == "r"), None)
    if r_col is None:
        raise ValueError(f"No R column found in {box_path}")

    team_a = str(table.iloc[0, 0]).strip()
    team_b = str(table.iloc[1, 0]).strip()
    runs_a = float(table.iloc[0][r_col])
    runs_b = float(table.iloc[1][r_col])

    winner = team_a if runs_a > runs_b else team_b

    # Derive game number from filename if possible
    m = re.search(r"game_box_(\d+)", box_path.name)
    game_id = m.group(1) if m else box_path.name

    return {
        "game_id": game_id,
        "date": date_display,
        "team_a": team_a,
        "team_b": team_b,
        "runs_a": runs_a,
        "runs_b": runs_b,
        "winner": winner,
        "path": box_path,
    }


def derive_gs_dates_from_grid(season: int, league_id: int) -> List[str]:
    """
    Derive GS dates from the schedule grid by taking the continuous tail block
    where exactly two teams play (the same two teams) ending on the last played date.
    """
    grid_path = (
        Path("csv")
        / "in"
        / "almanac_core"
        / str(season)
        / "leagues"
        / f"league_{league_id}_schedule_grid.html"
    )
    if not grid_path.exists():
        return []
    df = parse_schedule_grid(grid_path, season=season)
    # Build team set per date
    by_date = (
        df.groupby("date")["team_raw"]
        .apply(lambda s: sorted(set(str(x) for x in s if str(x).strip())))
        .reset_index()
    )
    played_dates = (
        df[df["played"]]
        .groupby("date")["team_raw"]
        .apply(lambda s: sorted(set(str(x) for x in s if str(x).strip())))
        .reset_index()
    )
    played = df[df["played"]]
    if played.empty:
        return []
    last_date = played["date"].max()
    final_set = played_dates[played_dates["date"] == last_date]["team_raw"].iloc[0]
    if len(final_set) != 2:
        return []

    # Walk backward to collect the continuous tail block with the same two teams
    dates_sets = list(
        zip(
            played_dates.sort_values("date")["date"],
            played_dates.sort_values("date")["team_raw"],
        )
    )
    tail: List[str] = []
    for date, teams in reversed(dates_sets):
        if teams == final_set:
            tail.append(pd.to_datetime(date).strftime("%Y-%m-%d"))
        else:
            if tail:
                break  # stop at first non-matching once we've started collecting
    return list(reversed(tail))


def summarize_series(box_files: List[Path], season: Optional[int], league_id: Optional[int]) -> List[dict]:
    games = [parse_game(p, season=season) for p in box_files]

    derived_dates: List[str] = []
    if season is not None and league_id is not None:
        derived_dates = derive_gs_dates_from_grid(season, league_id)
    if derived_dates:
        if len(derived_dates) >= len(games):
            for g, dt_str in zip(games, derived_dates):
                g["date"] = dt_str
        else:
            # If fewer dates than games, extend by adding days to the last known date
            extended = list(derived_dates)
            if extended:
                last_dt = pd.to_datetime(extended[-1])
                while len(extended) < len(games):
                    last_dt = last_dt + pd.Timedelta(days=1)
                    extended.append(last_dt.strftime("%Y-%m-%d"))
                for g, dt_str in zip(games, extended):
                    g["date"] = dt_str

    # Enforce strictly increasing dates to avoid duplicates
    last_dt: Optional[pd.Timestamp] = None
    for g in games:
        dt = pd.to_datetime(g.get("date"), errors="coerce") if g.get("date") else None
        if dt is not None and not pd.isna(dt):
            if last_dt is not None and dt <= last_dt:
                dt = last_dt + pd.Timedelta(days=1)
            g["date"] = dt.strftime("%Y-%m-%d")
            last_dt = dt

    tally: dict[str, int] = {}
    for g in games:
        tally[g["winner"]] = tally.get(g["winner"], 0) + 1
        g["tally"] = dict(tally)
    return games


def build_markdown(
    games: List[dict], season: int, league_id: int, generated_at: datetime
) -> str:
    lines: List[str] = []
    lines.append(f"# Grand Series {season} Recap (League {league_id})")
    lines.append(
        f"Generated on: {generated_at.strftime('%Y-%m-%d %H:%M:%S')} (local time)"
    )
    lines.append("")
    for idx, g in enumerate(games, 1):
        date_str = g["date"] or "Unknown date"
        tally_str = ", ".join(f"{k}:{v}" for k, v in g["tally"].items())
        lines.append(
            f"{idx}. {date_str} — {g['team_a']} {int(g['runs_a'])} vs {g['team_b']} {int(g['runs_b'])} "
            f"(winner: {g['winner']}; tally: {tally_str})"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    season = args.season
    gs_dir = args.gs_dir or Path("csv") / "out" / "eb" / f"gs_{season}"
    if not gs_dir.exists():
        raise FileNotFoundError(f"GS directory not found: {gs_dir}")

    box_files = sorted(gs_dir.glob("game_box_*.html"))
    if not box_files:
        raise FileNotFoundError(f"No game_box_*.html found in {gs_dir}")

    games = summarize_series(box_files, season=season, league_id=args.league_id)
    generated_at = datetime.now()
    md = build_markdown(games, season=season, league_id=args.league_id, generated_at=generated_at)

    out_path = args.out_md or (Path("csv") / "out" / "eb" / f"gs_{season}_summary.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")

    logging.info("Wrote Grand Series summary to %s", out_path)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
