#!/usr/bin/env python
"""EB schedule context brief for any season/league (team-aware, defensive)."""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from eb_text_utils import normalize_eb_text


def log(msg: str) -> None:
    print(msg, flush=True)


ALL_STAR_INFO: Dict[int, Dict[str, Optional[str]]] = {
    1972: {"host": "Boston Patriots", "winner_conf": "NBC"},
    1973: {"host": "San Francisco Warriors", "winner_conf": "NBC"},
    1974: {"host": "St. Louis Stallions", "winner_conf": "NBC"},
    1975: {"host": "Miami Hurricanes", "winner_conf": "NBC"},
    1976: {"host": "Denver Rocketeers", "winner_conf": "NBC"},
    1977: {"host": "Atlanta Kings", "winner_conf": "ABC"},
    1978: {"host": "New York Aces", "winner_conf": "NBC"},
    1979: {"host": "Tampa Bay Storm", "winner_conf": "NBC"},
    1980: {"host": "Las Vegas Gamblers", "winner_conf": "NBC"},
    1981: {"host": "Phoenix Firebirds", "winner_conf": None},
    1982: {"host": "Pittsburgh Express", "winner_conf": None},
    1983: {"host": "Chicago Fire", "winner_conf": None},
    1984: {"host": "Philadelphia Fury", "winner_conf": None},
    1985: {"host": "San Diego Seraphs", "winner_conf": None},
    1986: {"host": "Nashville Blues", "winner_conf": None},
    1987: {"host": "Detroit Dukes", "winner_conf": None},
    1988: {"host": "Seattle Comets", "winner_conf": None},
    1989: {"host": "Minneapolis Blizzard", "winner_conf": None},
    1990: {"host": "Houston Mavericks", "winner_conf": None},
    1991: {"host": "Charlotte Colonels", "winner_conf": None},
    1992: {"host": "Portland Lumberjacks", "winner_conf": None},
    1993: {"host": "Los Angeles Cobras", "winner_conf": None},
    1994: {"host": "Cincinnati Cougars", "winner_conf": None},
    1995: {"host": "Dallas Rustlers", "winner_conf": None},
    1996: {"host": "Boston Patriots", "winner_conf": None},
    1997: {"host": "San Francisco Warriors", "winner_conf": None},
    1998: {"host": "St. Louis Stallions", "winner_conf": None},
    1999: {"host": "Miami Hurricanes", "winner_conf": None},
    2000: {"host": "Denver Rocketeers", "winner_conf": None},
    2001: {"host": "Atlanta Kings", "winner_conf": None},
    2002: {"host": "New York Aces", "winner_conf": None},
    2003: {"host": "Tampa Bay Storm", "winner_conf": None},
    2004: {"host": "Las Vegas Gamblers", "winner_conf": None},
    2005: {"host": "Phoenix Firebirds", "winner_conf": None},
    2006: {"host": "Pittsburgh Express", "winner_conf": None},
    2007: {"host": "Chicago Fire", "winner_conf": None},
    2008: {"host": "Philadelphia Fury", "winner_conf": None},
    2009: {"host": "San Diego Seraphs", "winner_conf": None},
    2010: {"host": "Nashville Blues", "winner_conf": None},
    2011: {"host": "Detroit Dukes", "winner_conf": None},
    2012: {"host": "Seattle Comets", "winner_conf": None},
    2013: {"host": "Minneapolis Blizzard", "winner_conf": None},
    2014: {"host": "Houston Mavericks", "winner_conf": None},
    2015: {"host": "Charlotte Colonels", "winner_conf": None},
    2016: {"host": "Portland Lumberjacks", "winner_conf": None},
    2017: {"host": "Los Angeles Cobras", "winner_conf": None},
    2018: {"host": "Cincinnati Cougars", "winner_conf": None},
    2019: {"host": "Dallas Rustlers", "winner_conf": None},
    2020: {"host": "St. Louis Stallions", "winner_conf": None},
    2021: {"host": "San Francisco Warriors", "winner_conf": None},
    2022: {"host": "Boston Patriots", "winner_conf": None},
}


def load_games_by_team(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        log(f"[WARN] Games-by-team file missing: {path}")
        return None
    df = pd.read_csv(path)
    log(f"[INFO] Loaded {len(df)} rows from {path}")
    log(f"[DEBUG] Columns in {path.name}: {df.columns.tolist()}")
    return df


def find_date_col(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if "date" in str(col).lower():
            return col
    return None


def games_per_team_overview(df: pd.DataFrame) -> Optional[dict]:
    team_col = None
    for cand in ["team_id", "team_abbr", "team", "Team"]:
        if cand in df.columns:
            team_col = cand
            break
    if team_col is None:
        return None
    counts = df.groupby(team_col).size().describe()
    return counts.to_dict()


def longest_road_trip_per_team(df: pd.DataFrame, date_col: str) -> List[Tuple[str, int]]:
    home_col = None
    for cand in ["home_away", "HA", "venue"]:
        if cand in df.columns:
            home_col = cand
            break
    team_col = None
    for cand in ["team_abbr", "team_id", "team", "Team"]:
        if cand in df.columns:
            team_col = cand
            break
    if home_col is None or team_col is None:
        return []
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    trips: List[Tuple[str, int]] = []
    for team, grp in df.groupby(team_col):
        grp = grp.sort_values(date_col)
        best = cur = 0
        for _, r in grp.iterrows():
            ha_val = str(r.get(home_col, "")).upper()
            if ha_val.startswith("A"):
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        trips.append((str(team), best))
    trips = sorted(trips, key=lambda x: x[1], reverse=True)
    return trips


def detect_allstar_break(df: pd.DataFrame, date_col: str, season: int) -> Tuple[Optional[date], Optional[date], Optional[date]]:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    if df.empty:
        return (None, None, None)
    dates = df[date_col].dt.date
    min_date = dates.min()
    max_date = dates.max()
    full_range = pd.date_range(min_date, max_date, freq="D").date
    game_dates = set(dates.unique())
    idle_july = [d for d in full_range if d not in game_dates and d.month == 7 and d.year == season]
    if not idle_july:
        return (None, None, None)
    runs: List[Tuple[date, date, int]] = []
    start = idle_july[0]
    prev = idle_july[0]
    for d in idle_july[1:]:
        if d == prev + timedelta(days=1):
            prev = d
        else:
            runs.append((start, prev, (prev - start).days + 1))
            start = prev = d
    runs.append((start, prev, (prev - start).days + 1))
    runs = sorted(runs, key=lambda x: x[2], reverse=True)
    best_start, best_end, best_len = runs[0]
    if best_len >= 3:
        return (best_start, best_start + timedelta(days=1), best_start + timedelta(days=2))
    return (None, None, None)


def build_allstar_section(season: int, league_id: int, games_df: Optional[pd.DataFrame], date_col: Optional[str]) -> List[str]:
    lines: List[str] = []
    lines.append("## All-Star Break")
    info = ALL_STAR_INFO.get(season, {"host": "Unknown host", "winner_conf": None})
    host = info.get("host") or "Unknown host"
    winner_conf = info.get("winner_conf")
    winner_str = winner_conf if winner_conf else "not yet recorded"
    derby_date = asg_date = travel_date = None
    if games_df is not None and date_col:
        derby_date, asg_date, travel_date = detect_allstar_break(games_df, date_col, season)
    if derby_date and asg_date and travel_date:
        lines.append(f"- Home Run Derby: {derby_date.isoformat()}")
        lines.append(f"- All-Star Game: {asg_date.isoformat()}")
        lines.append(f"- Travel/off day: {travel_date.isoformat()}")
    else:
        lines.append(f"- Host: {host}")
        lines.append(f"- Winner: {winner_str}")
        lines.append("- Exact calendar dates: [not auto-detected]")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="EB schedule context brief.")
    parser.add_argument("--season", type=int, default=1972)
    parser.add_argument("--league-id", type=int, default=200)
    args = parser.parse_args()
    season = args.season
    league_id = args.league_id

    base = Path("csv/out/almanac") / str(season)
    games_by_team_path = base / f"games_{season}_league{league_id}_by_team.csv"

    md: List[str] = []
    md.append(f"# EB Schedule Context {season} – Data Brief (DO NOT PUBLISH)")
    md.append(f"_League ID {league_id}_")
    md.append("")

    games_df = load_games_by_team(games_by_team_path)
    date_col = find_date_col(games_df) if games_df is not None else None

    # Schedule overview
    md.append("## Schedule Overview")
    overview = games_per_team_overview(games_df) if games_df is not None else None
    if overview:
        md.append(f"- Games per team (count describe): {overview}")
    else:
        md.append("- [WARN] Games-by-team file not found or missing team/date columns; schedule overview skipped.")
    md.append("")

    # Brutal stretches
    md.append("## Brutal Stretches")
    if games_df is not None and date_col:
        trips = longest_road_trip_per_team(games_df, date_col)
        trips = [t for t in trips if t[1] > 0][:5]
        if trips:
            for team, length in trips:
                md.append(f"- {team}: longest road trip {length} games")
        else:
            md.append("- [WARN] Unable to compute brutal stretches from schedule data.")
    else:
        md.append("- [WARN] Unable to compute brutal stretches from schedule data.")
    md.append("")

    # All-Star break (simplified)
    md.extend(build_allstar_section(season, league_id, games_df, date_col))
    md.append("")

    out_path = base / f"eb_schedule_context_{season}_league{league_id}.md"
    full_text = "\n".join(md)
    full_text = normalize_eb_text(full_text)
    out_path.write_text(full_text, encoding="utf-8")
    log(f"[OK] Wrote schedule context brief to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
