#!/usr/bin/env python
"""Helper utilities for parsing almanac HTML pages."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
from bs4 import BeautifulSoup


def _read_html_text(html_path: Path) -> str:
    if not html_path.exists():
        raise FileNotFoundError(f"HTML not found: {html_path}")
    return html_path.read_text(encoding="utf-8", errors="ignore")


def _normalize_table_columns(df: pd.DataFrame) -> Dict[str, str]:
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    result = {}
    for key in ("player", "team"):
        if key in lower_map:
            result[key] = lower_map[key]
    return result


def parse_preseason_predictions(html_path: Path) -> List[Dict[str, Any]]:
    """
    Parse a preseason predictions HTML page for a single league/season.

    Return a list of dicts with at least:
    - 'player_name'  (str)
    - 'team_name'    (str)  # city or full team name as given on the page
    - 'hype_role'    (str)  # e.g. 'MVP favorite', 'HR leader pick', etc. if detectable
    - 'rank'         (int)  # 1-based rank within that hype_role if detectable, else None
    - 'source'       (str)  # short string describing which block/table it came from
    """
    import re

    html_path = Path(html_path)
    if not html_path.exists():
        raise FileNotFoundError(f"Preseason prediction HTML not found: {html_path}")

    text = html_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(text, "html.parser")

    entries: List[Dict[str, Any]] = []
    pid_re = re.compile(r"player_(\d+)\.html")

    for tr in soup.find_all("tr"):
        player_link = tr.find("a", href=lambda h: h and "players/player_" in h)
        if not player_link:
            continue

        href = player_link.get("href", "")
        m = pid_re.search(href)
        player_id: Optional[int]
        if m:
            try:
                player_id = int(m.group(1))
            except Exception:
                player_id = None
        else:
            player_id = None

        raw_name = player_link.get_text(" ", strip=True)
        # Drop position and anything after the first comma.
        if "," in raw_name:
            short_name = raw_name.split(",", 1)[0].strip()
        else:
            short_name = raw_name.strip()

        team_link = tr.find("a", href=lambda h: h and "teams/team_" in h)
        team_name = ""
        if team_link:
            team_name = team_link.get_text(" ", strip=True)

        entries.append(
            {
                "player_name": short_name,
                "team_name": team_name,
                "hype_role": "preseason",
                "rank": None,
                "source": "preseason_prediction",
                "player_id": player_id,
            }
        )

    if not entries:
        # Last-resort single placeholder so downstream code doesn't explode
        entries.append(
            {
                "player_name": "Unknown",
                "team_name": "",
                "hype_role": "preseason",
                "rank": None,
                "source": "generic_preseason",
                "player_id": None,
            }
        )

    return entries


def parse_league_history_index(html_path: Path) -> List[Dict[str, Any]]:
    """Stub: parse league history index (Grand Series participants/champions)."""
    return []


def parse_schedule_grid(html_path, season: int):
    """
    Parse the almanac schedule_grid HTML into a per-team, per-date dataframe.

    This implementation targets the OOTP schedule grid table that has:
      - First column: month number
      - Second column: day number
      - Remaining columns: team headers (usually abbreviations).

    We pick the widest table in the HTML (most columns), treat its first row
    as the header, and then emit one record per (team, date) indicating whether
    that team has any entry (played=True) on that date. Home/away is not
    recoverable from the stripped table, so is_home is always None.
    """
    from pathlib import Path
    import datetime

    import pandas as pd

    path = Path(html_path)
    if not path.exists():
        raise FileNotFoundError(f"Schedule grid HTML not found: {path}")

    html_text = path.read_text(encoding="utf-8", errors="ignore")
    tables = pd.read_html(html_text)
    if not tables:
        raise ValueError(f"No tables found in schedule grid HTML: {path}")

    # Pick the table with the most columns (the actual grid).
    grid = max(tables, key=lambda t: t.shape[1])
    # Use the first row as header.
    header = grid.iloc[0]
    grid = grid.iloc[1:].copy()
    grid.columns = header

    # First two header entries are month/day; rest are team columns (team abbreviations).
    if len(grid.columns) < 3:
        raise ValueError("Schedule grid table does not have enough columns (need month, day, teams)")
    month_col = grid.columns[0]
    day_col = grid.columns[1]
    team_cols = list(grid.columns[2:])

    records: list[dict[str, object]] = []

    for _, row in grid.iterrows():
        try:
            month = int(str(row[month_col]).strip())
            day = int(str(row[day_col]).strip())
        except Exception:
            continue
        try:
            current_date = datetime.date(season, month, day)
        except Exception:
            continue

        for team_label in team_cols:
            val = row.get(team_label, "")
            text = "" if pd.isna(val) else str(val).strip()
            played = text != ""
            # crude home/away: OOTP often uses "@OPP" to mark away
            is_home = None
            opp = text
            if played:
                if text.startswith("@"):
                    is_home = False
                    opp = text[1:].strip()
                else:
                    is_home = True
            records.append(
                {
                    "season": int(season),
                    "date": current_date,
                    "month": month,
                    "day": day,
                    "team_raw": str(team_label),
                    "played": bool(played),
                    "is_home": is_home,
                    "cell_text": text,
                    "opponent_raw": opp if played else "",
                }
            )

    return pd.DataFrame.from_records(records)
