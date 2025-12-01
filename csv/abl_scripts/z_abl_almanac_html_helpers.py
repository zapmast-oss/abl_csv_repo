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

    Assumptions about the HTML (OOTP schedule grid):
    - Single <table> containing the grid.
    - Header row:
        * column 0: month label (e.g., "Apr", "April")
        * column 1: day number / day-of-season
        * columns 2..N: team headers (usually city or "City TeamName").
    - Data rows:
        * first two cells repeat the month and day.
        * remaining cells mirror the header-team columns.
        * cells for teams playing that day have CSS classes:
              "gh" => that team is at home
              "ga" => that team is away
          Blank cells => that team is idle that day.
    - Spring gaps and the All-Star break are represented as rows where
      the first two cells still carry month/day, but all team cells are blank.

    Returned dataframe columns:
      season      int, the season passed in
      date        datetime.date, using the given season + month/day
      month       int 3..10
      day         int day-of-month
      team_raw    str, raw header value for that column (before any dim lookup)
      played      bool, True if the team has any "gh"/"ga" marker on that date
      is_home     bool|None, True if only "gh" present, False if only "ga",
                  None if the team did not play that day

    One row per (date, team_raw). Off-days are included with played=False so
    downstream code can reason about rest and brutal stretches cleanly.
    """
    from pathlib import Path
    import datetime
    import re

    from bs4 import BeautifulSoup
    import pandas as pd

    path = Path(html_path)
    if not path.exists():
        raise FileNotFoundError(f"Schedule grid HTML not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    table = soup.find("table")
    if table is None:
        raise ValueError(f"No <table> found in schedule grid HTML: {path}")

    rows = table.find_all("tr")
    if not rows:
        raise ValueError(f"No <tr> rows found in schedule grid table: {path}")

    header_cells = rows[0].find_all(["th", "td"])
    if len(header_cells) < 3:
        raise ValueError("Expected at least 3 header cells (month, day, teams...)")

    # Raw header labels for each team column (index 2..N)
    team_labels = [c.get_text(strip=True) for c in header_cells[2:]]

    month_lookup = {
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
    }

    records: list[dict[str, object]] = []

    for row in rows[1:]:
        cells = row.find_all(["td", "th"])
        if len(cells) < 2:
            continue

        month_text = cells[0].get_text(strip=True)
        day_text = cells[1].get_text(strip=True)

        # Some spacer rows may have no month/day; skip those entirely.
        if not month_text or not day_text:
            continue

        mt = month_text.strip().lower()
        key = mt[:3]
        if key not in month_lookup:
            try:
                month_num = int(mt)
            except ValueError:
                # Unknown month representation; be conservative and skip.
                continue
        else:
            month_num = month_lookup[key]

        # Day-of-month may occasionally include extra characters; strip to first integer.
        day = None
        try:
            day = int(day_text)
        except ValueError:
            m = re.search(r"\d+", day_text)
            if m:
                day = int(m.group())
        if day is None:
            continue

        try:
            current_date = datetime.date(season, month_num, day)
        except ValueError:
            # Invalid calendar date (e.g., Feb 30) => skip the row.
            continue

        # Emit one record per (team, date) so that downstream can compute
        # both game days and off-days per club.
        for j, team_label in enumerate(team_labels, start=2):
            if j >= len(cells):
                break
            cell = cells[j]
            classes = cell.get("class", []) or []
            # "gh" = home, "ga" = away
            is_home = "gh" in classes
            is_away = "ga" in classes
            played = bool(is_home or is_away)

            records.append(
                {
                    "season": int(season),
                    "date": current_date,
                    "month": month_num,
                    "day": day,
                    "team_raw": team_label,
                    "played": played,
                    "is_home": is_home if played else None,
                }
            )

    return pd.DataFrame.from_records(records)
