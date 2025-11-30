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
    html_text = _read_html_text(html_path)
    soup = BeautifulSoup(html_text, "html.parser")

    entries: List[Dict[str, Any]] = []

    # First, try to parse with pandas for structured tables.
    try:
        tables = pd.read_html(html_text)
    except ValueError:
        tables = []

    for idx, df in enumerate(tables):
        col_map = _normalize_table_columns(df)
        if not col_map:
            continue
        player_col = col_map.get("player")
        team_col = col_map.get("team")
        if not player_col or not team_col:
            continue
        cols_lower = [str(c).lower() for c in df.columns]
        role = "preseason_pitcher" if any("era" in c or "ip" in c for c in cols_lower) else "preseason_hitter"
        for ridx, row in df.iterrows():
            player_name = str(row[player_col]).strip()
            team_name = str(row[team_col]).strip()
            if not player_name or player_name.lower() == "player":
                continue
            entries.append(
                {
                    "player_name": player_name,
                    "team_name": team_name,
                    "hype_role": role,
                    "rank": ridx + 1,
                    "source": f"table_{idx}",
                }
            )

    # Fallback: if nothing parsed, emit a generic placeholder to avoid empty output.
    if not entries:
        for li in soup.find_all("li"):
            text = li.get_text(" ", strip=True)
            if not text:
                continue
            entries.append(
                {
                    "player_name": text,
                    "team_name": "",
                    "hype_role": "preseason",
                    "rank": None,
                    "source": "generic_preseason",
                }
            )

    if not entries:
        # Last-resort single placeholder
        entries.append(
            {
                "player_name": "Unknown",
                "team_name": "",
                "hype_role": "preseason",
                "rank": None,
                "source": "generic_preseason",
            }
        )
    return entries


def parse_league_history_index(html_path: Path) -> List[Dict[str, Any]]:
    """Stub: parse league history index (Grand Series participants/champions)."""
    return []


def parse_schedule_grid(html_path: Path) -> List[Dict[str, Any]]:
    """Stub: parse schedule grid into (date, team, opp_team, home/away) rows."""
    return []
