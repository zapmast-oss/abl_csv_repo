#!/usr/bin/env python
"""
Extract Grand Series games from almanac scores/box HTML.

Usage example:
  python csv/abl_scripts/z_abl_grand_series_extractor.py \
      --season 1980 --league-id 200 --start 1980-10-21 --end 1980-11-01
"""

from __future__ import annotations

import argparse
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
import zipfile

from bs4 import BeautifulSoup
from io import StringIO
import pandas as pd


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract Grand Series games from almanac HTML.")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--league-id", type=int, default=200)
    p.add_argument("--start", type=str, required=True, help="GS start date YYYY-MM-DD")
    p.add_argument("--end", type=str, required=True, help="GS end date YYYY-MM-DD")
    return p.parse_args()


def daterange(start: datetime, end: datetime):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def extract_game_ids(zip_path: Path, season: int, league_id: int, dt: datetime) -> List[int]:
    """Extract game_ids from a scores page for a given date."""
    rel = f"almanac_{season}/leagues/league_{league_id}_scores_{dt.strftime('%Y_%m_%d')}.html"
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            html = zf.read(rel).decode("utf-8", errors="ignore")
    except KeyError:
        return []
    soup = BeautifulSoup(html, "html.parser")
    ids: List[int] = []
    for td in soup.find_all("td", class_="boxtitle"):
        a = td.find("a", href=True, string=re.compile("Box Score"))
        if not a:
            continue
        m = re.search(r"game_box_(\d+)\.html", a["href"])
        if m:
            ids.append(int(m.group(1)))
    return ids


def parse_box(zip_path: Path, season: int, game_id: int) -> Dict[str, object]:
    """Parse a single box score page to pull date, teams, and final score."""
    rel = f"almanac_{season}/box_scores/game_box_{game_id}.html"
    with zipfile.ZipFile(zip_path, "r") as zf:
        html = zf.read(rel).decode("utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    # date from subtitle if present
    date_iso = ""
    sub = soup.find("div", class_="repsubtitle")
    if sub:
        # look for "TUESDAY, OCTOBER 7TH , 1980" style
        m = re.search(r"([A-Za-z]+ \d{1,2})(?:st|nd|rd|th)?\s*,?\s*(\d{4})", sub.get_text(" ", strip=True))
        if m:
            try:
                dt = datetime.strptime(f"{m.group(1)} {m.group(2)}", "%B %d %Y")
                date_iso = dt.date().isoformat()
            except Exception:
                date_iso = ""

    # teams/scores from first linescore-style table
    home = away = ""
    home_r = away_r = None
    for tbl in soup.find_all("table"):
        header = [c.get_text(" ", strip=True) for c in tbl.find_all("th")]
        if "R" in header and "H" in header and len(tbl.find_all("tr")) >= 3:
            rows = tbl.find_all("tr")[1:3]
            away_cells = [c.get_text(" ", strip=True) for c in rows[0].find_all(["td", "th"])]
            home_cells = [c.get_text(" ", strip=True) for c in rows[1].find_all(["td", "th"])]
            if len(away_cells) >= len(header) and len(home_cells) >= len(header):
                away = away_cells[0]
                home = home_cells[0]
                try:
                    away_r = int(away_cells[header.index("R")])
                    home_r = int(home_cells[header.index("R")])
                except Exception:
                    pass
            break

    return {
        "game_id": game_id,
        "date": date_iso,
        "home": home,
        "away": away,
        "home_r": home_r,
        "away_r": away_r,
    }


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    zip_path = Path("data_raw/ootp_html") / f"almanac_{args.season}.zip"
    if not zip_path.exists():
        logger.error("Almanac zip not found: %s", zip_path)
        return 1

    games = []
    for d in daterange(start, end):
        for gid in extract_game_ids(zip_path, args.season, args.league_id, d):
            games.append(parse_box(zip_path, args.season, gid))

    # Filter to games that have teams
    games = [g for g in games if g["home"] or g["away"]]
    games.sort(key=lambda g: g.get("date", ""))

    # Running tally
    tally: Dict[str, int] = {}
    recap = []
    for g in games:
        if g["home_r"] is None or g["away_r"] is None:
            result = "score N/A"
        elif g["home_r"] > g["away_r"]:
            tally[g["home"]] = tally.get(g["home"], 0) + 1
            result = f"{g['away']} {g['away_r']} - {g['home']} {g['home_r']}"
        else:
            tally[g["away"]] = tally.get(g["away"], 0) + 1
            result = f"{g['away']} {g['away_r']} - {g['home']} {g['home_r']}"
        recap.append(
            {
                "date": g["date"],
                "game_id": g["game_id"],
                "result": result,
                "series_tally": dict(tally),
            }
        )

    # Print recap
    for idx, r in enumerate(recap, 1):
        tally_str = ", ".join(f"{team}:{wins}" for team, wins in r["series_tally"].items())
        print(f"Game {idx}: {r['date']}  {r['result']}  (tally: {tally_str})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
