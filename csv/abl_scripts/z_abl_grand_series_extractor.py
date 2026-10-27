#!/usr/bin/env python
"""
Extract Grand Series games from almanac scores/box HTML.

If --start/--end are omitted, the script will derive the GS span from
the schedule grid (last contiguous block where only two teams play).

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

from z_abl_almanac_html_helpers import parse_schedule_grid
import re


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract Grand Series games from almanac HTML.")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--league-id", type=int, default=200)
    p.add_argument("--start", type=str, required=False, help="GS start date YYYY-MM-DD (optional; derive from grid if omitted)")
    p.add_argument("--end", type=str, required=False, help="GS end date YYYY-MM-DD (optional; derive from grid if omitted)")
    p.add_argument("--out-dir", type=str, required=False, help="Optional output dir to save box/log HTML for each GS game")
    return p.parse_args()


def _build_team_keys(dim_path: Path) -> set[str]:
    """Build a set of normalized team keys from dim_team_park."""
    if not dim_path.exists():
        return set()
    df = pd.read_csv(dim_path)
    if "Team Name" not in df.columns or "Abbr" not in df.columns:
        return set()

    def norm(val: str) -> str:
        v = (val or "").strip().lower()
        v = re.sub(r"\(.*?\)", "", v)
        v = re.sub(r"[^a-z0-9]+", "", v)
        return v

    keys = set()
    for _, row in df.iterrows():
        abbr = str(row["Abbr"])
        team = str(row["Team Name"])
        city = str(row.get("City", "")).split("(")[0].strip()
        for cand in [abbr, team, city, f"{city} {team}".strip(), f"{team} ({abbr})"]:
            k = norm(cand)
            if k:
                keys.add(k)
    return keys


def derive_gs_span_from_grid(season: int, league_id: int):
    """Find the last contiguous block of dates where exactly two teams play (GS)."""
    grid_path = Path("csv") / "in" / "almanac_core" / str(season) / "leagues" / f"league_{league_id}_schedule_grid.html"
    if not grid_path.exists():
        return None, None
    df = parse_schedule_grid(grid_path, season=season)
    if df.empty:
        return None, None
    daily = df[df["played"]].groupby("date")["team_raw"].nunique().reset_index(name="teams").sort_values("date")
    if daily.empty:
        return None, None
    # Find the last date where exactly two teams played
    last_two = daily[daily["teams"] == 2]
    if last_two.empty:
        return None, None
    end_date = last_two["date"].max()
    # Walk backwards from end_date while teams==2 to get start_date
    dates_desc = daily[daily["date"] <= end_date].sort_values("date", ascending=False)
    start_date = end_date
    prev = end_date
    for _, row in dates_desc.iterrows():
        d = row["date"]
        if d == prev and row["teams"] == 2:
            start_date = d
            prev = d - pd.Timedelta(days=1)
            continue
        # break when we hit a gap or teams !=2
        break
    return datetime.combine(start_date, datetime.min.time()), datetime.combine(end_date, datetime.min.time())


def derive_gs_span_from_scores(zip_path: Path, season: int, league_id: int, grid_start: datetime | None) -> tuple[datetime | None, datetime | None]:
    """Fallback: use scores pages to find the last contiguous run where exactly one game is played and only two teams appear."""
    if grid_start is None:
        start_date = datetime(season, 10, 1)
    else:
        start_date = grid_start
    window = 60  # scan forward up to 60 days from postseason start
    day_info = []
    for i in range(window):
        dt = start_date + timedelta(days=i)
        info = extract_scores_day(zip_path, season, league_id, dt)
        if info["games"] > 0:
            day_info.append((dt, info["games"], info["teams"]))

    # find last contiguous run where games==1 and team set size<=2
    best_start = best_end = None
    best_teams = set()
    run_start = run_end = None
    run_teams = set()
    prev = None
    for dt, gcount, teams in day_info:
        if gcount == 1:
            if run_start is None or (prev and (dt - prev).days > 1):
                run_start = dt
                run_teams = set()
            run_end = dt
            run_teams.update([t for t in teams if t])
        else:
            if run_start is not None and len(run_teams) == 2:
                best_start, best_end, best_teams = run_start, run_end, run_teams.copy()
            run_start = run_end = None
            run_teams = set()
        prev = dt
    if run_start is not None and len(run_teams) == 2:
        best_start, best_end, best_teams = run_start, run_end, run_teams.copy()
    return best_start, best_end


def derive_gs_dates_from_grid(season: int, league_id: int) -> List[datetime]:
    """Return the list of GS dates: all post-September dates where the team set equals the team set on the final played date."""
    grid_path = Path("csv") / "in" / "almanac_core" / str(season) / "leagues" / f"league_{league_id}_schedule_grid.html"
    if not grid_path.exists():
        return []
    df = parse_schedule_grid(grid_path, season=season)
    if df.empty:
        return []
    df["date"] = pd.to_datetime(df["date"])
    last_date = df[df["played"]]["date"].max()
    if pd.isna(last_date):
        return []
    last_teams = set(df[(df["date"] == last_date) & (df["played"])]["team_raw"])
    dates = []
    post = df[(df["played"]) & (df["date"] > pd.Timestamp(f"{season}-09-30"))]
    for d, sub in post.groupby("date"):
        teams = set(sub["team_raw"])
        if len(teams) == 2 and teams == last_teams:
            dates.append(d)
    dates = sorted(set(dates))
    return [datetime.combine(d.date(), datetime.min.time()) for d in dates]


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


def extract_scores_day(zip_path: Path, season: int, league_id: int, dt: datetime) -> Dict[str, object]:
    """Return number of games and team labels for a given scores page."""
    rel = f"almanac_{season}/leagues/league_{league_id}_scores_{dt.strftime('%Y_%m_%d')}.html"
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            html = zf.read(rel).decode("utf-8", errors="ignore")
    except KeyError:
        return {"games": 0, "teams": []}
    soup = BeautifulSoup(html, "html.parser")
    games = 0
    teams = []
    for td in soup.find_all("td", class_="boxtitle"):
        a = td.find("a", href=True, string=re.compile("Box Score"))
        if a:
            games += 1
        # try to capture team names from subsequent linescore table
    # grab first linescore table and collect team anchors
    for tbl in soup.find_all("table"):
        header = [c.get_text(" ", strip=True) for c in tbl.find_all("th")]
        if "R" in header and "H" in header and len(tbl.find_all("tr")) >= 3:
            rows = tbl.find_all("tr")[1:3]
            for r in rows:
                a = r.find("a", href=True)
                if a:
                    teams.append(a.get_text(strip=True))
            break
    return {"games": games, "teams": teams}


def parse_box(zip_path: Path, season: int, game_id: int, save_dir: Path | None = None) -> Dict[str, object]:
    """Parse a single box score page to pull date, teams, and final score. Optionally save the box/log HTML."""
    rel = f"almanac_{season}/box_scores/game_box_{game_id}.html"
    with zipfile.ZipFile(zip_path, "r") as zf:
        html = zf.read(rel).decode("utf-8", errors="ignore")
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            (save_dir / f"game_box_{game_id}.html").write_text(html, encoding="utf-8")
            # also save game log if present
            log_rel = f"almanac_{season}/game_logs/log_{game_id}.html"
            if log_rel in zf.namelist():
                log_html = zf.read(log_rel).decode("utf-8", errors="ignore")
                (save_dir / f"log_{game_id}.html").write_text(log_html, encoding="utf-8")
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

    home = away = ""
    home_r = away_r = None

    # Try BeautifulSoup linescore-style table
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

    # Fallback: use pandas to locate a table with an 'R' column
    if home_r is None or away_r is None:
        try:
            tables = pd.read_html(StringIO(html))
        except Exception:
            tables = []
        for df in tables:
            cols = [str(c) for c in df.columns]
            if "R" in cols and len(df) >= 2:
                try:
                    away = str(df.iloc[0, 0])
                    home = str(df.iloc[1, 0])
                    away_r = int(df.loc[0, "R"])
                    home_r = int(df.loc[1, "R"])
                    break
                except Exception:
                    continue

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
    zip_path = Path("data_raw/ootp_html") / f"almanac_{args.season}.zip"
    if not zip_path.exists():
        logger.error("Almanac zip not found: %s", zip_path)
        return 1

    # Build list of dates to scan
    date_list: List[datetime] = []
    derived = "manual"
    if args.start and args.end:
        start = datetime.strptime(args.start, "%Y-%m-%d")
        end = datetime.strptime(args.end, "%Y-%m-%d")
        cur = start
        while cur <= end:
            date_list.append(cur)
            cur += timedelta(days=1)
        start_date = start
        end_date = end
    else:
        derived = "grid"
        date_list = derive_gs_dates_from_grid(args.season, args.league_id)
        if not date_list:
            # fallback to scores-derived span if grid failed
            ds, de = derive_gs_span_from_scores(zip_path, args.season, args.league_id, None)
            if ds and de:
                cur = ds
                while cur <= de:
                    date_list.append(cur)
                    cur += timedelta(days=1)
                derived = "scores"
        if not date_list:
            logger.error("Could not derive GS dates and no dates provided.")
            return 1
        start_date = min(date_list)
        end_date = max(date_list)

    games = []
    out_dir = Path(args.out_dir) if args.out_dir else None

    for d in date_list:
        for gid in extract_game_ids(zip_path, args.season, args.league_id, d):
            games.append(parse_box(zip_path, args.season, gid, save_dir=out_dir))

    # Filter to games that have teams
    games = [g for g in games if g["home"] or g["away"]]
    games.sort(key=lambda g: g.get("date", ""))

    # Restrict to two teams (GS is two clubs, max 7 games)
    team_counts: Dict[str, int] = {}
    for g in games:
        for t in [g["home"], g["away"]]:
            if t:
                team_counts[t] = team_counts.get(t, 0) + 1
    if len(team_counts) > 2:
        top_two = sorted(team_counts.items(), key=lambda kv: kv[1], reverse=True)[:2]
        allowed = {t for t, _ in top_two}
        games = [g for g in games if g["home"] in allowed and g["away"] in allowed]

    # Cap at 7 games (best-of-7)
    games = games[:7]

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

    # Print recap with generation timestamp
    generated = datetime.now().strftime("Generated on: %Y-%m-%d %H:%M:%S (local time)")
    print(generated)
    if derived == "manual":
        print(f"Grand Series window (manual): {start_date.date()} to {end_date.date()}")
    elif derived == "scores":
        print(f"Grand Series window (derived from scores): {start_date.date()} to {end_date.date()}")
    else:
        print(f"Grand Series window (derived from grid): {start_date.date()} to {end_date.date()}")
    for idx, r in enumerate(recap, 1):
        tally_str = ", ".join(f"{team}:{wins}" for team, wins in r["series_tally"].items())
        print(f"Game {idx}: {r['date']}  {r['result']}  (tally: {tally_str})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
