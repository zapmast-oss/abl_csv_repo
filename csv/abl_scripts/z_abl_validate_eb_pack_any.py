#!/usr/bin/env python
"""
Read-only validator for EB regular-season pack markdown.

Validates (for a given season/league):
  A) Preseason hype block vs source HTML/data (reuses hype script helpers)
  B) Schedule context dates vs schedule grid HTML
  C) Month-of-Glory top entry vs games-by-team monthly splits

If required inputs are missing, it logs a warning and skips that check.
"""

from __future__ import annotations

import argparse
import logging
import re
from datetime import datetime, date
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import pandas as pd
from bs4 import BeautifulSoup

# Import helpers from existing scripts to keep logic in sync
from z_abl_almanac_html_helpers import parse_schedule_grid
# Hype script helpers
from z_abl_preseason_hype_any import (
    parse_preseason_players,
    load_dim_player_profile,
    load_players_csv,
    load_dim_team_lookup,
    load_html_war_and_team,
    resolve_team_abbr,
    bucket_war,
)
# Month-of-glory source (reuse if present)
try:
    from z_abl_month_glory_misery_any import (
        # If available, adapt as needed; otherwise we do a simple recompute below.
    )
except Exception:
    pass


logger = logging.getLogger(__name__)


# -----------------------------
# Markdown parsing helpers
# -----------------------------
def load_pack_md(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines()


def extract_section(lines: List[str], heading: str) -> List[str]:
    out: List[str] = []
    start = -1
    for i, line in enumerate(lines):
        if line.strip() == heading:
            start = i
            break
    if start == -1:
        return []
    for line in lines[start + 1 :]:
        if line.startswith("#"):
            break
        out.append(line)
    return out


def parse_hype_lines(section_lines: List[str]) -> pd.DataFrame:
    data = []
    bucket = None
    for line in section_lines:
        if line.startswith("**") and line.endswith("**"):
            bucket = line.strip("*").strip()
            continue
        m = re.match(r"-\s+(.*)\s+\(([^)]+)\)\s+- WAR:\s+([-\d\.]+)", line.strip())
        if m and bucket:
            full_name = m.group(1).strip()
            team_abbr = m.group(2).strip().split(",")[-1].strip()  # handle "(P, CHI)" -> CHI
            war_val = float(m.group(3))
            data.append(
                {
                    "full_name": full_name,
                    "team_abbr": team_abbr,
                    "war_value": war_val,
                    "bucket_md": bucket,
                }
            )
    return pd.DataFrame(data)


def parse_schedule_section(section_lines: List[str]) -> Dict[str, str]:
    fields = {}
    for line in section_lines:
        line = line.strip()
        if line.startswith("- Opening Day:"):
            fields["opening"] = line.split(":", 1)[1].strip()
        elif line.startswith("- Final Day:"):
            fields["final"] = line.split(":", 1)[1].strip()
        elif line.startswith("- HR Derby:"):
            fields["derby"] = line.split(":", 1)[1].strip()
        elif line.startswith("- All-Star Game:"):
            fields["asg"] = line.split(":", 1)[1].strip()
        elif line.startswith("- Extra off-day:"):
            fields["extra"] = line.split(":", 1)[1].strip()
    return fields


def parse_month_glory_top(section_lines: List[str]) -> Optional[Dict[str, object]]:
    # Expect first bullet under Month of Glory – Overachievers
    capture = False
    for line in section_lines:
        if line.strip().startswith("## Month of Glory"):
            capture = True
            continue
        if capture and line.strip().startswith("- "):
            txt = line.strip()[2:]
            # Example: "Atlanta (Atlanta) — in June went 18-8 (0.692), delta vs season=+0.203"
            m = re.match(
                r"(.+)\s+—\s+in\s+([A-Za-z]+)\s+went\s+(\d+)-(\d+)\s+\(([\d\.]+)\),\s+delta vs season=([+\-]?\d+\.\d+)",
                txt,
            )
            if m:
                return {
                    "team_label_raw": m.group(1).strip(),
                    "month": m.group(2),
                    "w": int(m.group(3)),
                    "l": int(m.group(4)),
                    "pct": float(m.group(5)),
                    "delta": float(m.group(6)),
                }
            break
    return None


# -----------------------------
# Preseason hype validation
# -----------------------------
def validate_preseason_hype(season: int, league_id: int, pack_lines: List[str], root: Path) -> Tuple[bool, str]:
    hype_md_lines = extract_section(pack_lines, "## Preseason hype - who delivered?")
    if not hype_md_lines:
        return False, "Preseason hype section not found in pack."

    # Paths
    preseason_html = root / "csv" / "in" / "almanac_core" / str(season) / "leagues" / f"league_{league_id}_preseason_prediction_report.html"
    if not preseason_html.exists():
        logger.warning("Preseason HTML missing; skipping hype validation.")
        return False, "Skipped (missing preseason HTML)"

    # Reuse hype script helpers to build expected
    hype_raw = parse_preseason_players(preseason_html)
    profiles = load_dim_player_profile(root)
    players = load_players_csv(root)
    team_lookup = load_dim_team_lookup(root)
    html_war, html_team = load_html_war_and_team(root, season, hype_raw["player_id"].tolist(), team_lookup)
    resolved = resolve_hype_pipeline_only(hype_raw, profiles, players, html_war, html_team)
    over, delivered, under = bucket_war(resolved)
    over["bucket_expected"] = "Over-delivered"
    delivered["bucket_expected"] = "Delivered"
    under["bucket_expected"] = "Under-delivered"
    df_expected = pd.concat([over, delivered, under], ignore_index=True)[["full_name", "team_abbr", "war_value", "bucket_expected"]]

    # Parse markdown
    df_md = parse_hype_lines(hype_md_lines)
    if df_md.empty:
        return False, "Markdown hype block could not be parsed."

    # Compare sets and values
    merged = df_expected.merge(df_md, on=["full_name", "team_abbr"], how="outer", suffixes=("_exp", "_md"))
    mismatches = []
    for _, row in merged.iterrows():
        if pd.isna(row["war_value_exp"]) or pd.isna(row["war_value_md"]):
            mismatches.append(row)
            continue
        if abs(row["war_value_exp"] - row["war_value_md"]) > 0.01:
            mismatches.append(row)
            continue
        if str(row["bucket_expected"]) != str(row["bucket_md"]):
            mismatches.append(row)

    if mismatches:
        logger.error("[FAIL] Preseason hype mismatches (showing up to 10):")
        logger.error(pd.DataFrame(mismatches)[["full_name", "team_abbr", "war_value_exp", "war_value_md", "bucket_expected", "bucket_md"]].head(10))
        return False, "Preseason hype mismatches found."

    logger.info("[OK] Preseason hype block matches data for season %s.", season)
    return True, "PASS"


def resolve_hype_pipeline_only(hype: pd.DataFrame, profiles: pd.DataFrame, players: pd.DataFrame, html_war: Dict[int, float], html_team: Dict[int, str]) -> pd.DataFrame:
    merged = hype.merge(profiles, left_on="player_id", right_on="ID", how="left", suffixes=("", "_prof"))
    if not players.empty:
        merged = merged.merge(players, left_on="player_id", right_on="ID", how="left", suffixes=("", "_ply"))

    if html_team:
        merged["team_abbr_html"] = merged["player_id"].map(html_team)

    merged["First Name"] = merged["First Name"].fillna("").astype(str).str.strip()
    merged["Last Name"] = merged["Last Name"].fillna("").astype(str).str.strip()

    def build_full_name(row: pd.Series) -> str:
        fn = row.get("First Name", "")
        ln = row.get("Last Name", "")
        if fn and ln:
            return f"{fn} {ln}"
        for alt in ["Name", "Name.1", "players_full"]:
            if alt in row and isinstance(row[alt], str) and row[alt].strip():
                return row[alt].strip()
        raw = str(row.get("raw_name", "")).strip()
        if "," in raw:
            raw = raw.split(",", 1)[0].strip()
        return raw

    merged["full_name"] = merged.apply(build_full_name, axis=1)
    merged["team_abbr"] = merged.apply(resolve_team_abbr, axis=1)

    def pick_position(row: pd.Series) -> str:
        pos = row.get("players_position", "")
        if isinstance(pos, str) and pos.strip():
            return pos.strip()
        raw = str(row.get("raw_name", "")).strip()
        if "," in raw:
            return raw.split(",", 1)[1].strip()
        return ""
    merged["position"] = merged.apply(pick_position, axis=1)

    def apply_html_war(row: pd.Series) -> float:
        pid = row.get("player_id")
        if isinstance(pid, (int, float)) and not pd.isna(pid) and int(pid) in html_war:
            return float(html_war[int(pid)])
        return 0.0

    merged["war_value"] = merged.apply(apply_html_war, axis=1)
    return merged[["player_id", "full_name", "team_abbr", "position", "war_value"]]


# -----------------------------
# Schedule context validation
# -----------------------------
def detect_asg_dates_from_grid(df: pd.DataFrame) -> Tuple[Optional[date], Optional[date], Optional[date]]:
    july = df[df["date"].apply(lambda d: d.month == 7)].copy()
    july = july.sort_values("date")
    daily = july.groupby("date")["played"].sum().reset_index()
    off_dates = daily[daily["played"] == 0]["date"].tolist()
    if not off_dates:
        return None, None, None
    run = [off_dates[0]]
    for d in off_dates[1:]:
        if (d - run[-1]).days == 1:
            run.append(d)
        else:
            if len(run) >= 3:
                break
            run = [d]
    if len(run) >= 3:
        return run[0], run[1], run[2]
    return None, None, None


def validate_schedule_context(season: int, league_id: int, pack_lines: List[str], root: Path) -> Tuple[bool, str]:
    section = extract_section(pack_lines, "## EB Schedule Context")
    if not section:
        return False, "Schedule context section not found."

    schedule_html = root / "csv" / "in" / "almanac_core" / str(season) / "leagues" / f"league_{league_id}_schedule_grid.html"
    if not schedule_html.exists():
        logger.warning("Schedule grid HTML missing; skipping schedule validation.")
        return False, "Skipped (missing schedule grid)"

    # parse markdown dates
    md_fields = parse_schedule_section(section)

    # parse grid
    df = parse_schedule_grid(schedule_html, season=season)
    if df.empty:
        return False, "Schedule grid parsed empty."
    df_played = df[df["played"]]
    if df_played.empty:
        return False, "No played games in grid."
    opening_date = df_played["date"].min()
    final_date = df_played["date"].max()
    derby_date, asg_date, extra_date = detect_asg_dates_from_grid(df)

    def parse_date_str(s: str) -> Optional[date]:
        if not s:
            return None
        try:
            return datetime.strptime(s, "%B %d, %Y").date()
        except Exception:
            return None

    md_open = parse_date_str(md_fields.get("opening"))
    md_final = parse_date_str(md_fields.get("final"))
    md_derby = parse_date_str(md_fields.get("derby"))
    md_asg = parse_date_str(md_fields.get("asg"))
    md_extra = parse_date_str(md_fields.get("extra"))

    mismatches = []
    if md_open != opening_date:
        mismatches.append(f"Opening Day: pack={md_open}, grid={opening_date}")
    if md_final != final_date:
        mismatches.append(f"Final Day: pack={md_final}, grid={final_date}")
    if md_derby != derby_date:
        mismatches.append(f"HR Derby: pack={md_derby}, grid={derby_date}")
    if md_asg != asg_date:
        mismatches.append(f"ASG: pack={md_asg}, grid={asg_date}")
    if md_extra != extra_date:
        mismatches.append(f"Extra off: pack={md_extra}, grid={extra_date}")

    if mismatches:
        for m in mismatches:
            logger.error("[FAIL] Schedule context mismatch: %s", m)
        return False, "Schedule context mismatches found."

    logger.info("[OK] Schedule context dates match grid for season %s.", season)
    return True, "PASS"


# -----------------------------
# Month-of-Glory spot-check
# -----------------------------
def recompute_monthly_splits(games_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(games_csv)
    # heuristics: find date, team name/abbr, runs
    date_col = next((c for c in df.columns if str(c).lower() in {"date", "game_date"}), None)
    team_col = None
    for cand in ["team_abbr", "team", "team_name"]:
        if cand in df.columns:
            team_col = cand
            break
    win_col = None
    for cand in ["is_win", "win", "result"]:
        if cand in df.columns:
            win_col = cand
            break
    if date_col is None or team_col is None:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["date"])
    if win_col and df[win_col].dtype == object:
        df[win_col] = df[win_col].str.upper().map({"W": 1, "L": 0})
    if win_col is None and "runs" in df.columns and "opp_runs" in df.columns:
        win_col = "is_win_tmp"
        df[win_col] = (df["runs"] > df["opp_runs"]).astype(int)
    if win_col is None:
        return pd.DataFrame()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    season_year = df["year"].mode().iloc[0]
    df = df[df["year"] == season_year]
    grouped = df.groupby([team_col, "month"]).agg(G=("date", "count"), W=(win_col, "sum"))
    grouped["L"] = grouped["G"] - grouped["W"]
    grouped["month_pct"] = grouped["W"] / grouped["G"]
    season_group = grouped.groupby(level=0).agg(G_season=("G", "sum"), W_season=("W", "sum"))
    season_group["season_pct"] = season_group["W_season"] / season_group["G_season"]
    grouped = grouped.join(season_group, on=team_col)
    grouped["delta"] = grouped["month_pct"] - grouped["season_pct"]
    grouped = grouped.reset_index()
    grouped["team_label_raw"] = grouped[team_col]
    return grouped


def validate_month_glory(season: int, league_id: int, pack_lines: List[str], root: Path) -> Tuple[bool, str]:
    games_csv = root / "csv" / "out" / "almanac" / str(season) / f"games_{season}_league{league_id}_by_team.csv"
    if not games_csv.exists():
        logger.warning("Games-by-team CSV missing; skipping Month-of-Glory validation.")
        return False, "Skipped (missing games CSV)"
    section = extract_section(pack_lines, "## 3k View – Flashback Story Candidates")
    if not section:
        return False, "Flashback section not found."
    top_entry = parse_month_glory_top(section)
    if not top_entry:
        return False, "Could not parse top Month-of-Glory entry."

    splits = recompute_monthly_splits(games_csv)
    if splits.empty:
        return False, "Could not recompute monthly splits."

    # Find matching team/month
    mname_to_num = {name: idx for idx, name in enumerate(["", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])}
    mnum = mname_to_num.get(top_entry["month"], None)
    if mnum is None:
        return False, "Month name not recognized."
    match = splits[(splits["team_label_raw"] == top_entry["team_label_raw"]) & (splits["month"] == mnum)]
    if match.empty:
        logger.error("[FAIL] Month-of-Glory top entry not found in recomputed splits: %s", top_entry)
        return False, "Month-of-Glory mismatch."
    row = match.iloc[0]
    if (
        row["W"] == top_entry["w"]
        and row["L"] == top_entry["l"]
        and abs(row["month_pct"] - top_entry["pct"]) <= 0.001
        and abs(row["delta"] - top_entry["delta"]) <= 0.001
    ):
        logger.info("[OK] Month-of-Glory top entry matches monthly splits for season %s.", season)
        return True, "PASS"
    logger.error("[FAIL] Month-of-Glory mismatch: pack=%s recomputed=%s", top_entry, {"W": row["W"], "L": row["L"], "pct": row["month_pct"], "delta": row["delta"]})
    return False, "Month-of-Glory mismatch."


# -----------------------------
# Main orchestration
# -----------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Validate EB pack against almanac sources.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--league-id", type=int, default=200)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    season = args.season
    league_id = args.league_id
    root = Path(__file__).resolve().parents[2]

    pack_path = root / "csv" / "out" / "eb" / f"eb_regular_season_pack_{season}_league{league_id}.md"
    if not pack_path.exists():
        logger.error("EB pack markdown not found: %s", pack_path)
        return 1
    pack_lines = load_pack_md(pack_path)

    results = {}

    try:
        ok, msg = validate_preseason_hype(season, league_id, pack_lines, root)
        results["Preseason hype"] = "PASS" if ok else msg
    except Exception as e:
        logger.exception("Preseason hype validation error: %s", e)
        results["Preseason hype"] = "ERROR"

    try:
        ok, msg = validate_schedule_context(season, league_id, pack_lines, root)
        results["Schedule context"] = "PASS" if ok else msg
    except Exception as e:
        logger.exception("Schedule context validation error: %s", e)
        results["Schedule context"] = "ERROR"

    try:
        ok, msg = validate_month_glory(season, league_id, pack_lines, root)
        results["Month-of-Glory/Misery"] = "PASS" if ok else msg
    except Exception as e:
        logger.exception("Month-of-Glory validation error: %s", e)
        results["Month-of-Glory/Misery"] = "ERROR"

    # Summary
    summary_lines = [f"[SUMMARY] Season {season}, league {league_id}:"]
    for k, v in results.items():
        summary_lines.append(f"  - {k}: {v}")
    print("\n".join(summary_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

