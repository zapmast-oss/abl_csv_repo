"""Build EB WAR leaders (top 3 batters and pitchers per conference) for 1981.

Assumptions:
- Season-to-date WAR is used as proxy leaders (no week-split exports available).
- Hitters: abl_statistics_player_statistics_-_sortable_stats_player_bat_stats.csv (WAR).
- Pitchers: abl_statistics_player_statistics_-_sortable_stats_player_pitch_stats_2.csv (WAR)
  merged with pitch_stats_1 for ERA/IP context.
- Conference (ABC/NBC) derived from dim_team_park.csv via team Abbr -> SL.

Output: csv/out/text_out/eb_war_leaders_1981.json with shape:
{
  "ABC": {"bat": [...], "pitch": [...]},
  "NBC": {"bat": [...], "pitch": [...]}
}
Each entry includes rank, player_name, team_abbr, team_name, war, position/role, line.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[2]
CSV_DIR = ROOT / "csv"
STATS_DIR = CSV_DIR / "abl_statistics"
STAR_DIR = CSV_DIR / "out" / "star_schema"
TEXT_OUT_DIR = CSV_DIR / "out" / "text_out"

BAT_STATS_PATH = STATS_DIR / "abl_statistics_player_statistics_-_sortable_stats_player_bat_stats.csv"
PITCH_STATS1_PATH = STATS_DIR / "abl_statistics_player_statistics_-_sortable_stats_player_pitch_stats_1.csv"
PITCH_STATS2_PATH = STATS_DIR / "abl_statistics_player_statistics_-_sortable_stats_player_pitch_stats_2.csv"
DIM_TEAM_PARK_PATH = STAR_DIR / "dim_team_park.csv"
OUT_PATH = TEXT_OUT_DIR / "eb_war_leaders_1981.json"


def find_header_line(path: Path) -> int:
    for idx, line in enumerate(path.read_text().splitlines(), start=1):
        if line.strip() and not line.lstrip().startswith("#"):
            return idx
    raise SystemExit(f"Could not locate header row in {path}")


def load_dim_team() -> Dict[str, Tuple[str, str, int]]:
    if not DIM_TEAM_PARK_PATH.exists():
        raise SystemExit(f"dim_team_park.csv missing at {DIM_TEAM_PARK_PATH}")
    dim = pd.read_csv(DIM_TEAM_PARK_PATH)
    id_col = None
    for cand in ["ID", "team_id", "Team ID"]:
        if cand in dim.columns:
            id_col = cand
            break
    if id_col is None:
        raise SystemExit("dim_team_park missing ID column")
    abbr_col = "Abbr" if "Abbr" in dim.columns else None
    if not abbr_col:
        for col in dim.columns:
            if "abbr" in col.lower():
                abbr_col = col
                break
    if not abbr_col:
        raise SystemExit("dim_team_park missing Abbr column")
    name_col = None
    for cand in ["Team Name", "Name"]:
        if cand in dim.columns:
            name_col = cand
            break
    if name_col is None:
        raise SystemExit("dim_team_park missing team name column")
    sl_col = None
    for cand in ["SL", "sub_league", "league"]:
        if cand in dim.columns:
            sl_col = cand
            break
    if sl_col is None:
        raise SystemExit("dim_team_park missing SL/sub_league column")

    dim = dim[[id_col, abbr_col, name_col, sl_col]].rename(
        columns={id_col: "team_id", abbr_col: "team_abbr", name_col: "team_name", sl_col: "sub_league"}
    )
    dim["team_abbr"] = dim["team_abbr"].astype(str).str.strip()
    dim["sub_league"] = dim["sub_league"].astype(str).str.strip().str.upper()
    return {
        row["team_abbr"]: (row["team_name"], row["sub_league"], int(row["team_id"]))
        for _, row in dim.iterrows()
    }


def load_batters(abbr_map: Dict[str, Tuple[str, str, int]]) -> pd.DataFrame:
    h = find_header_line(BAT_STATS_PATH)
    df = pd.read_csv(BAT_STATS_PATH, skiprows=h - 1)
    df["team_abbr"] = df["TM"].astype(str).str.strip()
    df = df[df["team_abbr"].isin(abbr_map.keys())].copy()
    df["team_name"] = df["team_abbr"].map(lambda ab: abbr_map[ab][0])
    df["sub_league"] = df["team_abbr"].map(lambda ab: abbr_map[ab][1])
    df["team_id"] = df["team_abbr"].map(lambda ab: abbr_map[ab][2])
    df["WAR"] = pd.to_numeric(df["WAR"], errors="coerce").fillna(0)
    return df


def load_pitchers(abbr_map: Dict[str, Tuple[str, str, int]]) -> pd.DataFrame:
    h1 = find_header_line(PITCH_STATS1_PATH)
    h2 = find_header_line(PITCH_STATS2_PATH)
    p1 = pd.read_csv(PITCH_STATS1_PATH, skiprows=h1 - 1)
    p2 = pd.read_csv(PITCH_STATS2_PATH, skiprows=h2 - 1)
    merge_keys = [k for k in ["ID", "Name", "TM", "ORG", "POS"] if k in p1.columns and k in p2.columns]
    merged = p2.merge(p1, on=merge_keys, how="left", suffixes=("_p2", "_p1"))
    merged["team_abbr"] = merged["TM"].astype(str).str.strip()
    merged = merged[merged["team_abbr"].isin(abbr_map.keys())].copy()
    merged["team_name"] = merged["team_abbr"].map(lambda ab: abbr_map[ab][0])
    merged["sub_league"] = merged["team_abbr"].map(lambda ab: abbr_map[ab][1])
    merged["team_id"] = merged["team_abbr"].map(lambda ab: abbr_map[ab][2])
    for col in ["WAR", "ERA", "FIP", "IP", "K/BB"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
    return merged


def top_war(df: pd.DataFrame, league: str, role: str, top_n: int = 3) -> List[dict]:
    subset = df[df["sub_league"] == league].copy()
    subset = subset.sort_values("WAR", ascending=False).head(top_n)
    records = []
    for rank, (_, row) in enumerate(subset.iterrows(), start=1):
        entry = {
            "conference": league,
            "role": role,
            "rank": rank,
            "player_id": int(row["ID"]),
            "player_name": str(row["Name"]),
            "team_id": int(row["team_id"]),
            "team_abbr": str(row["team_abbr"]),
            "team_name": str(row["team_name"]),
            "war": float(row.get("WAR", 0) or 0),
            "position": str(row.get("POS", "")),
        }
        if role == "batter":
            entry["line"] = f"OPS {row.get('OPS', 0):.3f}, HR {int(row.get('HR', 0) or 0)}, RBI {int(row.get('RBI', 0) or 0)}"
        else:
            entry["line"] = f"ERA {row.get('ERA', 0):.2f}, FIP {row.get('FIP', 0):.2f}, IP {row.get('IP', 0):.1f}"
        records.append(entry)
    return records


def build_payload(bat: pd.DataFrame, pitch: pd.DataFrame) -> Dict[str, dict]:
    payload: Dict[str, dict] = {}
    for league in ["ABC", "NBC"]:
        payload[league] = {
            "bat": top_war(bat, league, role="batter"),
            "pitch": top_war(pitch, league, role="pitcher"),
        }
    return payload


def main() -> None:
    abbr_map = load_dim_team()
    bat = load_batters(abbr_map)
    pitch = load_pitchers(abbr_map)

    payload = build_payload(bat, pitch)
    TEXT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

    print(f"[INFO] WAR leaders written to {OUT_PATH}")
    for lg, sections in payload.items():
        bat_top = sections.get("bat", [])
        pitch_top = sections.get("pitch", [])
        if bat_top:
            print(f"  {lg} bat leader: {bat_top[0]['player_name']} ({bat_top[0]['team_abbr']}) WAR {bat_top[0]['war']:.1f}")
        if pitch_top:
            print(f"  {lg} pitch leader: {pitch_top[0]['player_name']} ({pitch_top[0]['team_abbr']}) WAR {pitch_top[0]['war']:.1f}")


if __name__ == "__main__":
    main()
