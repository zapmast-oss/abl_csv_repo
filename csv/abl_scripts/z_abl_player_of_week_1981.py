"""Build EB Players of the Week for ABC and NBC (season-to-date proxy).

Assumptions:
- Uses season-to-date WAR as a proxy for weekly dominance due to lack of
  week-split exports in the provided CSVs. Once weekly splits are available,
  replace the selection metric accordingly.
- Hitters: abl_statistics_player_statistics_-_sortable_stats_player_bat_stats.csv
  (WAR column present after skipping header commentary).
- Pitchers: abl_statistics_player_statistics_-_sortable_stats_player_pitch_stats_2.csv
  for WAR, merged with pitch_stats_1 for ERA/WHIP/IP context.
- Conference (ABC/NBC) derived from dim_team_park.csv via team Abbr -> SL.

Output: csv/out/text_out/eb_player_of_week_1981.json
Structure:
{
  "ABC": { ... },
  "NBC": { ... }
}
Each entry carries league, player_name, team_abbr, team_name, position,
slash_line, line, note, player_id, team_id.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

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
OUT_PATH = TEXT_OUT_DIR / "eb_player_of_week_1981.json"


def find_header_line(path: Path) -> int:
    """Return 1-based header line index (first non-comment/non-empty)."""
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
    header_line = find_header_line(BAT_STATS_PATH)
    df = pd.read_csv(BAT_STATS_PATH, skiprows=header_line - 1)
    keep_cols = ["ID", "Name", "POS", "TM", "WAR", "AVG", "OBP", "SLG", "OPS", "HR", "RBI"]
    df = df[[c for c in keep_cols if c in df.columns]].copy()
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
    for col in ["WAR", "ERA", "WHIP", "IP", "FIP", "K/BB"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
    return merged


def pick_player_of_week(bat: pd.DataFrame, pitch: pd.DataFrame) -> Dict[str, dict]:
    result: Dict[str, dict] = {}
    for league in ["ABC", "NBC"]:
        bat_top = bat[bat["sub_league"] == league].sort_values("WAR", ascending=False).head(1)
        pitch_top = pitch[pitch["sub_league"] == league].sort_values("WAR", ascending=False).head(1)

        bat_row = bat_top.iloc[0] if not bat_top.empty else None
        pitch_row = pitch_top.iloc[0] if not pitch_top.empty else None

        use_pitch = False
        if pitch_row is not None:
            if bat_row is None or float(pitch_row.get("WAR", 0) or 0) > float(bat_row.get("WAR", 0) or 0):
                use_pitch = True

        if use_pitch and pitch_row is not None:
            entry = {
                "league": league,
                "player_id": int(pitch_row["ID"]),
                "player_name": str(pitch_row["Name"]),
                "position": str(pitch_row.get("POS", "")),
                "team_id": int(pitch_row["team_id"]),
                "team_abbr": str(pitch_row["team_abbr"]),
                "team_name": str(pitch_row["team_name"]),
                "slash_line": f"{pitch_row.get('ERA', 0):.2f} ERA, WHIP {pitch_row.get('WHIP', 0):.2f}",
                "line": f"{pitch_row.get('IP', 0):.1f} IP, FIP {pitch_row.get('FIP', 0):.2f}, WAR {pitch_row.get('WAR', 0):.1f}",
                "note": "Top pitcher by WAR (season-to-date proxy).",
            }
        elif bat_row is not None:
            entry = {
                "league": league,
                "player_id": int(bat_row["ID"]),
                "player_name": str(bat_row["Name"]),
                "position": str(bat_row.get("POS", "")),
                "team_id": int(bat_row["team_id"]),
                "team_abbr": str(bat_row["team_abbr"]),
                "team_name": str(bat_row["team_name"]),
                "slash_line": f"{bat_row.get('AVG', 0):.3f}/{bat_row.get('OBP', 0):.3f}/{bat_row.get('SLG', 0):.3f}",
                "line": f"{int(bat_row.get('HR', 0) or 0)} HR, {int(bat_row.get('RBI', 0) or 0)} RBI, OPS {bat_row.get('OPS', 0):.3f}, WAR {bat_row.get('WAR', 0):.1f}",
                "note": "Top hitter by WAR (season-to-date proxy).",
            }
        else:
            continue

        result[league] = entry

    return result


def main() -> None:
    abbr_map = load_dim_team()
    bat = load_batters(abbr_map)
    pitch = load_pitchers(abbr_map)

    pofw = pick_player_of_week(bat, pitch)

    TEXT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(pofw, f, indent=2, ensure_ascii=False, default=str)

    print(f"[INFO] Players of the Week written to {OUT_PATH}")
    for lg, entry in pofw.items():
        print(f"  {lg}: {entry['player_name']} ({entry['team_abbr']}) – {entry['line']}")


if __name__ == "__main__":
    main()
