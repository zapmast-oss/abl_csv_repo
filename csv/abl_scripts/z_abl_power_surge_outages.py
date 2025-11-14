"""ABL Power Surge & Outage tracker."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

TEAM_MIN, TEAM_MAX = 1, 24

LOG_CANDIDATES = [
    "team_game_log.csv",
    "team_batting_gamelog.csv",
    "games_team_batting.csv",
    "team_batting_stats.csv",
]
BOX_CANDIDATES = [
    "team_box_batting.csv",
    "batting_box_by_team.csv",
]
GAMES_CANDIDATES = [
    "schedule.csv",
    "games.csv",
]
TEAM_INFO_CANDIDATES = [
    "team_info.csv",
    "teams.csv",
    "standings.csv",
]
PARK_CANDIDATES = [
    "parks.csv",
    "park_info.csv",
]


def pick_column(df: pd.DataFrame, *names: str) -> Optional[str]:
    lowered = {col.lower(): col for col in df.columns}
    for name in names:
        key = name.lower()
        if key in lowered:
            return lowered[key]
    return None


def read_first(base: Path, override: Optional[Path], candidates: Sequence[str]) -> Optional[pd.DataFrame]:
    if override:
        path = override
        if not path.exists():
            raise FileNotFoundError(f"Specified file not found: {path}")
        return pd.read_csv(path)
    for name in candidates:
        path = base / name
        if path.exists():
            return pd.read_csv(path)
    return None


def resolve_path(base: Path, value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path


def load_team_names(base: Path, override: Optional[Path]) -> Dict[int, str]:
    df = read_first(base, override, TEAM_INFO_CANDIDATES)
    if df is None:
        return {}
    team_col = pick_column(df, "team_id", "teamid", "TeamID")
    name_col = pick_column(df, "team_display", "team_name", "name", "abbr")
    names: Dict[int, str] = {}
    if not team_col:
        return names
    for _, row in df.iterrows():
        tid_val = row.get(team_col)
        if pd.isna(tid_val):
            continue
        try:
            tid = int(tid_val)
        except (TypeError, ValueError):
            continue
        if TEAM_MIN <= tid <= TEAM_MAX and name_col and pd.notna(row.get(name_col)):
            names[tid] = str(row.get(name_col))
    return names


def load_park_names(base: Path, override: Optional[Path]) -> Dict[str, str]:
    df = read_first(base, override, PARK_CANDIDATES)
    if df is None:
        return {}
    park_col = pick_column(df, "park_id", "ParkID", "park")
    name_col = pick_column(df, "park_name", "name")
    parks: Dict[str, str] = {}
    if not park_col:
        return parks
    for _, row in df.iterrows():
        pid = row.get(park_col)
        if pd.isna(pid):
            continue
        if name_col and pd.notna(row.get(name_col)):
            parks[str(pid)] = str(row.get(name_col))
    return parks


def load_logs(base: Path, override_logs: Optional[Path], override_boxes: Optional[Path], override_games: Optional[Path]) -> pd.DataFrame:
    logs = read_first(base, override_logs, LOG_CANDIDATES)
    if logs is not None:
        team_col = pick_column(logs, "team_id", "teamid")
        date_col = pick_column(logs, "game_date", "date")
        park_col = pick_column(logs, "park_id", "park")
        hr_col = pick_column(logs, "hr", "HR")
        pa_col = pick_column(logs, "pa", "PA")
        if not team_col or not date_col or not hr_col:
            logs = None
        else:
            data = logs.copy()
            data["team_id"] = pd.to_numeric(data[team_col], errors="coerce").astype("Int64")
            data = data.dropna(subset=["team_id"])
            data = data[(data["team_id"] >= TEAM_MIN) & (data["team_id"] <= TEAM_MAX)]
            data["game_date"] = pd.to_datetime(data[date_col])
            data["park_id"] = data[park_col].astype(str) if park_col else ""
            if pa_col:
                data["PA"] = pd.to_numeric(data[pa_col], errors="coerce")
            else:
                ab_col = pick_column(logs, "ab")
                bb_col = pick_column(logs, "bb")
                hbp_col = pick_column(logs, "hbp", "hp")
                sf_col = pick_column(logs, "sf")
                sh_col = pick_column(logs, "sh")
                pa = (
                    pd.to_numeric(data[ab_col], errors="coerce").fillna(0)
                    + pd.to_numeric(data[bb_col], errors="coerce").fillna(0)
                    + pd.to_numeric(data[hbp_col], errors="coerce").fillna(0)
                    + pd.to_numeric(data[sf_col], errors="coerce").fillna(0)
                    + pd.to_numeric(data[sh_col], errors="coerce").fillna(0)
                )
                data["PA"] = pa
            data["HR"] = pd.to_numeric(data[hr_col], errors="coerce").fillna(0)
            return data[["team_id", "game_date", "park_id", "HR", "PA"]]
    boxes = read_first(base, override_boxes, BOX_CANDIDATES)
    games = read_first(base, override_games, GAMES_CANDIDATES)
    if boxes is None or games is None:
        raise FileNotFoundError("Unable to find suitable logs/boxes+games data.")
    team_col = pick_column(boxes, "team_id", "teamid")
    date_col = pick_column(boxes, "game_date", "date")
    hr_col = pick_column(boxes, "hr", "HR")
    pa_col = pick_column(boxes, "pa", "PA")
    game_id_col = pick_column(boxes, "game_id", "game_key")
    if not team_col or not date_col or not hr_col or not game_id_col:
        raise ValueError("Box file missing key columns.")
    box_data = boxes.copy()
    box_data["team_id"] = pd.to_numeric(boxes[team_col], errors="coerce").astype("Int64")
    box_data = box_data.dropna(subset=["team_id"])
    box_data = box_data[(box_data["team_id"] >= TEAM_MIN) & (box_data["team_id"] <= TEAM_MAX)]
    box_data["game_id"] = boxes[game_id_col].astype(str)
    box_data["game_date"] = pd.to_datetime(boxes[date_col])
    if pa_col:
        box_data["PA"] = pd.to_numeric(boxes[pa_col], errors="coerce")
    else:
        ab_col = pick_column(boxes, "ab")
        bb_col = pick_column(boxes, "bb")
        hbp_col = pick_column(boxes, "hbp", "hp")
        sf_col = pick_column(boxes, "sf")
        sh_col = pick_column(boxes, "sh")
        pa = (
            pd.to_numeric(box_data[ab_col], errors="coerce").fillna(0)
            + pd.to_numeric(box_data[bb_col], errors="coerce").fillna(0)
            + pd.to_numeric(box_data[hbp_col], errors="coerce").fillna(0)
            + pd.to_numeric(box_data[sf_col], errors="coerce").fillna(0)
            + pd.to_numeric(box_data[sh_col], errors="coerce").fillna(0)
        )
        box_data["PA"] = pa
    box_data["HR"] = pd.to_numeric(box_data[hr_col], errors="coerce").fillna(0)

    game_park_col = pick_column(games, "park_id", "park")
    game_id_col2 = pick_column(games, "game_id", "game_key")
    if not game_park_col or not game_id_col2:
        raise ValueError("Games file missing park info.")
    game_info = games[[game_id_col2, game_park_col]].copy()
    game_info.columns = ["game_id", "park_id"]
    merged = box_data.merge(game_info, on="game_id", how="left")
    merged["park_id"] = merged["park_id"].astype(str).fillna("")
    return merged[["team_id", "game_date", "park_id", "HR", "PA"]]


def determine_weeks(dates: pd.Series, week_end: Optional[str]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    if week_end:
        week_end_date = pd.to_datetime(week_end)
    else:
        max_date = dates.max()
        dow = max_date.weekday()
        offset = dow - 6
        if offset < 0:
            offset += 7
        week_end_date = max_date - timedelta(days=offset)
    week_end_date = week_end_date.normalize()
    week_start_date = week_end_date - timedelta(days=6)
    prior_end = week_start_date - timedelta(days=1)
    prior_start = prior_end - timedelta(days=6)
    return (week_start_date, week_end_date), (prior_start, prior_end)


def aggregate_week(data: pd.DataFrame, key_cols: Sequence[str], week_start: pd.Timestamp, week_end: pd.Timestamp) -> pd.DataFrame:
    mask = (data["game_date"] >= week_start) & (data["game_date"] <= week_end)
    subset = data.loc[mask].copy()
    subset["games"] = 1
    agg = subset.groupby(list(key_cols), as_index=False).agg(
        games=("games", "sum"),
        HR=("HR", "sum"),
        PA=("PA", "sum"),
    )
    agg["HR_per_PA"] = agg.apply(lambda r: r["HR"] / r["PA"] if r["PA"] > 0 else np.nan, axis=1)
    agg["week_start"] = week_start
    agg["week_end"] = week_end
    return agg


def merge_weeks(current: pd.DataFrame, prior: pd.DataFrame, key_cols: Sequence[str]) -> pd.DataFrame:
    cols = list(key_cols) + ["week_start", "week_end", "games", "HR", "PA", "HR_per_PA"]
    current = current[cols]
    prior = prior[list(key_cols) + ["games", "HR", "PA", "HR_per_PA"]].rename(
        columns={
            "games": "games_prev",
            "HR": "HR_prev",
            "PA": "PA_prev",
            "HR_per_PA": "HR_per_PA_prev",
        }
    )
    merged = current.merge(prior, on=list(key_cols), how="left")
    merged["delta_HR_per_PA"] = merged["HR_per_PA"] - merged["HR_per_PA_prev"]
    merged["pct_change"] = merged.apply(
        lambda r: (r["delta_HR_per_PA"] / r["HR_per_PA_prev"]) if pd.notna(r["HR_per_PA_prev"]) and r["HR_per_PA_prev"] != 0 else np.nan,
        axis=1,
    )
    merged["surge_flag"] = merged["delta_HR_per_PA"].apply(
        lambda d: "SURGE" if pd.notna(d) and d >= 0.005 else ("OUTAGE" if pd.notna(d) and d <= -0.005 else "")
    )
    return merged


def text_table(
    df: pd.DataFrame,
    columns: Sequence[Tuple[str, str, int, bool, str]],
    title: str,
    subtitle: str,
    key_lines: Sequence[str],
    def_lines: Sequence[str],
) -> str:
    lines = [title, "=" * len(title), subtitle, ""]
    header = " ".join(
        f"{label:<{width}}" if not align_right else f"{label:>{width}}"
        for label, _, width, align_right, _ in columns
    )
    lines.append(header)
    lines.append("-" * len(header))
    if df.empty:
        lines.append("(No qualifiers.)")
    else:
        for _, row in df.iterrows():
            parts = []
            for _, col_name, width, align_right, fmt in columns:
                value = row.get(col_name, "")
