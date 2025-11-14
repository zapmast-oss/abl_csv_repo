"""ABL Milestones On The Horizon: highlight players nearing career marks."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from abl_config import stamp_text_block

TEAM_MIN, TEAM_MAX = 1, 24

PLAYER_FILES = [
    "players.csv",
    "player_register.csv",
    "rosters.csv",
]
TEAM_FILES = [
    "team_info.csv",
    "teams.csv",
    "standings.csv",
    "team_record.csv",
]
BATTING_FILES = [
    "players_career_batting_stats.csv",
    "players_batting_career.csv",
    "batting_career.csv",
]
PITCHING_FILES = [
    "players_career_pitching_stats.csv",
    "players_pitching_career.csv",
    "pitching_career.csv",
]
BATTING_LOG_FILES = [
    "players_game_batting.csv",
    "players_game_batting_stats.csv",
]
PITCHING_LOG_FILES = [
    "players_game_pitching_stats.csv",
    "players_pitching_gamelog.csv",
]

HITTER_MILESTONES: Dict[str, List[int]] = {
    "H": [1000, 1500],
    "RBI": [500, 750, 900],
    "R": [500, 750, 1000],
    "2B": [300, 350],
    "3B": [50, 75],
    "SB": [200, 300, 400, 500],
    "BB": [500, 750, 1000],
}
PITCHER_MILESTONES: Dict[str, List[int]] = {
    "SO": [1000, 1500],
    "W": [100, 125, 150],
    "SV": [100, 150, 200],
    "IP": [2000, 2500, 3000, 3500],
}
HR_BIG_MARKS = [200, 250, 300]
STAT_LABELS = {
    "HR": "HR",
    "H": "Hit",
    "RBI": "RBI",
    "R": "Run",
    "2B": "Double",
    "3B": "Triple",
    "SB": "SB",
    "BB": "Walk",
    "SO": "Strikeout",
    "W": "Win",
    "SV": "Save",
    "IP": "IP",
}


def pick_column(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    lowered = {col.lower(): col for col in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in lowered:
            return lowered[key]
    return None


def read_first(base: Path, candidates: Sequence[str]) -> Optional[pd.DataFrame]:
    for name in candidates:
        path = base / name
        if path.exists():
            return pd.read_csv(path)
    return None


def load_games(base: Path) -> Optional[pd.DataFrame]:
    path = base / "games.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, usecols=["game_id", "date"])
    df["game_id"] = df["game_id"].astype(str)
    df["game_date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["game_date"])
    return df[["game_id", "game_date"]]


def load_team_info(base: Path) -> Tuple[
    Dict[int, str], Dict[int, str], Dict[int, str], Dict[int, int]
]:
    df = read_first(base, TEAM_FILES)
    if df is None:
        return {}, {}, {}, {}
    team_col = pick_column(df, "team_id", "teamid", "teamID", "TeamID")
    name_col = pick_column(df, "team_display", "team_name", "name", "TeamName")
    abbr_col = pick_column(df, "abbr", "team_abbr")
    div_col = pick_column(df, "division_id", "DivisionID", "div_id")
    sub_col = pick_column(df, "sub_league_id", "subleague_id", "subleague", "sub_id")
    league_col = pick_column(df, "league_id", "LeagueID")
    if not team_col:
        return {}, {}, {}, {}
    meta = pd.DataFrame()
    meta["team_id"] = pd.to_numeric(df[team_col], errors="coerce").astype("Int64")
    meta = meta[(meta["team_id"] >= TEAM_MIN) & (meta["team_id"] <= TEAM_MAX)]
    if name_col:
        meta["team_display"] = df[name_col].fillna("")
    else:
        meta["team_display"] = ""
    if abbr_col:
        meta["team_abbr"] = df[abbr_col].fillna("")
    else:
        meta["team_abbr"] = ""
    if div_col:
        meta["division_id"] = pd.to_numeric(df[div_col], errors="coerce").astype("Int64")
    else:
        meta["division_id"] = pd.NA
    if sub_col:
        meta["sub_league_id"] = pd.to_numeric(df[sub_col], errors="coerce").astype("Int64")
    else:
        meta["sub_league_id"] = pd.NA
    if league_col:
        meta["league_id"] = pd.to_numeric(df[league_col], errors="coerce").astype("Int64")
    else:
        meta["league_id"] = pd.NA
    conf_map = {0: "N", 1: "A"}
    div_map = {0: "E", 1: "C", 2: "W"}
    meta["conf_div"] = (
        meta["sub_league_id"].map(conf_map).fillna("")
        + "-"
        + meta["division_id"].map(div_map).fillna("")
    ).str.strip("-")
    names = meta.set_index("team_id")["team_display"].to_dict()
    confs = meta.set_index("team_id")["conf_div"].to_dict()
    abbrs = meta.set_index("team_id")["team_abbr"].to_dict()
    leagues = meta.set_index("team_id")["league_id"].to_dict()
    return names, confs, abbrs, leagues


def load_players(base: Path) -> pd.DataFrame:
    df = read_first(base, PLAYER_FILES)
    if df is None:
        raise FileNotFoundError("Unable to locate player master file.")
    pid_col = pick_column(df, "player_id", "PlayerID", "pid")
    team_col = pick_column(df, "team_id", "teamID", "TeamID", "current_team_id")
    if not pid_col or not team_col:
        raise ValueError("Player master requires player_id and team_id columns.")
    data = pd.DataFrame()
    data["player_id"] = pd.to_numeric(df[pid_col], errors="coerce").astype("Int64")
    data["team_id"] = pd.to_numeric(df[team_col], errors="coerce").astype("Int64")
    first_col = pick_column(df, "first_name", "firstname", "FirstName")
    last_col = pick_column(df, "last_name", "lastname", "LastName")
    full_col = pick_column(df, "name_full", "name", "player_name")
    status_col = pick_column(df, "status")
    level_col = pick_column(df, "level", "level_id")
    active_col = pick_column(df, "is_active")
    data["first_name"] = df[first_col].fillna("") if first_col else ""
    data["last_name"] = df[last_col].fillna("") if last_col else ""
    data["name_full"] = df[full_col].fillna("") if full_col else ""
    data["status"] = df[status_col].astype(str) if status_col else ""
    data["level"] = df[level_col].astype(str) if level_col else ""
    data["is_active"] = pd.to_numeric(df[active_col], errors="coerce") if active_col else np.nan
    data = data.dropna(subset=["player_id", "team_id"])
    data = data[(data["team_id"] >= TEAM_MIN) & (data["team_id"] <= TEAM_MAX)]
    if status_col:
        status_lower = data["status"].str.lower()
        inactive_mask = status_lower.str.contains("retir|inactive|il60|60-day|60 day", na=False)
        data = data[~inactive_mask]
    if active_col:
        data = data[(data["is_active"].isna()) | (data["is_active"] != 0)]
    return data.reset_index(drop=True)


def load_career_batting(base: Path, allowed_leagues: Optional[Set[int]] = None) -> pd.DataFrame:
    df = read_first(base, BATTING_FILES)
    if df is None:
        return pd.DataFrame(columns=["player_id"])
    pid_col = pick_column(df, "player_id", "PlayerID")
    if not pid_col:
        return pd.DataFrame(columns=["player_id"])
    df = df.copy()
    df["player_id"] = pd.to_numeric(df[pid_col], errors="coerce").astype("Int64")
    level_col = pick_column(df, "level_id", "level")
    if level_col:
        levels = pd.to_numeric(df[level_col], errors="coerce")
        df = df[levels == 1]
    league_col = pick_column(df, "league_id", "LeagueID")
    if allowed_leagues and league_col:
        leagues = pd.to_numeric(df[league_col], errors="coerce")
        df = df[leagues.isin(list(allowed_leagues))]
    split_col = pick_column(df, "split_id", "splitID")
    if split_col:
        splits = pd.to_numeric(df[split_col], errors="coerce")
        if (splits == 1).any():
            df = df[splits == 1]
        elif (splits == 0).any():
            df = df[splits == 0]
    stat_map = {
        "HR": ["hr", "HR"],
        "H": ["h", "H"],
        "RBI": ["rbi", "RBI"],
        "R": ["r", "R"],
        "2B": ["d", "D", "2b", "two_b"],
        "3B": ["t", "T", "3b", "three_b"],
        "SB": ["sb", "SB"],
        "BB": ["bb", "BB"],
    }
    for key, options in stat_map.items():
        col = pick_column(df, *options)
        df[key] = pd.to_numeric(df[col], errors="coerce").fillna(0) if col else 0
    grouped = df.groupby("player_id")[list(stat_map.keys())].sum().reset_index()
    return grouped


def load_career_pitching(base: Path, allowed_leagues: Optional[Set[int]] = None) -> pd.DataFrame:
    df = read_first(base, PITCHING_FILES)
    if df is None:
        return pd.DataFrame(columns=["player_id"])
    pid_col = pick_column(df, "player_id", "PlayerID")
    if not pid_col:
        return pd.DataFrame(columns=["player_id"])
    df = df.copy()
    df["player_id"] = pd.to_numeric(df[pid_col], errors="coerce").astype("Int64")
    level_col = pick_column(df, "level_id", "level")
    if level_col:
        levels = pd.to_numeric(df[level_col], errors="coerce")
        df = df[levels == 1]
    league_col = pick_column(df, "league_id", "LeagueID")
    if allowed_leagues and league_col:
        leagues = pd.to_numeric(df[league_col], errors="coerce")
        df = df[leagues.isin(list(allowed_leagues))]
    split_col = pick_column(df, "split_id", "splitID")
    if split_col:
        splits = pd.to_numeric(df[split_col], errors="coerce")
        if (splits == 1).any():
            df = df[splits == 1]
        elif (splits == 0).any():
            df = df[splits == 0]
    stat_map = {
        "SO": ["so", "SO", "k", "K"],
        "W": ["w", "W"],
        "SV": ["sv", "SV"],
    }
    for key, options in stat_map.items():
        col = pick_column(df, *options)
        df[key] = pd.to_numeric(df[col], errors="coerce").fillna(0) if col else 0
    ip_col = pick_column(df, "ip", "IP")
    outs_col = pick_column(df, "outs", "ip_outs")
    df["IP_raw"] = pd.to_numeric(df[ip_col], errors="coerce") if ip_col else np.nan
    if outs_col:
        outs_val = pd.to_numeric(df[outs_col], errors="coerce")
        df.loc[df["IP_raw"].isna() & outs_val.notna(), "IP_raw"] = outs_val / 3.0
    df["IP_raw"] = df["IP_raw"].fillna(0)
    grouped = (
        df.groupby("player_id")[["SO", "W", "SV", "IP_raw"]]
        .sum()
        .rename(columns={"IP_raw": "IP"})
        .reset_index()
    )
    return grouped


def compute_anchor_window(dates: pd.Series, days: int = 30) -> Tuple[pd.Timestamp, pd.Timestamp]:
    if dates.empty:
        raise ValueError("No dates available for pace computation.")
    anchor = dates.max()
    start = anchor - pd.Timedelta(days=days - 1)
    return start, anchor


def build_batting_pace(base: Path, games: Optional[pd.DataFrame]) -> pd.DataFrame:
    if games is None:
        return pd.DataFrame()
    df = read_first(base, BATTING_LOG_FILES)
    if df is None:
        return pd.DataFrame()
    pid_col = pick_column(df, "player_id", "PlayerID")
    team_col = pick_column(df, "team_id", "teamID")
    game_col = pick_column(df, "game_id", "GameID")
    if not pid_col or not team_col or not game_col:
        return pd.DataFrame()
    df = df.copy()
    df["player_id"] = pd.to_numeric(df[pid_col], errors="coerce").astype("Int64")
    df["team_id"] = pd.to_numeric(df[team_col], errors="coerce").astype("Int64")
    df = df[(df["team_id"] >= TEAM_MIN) & (df["team_id"] <= TEAM_MAX)]
    df["game_id"] = df[game_col].astype(str)
    df = df.merge(games, on="game_id", how="left")
    df = df.dropna(subset=["game_date"])
    if df.empty:
        return pd.DataFrame()
    start, anchor = compute_anchor_window(df["game_date"])
    window = df[(df["game_date"] >= start) & (df["game_date"] <= anchor)].copy()
    if window.empty:
        return pd.DataFrame()
    stat_map = {
        "HR": pick_column(df, "hr"),
        "H": pick_column(df, "h"),
        "RBI": pick_column(df, "rbi"),
        "R": pick_column(df, "r"),
        "2B": pick_column(df, "d", "2b"),
        "3B": pick_column(df, "t", "3b"),
        "SB": pick_column(df, "sb"),
        "BB": pick_column(df, "bb"),
    }
    for key, col in stat_map.items():
        if col:
            window[key] = pd.to_numeric(window[col], errors="coerce").fillna(0)
        else:
            window[key] = 0.0
    grouped = window.groupby("player_id")
    games_played = grouped["game_id"].nunique().rename("games_30")
    totals = grouped[list(stat_map.keys())].sum()
    pace = totals.div(games_played, axis=0)
    pace["games_30"] = games_played
    pace = pace.reset_index()
    return pace


def build_pitching_pace(base: Path, games: Optional[pd.DataFrame]) -> pd.DataFrame:
    if games is None:
        return pd.DataFrame()
    df = read_first(base, PITCHING_LOG_FILES)
    if df is None:
        return pd.DataFrame()
    pid_col = pick_column(df, "player_id", "PlayerID")
    team_col = pick_column(df, "team_id", "teamID")
    game_col = pick_column(df, "game_id", "GameID")
    if not pid_col or not team_col or not game_col:
        return pd.DataFrame()
    df = df.copy()
    df["player_id"] = pd.to_numeric(df[pid_col], errors="coerce").astype("Int64")
    df["team_id"] = pd.to_numeric(df[team_col], errors="coerce").astype("Int64")
    df = df[(df["team_id"] >= TEAM_MIN) & (df["team_id"] <= TEAM_MAX)]
    df["game_id"] = df[game_col].astype(str)
    df = df.merge(games, on="game_id", how="left")
    df = df.dropna(subset=["game_date"])
    if df.empty:
        return pd.DataFrame()
    so_col = pick_column(df, "so", "k")
    w_col = pick_column(df, "w")
    sv_col = pick_column(df, "sv", "s")
    ip_col = pick_column(df, "ip")
    outs_col = pick_column(df, "outs", "ip_outs")
    df["SO"] = pd.to_numeric(df[so_col], errors="coerce").fillna(0) if so_col else 0.0
    df["W"] = pd.to_numeric(df[w_col], errors="coerce").fillna(0) if w_col else 0.0
    df["SV"] = pd.to_numeric(df[sv_col], errors="coerce").fillna(0) if sv_col else 0.0
    df["IP_val"] = pd.to_numeric(df[ip_col], errors="coerce") if ip_col else np.nan
    if outs_col:
        outs_val = pd.to_numeric(df[outs_col], errors="coerce")
        df.loc[df["IP_val"].isna() & outs_val.notna(), "IP_val"] = outs_val / 3.0
    df["IP_val"] = df["IP_val"].fillna(0)
    per_game = (
        df.groupby(["player_id", "game_id"])
        [["SO", "W", "SV", "IP_val", "game_date"]]
        .agg({"SO": "sum", "W": "sum", "SV": "sum", "IP_val": "sum", "game_date": "first"})
        .reset_index()
    )
    start, anchor = compute_anchor_window(per_game["game_date"])
    window = per_game[(per_game["game_date"] >= start) & (per_game["game_date"] <= anchor)].copy()
    if window.empty:
        return pd.DataFrame()
    grouped = window.groupby("player_id")
    games_played = grouped["game_id"].nunique().rename("games_30")
    totals = grouped[["SO", "W", "SV", "IP_val"]].sum()
    pace = totals.div(games_played, axis=0)
    pace = pace.rename(columns={"IP_val": "IP"})
    pace["games_30"] = games_played
    pace = pace.reset_index()
    return pace


def build_pace_lookup(base: Path) -> Dict[Tuple[int, str], float]:
    games = load_games(base)
    pace_map: Dict[Tuple[int, str], float] = {}
    batting_pace = build_batting_pace(base, games)
    if not batting_pace.empty:
        for row in batting_pace.itertuples(index=False):
            for stat in ["HR", "H", "RBI", "R", "2B", "3B", "SB", "BB"]:
                val = getattr(row, stat, np.nan)
                if pd.notna(val) and val > 0:
                    pace_map[(int(row.player_id), stat)] = float(val)
    pitching_pace = build_pitching_pace(base, games)
    if not pitching_pace.empty:
        for row in pitching_pace.itertuples(index=False):
            for stat in ["SO", "W", "SV", "IP"]:
                val = getattr(row, stat, np.nan)
                if pd.notna(val) and val > 0:
                    pace_map[(int(row.player_id), stat)] = float(val)
    return pace_map


def build_player_name(row: pd.Series) -> str:
    first = str(row.get("first_name", "") or "").strip()
    last = str(row.get("last_name", "") or "").strip()
    full = str(row.get("name_full", "") or "").strip()
    if first or last:
        return f"{first} {last}".strip()
    if full:
        return full
    return f"Player {int(row['player_id'])}"


def ordinal(n: int) -> str:
    n = int(n)
    suffix = "th"
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        remainder = n % 10
        if remainder == 1:
            suffix = "st"
        elif remainder == 2:
            suffix = "nd"
        elif remainder == 3:
            suffix = "rd"
    return f"{n}{suffix}"


def next_target(current: float, milestones: Iterable[int]) -> Optional[int]:
    for mark in milestones:
        if mark > current:
            return int(mark)
    return None


def hr_targets(current: float) -> List[Tuple[int, str]]:
    current_int = int(math.floor(current))
    base = ((current_int // 10) + 1) * 10
    if current % 10 == 0:
        base = current_int + 10
    targets = []
    targets.append((base, f"next 10 HR (-> {base})"))
    big = next_target(current, HR_BIG_MARKS)
    if big:
        targets.append((big, f"{ordinal(big)} HR"))
    return targets


def fixed_target_label(stat_key: str, target: int) -> str:
    label = STAT_LABELS.get(stat_key, stat_key)
    return f"{ordinal(target)} {label}"


def build_milestones(row: pd.Series, within: int) -> List[Dict[str, object]]:
    stats = []
    hr_val = row.get("HR")
    if pd.notna(hr_val):
        for target, label in hr_targets(float(hr_val)):
            stats.append(("HR", float(hr_val), target, label))
    for key, marks in HITTER_MILESTONES.items():
        val = row.get(key)
        if pd.isna(val):
            continue
        target = next_target(float(val), marks)
        if target:
            stats.append((key, float(val), target, fixed_target_label(key, target)))
    for key, marks in PITCHER_MILESTONES.items():
        val = row.get(key)
        if pd.isna(val):
            continue
        target = next_target(float(val), marks)
        if target:
            label = fixed_target_label(key, target)
            stats.append((key, float(val), target, label))
    results: List[Dict[str, object]] = []
    seen = set()
    for stat_key, current, target, label in stats:
        raw_to_go = target - current
        if not (0 < raw_to_go <= within):
            continue
        dedup_key = (stat_key, target, label)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        if stat_key == "IP":
            current_display = round(current, 1)
            to_go_display = round(raw_to_go, 1)
        else:
            current_display = int(round(current))
            to_go_display = int(math.ceil(raw_to_go))
        results.append(
            {
                "stat_key": stat_key,
                "current_total": current_display,
                "current_raw": float(current),
                "target": int(target),
                "to_go": to_go_display,
                "to_go_raw": float(raw_to_go),
                "milestone_label": label,
            }
        )
    return results


def format_stat_display(stat_key: str, value: object) -> str:
    if value == "" or pd.isna(value):
        return ""
    if stat_key == "IP":
        return f"{float(value):.1f}"
    return f"{int(round(float(value)))}"


def build_report(base: Path, within: int, pace_lookup: Dict[Tuple[int, str], float]) -> pd.DataFrame:
    names_map, conf_div_map, abbr_map, league_map = load_team_info(base)
    allowed_leagues = {int(v) for v in league_map.values() if pd.notna(v)}
    players = load_players(base)
    batting = load_career_batting(base, allowed_leagues if allowed_leagues else None)
    pitching = load_career_pitching(base, allowed_leagues if allowed_leagues else None)
    data = players.merge(batting, on="player_id", how="left")
    data = data.merge(pitching, on="player_id", how="left", suffixes=("", "_pitch"))
    stat_columns = list(STAT_LABELS.keys())
    for col in stat_columns:
        if col not in data.columns:
            data[col] = 0.0
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0.0)
    data["team_display"] = data["team_id"].map(names_map).fillna("")
    data["conf_div"] = data["team_id"].map(conf_div_map).fillna("")
    data["team_abbr"] = data["team_id"].map(abbr_map).fillna("")
    data["player_name"] = data.apply(build_player_name, axis=1)
    rows: List[Dict[str, object]] = []
    for row in data.itertuples(index=False):
        milestone_entries = build_milestones(pd.Series(row._asdict()), within)
        for entry in milestone_entries:
            pace = pace_lookup.get((int(row.player_id), entry["stat_key"]), 0.0)
            proj = ""
            if pace and pace > 0:
                proj = int(math.ceil(entry["to_go_raw"] / pace))
            rows.append(
                {
                    "team_id": int(row.team_id),
                    "team_display": row.team_display,
                    "player_id": int(row.player_id),
                    "player_name": row.player_name,
                    "stat_key": entry["stat_key"],
                    "current_total": entry["current_total"],
                    "current_raw": entry["current_raw"],
                    "target": entry["target"],
                    "to_go": entry["to_go"],
                    "to_go_raw": entry["to_go_raw"],
                    "milestone_label": entry["milestone_label"],
                    "proj_games_to_target": proj,
                }
            )
    report = pd.DataFrame(rows)
    if report.empty:
        return report
    report = report.sort_values(
        by=["to_go", "milestone_label", "player_name"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    report["proj_games_to_target"] = report["proj_games_to_target"].replace(0, "")
    report["current_total"] = report.apply(lambda r: format_stat_display(r["stat_key"], r["current_total"]), axis=1)
    report["to_go"] = report.apply(lambda r: format_stat_display(r["stat_key"], r["to_go"]), axis=1)
    report["proj_games_to_target"] = report["proj_games_to_target"].apply(
        lambda x: "" if x == "" or pd.isna(x) else str(int(x))
    )
    report = report.drop(columns=["current_raw", "to_go_raw"], errors="ignore")
    return report


def build_text_report(df: pd.DataFrame, limit: int = 25, within: int = 5) -> str:
    lines = [
        "ABL Milestones On The Horizon",
        "=" * 32,
        f"Shows players within {within} of career benchmarks for quick shoutouts and broadcast nuggets.",
        "Handy for spotting imminent highlight moments before they happen.",
        "",
    ]
    header = (
        f"{'Player':<24} {'Team':<12} {'Milestone':<24} {'Current':>10} {'Target':>10} "
        f"{'Need':>8} {'ProjG':>6}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    rows = df.head(limit)
    if rows.empty:
        lines.append("No players within the configured window.")
    else:
        for _, row in rows.iterrows():
            lines.append(
                f"{row['player_name']:<24} {row['team_display']:<12} {row['milestone_label']:<24} "
                f"{row['current_total']:>10} {row['target']:>10} {row['to_go']:>8} "
                f"{(row['proj_games_to_target'] or ''):>6}"
            )
    lines.append("")
    lines.append("Need = difference to milestone; ProjG estimates games at recent pace when available.")
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Highlight ABL milestones on the horizon.")
    parser.add_argument("--base", type=str, default=".", help="Base directory for CSVs.")
    parser.add_argument(
        "--within",
        type=int,
        default=5,
        help="Include milestones where to-go is within this threshold.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="out/csv_out/z_ABL_Milestones_On_The_Horizon.csv",
        help="Output CSV path (default inside out/csv_out).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    base = Path(args.base).resolve()
    pace_lookup = build_pace_lookup(base)
    report = build_report(base, within=args.within, pace_lookup=pace_lookup)
    out_path = (base / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if report.empty:
        report = pd.DataFrame(
            columns=[
                "team_id",
                "team_display",
                "player_id",
                "player_name",
                "stat_key",
                "current_total",
                "target",
                "to_go",
                "milestone_label",
                "proj_games_to_target",
            ]
        )
    report.to_csv(out_path, index=False)
    text_filename = out_path.with_suffix(".txt").name
    if out_path.parent.name.lower() in {'csv_out'}:
        text_dir = out_path.parent.parent / "txt_out"
    else:
        text_dir = out_path.parent
    text_dir.mkdir(parents=True, exist_ok=True)
    text_path = text_dir / text_filename
    text_path.write_text(stamp_text_block(build_text_report(report, within=args.within)), encoding="utf-8")
    print("Milestones (top 25):")
    if report.empty:
        print("  No players currently within range.")
    else:
        print(report.head(25).to_string(index=False))
    print(f"\nWrote {len(report)} rows to {out_path} and summary to {text_path}.")


if __name__ == "__main__":
    main()

