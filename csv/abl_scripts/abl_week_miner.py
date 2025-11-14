"""Mine weekly ABL games for highlight-worthy events."""

from __future__ import annotations

import argparse
import math
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

LEAGUE_ID = 200
TEAM_MIN, TEAM_MAX = 1, 24
DATA_DIR = Path.cwd()

HERO_PITCH_THRESHOLD = 120
RELIEF_LONG_IP = 3.0
STREAK_THRESHOLD = 8
SKID_THRESHOLD = 7
GIANT_WIN_MAX = 0.45
GIANT_LOSE_MIN = 0.60
HIT_MILESTONES = [2000]
HR_MILESTONES = [200, 300]


def pick(df: pd.DataFrame, *names: str) -> Optional[str]:
    lowered = {col.lower(): col for col in df.columns}
    for name in names:
        key = name.lower()
        if key in lowered:
            return lowered[key]
    return None


def read_csv_smart(*names: str) -> Optional[pd.DataFrame]:
    variants = []
    for n in names:
        variants.extend(
            [
                n,
                n.lower(),
                n.upper(),
                n.replace(" ", "_"),
                n.replace("_", " "),
                n.title(),
                n.capitalize(),
            ]
        )
    seen = set()
    for candidate in variants:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        path = DATA_DIR / candidate
        if path.exists():
            return pd.read_csv(path)
    return None


def safe_int(value) -> Optional[int]:
    if pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def safe_float(value) -> Optional[float]:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class LineScoreHelper:
    """Pull per-inning line scores when the export provides them."""

    def __init__(self, df: pd.DataFrame):
        columns = list(df.columns)
        self.line_cols = [
            col
            for col in columns
            if "line" in col.lower() and "score" in col.lower()
        ]
        self.away_inning_cols = self._collect_inning_cols(
            columns, ["away", "visitor", "team0", "runs0", "score0"]
        )
        self.home_inning_cols = self._collect_inning_cols(
            columns, ["home", "team1", "runs1", "score1"]
        )
        self.available = bool(
            self.line_cols or self.away_inning_cols or self.home_inning_cols
        )

    @staticmethod
    def _collect_inning_cols(columns: List[str], tokens: List[str]) -> List[str]:
        candidates: List[Tuple[int, str]] = []
        for col in columns:
            lower = col.lower()
            if "inn" not in lower:
                continue
            if not any(token in lower for token in tokens):
                continue
            match = re.search(r"(\d+)", lower)
            if not match:
                continue
            candidates.append((int(match.group(1)), col))
        candidates.sort(key=lambda item: item[0])
        return [col for _, col in candidates]

    @staticmethod
    def _parse_line_score_sequences(value) -> List[List[int]]:
        if pd.isna(value):
            return []
        if isinstance(value, (list, tuple)):
            sequences: List[List[int]] = []
            for part in value:
                sequences.extend(LineScoreHelper._parse_line_score_sequences(part))
            return sequences
        text = str(value).strip()
        if not text:
            return []
        parts = [part.strip() for part in text.split("|")] if "|" in text else [text]
        sequences = []
        for part in parts:
            numbers = [int(chunk) for chunk in re.findall(r"-?\d+", part)]
            if numbers:
                sequences.append(numbers)
        return sequences

    def _values_from_cols(
        self, row: pd.Series, cols: List[str], expected_total: Optional[int]
    ) -> List[int]:
        if not cols:
            return []
        values: List[int] = []
        for col in cols:
            num = safe_int(row.get(col))
            values.append(num if num is not None else 0)
        if expected_total is None or sum(values) == expected_total:
            return values
        return []

    def _extract_from_line_strings(
        self,
        row: pd.Series,
        away_total: int,
        home_total: int,
        away_seq: List[int],
        home_seq: List[int],
    ) -> Tuple[List[int], List[int]]:
        if self.line_cols:
            for col in self.line_cols:
                sequences = self._parse_line_score_sequences(row.get(col))
                for seq in sequences:
                    total = sum(seq)
                    if not away_seq and total == away_total:
                        away_seq = seq
                    elif not home_seq and total == home_total:
                        home_seq = seq
                if away_seq and home_seq:
                    break
        return away_seq, home_seq

    def extract(
        self, row: pd.Series, away_total: int, home_total: int
    ) -> Optional[Tuple[List[int], List[int]]]:
        if not self.available:
            return None
        away_seq = self._values_from_cols(row, self.away_inning_cols, away_total)
        home_seq = self._values_from_cols(row, self.home_inning_cols, home_total)
        if (not away_seq or not home_seq) and self.line_cols:
            away_seq, home_seq = self._extract_from_line_strings(
                row, away_total, home_total, away_seq, home_seq
            )
        if away_seq and home_seq:
            return away_seq, home_seq
        return None


def pad_scores(values: List[int], length: int) -> List[int]:
    padded = list(values)
    if len(padded) < length:
        padded.extend([0] * (length - len(padded)))
    return padded[:length]


def cumulative_runs(values: List[int]) -> List[int]:
    totals: List[int] = []
    running = 0
    for val in values:
        running += val
        totals.append(running)
    return totals


def evaluate_line_events(
    away_line: List[int],
    home_line: List[int],
    home_win: bool,
    innings: float,
) -> Tuple[bool, bool]:
    if not away_line or not home_line:
        return False, False
    base_len = max(len(away_line), len(home_line))
    innings_len = 9
    if not pd.isna(innings):
        innings_len = max(int(math.ceil(float(innings))), 9)
    target_len = max(base_len, innings_len)
    away_seq = pad_scores(away_line, target_len)
    home_seq = pad_scores(home_line, target_len)
    away_cum = cumulative_runs(away_seq)
    home_cum = cumulative_runs(home_seq)
    winner = home_cum if home_win else away_cum
    loser = away_cum if home_win else home_cum
    comeback = any((loser_val - winner_val) >= 3 for winner_val, loser_val in zip(winner, loser))
    behind = any(
        inning >= 6 and winner_val < loser_val
        for inning, (winner_val, loser_val) in enumerate(zip(winner, loser), start=1)
    )
    return comeback, behind


def filter_abl(df: pd.DataFrame) -> pd.DataFrame:
    result = df
    league_col = pick(df, "league_id", "league", "lg_id")
    if league_col:
        result = result[result[league_col] == LEAGUE_ID]
    team_col = pick(df, "team_id", "teamid", "tid")
    if team_col:
        result = result[result[team_col].between(TEAM_MIN, TEAM_MAX)]
    return result


def infer_current_week(games_df: pd.DataFrame) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    if games_df is None or games_df.empty:
        return None
    date_col = pick(games_df, "date", "game_date")
    if not date_col:
        return None
    games_df["date"] = pd.to_datetime(games_df[date_col], errors="coerce")
    played_col = pick(games_df, "played", "completed")
    if played_col:
        mask_played = games_df[played_col].astype(int) != 0
    else:
        score_pairs = [
            ("runs0", "runs1"),
            ("score0", "score1"),
            ("away_score", "home_score"),
            ("home_score", "away_score"),
        ]
        mask_played = pd.Series(False, index=games_df.index)
        for a_col, b_col in score_pairs:
            if a_col in games_df.columns and b_col in games_df.columns:
                a_vals = pd.to_numeric(games_df[a_col], errors="coerce")
                b_vals = pd.to_numeric(games_df[b_col], errors="coerce")
                mask_pair = a_vals.notna() & b_vals.notna()
                mask_played = mask_played | mask_pair
    played_dates = games_df[mask_played]["date"]
    if played_dates.empty:
        return None
    last_played = played_dates.max()
    if pd.isna(last_played):
        return None
    start = last_played - pd.Timedelta(days=5)
    return start.normalize(), last_played.normalize()


def parse_dates(args, games_df: Optional[pd.DataFrame]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    if args.start and args.end:
        start = pd.to_datetime(args.start)
        end = pd.to_datetime(args.end)
        return start.normalize(), end.normalize()

    inferred = None
    if games_df is not None:
        games_df = games_df.copy()
        date_col = pick(games_df, "date", "game_date")
        if date_col:
            games_df["date"] = pd.to_datetime(games_df[date_col], errors="coerce")
            if args.mode == "sim":
                inferred = infer_sim_window(games_df.copy())
                if not inferred:
                    inferred = infer_last_played_window(games_df.copy())
            else:
                inferred = infer_last_played_window(games_df.copy())

    if inferred:
        return inferred

    end = pd.Timestamp(datetime.now().date()) - pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=5)
    return start.normalize(), end.normalize()


def compute_played_mask(games_df: pd.DataFrame) -> pd.Series:
    played_col = pick(games_df, "played", "completed")
    if played_col:
        return games_df[played_col].astype(int) != 0
    score_pairs = [
        ("runs0", "runs1"),
        ("score0", "score1"),
        ("away_score", "home_score"),
        ("home_score", "away_score"),
    ]
    mask = pd.Series(False, index=games_df.index)
    for a_col, b_col in score_pairs:
        if a_col in games_df.columns and b_col in games_df.columns:
            a_vals = pd.to_numeric(games_df[a_col], errors="coerce")
            b_vals = pd.to_numeric(games_df[b_col], errors="coerce")
            mask |= a_vals.notna() & b_vals.notna()
    return mask


def infer_last_played_window(games_df: pd.DataFrame) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    mask_played = compute_played_mask(games_df)
    played_dates = games_df[mask_played]["date"]
    if played_dates.empty:
        return None
    last_played = played_dates.max()
    if pd.isna(last_played):
        return None
    start = last_played - pd.Timedelta(days=5)
    return start.normalize(), last_played.normalize()


def infer_sim_window(games_df: pd.DataFrame) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    mask_played = compute_played_mask(games_df)
    unplayed = games_df[~mask_played]
    if unplayed.empty:
        return None
    next_date = unplayed["date"].min()
    if pd.isna(next_date):
        return None
    start = next_date - pd.Timedelta(days=6)
    end = next_date - pd.Timedelta(days=1)
    return start.normalize(), end.normalize()


def load_team_info() -> Dict[int, dict]:
    info: Dict[int, dict] = {}
    df = read_csv_smart("team_record.csv", "team_records.csv", "Team_Record.csv")
    if df is not None:
        df = filter_abl(df)
        team_col = pick(df, "team_id", "teamid", "tid")
        name_col = pick(df, "team_display", "team_name", "name", "nickname")
        conf_col = pick(df, "conference", "conf")
        div_col = pick(df, "division", "div")
        w_col = pick(df, "w", "wins")
        l_col = pick(df, "l", "losses")
        streak_col = pick(df, "streak")
        if team_col:
            for row in df.itertuples():
                row_d = row._asdict()
                tid = int(row_d[team_col])
                wins = float(row_d[w_col]) if w_col and row_d.get(w_col) is not None else 0.0
                losses = float(row_d[l_col]) if l_col and row_d.get(l_col) is not None else 0.0
                wpct = wins / (wins + losses) if (wins + losses) > 0 else 0.0
                info[tid] = {
                    "name": row_d.get(name_col) if name_col else f"Team {tid}",
                    "conference": row_d.get(conf_col, ""),
                    "division": row_d.get(div_col, ""),
                    "wpct": wpct,
                    "streak": row_d.get(streak_col, "") if streak_col else "",
                }
    teams_csv = read_csv_smart("teams.csv", "Teams.csv")
    if teams_csv is not None:
        teams_csv = filter_abl(teams_csv)
        id_col = pick(teams_csv, "team_id", "teamid", "tid")
        city_col = pick(teams_csv, "name", "city_name")
        abbr_col = pick(teams_csv, "abbr")
        display_col = pick(teams_csv, "team_display", "nickname")
        sub_col = pick(teams_csv, "sub_league_id", "sub_league")
        div_col = pick(teams_csv, "division_id", "division")
        if id_col:
            for row in teams_csv.itertuples():
                row_d = row._asdict()
                tid = int(row_d[id_col])
                entry = info.setdefault(
                    tid,
                    {
                        "name": f"Team {tid}",
                        "conference": "",
                        "division": "",
                        "wpct": 0.0,
                        "streak": "",
                    },
                )
                city_name = row_d.get(city_col)
                nickname = row_d.get(display_col)
                if city_name and nickname:
                    entry["name"] = f"{city_name} {nickname}"
                elif city_name:
                    entry["name"] = city_name
                elif nickname:
                    entry["name"] = nickname
                if abbr_col and row_d.get(abbr_col):
                    entry["abbr"] = row_d[abbr_col]
                if entry.get("conference") in ("", None):
                    if sub_col and row_d.get(sub_col) is not None:
                        entry["conference"] = "NBC" if int(row_d[sub_col]) == 0 else "ABC"
                if entry.get("division") in ("", None):
                    if div_col and row_d.get(div_col) is not None:
                        entry["division"] = str(row_d[div_col])
                if entry.get("streak") in ("", None):
                    entry["streak"] = ""
    return info


def load_career_batting_totals() -> Dict[int, Dict[str, int]]:
    df = read_csv_smart("players_career_batting_stats.csv")
    totals: Dict[int, Dict[str, int]] = {}
    if df is None or "player_id" not in df.columns:
        return totals
    grouped = (
        df.groupby("player_id")[["h", "hr"]]
        .sum(min_count=1)
        .reset_index()
    )
    for row in grouped.itertuples():
        totals[int(row.player_id)] = {
            "h": int(row.h) if not pd.isna(row.h) else 0,
            "hr": int(row.hr) if not pd.isna(row.hr) else 0,
        }
    return totals


def update_team_state(
    team_records: Dict[int, Dict[str, int]],
    team_streak: Dict[int, Dict[str, Optional[str]]],
    winner_team: int,
    loser_team: int,
) -> None:
    for tid in (winner_team, loser_team):
        team_records.setdefault(tid, {"w": 0, "l": 0})
        team_streak.setdefault(tid, {"type": None, "len": 0})
    team_records[winner_team]["w"] += 1
    team_records[loser_team]["l"] += 1

    win_streak = team_streak[winner_team]
    if win_streak["type"] == "W":
        win_streak["len"] += 1
    else:
        win_streak["type"] = "W"
        win_streak["len"] = 1
    lose_streak = team_streak[loser_team]
    if lose_streak["type"] == "L":
        lose_streak["len"] += 1
    else:
        lose_streak["type"] = "L"
        lose_streak["len"] = 1


def get_team_wpct(
    team_id: int,
    team_records: Dict[int, Dict[str, int]],
    team_info: Dict[int, dict],
) -> float:
    rec = team_records.get(team_id)
    if rec:
        total = rec["w"] + rec["l"]
        if total > 0:
            return rec["w"] / total
    fallback = team_info.get(team_id, {}).get("wpct")
    if fallback is not None:
        return fallback
    return 0.5


def determine_winner_and_loser(game: pd.Series) -> Tuple[Optional[int], Optional[int]]:
    try:
        home_score = int(game["home_score"])
        away_score = int(game["away_score"])
        home_id = int(game["home_id"])
        away_id = int(game["away_id"])
    except (KeyError, TypeError, ValueError):
        return None, None
    if pd.isna(home_score) or pd.isna(away_score):
        return None, None
    if home_score == away_score:
        return None, None
    if home_score > away_score:
        return home_id, away_id
    return away_id, home_id


def update_team_state(
    team_records: Dict[int, Dict[str, int]],
    team_streak: Dict[int, Dict[str, Optional[str]]],
    winner_team: int,
    loser_team: int,
) -> None:
    for tid in (winner_team, loser_team):
        team_records.setdefault(tid, {"w": 0, "l": 0})
        team_streak.setdefault(tid, {"type": None, "len": 0})
    team_records[winner_team]["w"] += 1
    team_records[loser_team]["l"] += 1

    win_streak = team_streak[winner_team]
    if win_streak["type"] == "W":
        win_streak["len"] += 1
    else:
        win_streak["type"] = "W"
        win_streak["len"] = 1
    lose_streak = team_streak[loser_team]
    if lose_streak["type"] == "L":
        lose_streak["len"] += 1
    else:
        lose_streak["type"] = "L"
        lose_streak["len"] = 1


def get_team_wpct(
    team_id: int,
    team_records: Dict[int, Dict[str, int]],
    team_info: Dict[int, dict],
) -> float:
    rec = team_records.get(team_id)
    if rec:
        total = rec["w"] + rec["l"]
        if total > 0:
            return rec["w"] / total
    fallback = team_info.get(team_id, {}).get("wpct")
    if fallback is not None:
        return fallback
    return 0.5


def build_series_tags(games: pd.DataFrame, key_col: str) -> Dict[str, str]:
    if games.empty:
        return {}
    series_tags: Dict[str, str] = {}
    games_sorted = games.sort_values(["home_id", "away_id", "date"])
    current: List[pd.Series] = []
    last_date: Optional[pd.Timestamp] = None
    last_home = last_away = None
    max_series_length = 4

    def finalize(series: List[pd.Series]):
        if not series:
            return
        length = len(series)
        wins_home = sum(1 for g in series if g["home_score"] > g["away_score"])
        wins_away = length - wins_home
        last_game = series[-1]
        game_id = last_game[key_col]
        if length not in {2, 3, 4}:
            return
        if wins_home == length or wins_away == length:
            series_tags[game_id] = "SWEEP"
            return
        games_before_last = length - 1
        if games_before_last == 0:
            return
        last_home_win = last_game["home_score"] > last_game["away_score"]
        if last_home_win:
            home_wins_before = wins_home - 1
            if home_wins_before == 0 and wins_away == games_before_last:
                series_tags[game_id] = "AVOID_SWEEP"
        else:
            away_wins_before = wins_away - 1
            if away_wins_before == 0 and wins_home == games_before_last:
                series_tags[game_id] = "AVOID_SWEEP"

    for _, row in games_sorted.iterrows():
        date = row["date"]
        home_id = row["home_id"]
        away_id = row["away_id"]
        continue_series = (
            current
            and home_id == last_home
            and away_id == last_away
            and last_date is not None
            and (date - last_date).days <= 2
            and len(current) < max_series_length
        )
        if continue_series:
            current.append(row)
        else:
            finalize(current)
            current = [row]
        last_date = date
        last_home = home_id
        last_away = away_id
    finalize(current)
    return series_tags


def summarize_category(rows: pd.DataFrame, title: str) -> str:
    if rows.empty:
        return ""
    return f"=== {title} ===\n{rows.to_string(index=False)}"


def main():
    parser = argparse.ArgumentParser(description="ABL weekly highlight miner")
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="End date YYYY-MM-DD")
    parser.add_argument("--max_rows", type=int, default=200)
    parser.add_argument("--mode", choices=["weekly", "sim"], help="weekly recap or sim-through-Saturday")
    args = parser.parse_args()
    base_dir = DATA_DIR

    mode = args.mode
    if not mode:
        if sys.stdin is not None and sys.stdin.isatty():
            try:
                user_input = input(
                    "Press Enter for weekly recap or type 'sim' for Sim-through-Saturday mode: "
                ).strip().lower()
            except EOFError:
                user_input = ""
        else:
            user_input = ""
        mode = "weekly" if not user_input else user_input
        if mode not in {"weekly", "sim"}:
            print("Invalid mode entered; defaulting to weekly.")
            mode = "weekly"
    args.mode = mode

    games_raw = read_csv_smart(
        "team_game_log.csv",
        "team_game_logs.csv",
        "games.csv",
        "Games.csv",
    )
    if games_raw is None:
        print("No team_game_log/games CSV found; nothing to mine.")
        return

    games_raw = filter_abl(games_raw)
    start, end = parse_dates(args, games_raw.copy())

    games_all = games_raw.copy()
    date_col = pick(games_all, "date", "game_date")
    games_all["date"] = pd.to_datetime(games_all[date_col], errors="coerce")
    games_all = games_all.dropna(subset=["date"])

    ootp_game_col = pick(games_all, "game_id", "gameid")
    if ootp_game_col:
        games_all["game_identifier"] = games_all[ootp_game_col].astype(str)
    else:
        games_all["game_identifier"] = games_all.apply(
            lambda r: f"{r['date'].date()}_{int(r['away_id'])}@{int(r['home_id'])}",
            axis=1,
        )

    away_id_col = pick(games_all, "away_team_id", "away_id", "team_id_away", "away_team", "team0")
    home_id_col = pick(games_all, "home_team_id", "home_id", "team_id_home", "home_team", "team1")
    games_all["away_id"] = games_all[away_id_col].astype(int)
    games_all["home_id"] = games_all[home_id_col].astype(int)

    score_cols_away = [
        "away_runs",
        "r_away",
        "runs_away",
        "score0",
        "runs0",
        "away_score",
    ]
    score_cols_home = [
        "home_runs",
        "r_home",
        "runs_home",
        "score1",
        "runs1",
        "home_score",
    ]
    away_score_col = pick(games_all, *score_cols_away)
    home_score_col = pick(games_all, *score_cols_home)
    games_all["away_score"] = pd.to_numeric(games_all[away_score_col], errors="coerce")
    games_all["home_score"] = pd.to_numeric(games_all[home_score_col], errors="coerce")

    innings_col = pick(games_all, "innings", "inn", "inning", "ipd")
    games_all["innings"] = pd.to_numeric(games_all[innings_col], errors="coerce").fillna(9)

    games_week = games_all[
        (games_all["date"] >= start) & (games_all["date"] <= end)
    ].copy()
    weekday_cutoff = 5 if mode == "sim" else 6  # sim: Mon-Sat, weekly: Mon-Sun
    games_week = games_week[games_week["date"].dt.weekday <= weekday_cutoff]
    games_week = games_week[compute_played_mask(games_week)]
    if games_week.empty:
        print("No games in the selected window.")
        return
    games_week = games_week.sort_values(["date", "game_identifier"]).reset_index(drop=True)

    score_cols_away = [
        "away_runs",
        "r_away",
        "runs_away",
        "score0",
        "runs0",
        "away_score",
    ]
    score_cols_home = [
        "home_runs",
        "r_home",
        "runs_home",
        "score1",
        "runs1",
        "home_score",
    ]
    away_score_col = pick(games_all, *score_cols_away)
    home_score_col = pick(games_all, *score_cols_home)
    games_week = games_week.dropna(subset=["away_score", "home_score"])

    home_name_col = pick(games_week, "home_team_name", "home_name")
    away_name_col = pick(games_week, "away_team_name", "away_name")

    team_info = load_team_info()
    career_batting_totals = load_career_batting_totals()
    games_week["home_name"] = games_week.apply(
        lambda r: team_info.get(int(r["home_id"]), {}).get("name")
        or (r[home_name_col] if home_name_col else f"Team {int(r['home_id'])}"),
        axis=1,
    )
    games_week["away_name"] = games_week.apply(
        lambda r: team_info.get(int(r["away_id"]), {}).get("name")
        or (r[away_name_col] if away_name_col else f"Team {int(r['away_id'])}"),
        axis=1,
    )

    ootp_game_col = pick(games_df, "game_id", "gameid")
    if ootp_game_col:
        games_df["game_identifier"] = games_df[ootp_game_col].astype(str)
    else:
        games_df["game_identifier"] = games_df.apply(
            lambda r: f"{r['date'].date()}_{int(r['away_id'])}@{int(r['home_id'])}",
            axis=1,
        )

    series_tags = build_series_tags(games_week, "game_identifier")
    line_helper = LineScoreHelper(games_week)

    team_records = {tid: {"w": 0, "l": 0} for tid in range(TEAM_MIN, TEAM_MAX + 1)}
    team_streak = {tid: {"type": None, "len": 0} for tid in range(TEAM_MIN, TEAM_MAX + 1)}
    pre_games = all_games_sorted[all_games_sorted["date"] < start]
    for _, game in pre_games.iterrows():
        winner_team, loser_team = determine_winner_and_loser(game)
        if winner_team is None or loser_team is None:
            continue
        if not (TEAM_MIN <= winner_team <= TEAM_MAX and TEAM_MIN <= loser_team <= TEAM_MAX):
            continue
        update_team_state(team_records, team_streak, winner_team, loser_team)

    bat_cols: Dict[str, Optional[str]] = {}
    bat_logs = read_csv_smart(
        "player_game_log_batting.csv",
        "player_game_logs_batting.csv",
    )
    if bat_logs is not None:
        bat_logs = filter_abl(bat_logs)
        bat_date_col = pick(bat_logs, "date", "game_date")
        bat_logs["date"] = pd.to_datetime(bat_logs[bat_date_col], errors="coerce")
        bat_logs = bat_logs[(bat_logs["date"] >= start) & (bat_logs["date"] <= end)]
        bat_cols = {
            "team": pick(bat_logs, "team_id", "teamid", "tid"),
            "opp": pick(bat_logs, "opponent_id", "opp_team_id", "opp"),
            "player": pick(bat_logs, "player_name", "name"),
            "player_id": pick(bat_logs, "player_id", "pid"),
            "hits": pick(bat_logs, "h", "hits"),
            "doubles": pick(bat_logs, "2b", "doubles"),
            "triples": pick(bat_logs, "3b", "triples"),
            "hrs": pick(bat_logs, "hr", "home_runs"),
            "sb": pick(bat_logs, "sb", "stolen_bases"),
            "rbi": pick(bat_logs, "rbi"),
        }
    pitch_cols: Dict[str, Optional[str]] = {}
    pitch_logs = read_csv_smart(
        "player_game_log_pitching.csv",
        "player_game_logs_pitching.csv",
    )
    if pitch_logs is not None:
        pitch_logs = filter_abl(pitch_logs)
        pitch_date_col = pick(pitch_logs, "date", "game_date")
        pitch_logs["date"] = pd.to_datetime(pitch_logs[pitch_date_col], errors="coerce")
        pitch_logs = pitch_logs[
            (pitch_logs["date"] >= start) & (pitch_logs["date"] <= end)
        ]
        pitch_cols = {
            "team": pick(pitch_logs, "team_id", "teamid", "tid"),
            "opp": pick(pitch_logs, "opponent_id", "opp_team_id", "opp"),
            "player": pick(pitch_logs, "player_name", "name"),
            "so": pick(pitch_logs, "so", "k"),
            "ip": pick(pitch_logs, "ip", "innings_pitched"),
            "er": pick(pitch_logs, "er", "earned_runs"),
            "gs": pick(pitch_logs, "gs", "games_started"),
            "cg": pick(pitch_logs, "cg"),
            "sho": pick(pitch_logs, "sho"),
            "hits": pick(pitch_logs, "h", "hits"),
            "pitches": pick(pitch_logs, "pi", "pitches"),
            "ir": pick(pitch_logs, "ir", "irs_inherited"),
            "irs": pick(pitch_logs, "irs"),
        }

    def player_log_filter(df, date, team_id, opp_id, team_col, opp_col):
        if df is None or not team_col:
            return pd.DataFrame()
        mask = (df["date"] == date) & (
            pd.to_numeric(df[team_col], errors="coerce") == int(team_id)
        )
        if opp_col:
            mask &= pd.to_numeric(df[opp_col], errors="coerce") == int(opp_id)
        return df[mask]

    events_records = []
    for _, game in games_df.iterrows():
        date = game["date"]
        away_score = int(game["away_score"])
        home_score = int(game["home_score"])
        innings = float(game["innings"])
        home_win = home_score > away_score
        diff = abs(home_score - away_score)
        opponent_score = away_score if home_win else home_score
        winner_team = int(game["home_id"] if home_win else game["away_id"])
        loser_team = int(game["away_id"] if home_win else game["home_id"])

        event_tags: List[str] = []
        player_tags: List[str] = []
        base_points = 0
        individual_points = 0

        def apply_event(tag: str, value: int):
            nonlocal base_points
            if tag not in event_tags:
                event_tags.append(tag)
            base_points = max(base_points, value)

        if home_win and (innings > 9 or diff <= 1):
            apply_event("WALKOFF", 12)
        elif diff == 1:
            apply_event("ONE_RUN", 6)

        if innings > 9:
            apply_event("EXTRA", 6)

        if (away_score >= 9 and home_score >= 9) or (away_score + home_score >= 18):
            apply_event("SLUGFEST", 7)

        if diff >= 8:
            apply_event("BLOWOUT", 5)

        tag = series_tags.get(game["game_identifier"])
        if tag == "SWEEP":
            apply_event("SWEEP", 6)
        elif tag == "AVOID_SWEEP":
            apply_event("AVOID_SWEEP", 6)

        if line_helper.available:
            line_data = line_helper.extract(game, away_score, home_score)
            if line_data:
                away_line, home_line = line_data
                comeback_flag, behind_flag = evaluate_line_events(
                    away_line, home_line, home_win, innings
                )
                if comeback_flag:
                    apply_event("COMEBACK", 10)
                if behind_flag:
                    apply_event("BEHIND", 8)

        if (
            pitch_logs is not None
            and pitch_cols.get("team")
            and pitch_cols.get("ip")
            and pitch_cols.get("er")
            and opponent_score == 0
        ):
            logs = player_log_filter(
                pitch_logs,
                date,
                winner_team,
                loser_team,
                pitch_cols.get("team"),
                pitch_cols.get("opp"),
            )
            ip_col = pitch_cols.get("ip")
            er_col = pitch_cols.get("er")
            if not logs.empty and ip_col in logs.columns and er_col in logs.columns:
                cg_col = pitch_cols.get("cg")
                sho_col = pitch_cols.get("sho")
                so_col = pitch_cols.get("so")
                hits_col = pitch_cols.get("hits")
                gs_col = pitch_cols.get("gs")

                def pick_by_ip(df_subset: pd.DataFrame) -> pd.Series:
                    if ip_col and ip_col in df_subset.columns:
                        order = (
                            pd.to_numeric(df_subset[ip_col], errors="coerce")
                            .fillna(0)
                            .sort_values(ascending=False)
                        )
                        if not order.empty:
                            return df_subset.loc[order.index[0]]
                    return df_subset.iloc[0]

                starter_row = None
                if gs_col and gs_col in logs.columns:
                    starters_mask = (
                        pd.to_numeric(logs[gs_col], errors="coerce").fillna(0) >= 1
                    )
                    starters = logs[starters_mask]
                    if not starters.empty:
                        starter_row = pick_by_ip(starters)
                if starter_row is None:
                    starter_row = pick_by_ip(logs)

                def grab(row: pd.Series, col: Optional[str], converter):
                    if not col or col not in row.index:
                        return None
                    return converter(row[col])

                ip_val = grab(starter_row, ip_col, safe_float)
                er_val = grab(starter_row, er_col, safe_float)
                cg_flag = bool(grab(starter_row, cg_col, safe_int))
                sho_flag = bool(grab(starter_row, sho_col, safe_int))
                so_val = grab(starter_row, so_col, safe_int)
                hits_val = grab(starter_row, hits_col, safe_int)

                is_complete = cg_flag or (ip_val is not None and ip_val >= 9)
                allowed_zero = sho_flag or (er_val is not None and er_val == 0)
                dominant = False
                if hits_val is not None:
                    dominant = hits_val <= 3
                if not dominant and so_val is not None:
                    dominant = so_val >= 10
                if is_complete and allowed_zero and dominant:
                    apply_event("SHUTOUT", 9)

        # Pitchers' duel
        duel = False
        if (
            pitch_logs is not None
            and pitch_cols.get("team")
            and pitch_cols.get("ip")
            and pitch_cols.get("er")
            and pitch_cols.get("gs")
        ):
            duel = True
            for team_id, opp_id in [
                (int(game["away_id"]), int(game["home_id"])),
                (int(game["home_id"]), int(game["away_id"])),
            ]:
                logs = player_log_filter(
                    pitch_logs,
                    date,
                    team_id,
                    opp_id,
                    pitch_cols.get("team"),
                    pitch_cols.get("opp"),
                )
                ip_col = pitch_cols.get("ip")
                er_col = pitch_cols.get("er")
                gs_col = pitch_cols.get("gs")
                if (
                    logs.empty
                    or ip_col not in logs.columns
                    or er_col not in logs.columns
                    or gs_col not in logs.columns
                ):
                    duel = False
                    break
                starters_mask = (
                    pd.to_numeric(logs[gs_col], errors="coerce").fillna(0) >= 1
                )
                starters = logs[starters_mask]
                if starters.empty:
                    duel = False
                    break
                order = (
                    pd.to_numeric(starters[ip_col], errors="coerce")
                    .fillna(0)
                    .sort_values(ascending=False)
                )
                starter = (
                    starters.loc[order.index[0]] if not order.empty else starters.iloc[0]
                )
                ip_val = safe_float(starter[ip_col])
                er_val = safe_float(starter[er_col])
                if ip_val is None or ip_val < 7 or er_val is None or er_val > 2:
                    duel = False
                    break
        if duel:
            apply_event("DUEL", 9)

        # Individual feats
        if bat_logs is not None and bat_cols.get("team"):
            for team_id, opp_id in [
                (int(game["away_id"]), int(game["home_id"])),
                (int(game["home_id"]), int(game["away_id"])),
            ]:
                logs = player_log_filter(
                    bat_logs,
                    date,
                    team_id,
                    opp_id,
                    bat_cols.get("team"),
                    bat_cols.get("opp"),
                )
                if logs.empty:
                    continue
                for row in logs.itertuples():
                    data = row._asdict()
                    name = data.get(bat_cols.get("player")) if bat_cols.get("player") else None
                    name = name or "Player"
                    player_id_val = (
                        safe_int(data.get(bat_cols.get("player_id"))) if bat_cols.get("player_id") else None
                    )
                    hits_val = (
                        safe_int(data.get(bat_cols.get("hits"))) if bat_cols.get("hits") else None
                    )
                    doubles_val = (
                        safe_int(data.get(bat_cols.get("doubles"))) if bat_cols.get("doubles") else None
                    )
                    triples_val = (
                        safe_int(data.get(bat_cols.get("triples"))) if bat_cols.get("triples") else None
                    )
                    hrs_val = (
                        safe_int(data.get(bat_cols.get("hrs"))) if bat_cols.get("hrs") else None
                    )
                    sb_val = (
                        safe_int(data.get(bat_cols.get("sb"))) if bat_cols.get("sb") else None
                    )
                    rbi_val = (
                        safe_int(data.get(bat_cols.get("rbi"))) if bat_cols.get("rbi") else None
                    )
                    singles_val = None
                    if hits_val is not None:
                        singles_val = hits_val - (doubles_val or 0) - (triples_val or 0) - (hrs_val or 0)
                    if hits_val is not None and hits_val >= 4:
                        player_tags.append(f"4H: {name}")
                        individual_points += 6
                    if hrs_val is not None and hrs_val >= 3:
                        player_tags.append(f"3HR: {name}")
                        individual_points += 8
                        apply_event("THREE_HR", 10)
                    elif hrs_val is not None and hrs_val >= 2:
                        player_tags.append(f"Multi-HR: {name}")
                        individual_points += 5
                    if (
                        hrs_val is not None
                        and hrs_val >= 1
                        and rbi_val is not None
                        and rbi_val >= 4
                    ):
                        player_tags.append(f"Grand Slam: {name}")
                        individual_points += 6
                        apply_event("GRAND_SLAM", 9)
                    if (
                        singles_val is not None
                        and singles_val >= 1
                        and (doubles_val or 0) >= 1
                        and (triples_val or 0) >= 1
                        and (hrs_val or 0) >= 1
                    ):
                        player_tags.append(f"Cycle: {name}")
                        individual_points += 15
                    if sb_val is not None and sb_val >= 3:
                        player_tags.append(f"3SB: {name}")
                        individual_points += 5
                    if player_id_val and career_batting_totals:
                        totals = career_batting_totals.get(player_id_val)
                        if totals:
                            hits_total = totals.get("h")
                            hrs_total = totals.get("hr")
                            if hits_total is not None and hits_val is not None:
                                prior_hits = hits_total - hits_val
                                for threshold in HIT_MILESTONES:
                                    if prior_hits < threshold <= hits_total:
                                        player_tags.append(f"{threshold} Hits: {name}")
                                        apply_event("MILESTONE", 12)
                                        break
                            if hrs_total is not None and hrs_val is not None:
                                prior_hr = hrs_total - hrs_val
                                for threshold in HR_MILESTONES:
                                    if prior_hr < threshold <= hrs_total:
                                        player_tags.append(f"{threshold} HR: {name}")
                                        apply_event("MILESTONE", 12)
                                        break

        if pitch_logs is not None and pitch_cols.get("team"):
            for team_id, opp_id in [
                (int(game["away_id"]), int(game["home_id"])),
                (int(game["home_id"]), int(game["away_id"])),
            ]:
                logs = player_log_filter(
                    pitch_logs,
                    date,
                    team_id,
                    opp_id,
                    pitch_cols.get("team"),
                    pitch_cols.get("opp"),
                )
                if logs.empty:
                    continue
                for row in logs.itertuples():
                    data = row._asdict()
                    name = data.get(pitch_cols.get("player")) if pitch_cols.get("player") else None
                    name = name or "Pitcher"
                    so_val = (
                        safe_int(data.get(pitch_cols.get("so"))) if pitch_cols.get("so") else None
                    )
                    ip_val = (
                        safe_float(data.get(pitch_cols.get("ip"))) if pitch_cols.get("ip") else None
                    )
                    er_val = (
                        safe_float(data.get(pitch_cols.get("er"))) if pitch_cols.get("er") else None
                    )
                    gs_val = (
                        safe_int(data.get(pitch_cols.get("gs"))) if pitch_cols.get("gs") else None
                    )
                    cg_val = (
                        safe_int(data.get(pitch_cols.get("cg"))) if pitch_cols.get("cg") else None
                    )
                    sho_val = (
                        safe_int(data.get(pitch_cols.get("sho"))) if pitch_cols.get("sho") else None
                    )
                    pitches_val = (
                        safe_int(data.get(pitch_cols.get("pitches"))) if pitch_cols.get("pitches") else None
                    )
                    ir_val = (
                        safe_int(data.get(pitch_cols.get("ir"))) if pitch_cols.get("ir") else None
                    )
                    irs_val = (
                        safe_int(data.get(pitch_cols.get("irs"))) if pitch_cols.get("irs") else None
                    )
                    if so_val is not None and so_val >= 10:
                        player_tags.append(f"10K: {name}")
                        individual_points += 6
                    if (
                        gs_val is not None
                        and gs_val >= 1
                        and (
                            (cg_val is not None and cg_val >= 1)
                            or (ip_val is not None and ip_val >= 9)
                        )
                    ):
                        player_tags.append(f"CG: {name}")
                        individual_points += 6
                        if (er_val is not None and er_val == 0) or (
                            sho_val is not None and sho_val >= 1
                        ):
                            player_tags.append(f"SHO: {name}")
                            individual_points += 8
                    if (
                        gs_val is not None
                        and gs_val >= 1
                        and pitches_val is not None
                        and pitches_val >= HERO_PITCH_THRESHOLD
                    ):
                        player_tags.append(f"{pitches_val} pitches: {name}")
                        individual_points += 8
                        apply_event("HERO_START", 11)
                    if (gs_val or 0) == 0:
                        if ip_val is not None and ip_val >= RELIEF_LONG_IP:
                            player_tags.append(f"{ip_val:.1f} IP relief: {name}")
                            individual_points += 5
                            apply_event("RELIEF_HERO", 10)
                        if (
                            ir_val is not None
                            and ir_val >= 2
                            and irs_val is not None
                            and irs_val == 0
                        ):
                            player_tags.append(f"Clutch Relief ({ir_val} IR stranded): {name}")
                            individual_points += 6
                            apply_event("CLUTCH_RELIEF", 9)

        context_points = 0
        home_info = team_info.get(int(game["home_id"]), {})
        away_info = team_info.get(int(game["away_id"]), {})
        if home_info.get("division") and home_info.get("division") == away_info.get("division"):
            context_points += 2
        if home_info.get("wpct") is not None and away_info.get("wpct") is not None:
            if abs(home_info["wpct"] - away_info["wpct"]) <= 0.02:
                context_points += 2
        winner_info = team_info.get(winner_team, {})
        streak_str = winner_info.get("streak") if winner_info else ""
        if isinstance(streak_str, str) and streak_str:
            match = re.match(r"([WL])(\d+)", streak_str.strip())
            if match:
                direction, amount = match.groups()
                streak_len = int(amount)
                if direction.upper() == "W" and streak_len >= STREAK_THRESHOLD:
                    apply_event("STREAK_WATCH", 10)
                    player_tags.append(
                        f"Streak Watch: {winner_info.get('name', 'Team')} W{streak_len}"
                    )

        highlight_score = base_points + individual_points + context_points
        if highlight_score < 6:
            continue

        conference = home_info.get("conference") or ""
        primary_tag = event_tags[0] if event_tags else "HIGHLIGHT"
        innings_display = int(innings) if not pd.isna(innings) else 9
        scoreline = f"{game['away_name']} {away_score}-{home_score} {game['home_name']} ({innings_display} inn)"
        description = f"{primary_tag.replace('_', '-')} -- {scoreline}"
        if player_tags:
            description += " | " + "; ".join(player_tags)

        category = "INDIVIDUAL FEATS"
        if "WALKOFF" in event_tags:
            category = "WALK-OFFS"
        elif "STREAK_WATCH" in event_tags:
            category = "STREAK WATCH"
        elif any(tag in event_tags for tag in ("DUEL", "SHUTOUT")):
            category = "ACE DUELS"
        elif "SLUGFEST" in event_tags:
            category = "SLUGFESTS"
        elif any(tag in event_tags for tag in ("COMEBACK", "BEHIND")):
            category = "COMEBACKS (3+)"
        elif "ONE_RUN" in event_tags:
            category = "ONE-RUN GAMES"
        elif "BLOWOUT" in event_tags:
            category = "BLOWOUTS"
        elif any(tag in event_tags for tag in ("SWEEP", "AVOID_SWEEP")):
            category = "SWEEPS / AVOID SWEEP"
        elif any(tag in event_tags for tag in ("HERO_START", "RELIEF_HERO")):
            category = "HEROIC OUTINGS"
        elif "CLUTCH_RELIEF" in event_tags:
            category = "CLUTCH RELIEF"
        elif any(tag in event_tags for tag in ("THREE_HR", "GRAND_SLAM")):
            category = "POWER SURGE"
        elif "MILESTONE" in event_tags:
            category = "MILESTONE MOMENTS"

        events_records.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "conference": conference,
                "game_id": game["game_identifier"],
                "away_name": game["away_name"],
                "home_name": game["home_name"],
                "final_away": int(away_score),
                "final_home": int(home_score),
                "innings": int(innings),
                "event_tags": ";".join(event_tags),
                "player_tags": ";".join(player_tags),
                "description": description,
                "highlight_score": int(highlight_score),
                "category": category,
            }
        )

    if not events_records:
        print("No highlight events detected for the window.")
        return

    highlights = pd.DataFrame(events_records)
    conference_order = {"NBC": 0, "ABC": 1}
    highlights["conference_rank"] = highlights["conference"].map(conference_order).fillna(99)
    highlights = highlights.sort_values(
        ["date", "conference_rank", "highlight_score"],
        ascending=[True, True, False],
    ).drop(columns="conference_rank")
    if len(highlights) > args.max_rows:
        highlights = highlights.head(args.max_rows)

    csv_cols = [
        "date",
        "conference",
        "game_id",
        "away_name",
        "home_name",
        "final_away",
        "final_home",
        "innings",
        "event_tags",
        "player_tags",
        "description",
        "highlight_score",
    ]
    csv_dir = base_dir / "out" / "csv_out"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "z_ABL_Week_Miner.csv"
    highlights[csv_cols].to_csv(csv_path, index=False)

    sections = []
    for title in [
        "WALK-OFFS",
        "STREAK WATCH",
        "ACE DUELS",
        "SLUGFESTS",
        "ONE-RUN GAMES",
        "COMEBACKS (3+)",
        "BLOWOUTS",
        "INDIVIDUAL FEATS",
        "SWEEPS / AVOID SWEEP",
        "HEROIC OUTINGS",
        "CLUTCH RELIEF",
        "POWER SURGE",
        "MILESTONE MOMENTS",
    ]:
        section_rows = highlights[highlights["category"] == title]
        text = summarize_category(section_rows, title)
        if text:
            sections.append(text)
    txt_dir = base_dir / "out" / "txt_out"
    txt_dir.mkdir(parents=True, exist_ok=True)
    txt_path = txt_dir / "z_ABL_Week_Miner.txt"
    txt_path.write_text("\n\n".join(sections), encoding="utf-8")
    print(f"Mined {len(highlights)} games; wrote {csv_path} and {txt_path}.")


if __name__ == "__main__":
    main()
