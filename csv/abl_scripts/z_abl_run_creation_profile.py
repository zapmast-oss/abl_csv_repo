"""ABL Run Creation Profile: HR share, SB pressure, situational RBI."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

TEAM_MIN, TEAM_MAX = 1, 24
RECORD_CANDIDATES = [
    "team_record.csv",
    "team_season.csv",
    "team_totals.csv",
    "teams_season.csv",
    "standings.csv",
]
BATTING_CANDIDATES = [
    "team_batting.csv",
    "teams_batting.csv",
    "team_batting_stats.csv",
    "teams_batting_stats.csv",
    "batting_team_totals.csv",
]
SCORING_CANDIDATES = [
    "team_scoring.csv",
    "batting_splits_situational.csv",
    "team_run_scoring.csv",
]
TEAM_INFO_CANDIDATES = [
    "team_info.csv",
    "teams.csv",
]
LOG_CANDIDATES = [
    "team_game_log.csv",
    "teams_game_log.csv",
    "game_log_team.csv",
    "team_log.csv",
    "schedule_results.csv",
]
PBP_CANDIDATES = ["game_logs.csv"]


def resolve_source(base: Path, override: Optional[Path], candidates: Sequence[str]) -> Optional[Path]:
    if override:
        override_path = Path(override)
        if not override_path.exists():
            raise FileNotFoundError(f"Specified file not found: {override_path}")
        return override_path
    for name in candidates:
        path = base / name
        if path.exists():
            return path
    return None


def pick_column(df: pd.DataFrame, *names: str) -> Optional[str]:
    lowered = {col.lower(): col for col in df.columns}
    for name in names:
        key = name.lower()
        if key in lowered:
            return lowered[key]
    return None


def read_first(base: Path, override: Optional[Path], candidates: Sequence[str]) -> Optional[pd.DataFrame]:
    path = resolve_source(base, override, candidates)
    if path is None:
        return None
    return pd.read_csv(path)


def read_first_with_path(
    base: Path, override: Optional[Path], candidates: Sequence[str]
) -> Tuple[Optional[pd.DataFrame], Optional[Path]]:
    path = resolve_source(base, override, candidates)
    if path is None:
        return None, None
    return pd.read_csv(path), path


def resolve_text_path(csv_path: Path) -> Path:
    text_name = csv_path.with_suffix(".txt").name
    parent = csv_path.parent
    parent_lower = parent.name.lower()
    if parent_lower in {"csv_out"}:
        text_dir = parent.parent / "txt_out"
    elif parent_lower in {"txt_out"}:
        text_dir = parent
    else:
        text_dir = parent
    text_dir.mkdir(parents=True, exist_ok=True)
    return text_dir / text_name


def load_record(base: Path, override: Optional[Path]) -> pd.DataFrame:
    df = read_first(base, override, RECORD_CANDIDATES)
    if df is None:
        raise FileNotFoundError("Unable to find team record/season file.")
    team_col = pick_column(df, "team_id", "teamid", "teamID", "TeamID")
    if not team_col:
        raise ValueError("team_id column missing in record data.")
    name_col = pick_column(df, "team_display", "team_name", "name", "TeamName")
    wins_col = pick_column(df, "wins", "w")
    losses_col = pick_column(df, "losses", "l")
    runs_col = pick_column(df, "runs_scored", "rs", "r", "runsscored")

    rec = pd.DataFrame()
    rec["team_id"] = pd.to_numeric(df[team_col], errors="coerce").astype("Int64")
    rec = rec[(rec["team_id"] >= TEAM_MIN) & (rec["team_id"] <= TEAM_MAX)]

    rec["team_display"] = df[name_col].fillna("") if name_col else ""
    rec["wins"] = pd.to_numeric(df[wins_col], errors="coerce") if wins_col else np.nan
    rec["losses"] = pd.to_numeric(df[losses_col], errors="coerce") if losses_col else np.nan
    rec["runs_scored"] = pd.to_numeric(df[runs_col], errors="coerce") if runs_col else np.nan
    return rec


def load_batting(base: Path, override: Optional[Path]) -> pd.DataFrame:
    df = read_first(base, override, BATTING_CANDIDATES)
    if df is None:
        raise FileNotFoundError("Unable to find team batting file.")
    team_col = pick_column(df, "team_id", "teamid", "teamID", "TeamID")
    if not team_col:
        raise ValueError("team_id missing in batting totals.")
    hr_col = pick_column(df, "hr")
    rbi_col = pick_column(df, "rbi")
    sb_col = pick_column(df, "sb")
    cs_col = pick_column(df, "cs")

    bat = pd.DataFrame()
    bat["team_id"] = pd.to_numeric(df[team_col], errors="coerce").astype("Int64")
    bat = bat[(bat["team_id"] >= TEAM_MIN) & (bat["team_id"] <= TEAM_MAX)]
    bat["HR"] = pd.to_numeric(df[hr_col], errors="coerce") if hr_col else np.nan
    bat["RBI"] = pd.to_numeric(df[rbi_col], errors="coerce") if rbi_col else np.nan
    bat["SB"] = pd.to_numeric(df[sb_col], errors="coerce") if sb_col else np.nan
    bat["CS"] = pd.to_numeric(df[cs_col], errors="coerce") if cs_col else np.nan
    return bat


def load_scoring(base: Path, override: Optional[Path]) -> Optional[pd.DataFrame]:
    df = read_first(base, override, SCORING_CANDIDATES)
    if df is None:
        return None
    team_col = pick_column(df, "team_id", "teamid", "teamID", "TeamID")
    if not team_col:
        return None
    hr_rbi_col = pick_column(df, "rbi_on_hr", "hr_rbi", "rbi_hr")
    rbi_2out_col = pick_column(df, "rbi_2out", "rbi_two_out", "two_out_rbi")
    scoring = pd.DataFrame()
    scoring["team_id"] = pd.to_numeric(df[team_col], errors="coerce").astype("Int64")
    scoring = scoring[(scoring["team_id"] >= TEAM_MIN) & (scoring["team_id"] <= TEAM_MAX)]
    scoring["rbi_on_hr"] = pd.to_numeric(df[hr_rbi_col], errors="coerce") if hr_rbi_col else np.nan
    scoring["rbi_2out"] = pd.to_numeric(df[rbi_2out_col], errors="coerce") if rbi_2out_col else np.nan
    return scoring


def load_team_names(base: Path, override: Optional[Path]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    df = read_first(base, override, TEAM_INFO_CANDIDATES)
    if df is None:
        return pd.DataFrame(columns=["team_id", "team_display"]), {}
    team_col = pick_column(df, "team_id", "teamid", "teamID", "TeamID")
    name_col = pick_column(df, "team_display", "team_name", "name", "TeamName")
    nickname_col = pick_column(df, "nickname")
    if not team_col or not name_col:
        return pd.DataFrame(columns=["team_id", "team_display"]), {}
    meta = pd.DataFrame()
    meta["team_id"] = pd.to_numeric(df[team_col], errors="coerce").astype("Int64")
    meta["team_display"] = df[name_col].fillna("")
    meta["city_name"] = df[name_col].fillna("")
    meta["nickname"] = df[nickname_col].fillna("") if nickname_col else ""
    meta = meta[(meta["team_id"] >= TEAM_MIN) & (meta["team_id"] <= TEAM_MAX)]
    name_map: Dict[str, int] = {}
    for _, row in meta.iterrows():
        tid = int(row["team_id"])
        combos = {
            str(row.get("team_display", "")).strip(),
            str(row.get("city_name", "")).strip(),
            str(row.get("nickname", "")).strip(),
            f"{row.get('city_name', '').strip()} {row.get('nickname', '').strip()}".strip(),
        }
        for label in combos:
            if label:
                name_map[label.lower()] = tid
    return meta[["team_id", "team_display"]], name_map


def extract_team_from_half(text: str, name_map: Dict[str, int]) -> Optional[int]:
    lower = text.lower()
    if "batting" not in lower:
        return None
    match = re.search(r"-\s*(.+?)\s+batting", text)
    if not match:
        return None
    team_label = match.group(1).strip().lower()
    return name_map.get(team_label)


def parse_hr_runs(text: str) -> int:
    text_lower = text.lower()
    if "grand slam" in text_lower:
        return 4
    match = re.search(r"(\d+)-run", text_lower)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return 1
    if "solo" in text_lower:
        return 1
    return 1


def is_final_play_line(text: str) -> bool:
    if ": " not in text:
        return False
    _, detail = text.split(":", 1)
    detail = detail.strip().lower()
    if not detail:
        return False
    non_results = (
        "ball",
        "called strike",
        "swinging strike",
        "foul ball",
        "bunts foul",
        "timeout",
        "pickoff attempt",
        "throw over",
        "defensive",
    )
    return not detail.startswith(non_results)


def outs_delta(text: str) -> int:
    text_lower = text.lower()
    if "triple play" in text_lower:
        return 3
    if "double play" in text_lower:
        return 2
    keywords = [
        " fly out",
        " line out",
        " pop out",
        " foul out",
        "ground out",
        "grounds out",
        "strike out",
        "strikes out",
        "caught looking",
        "fielders choice",
        "sacrifice fly",
        "sacrifice bunt",
        "sac fly",
        "infield fly",
        "bunt out",
        "is caught stealing",
        "picked off",
    ]
    if any(kw in text_lower for kw in keywords):
        return 1
    return 0


def is_rbi_play(text: str) -> bool:
    text_lower = text.lower()
    disqualifiers = ("error", "wild pitch", "passed ball", "balk")
    if any(word in text_lower for word in disqualifiers):
        return False
    qualifiers = (
        "single",
        "double",
        "triple",
        "home run",
        "walk",
        "hit by pitch",
        "sacrifice",
        "ground out",
        "fly out",
        "line out",
        "fielders choice",
    )
    return any(word in text_lower for word in qualifiers)


def is_scoring_line(text: str) -> bool:
    text_lower = text.lower()
    if "scores" in text_lower:
        return True
    if "runner from 3rd" in text_lower and ("safe" in text_lower or "scores" in text_lower):
        return True
    return False


def scoring_credit_allowed(text_lower: str) -> bool:
    disqualifiers = ("wild pitch", "passed ball", "error", "balk", "pb")
    return not any(word in text_lower for word in disqualifiers)


def compute_scoring_from_pbp(
    base: Path,
    override: Optional[Path],
    name_map: Dict[str, int],
    relevant_ids: Optional[set[str]] = None,
) -> pd.DataFrame:
    path = resolve_source(base, override, PBP_CANDIDATES)
    if path is None or not name_map:
        return pd.DataFrame(columns=["team_id", "rbi_on_hr", "rbi_2out"])
    hr_totals: Dict[int, int] = defaultdict(int)
    two_out_totals: Dict[int, int] = defaultdict(int)
    current_team: Optional[int] = None
    outs = 0
    play_team: Optional[int] = None
    play_outs = 0
    play_rbi = False

    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, escapechar="\\")
        next(reader, None)
        for row in reader:
            if len(row) < 4:
                continue
            game_id, _, _, text = row
            if relevant_ids is not None and game_id not in relevant_ids:
                continue
            text_stripped = text.strip()
            text_lower = text_stripped.lower()
            if ("top of the" in text_lower or "bottom of the" in text_lower) and "batting" in text_lower:
                team_id = extract_team_from_half(text_stripped, name_map)
                current_team = team_id
                outs = 0
                play_team = None
                play_rbi = False
                continue
            if current_team is None:
                continue
            if "home run" in text_lower:
                runs = parse_hr_runs(text_lower)
                if runs:
                    hr_totals[current_team] += runs
                    if outs >= 2:
                        two_out_totals[current_team] += runs
                play_team = None
                play_rbi = False
                continue
            if is_final_play_line(text_stripped):
                play_team = current_team
                play_outs = outs
                play_rbi = is_rbi_play(text_lower)
                outs += outs_delta(text_lower)
                continue
            if any(keyword in text_lower for keyword in ("wild pitch", "passed ball", "balk")):
                play_team = current_team
                play_outs = outs
                play_rbi = False
                continue
            if is_scoring_line(text_stripped):
                if play_team is not None and play_rbi and scoring_credit_allowed(text_lower):
                    if play_outs >= 2:
                        two_out_totals[play_team] += 1
                continue

    data = []
    for tid in range(TEAM_MIN, TEAM_MAX + 1):
        data.append(
            {
                "team_id": tid,
                "rbi_on_hr": hr_totals.get(tid, 0),
                "rbi_2out": two_out_totals.get(tid, 0),
            }
        )
    return pd.DataFrame(data)


def load_logs(base: Path, override: Optional[Path]) -> Tuple[Optional[pd.DataFrame], Optional[Path]]:
    df, path = read_first_with_path(base, override, LOG_CANDIDATES)
    if df is None:
        return None, path
    team_col = pick_column(df, "team_id", "teamid", "teamID", "TeamID")
    if not team_col:
        return None, path
    df = df.copy()
    df["team_id"] = pd.to_numeric(df[team_col], errors="coerce").astype("Int64")
    df = df[(df["team_id"] >= TEAM_MIN) & (df["team_id"] <= TEAM_MAX)]
    result_col = pick_column(df, "result")
    runs_for_col = pick_column(df, "runs_for", "rs", "r")
    runs_against_col = pick_column(df, "runs_against", "ra")
    mask_played = pd.Series(False, index=df.index)
    if result_col:
        mask_played |= df[result_col].notna()
    if runs_for_col and runs_against_col:
        mask_played |= df[runs_for_col].notna() & df[runs_against_col].notna()
    return df[mask_played], path


def load_team_names(base: Path, override: Optional[Path]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    df = read_first(base, override, TEAM_INFO_CANDIDATES)
    if df is None:
        return pd.DataFrame(columns=["team_id", "team_display"]), {}
    team_col = pick_column(df, "team_id", "teamid", "teamID", "TeamID")
    name_col = pick_column(df, "team_display", "team_name", "name", "TeamName")
    nickname_col = pick_column(df, "nickname")
    if not team_col or not name_col:
        return pd.DataFrame(columns=["team_id", "team_display"]), {}
    meta = pd.DataFrame()
    meta["team_id"] = pd.to_numeric(df[team_col], errors="coerce").astype("Int64")
    meta["team_display"] = df[name_col].fillna("")
    meta["city_name"] = df[name_col].fillna("")
    meta["nickname"] = df[nickname_col].fillna("") if nickname_col else ""
    meta = meta[(meta["team_id"] >= TEAM_MIN) & (meta["team_id"] <= TEAM_MAX)]
    name_map: Dict[str, int] = {}
    for _, row in meta.iterrows():
        tid = int(row["team_id"])
        combos = {
            str(row.get("team_display", "")).strip(),
            str(row.get("city_name", "")).strip(),
            str(row.get("nickname", "")).strip(),
            f"{row.get('city_name', '').strip()} {row.get('nickname', '').strip()}".strip(),
        }
        for label in combos:
            if label:
                name_map[label.lower()] = tid
    return meta[["team_id", "team_display"]], name_map


def derive_games(
    record_df: pd.DataFrame, logs_df: Optional[pd.DataFrame]
) -> pd.Series:
    games = pd.Series(np.nan, index=record_df.index, dtype="float64")
    wins = record_df["wins"]
    losses = record_df["losses"]
    games_mask = wins.notna() & losses.notna()
    games.loc[games_mask] = wins.loc[games_mask] + losses.loc[games_mask]
    if logs_df is not None:
        log_counts = logs_df.groupby("team_id").size()
        missing_ids = record_df.loc[games.isna() | (games == 0), "team_id"]
        for team_id in missing_ids.dropna().unique():
            if team_id in log_counts:
                idx = record_df["team_id"] == team_id
                games.loc[idx] = log_counts[team_id]
    return games


def classify_power(pct: float) -> str:
    if pd.isna(pct):
        return "Unknown"
    if pct >= 0.4:
        return "Slugging"
    if pct >= 0.3:
        return "Punchy"
    if pct >= 0.2:
        return "Balanced"
    return "Small-ball"


def classify_pressure(sb_pg: float) -> str:
    if pd.isna(sb_pg):
        return "Unknown"
    if sb_pg >= 1.2:
        return "Relentless"
    if sb_pg >= 0.8:
        return "Aggressive"
    if sb_pg >= 0.4:
        return "Selective"
    return "Station-to-Station"


def classify_clutch(pct: float) -> str:
    if pd.isna(pct):
        return "Unknown"
    if pct >= 0.3:
        return "Two-out Machine"
    if pct >= 0.2:
        return "Timely"
    if pct >= 0.1:
        return "Occasional"
    return "Needs Spark"


def build_text_report(df: pd.DataFrame, limit: int = 24) -> str:
    lines = [
        "ABL Run Creation Profile",
        "=" * 28,
        "",
        "Breaks down how each lineup manufactures runs: long ball share, stolen-base pressure, and two-out conversion rate.",
        "Great for spotting clubs that live on slugging versus those who pressure defenses or cash in late-inning chances.",
        "",
    ]
    header = (
        f"{'Team':<22} {'Power':<16} {'Pressure':<16} {'Clutch':<16} "
        f"{'HR%':>7} {'SB Att/G':>9} {'2-out RBI%':>11}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    subset = df.head(limit)
    if subset.empty:
        lines.append("(No data available.)")
    for _, row in subset.iterrows():
        name = row["team_display"] or f"Team {int(row['team_id'])}"
        hr_pct = f"{row['pct_runs_via_hr']:.3f}" if pd.notna(row["pct_runs_via_hr"]) else "NA"
        sb_pg = f"{row['sb_att_pg']:.3f}" if pd.notna(row["sb_att_pg"]) else "NA"
        out_pct = f"{row['pct_rbi_2out']:.3f}" if pd.notna(row["pct_rbi_2out"]) else "NA"
        lines.append(
            f"{name:<22} {row['power_profile']:<16} {row['pressure_profile']:<16} "
            f"{row['clutch_profile']:<16} {hr_pct:>7} {sb_pg:>9} {out_pct:>11}"
        )
    lines.append("")
    lines.append("Key:")
    lines.append("  HR% uses explicit HR RBIs parsed from game logs; SB Att/G = (SB+CS)/G.")
    lines.append("  PWR Slugging >=40%, Punchy 30-39%, Balanced 20-29%, Small-ball <20%.")
    lines.append("  SPD Relentless >=1.20 att/G, Aggressive 0.80-1.19, Selective 0.40-0.79, Station-to-Station <0.40.")
    lines.append("  CLT Two-out Machine >=30%, Timely 20-29%, Occasional 10-19%, Needs Spark <10%.")
    lines.append(
        "Definition: HR% is runs via HR / total runs; SB pressure measures attempts per game; 2-out RBI% shows production after two outs."
    )
    return "\n".join(lines)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ABL Run Creation Profile.")
    parser.add_argument("--base", type=str, default=".", help="Base directory for CSVs.")
    parser.add_argument("--record", type=str, help="Override team record file.")
    parser.add_argument("--batting", type=str, help="Override team batting file.")
    parser.add_argument("--scoring", type=str, help="Override scoring detail file.")
    parser.add_argument("--logs", type=str, help="Override team log file for games fallback.")
    parser.add_argument("--pbp", type=str, help="Override play-by-play log for run detail.")
    parser.add_argument("--teams", type=str, help="Override team info file for names.")
    parser.add_argument(
        "--out",
        type=str,
        default="out/csv_out/z_ABL_Run_Creation_Profile.csv",
        help="Output CSV path.",
    )
    return parser.parse_args(argv)


def load_relevant_game_ids(base: Path) -> Optional[set[str]]:
    games_path = base / "games.csv"
    if not games_path.exists():
        return None
    df = pd.read_csv(
        games_path,
        usecols=[
            "game_id",
            "home_team",
            "away_team",
            "played",
        ],
    )
    df["home_team"] = pd.to_numeric(df["home_team"], errors="coerce")
    df["away_team"] = pd.to_numeric(df["away_team"], errors="coerce")
    df["played"] = pd.to_numeric(df["played"], errors="coerce")
    df = df[df["played"] == 1]
    mask = (
        df["home_team"].between(TEAM_MIN, TEAM_MAX)
        | df["away_team"].between(TEAM_MIN, TEAM_MAX)
    )
    ids = df.loc[mask, "game_id"].astype(str)
    return set(ids)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    base_dir = Path(args.base).resolve()

    game_ids = load_relevant_game_ids(base_dir)
    record_df = load_record(base_dir, Path(args.record) if args.record else None)
    batting_df = load_batting(base_dir, Path(args.batting) if args.batting else None)
    scoring_df = load_scoring(base_dir, Path(args.scoring) if args.scoring else None)
    logs_df, _ = load_logs(base_dir, Path(args.logs) if args.logs else None)
    names_df, name_map = load_team_names(base_dir, Path(args.teams) if args.teams else None)
    pbp_df = compute_scoring_from_pbp(
        base_dir, Path(args.pbp) if args.pbp else None, name_map, game_ids
    )

    df = record_df.merge(batting_df, on="team_id", how="outer", suffixes=("", "_bat"))
    if scoring_df is not None:
        df = df.merge(scoring_df, on="team_id", how="left")
    else:
        df["rbi_on_hr"] = np.nan
        df["rbi_2out"] = np.nan
    if not pbp_df.empty:
        df = df.merge(pbp_df, on="team_id", how="left", suffixes=("", "_pbp"))
        df["rbi_on_hr"] = df["rbi_on_hr_pbp"]
        df["rbi_2out"] = df["rbi_2out_pbp"]
        df = df.drop(columns=["rbi_on_hr_pbp", "rbi_2out_pbp"])

    numeric_cols = [
        "wins",
        "losses",
        "runs_scored",
        "HR",
        "RBI",
        "SB",
        "CS",
        "rbi_on_hr",
        "rbi_2out",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    df["g"] = derive_games(df, logs_df)

    if "team_display" not in df.columns:
        df["team_display"] = ""
    df["team_display"] = df["team_display"].fillna("")
    if not names_df.empty:
        display_map = names_df.set_index("team_id")["team_display"]
        df["team_display"] = np.where(
            df["team_display"].astype(str).str.len() > 0,
            df["team_display"],
            df["team_id"].map(display_map).fillna(""),
        )

    missing_runs = df["runs_scored"].isna() & df["RBI"].notna()
    df.loc[missing_runs, "runs_scored"] = df.loc[missing_runs, "RBI"]

    df["sb_attempts"] = df[["SB", "CS"]].sum(axis=1, min_count=1)
    df["sb_att_pg"] = np.where(
        df["g"] > 0, df["sb_attempts"] / df["g"], np.nan
    )

    df["pct_runs_via_hr"] = np.where(
        (df["rbi_on_hr"].notna()) & (df["runs_scored"] > 0),
        df["rbi_on_hr"] / df["runs_scored"],
        np.nan,
    )

    df["pct_rbi_2out"] = np.where(
        (df["rbi_2out"].notna()) & (df["RBI"] > 0),
        df["rbi_2out"] / df["RBI"],
        np.nan,
    )

    df["sb_att_pg"] = df["sb_att_pg"].round(3)
    df["pct_runs_via_hr"] = df["pct_runs_via_hr"].round(3)
    df["pct_rbi_2out"] = df["pct_rbi_2out"].round(3)

    text_df = df.copy()
    text_df["power_profile"] = text_df["pct_runs_via_hr"].apply(classify_power)
    text_df["pressure_profile"] = text_df["sb_att_pg"].apply(classify_pressure)
    text_df["clutch_profile"] = text_df["pct_rbi_2out"].apply(classify_clutch)

    column_order = [
        "team_id",
        "team_display",
        "g",
        "runs_scored",
        "HR",
        "RBI",
        "SB",
        "CS",
        "sb_attempts",
        "sb_att_pg",
        "rbi_on_hr",
        "pct_runs_via_hr",
        "rbi_2out",
        "pct_rbi_2out",
    ]
    df_csv = df[column_order]

    df_csv = df_csv.sort_values(
        by=["pct_runs_via_hr", "sb_att_pg"],
        ascending=[False, False],
        na_position="last",
    )

    out_path = (base_dir / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_csv.to_csv(out_path, index=False)

    text_path = resolve_text_path(out_path)
    text_path.write_text(build_text_report(text_df), encoding="utf-8")

    preview = df_csv.head(12)
    print("Run Creation Profile (top 12):")
    print(preview.to_string(index=False))
    print(f"\nWrote {len(df_csv)} rows to {out_path} and summary to {text_path}.")


if __name__ == "__main__":
    main()

