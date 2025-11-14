"""ABL late-inning clutch report."""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from abl_config import stamp_text_block

TEAM_MIN, TEAM_MAX = 1, 24
LINESCORE_CANDIDATES = [
    "game_linescore.csv",
    "linescores.csv",
    "games_linescore.csv",
    "team_game_log.csv",
    "teams_game_log.csv",
    "game_log_team.csv",
    "team_log.csv",
    "schedule_results.csv",
    "games.csv",
    "games_score.csv",
]
BYSPLIT_CANDIDATES = [
    "team_splits_by_inning.csv",
    "team_inning_splits.csv",
    "team_splits_by_innings.csv",
]


def build_game_level_from_games_score(base: Path) -> Optional[pd.DataFrame]:
    score_path = base / "games_score.csv"
    if not score_path.exists():
        return None
    score_df = pd.read_csv(score_path)
    required = {"game_id", "team", "inning", "score"}
    if not required.issubset(score_df.columns):
        return None
    try:
        score_df["team"] = pd.to_numeric(score_df["team"], errors="coerce").astype("Int64")
        score_df["inning"] = pd.to_numeric(score_df["inning"], errors="coerce").astype("Int64")
        score_df["score"] = pd.to_numeric(score_df["score"], errors="coerce")
    except Exception:
        return None
    pivot = (
        score_df.pivot_table(index=["game_id", "team"], columns="inning", values="score", aggfunc="sum")
        .reset_index()
    )
    away = pivot[pivot["team"] == 0].drop(columns=["team"])
    home = pivot[pivot["team"] == 1].drop(columns=["team"])
    if away.empty or home.empty:
        return None

    def rename_cols(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        rename_map = {}
        for col in df.columns:
            if isinstance(col, (int, float)):
                rename_map[col] = f"{prefix}{int(col)}"
        return df.rename(columns=rename_map)

    away = rename_cols(away, "a")
    home = rename_cols(home, "h")
    merged = away.merge(home, on="game_id", how="inner", suffixes=("", ""))
    games_path = base / "games.csv"
    if games_path.exists():
        games_df = pd.read_csv(games_path)
        away_id_col = pick_column(games_df, "away_team_id", "away_team", "team0", "visteam")
        home_id_col = pick_column(games_df, "home_team_id", "home_team", "team1", "hometeam")
        date_col = pick_column(games_df, "game_date", "date", "gamedate", "GameDate")
        columns_to_keep = ["game_id"]
        rename_map = {}
        if away_id_col:
            columns_to_keep.append(away_id_col)
            rename_map[away_id_col] = "away_team_id"
        if home_id_col:
            columns_to_keep.append(home_id_col)
            rename_map[home_id_col] = "home_team_id"
        if date_col:
            columns_to_keep.append(date_col)
            rename_map[date_col] = "game_date"
        merged = merged.merge(games_df[columns_to_keep].rename(columns=rename_map), on="game_id", how="left")
    return merged


def pick_column(df: pd.DataFrame, *names: str) -> Optional[str]:
    lowered = {col.lower(): col for col in df.columns}
    for name in names:
        key = name.lower()
        if key in lowered:
            return lowered[key]
    return None


def build_team_meta(base: Path) -> Dict[int, dict]:
    path = base / "teams.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    team_col = pick_column(df, "team_id", "teamid", "teamID", "TeamID")
    if not team_col:
        return {}
    name_col = pick_column(df, "team_display", "team_name", "name", "nickname")
    city_col = pick_column(df, "city", "city_name")
    nickname_col = pick_column(df, "nickname")
    abbr_col = pick_column(df, "abbr")
    sub_col = pick_column(df, "sub_league_id", "sub_league")
    div_col = pick_column(df, "division_id", "division")
    division_map = {0: "E", 1: "C", 2: "W"}
    meta: Dict[int, dict] = {}
    for _, row in df.iterrows():
        tid = row.get(team_col)
        if pd.isna(tid):
            continue
        tid_int = int(tid)
        if tid_int in meta:
            continue
        if name_col and pd.notna(row.get(name_col)):
            name_value = str(row.get(name_col))
        elif city_col and nickname_col and pd.notna(row.get(city_col)) and pd.notna(row.get(nickname_col)):
            name_value = f"{row.get(city_col)} {row.get(nickname_col)}"
        elif city_col and pd.notna(row.get(city_col)):
            name_value = str(row.get(city_col))
        elif nickname_col and pd.notna(row.get(nickname_col)):
            name_value = str(row.get(nickname_col))
        elif abbr_col and pd.notna(row.get(abbr_col)):
            name_value = str(row.get(abbr_col))
        else:
            name_value = ""
        conference_letter = ""
        if sub_col and pd.notna(row.get(sub_col)):
            conference_letter = "N" if int(row.get(sub_col)) == 0 else "A"
        division_letter = ""
        if div_col and pd.notna(row.get(div_col)):
            division_letter = division_map.get(int(row.get(div_col)), "")
        conf_div = ""
        if conference_letter and division_letter:
            conf_div = f"{conference_letter}-{division_letter}"
        elif conference_letter:
            conf_div = conference_letter
        elif division_letter:
            conf_div = division_letter
        meta[tid_int] = {"name": name_value, "conf_div": conf_div}
    return meta


def expand_games_to_team_rows(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    home_col = pick_column(df, "home_team", "home_team_id", "hometeam", "team1")
    away_col = pick_column(df, "away_team", "away_team_id", "awayteam", "team0")
    home_runs_col = pick_column(df, "home_runs", "runs_home", "score1", "runs1", "r_home")
    away_runs_col = pick_column(df, "away_runs", "runs_away", "score0", "runs0", "r_away")
    date_col = pick_column(df, "game_date", "date")
    if not all([home_col, away_col, home_runs_col, away_runs_col]):
        return None
    records = []
    for _, row in df.iterrows():
        home_id = pd.to_numeric(row[home_col], errors="coerce")
        away_id = pd.to_numeric(row[away_col], errors="coerce")
        if pd.isna(home_id) or pd.isna(away_id):
            continue
        home_runs = pd.to_numeric(row[home_runs_col], errors="coerce")
        away_runs = pd.to_numeric(row[away_runs_col], errors="coerce")
        date_val = pd.to_datetime(row[date_col], errors="coerce") if date_col else pd.NaT
        records.append(
            {
                "team_id": int(home_id),
                "runs_for": home_runs,
                "runs_against": away_runs,
                "result": "W" if home_runs > away_runs else "L" if home_runs < away_runs else "T",
                "game_date": date_val,
            }
        )
        records.append(
            {
                "team_id": int(away_id),
                "runs_for": away_runs,
                "runs_against": home_runs,
                "result": "W" if away_runs > home_runs else "L" if away_runs < home_runs else "T",
                "game_date": date_val,
            }
        )
    if not records:
        return None
    return pd.DataFrame(records)


def autodetect_linescore(base: Path, override: Optional[Path]) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
    score_game = build_game_level_from_games_score(base)
    if score_game is not None:
        return score_game, str(base / "games_score.csv"), "game"
    candidates = [Path(override)] if override else [base / name for name in LINESCORE_CANDIDATES]
    for path in candidates:
        if path is None or not path.exists():
            continue
        df = pd.read_csv(path)
        if "team_id" in df.columns and has_inning_cols_team(df):
            return df, str(path), "team"
        if has_home_away_ids(df) and has_inning_cols_game(df):
            return df, str(path), "game"
        expanded = expand_games_to_team_rows(df)
        if expanded is not None and "result" in expanded.columns:
            if has_inning_cols_team(expanded):
                return expanded, str(path), "team"
    return None, None, None


def has_home_away_ids(df: pd.DataFrame) -> bool:
    away = pick_column(df, "away_team_id", "away_id", "team0", "visteam", "visitor_team_id")
    home = pick_column(df, "home_team_id", "home_id", "team1", "hometeam", "home_team")
    return bool(away and home)


def has_inning_cols_team(df: pd.DataFrame) -> bool:
    for_cols, against_cols = detect_team_inning_columns(df.columns)
    return bool(for_cols and against_cols)


def detect_team_inning_columns(columns: List[str]) -> Tuple[Dict[int, str], Dict[int, str]]:
    pattern = re.compile(r"(?:^|_)(?:inn|i)?(\d+)[^0-9]*(for|against)$")
    for_cols: Dict[int, str] = {}
    against_cols: Dict[int, str] = {}
    for col in columns:
        lower = col.lower()
        match = pattern.search(lower)
        if match:
            inning = int(match.group(1))
            label = match.group(2)
            if label == "for":
                for_cols[inning] = col
            else:
                against_cols[inning] = col
    return for_cols, against_cols


def has_inning_cols_game(df: pd.DataFrame) -> bool:
    away, home = detect_game_inning_columns(df.columns)
    return bool(away and home)


def detect_game_inning_columns(columns: List[str]) -> Tuple[Dict[int, str], Dict[int, str]]:
    away_cols: Dict[int, str] = {}
    home_cols: Dict[int, str] = {}
    patterns = [
        (re.compile(r"^a(\d+)$"), "away"),
        (re.compile(r"^h(\d+)$"), "home"),
        (re.compile(r"^(?:away|visitor|vis)[ _-]?(\d+)$"), "away"),
        (re.compile(r"^(?:home|host)[ _-]?(\d+)$"), "home"),
        (re.compile(r"^(?:away|visitor)[ _-]?inning[ _-]?(\d+)$"), "away"),
        (re.compile(r"^(?:home)[ _-]?inning[ _-]?(\d+)$"), "home"),
    ]
    for col in columns:
        lower = col.lower()
        for pattern, label in patterns:
            match = pattern.match(lower)
            if match:
                inning = int(match.group(1))
                if label == "away":
                    away_cols[inning] = col
                else:
                    home_cols[inning] = col
                break
    return away_cols, home_cols


def autodetect_splits(base: Path, override: Optional[Path]) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    candidates = [Path(override)] if override else [base / name for name in BYSPLIT_CANDIDATES]
    for path in candidates:
        if path is None or not path.exists():
            continue
        df = pd.read_csv(path)
        if "team_id" in df.columns and has_inning_cols_team(df):
            return df, str(path)
    return None, None


def build_records_from_team_rows(df: pd.DataFrame) -> pd.DataFrame:
    for_cols, against_cols = detect_team_inning_columns(df.columns)
    if not for_cols or not against_cols:
        return pd.DataFrame()
    rows = []
    date_col = pick_column(df, "game_date", "date", "gamedate", "GameDate")
    game_id_col = pick_column(df, "game_id", "gameid")
    for _, row in df.iterrows():
        team_id = row.get("team_id")
        if pd.isna(team_id):
            continue
        team_id = int(team_id)
        for_innings = {inning: pd.to_numeric(row.get(col), errors="coerce") for inning, col in for_cols.items()}
        against_innings = {inning: pd.to_numeric(row.get(col), errors="coerce") for inning, col in against_cols.items()}
        if not for_innings or not against_innings:
            continue
        rows.append(
            build_record_from_innings(
                team_id=team_id,
                for_innings=for_innings,
                against_innings=against_innings,
                game_date=pd.to_datetime(row[date_col], errors="coerce") if date_col else pd.NaT,
                game_id=row.get(game_id_col),
            )
        )
    return pd.DataFrame(rows)


def build_records_from_game_rows(df: pd.DataFrame) -> pd.DataFrame:
    away_id_col = pick_column(df, "away_team_id", "away_id", "visitor_team_id", "team0", "visteam", "away_team")
    home_id_col = pick_column(df, "home_team_id", "home_id", "team1", "hometeam", "home_team")
    if not away_id_col or not home_id_col:
        return pd.DataFrame()
    away_cols, home_cols = detect_game_inning_columns(df.columns)
    if not away_cols or not home_cols:
        return pd.DataFrame()
    date_col = pick_column(df, "game_date", "date", "gamedate", "GameDate")
    game_id_col = pick_column(df, "game_id", "gameid")
    records = []
    for _, row in df.iterrows():
        away_id = pd.to_numeric(row[away_id_col], errors="coerce")
        home_id = pd.to_numeric(row[home_id_col], errors="coerce")
        if pd.isna(away_id) or pd.isna(home_id):
            continue
        for team_label, team_id, cols in [
            ("away", int(away_id), away_cols),
            ("home", int(home_id), home_cols),
        ]:
            for_innings = {inning: pd.to_numeric(row.get(col), errors="coerce") for inning, col in cols.items()}
            opp_innings_map = home_cols if team_label == "away" else away_cols
            against_innings = {inning: pd.to_numeric(row.get(col), errors="coerce") for inning, col in opp_innings_map.items()}
            if not for_innings or not against_innings:
                continue
            records.append(
                build_record_from_innings(
                    team_id=team_id,
                    for_innings=for_innings,
                    against_innings=against_innings,
                    game_date=pd.to_datetime(row[date_col], errors="coerce") if date_col else pd.NaT,
                    game_id=row.get(game_id_col),
                )
            )
    return pd.DataFrame(records)


def build_record_from_innings(
    team_id: int,
    for_innings: Dict[int, float],
    against_innings: Dict[int, float],
    game_date,
    game_id,
) -> Dict[str, object]:
    innings = set(for_innings.keys()) | set(against_innings.keys())
    if not innings:
        return {}
    runs_for_total = sum(v for v in for_innings.values() if pd.notna(v))
    runs_against_total = sum(v for v in against_innings.values() if pd.notna(v))
    runs_for_6 = sum(v for inning, v in for_innings.items() if inning <= 6 and pd.notna(v))
    runs_against_6 = sum(v for inning, v in against_innings.items() if inning <= 6 and pd.notna(v))
    runs_for_7p = sum(v for inning, v in for_innings.items() if inning >= 7 and pd.notna(v))
    runs_against_7p = sum(v for inning, v in against_innings.items() if inning >= 7 and pd.notna(v))
    if math.isnan(runs_for_total) or math.isnan(runs_against_total):
        return {}
    win_flag = 1 if runs_for_total > runs_against_total else 0
    trailing_after6 = runs_for_6 < runs_against_6
    leading_after6 = runs_for_6 > runs_against_6
    comeback = 1 if trailing_after6 and win_flag else 0
    blown = 1 if leading_after6 and not win_flag else 0
    return {
        "team_id": team_id,
        "runs_for_7p": runs_for_7p,
        "runs_against_7p": runs_against_7p,
        "comeback_win": comeback,
        "blown_lead": blown,
        "game_date": game_date,
        "game_id": game_id,
    }


def aggregate_team_metrics(records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame()
    records = records[(records["team_id"] >= TEAM_MIN) & (records["team_id"] <= TEAM_MAX)]
    if records.empty:
        return pd.DataFrame()
    grouped = records.groupby("team_id")
    agg = grouped.agg(
        g=("team_id", "size"),
        runs_for_7p=("runs_for_7p", "sum"),
        runs_against_7p=("runs_against_7p", "sum"),
        comeback_wins=("comeback_win", "sum"),
        blown_leads=("blown_lead", "sum"),
    ).reset_index()
    agg["run_diff_7p"] = agg["runs_for_7p"] - agg["runs_against_7p"]
    agg["late_runs_per_game_for"] = agg["runs_for_7p"] / agg["g"]
    agg["late_runs_per_game_against"] = agg["runs_against_7p"] / agg["g"]
    return agg


def aggregate_from_splits(df: pd.DataFrame) -> pd.DataFrame:
    for_cols, against_cols = detect_team_inning_columns(df.columns)
    if not for_cols or not against_cols:
        return pd.DataFrame()
    rows = []
    for _, row in df.iterrows():
        team_id = row.get("team_id")
        if pd.isna(team_id):
            continue
        team_id = int(team_id)
        runs_for_7p = sum(pd.to_numeric(row.get(col), errors="coerce") for inning, col in for_cols.items() if inning >= 7)
        runs_against_7p = sum(pd.to_numeric(row.get(col), errors="coerce") for inning, col in against_cols.items() if inning >= 7)
        rows.append(
            {
                "team_id": team_id,
                "g": pd.NA,
                "runs_for_7p": runs_for_7p,
                "runs_against_7p": runs_against_7p,
                "run_diff_7p": runs_for_7p - runs_against_7p,
                "late_runs_per_game_for": pd.NA,
                "late_runs_per_game_against": pd.NA,
                "comeback_wins": pd.NA,
                "blown_leads": pd.NA,
            }
        )
    result = pd.DataFrame(rows)
    result = result[(result["team_id"] >= TEAM_MIN) & (result["team_id"] <= TEAM_MAX)]
    return result


def build_text_report(df: pd.DataFrame, limit: int = 24) -> str:
    lines = [
        "ABL Late-Inning Clutch",
        "=" * 26,
        "Shows which clubs win the 7th inning onward: run differential, comeback wins, and blown leads.",
        "Great for identifying playoff-ready resilience and spotting bullpens that spring leaks late.",
        "",
    ]
    header = f"{'Team':<20} {'CD':<4} {'Profile':<8} {'RunDiff 7+':>11} {'7+ RF/RA':>14} {'Comebacks':>11} {'Blown L':>9}"
    lines.append(header)
    lines.append("-" * len(header))
    for _, row in df.head(limit).iterrows():
        name = row["team_display"] or f"Team {int(row['team_id'])}"
        conf_div = row.get("conf_div") or "--"
        diff = row["run_diff_7p"]
        tag = "Surge" if diff >= 10 else "Steady" if diff >= 0 else "Fade"
        runs_for = f"{row['runs_for_7p']:.1f}" if pd.notna(row["runs_for_7p"]) else "NA "
        runs_against = f"{row['runs_against_7p']:.1f}" if pd.notna(row["runs_against_7p"]) else "NA "
        comeback = f"{int(row['comeback_wins'])}" if pd.notna(row["comeback_wins"]) else "NA"
        blown = f"{int(row['blown_leads'])}" if pd.notna(row["blown_leads"]) else "NA"
        diff_txt = f"{diff:+.1f}" if pd.notna(diff) else " NA "
        lines.append(
            f"{name:<20} {conf_div:<4} {tag:<8} {diff_txt:>11} {f'{runs_for}/{runs_against}':>14} {comeback:>11} {blown:>9}"
        )
    lines.append("")
    lines.append("Key:")
    lines.append("  Surge  -> 7th+ run differential >= +10.")
    lines.append("  Steady -> between 0 and +9.9.")
    lines.append("  Fade   -> negative run differential in late innings.")
    lines.append("")
    lines.append("Definitions:")
    lines.append("  Late innings = 7th inning onward (including extras).")
    lines.append("  Comeback win = trailing after 6 complete and still win.")
    lines.append("  Blown lead = leading after 6 complete but lose the game.")
    return "\n".join(lines)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ABL late-inning clutch report.")
    parser.add_argument("--base", type=str, default=".", help="Base directory for CSVs.")
    parser.add_argument("--linescore", type=str, help="Explicit linescore file.")
    parser.add_argument("--pbp", type=str, help="Explicit play-by-play file (not yet supported).")
    parser.add_argument("--splits", type=str, help="Explicit inning splits file.")
    parser.add_argument(
        "--out",
        type=str,
        default="out/csv_out/z_ABL_Late_Inning_Clutch.csv",
        help="Output CSV (default: out/csv_out/z_ABL_Late_Inning_Clutch.csv).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    base_dir = Path(args.base).resolve()
    meta = build_team_meta(base_dir)

    linescore_df, source_path, mode = autodetect_linescore(base_dir, Path(args.linescore) if args.linescore else None)
    report_df: Optional[pd.DataFrame] = None
    data_source = None
    if linescore_df is not None and mode:
        if mode == "team":
            records = build_records_from_team_rows(linescore_df)
        else:
            records = build_records_from_game_rows(linescore_df)
        report_df = aggregate_team_metrics(records)
        data_source = source_path

    if report_df is None or report_df.empty:
        splits_df, splits_path = autodetect_splits(base_dir, Path(args.splits) if args.splits else None)
        if splits_df is not None:
            report_df = aggregate_from_splits(splits_df)
            data_source = splits_path

    if report_df is None or report_df.empty:
        raise FileNotFoundError("No usable linescore, pbp, or inning splits file found.")

    report_df["team_display"] = ""
    report_df["conf_div"] = ""
    for idx, row in report_df.iterrows():
        tid = int(row["team_id"])
        info = meta.get(tid, {})
        if not report_df.at[idx, "team_display"]:
            report_df.at[idx, "team_display"] = info.get("name", "")
        report_df.at[idx, "conf_div"] = info.get("conf_div", "")

    report_df["run_diff_7p"] = report_df["run_diff_7p"].round(1)
    if "late_runs_per_game_for" in report_df.columns:
        report_df["late_runs_per_game_for"] = report_df["late_runs_per_game_for"].round(2)
    if "late_runs_per_game_against" in report_df.columns:
        report_df["late_runs_per_game_against"] = report_df["late_runs_per_game_against"].round(2)

    report_df = report_df.sort_values(
        by=["run_diff_7p", "comeback_wins", "blown_leads"],
        ascending=[False, False, True],
        na_position="last",
    )

    column_order = [
        "team_id",
        "team_display",
        "g",
        "runs_for_7p",
        "runs_against_7p",
        "run_diff_7p",
        "late_runs_per_game_for",
        "late_runs_per_game_against",
        "comeback_wins",
        "blown_leads",
    ]
    for col in column_order:
        if col not in report_df.columns:
            report_df[col] = pd.NA
    export_df = report_df[column_order].copy()

    output_path = (base_dir / args.out).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text_filename = output_path.with_suffix(".txt").name
    if output_path.parent.name.lower() in {'csv_out'}:
        text_dir = output_path.parent.parent / "text_out"
    else:
        text_dir = output_path.parent
    text_dir.mkdir(parents=True, exist_ok=True)
    text_path = text_dir / text_filename

    export_df.to_csv(output_path, index=False)
    text_path.write_text(stamp_text_block(build_text_report(report_df)), encoding="utf-8")

    preview = report_df.head(12)
    print("Late-inning clutch (top 12):")
    print(preview.to_string(index=False))
    print(f"\nWrote {len(report_df)} rows to {output_path} and summary to {text_path} (source: {data_source}).")


if __name__ == "__main__":
    main()

