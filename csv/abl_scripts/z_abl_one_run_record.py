"""ABL one-run game performance report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from abl_config import stamp_text_block

TEAM_MIN, TEAM_MAX = 1, 24
LOG_CANDIDATES = [
    "team_game_log.csv",
    "teams_game_log.csv",
    "game_log_team.csv",
    "team_log.csv",
    "schedule_results.csv",
    "games.csv",
    "games_score.csv",
]


def pick_column(df: pd.DataFrame, *names: str) -> Optional[str]:
    lowered = {col.lower(): col for col in df.columns}
    for name in names:
        key = name.lower()
        if key in lowered:
            return lowered[key]
    return None


def expand_games_to_team_rows(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    home_col = pick_column(df, "home_team", "home_team_id", "hometeam", "team1")
    away_col = pick_column(df, "away_team", "away_team_id", "awayteam", "team0")
    home_runs_col = pick_column(df, "home_runs", "runs_home", "score1", "runs1", "r_home")
    away_runs_col = pick_column(df, "away_runs", "runs_away", "score0", "runs0", "r_away")
    date_col = pick_column(df, "game_date", "date")
    played_col = pick_column(df, "played")
    if not all([home_col, away_col, home_runs_col, away_runs_col]):
        return None
    if played_col:
        played_values = pd.to_numeric(df[played_col], errors="coerce")
        df = df[played_values == 1]
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


def autodetect_logs(base: Path, override: Optional[Path]) -> Tuple[pd.DataFrame, Path]:
    if override:
        if not override.exists():
            raise FileNotFoundError(f"Specified log file not found: {override}")
        return pd.read_csv(override), override
    for candidate in LOG_CANDIDATES:
        path = base / candidate
        if not path.exists():
            continue
        if candidate.lower().startswith("games"):
            raw = pd.read_csv(path)
            expanded = expand_games_to_team_rows(raw)
            if expanded is not None:
                return expanded, path
            continue
        return pd.read_csv(path), path
    raise FileNotFoundError(
        f"Could not locate a team log in {base}. Looked for: {', '.join(LOG_CANDIDATES)}"
    )


def build_team_meta(base: Path) -> Dict[int, Dict[str, str]]:
    teams_path = base / "teams.csv"
    if not teams_path.exists():
        return {}
    df = pd.read_csv(teams_path)
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

    meta: Dict[int, Dict[str, str]] = {}
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


def load_team_game_limits(base: Path) -> Dict[int, int]:
    record_path = base / "team_record.csv"
    if not record_path.exists():
        return {}
    df = pd.read_csv(record_path, usecols=["team_id", "g"])
    df = df.dropna(subset=["team_id", "g"])
    df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce").astype("Int64")
    df["g"] = pd.to_numeric(df["g"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["team_id"])
    grouped = df.groupby("team_id")["g"].max()
    return {int(tid): int(games) for tid, games in grouped.items()}


def determine_win_and_margin(df: pd.DataFrame, result_col: Optional[str], runs_for_col: Optional[str], runs_against_col: Optional[str]) -> pd.DataFrame:
    df = df.copy()
    win_flag = pd.Series(pd.NA, index=df.index, dtype="Float64")
    if result_col:
        win_flag = df[result_col].astype(str).str.upper().str.startswith("W")
    if runs_for_col and runs_against_col:
        runs_for = pd.to_numeric(df[runs_for_col], errors="coerce")
        runs_against = pd.to_numeric(df[runs_against_col], errors="coerce")
    else:
        runs_for = pd.Series(pd.NA, index=df.index, dtype="Float64")
        runs_against = pd.Series(pd.NA, index=df.index, dtype="Float64")

    if result_col is None:
        mask = runs_for.notna() & runs_against.notna()
        win_flag = pd.Series(pd.NA, index=df.index, dtype="Float64")
        win_flag[mask] = runs_for[mask] > runs_against[mask]
    else:
        missing_mask = win_flag.isna()
        mask = missing_mask & runs_for.notna() & runs_against.notna()
        win_flag[mask] = runs_for[mask] > runs_against[mask]

    win_flag = pd.Series(win_flag, dtype="Float64")
    margin = pd.Series(pd.NA, index=df.index, dtype="Float64")
    played_mask = runs_for.notna() & runs_against.notna()
    zero_mask = (runs_for == 0) & (runs_against == 0)
    mask_margin = played_mask & ~zero_mask
    margin[mask_margin] = (runs_for[mask_margin] - runs_against[mask_margin]).abs()
    win_flag[~mask_margin] = pd.NA

    df["win_flag"] = win_flag
    df["runs_for"] = runs_for
    df["runs_against"] = runs_against
    df["margin"] = margin
    return df


def build_report(
    df: pd.DataFrame,
    meta: Dict[int, Dict[str, str]],
    team_limits: Optional[Dict[int, int]] = None,
) -> pd.DataFrame:
    team_col = pick_column(df, "team_id", "teamid", "teamID", "TeamID")
    if not team_col:
        raise ValueError("team_id column is required in the log.")
    result_col = pick_column(df, "result")
    runs_for_col = pick_column(df, "runs_for", "r", "rs", "runsfor")
    runs_against_col = pick_column(df, "runs_against", "ra", "runsagainst")
    date_col = pick_column(df, "game_date", "date", "gamedate", "GameDate")
    display_col = pick_column(df, "team_display", "team_name", "name", "TeamName")

    work = df.copy()
    work["team_id"] = pd.to_numeric(work[team_col], errors="coerce").astype("Int64")
    work = work[(work["team_id"] >= TEAM_MIN) & (work["team_id"] <= TEAM_MAX)]
    if date_col:
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        valid_dates = work[date_col].dropna()
        if not valid_dates.empty:
            latest_year = valid_dates.dt.year.max()
            work = work[work[date_col].dt.year == latest_year]
        work = work.sort_values(date_col)
    else:
        work = work.sort_values(team_col)

    if team_limits:
        pieces = []
        for tid, group in work.groupby("team_id", sort=False):
            limit = int(team_limits.get(int(tid), len(group)))
            if limit <= 0:
                continue
            pieces.append(group.tail(limit))
        if pieces:
            work = pd.concat(pieces, ignore_index=True)

    work = determine_win_and_margin(work, result_col, runs_for_col, runs_against_col)
    work["win_flag"] = pd.to_numeric(work["win_flag"], errors="coerce")

    overall_df = work.dropna(subset=["win_flag"])
    if overall_df.empty:
        return pd.DataFrame(
            columns=[
                "team_id",
                "team_display",
                "conf_div",
                "overall_g",
                "overall_w",
                "overall_l",
                "overall_winpct",
                "one_run_g",
                "one_run_w",
                "one_run_l",
                "one_run_winpct",
                "one_run_diff_winpct",
                "one_run_share",
            ]
        )

    grouped = overall_df.groupby("team_id")
    overall_stats = grouped["win_flag"].agg(["sum", "count"])
    overall_stats.rename(columns={"sum": "overall_w", "count": "overall_g"}, inplace=True)
    overall_stats["overall_l"] = overall_stats["overall_g"] - overall_stats["overall_w"]
    overall_stats["overall_winpct"] = overall_stats["overall_w"] / overall_stats["overall_g"]

    one_run_df = overall_df[overall_df["margin"] == 1]
    one_grouped = one_run_df.groupby("team_id")
    one_stats = one_grouped["win_flag"].agg(["sum", "count"])
    one_stats.rename(columns={"sum": "one_run_w", "count": "one_run_g"}, inplace=True)
    one_stats["one_run_l"] = one_stats["one_run_g"] - one_stats["one_run_w"]
    one_stats["one_run_winpct"] = one_stats["one_run_w"] / one_stats["one_run_g"]

    result = overall_stats.join(one_stats, how="left")
    result["one_run_diff_winpct"] = result["one_run_winpct"] - result["overall_winpct"]
    result["one_run_share"] = result["one_run_g"] / result["overall_g"]

    result.reset_index(inplace=True)
    result["team_display"] = ""
    result["conf_div"] = ""
    if display_col:
        latest_names = (
            work.dropna(subset=["team_id"])
            .drop_duplicates(subset=["team_id"], keep="last")
            .set_index("team_id")
        )
        for idx, row in result.iterrows():
            tid = int(row["team_id"])
            if tid in latest_names.index and pd.notna(latest_names.at[tid, display_col]):
                result.at[idx, "team_display"] = str(latest_names.at[tid, display_col])
    for idx, row in result.iterrows():
        tid = int(row["team_id"])
        if tid in meta:
            if not result.at[idx, "team_display"]:
                result.at[idx, "team_display"] = meta[tid].get("name", "")
            result.at[idx, "conf_div"] = meta[tid].get("conf_div", "")

    result["overall_g"] = result["overall_g"].astype(int)
    result["overall_w"] = result["overall_w"].astype(int)
    result["overall_l"] = result["overall_l"].astype(int)
    result["one_run_g"] = result["one_run_g"].fillna(0).astype(int)
    result["one_run_w"] = result["one_run_w"].fillna(0).astype(int)
    result["one_run_l"] = result["one_run_l"].fillna(0).astype(int)

    result["overall_winpct"] = result["overall_winpct"].round(3)
    result["one_run_winpct"] = result["one_run_winpct"].round(3)
    result["one_run_diff_winpct"] = result["one_run_diff_winpct"].round(3)
    result["one_run_share"] = result["one_run_share"].round(3)

    result = result.sort_values(
        by=["one_run_diff_winpct", "one_run_winpct"],
        ascending=[False, False],
        na_position="last",
    )

    column_order = [
        "team_id",
        "team_display",
        "conf_div",
        "overall_g",
        "overall_w",
        "overall_l",
        "overall_winpct",
        "one_run_g",
        "one_run_w",
        "one_run_l",
        "one_run_winpct",
        "one_run_diff_winpct",
        "one_run_share",
    ]
    return result[column_order]


def build_text_report(df: pd.DataFrame, limit: int = 24) -> str:
    lines = [
        "ABL One-Run Record",
        "=" * 22,
        "Shows how each club fares in one-run decisions versus its overall paceâ€”perfect for clutch/luck chatter.",
        "Use it to flag teams overachieving (or melting down) in coin-flip finishes.",
        "",
    ]
    header = (
        f"{'Team':<20} {'CD':<4} {'Profile':<8} {'1R W-L':>10} {'Pct':>6} "
        f"{'Overall':>8} {'dPct':>7}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for _, row in df.head(limit).iterrows():
        name = row["team_display"] or f"Team {int(row['team_id'])}"
        conf_div = row.get("conf_div") or "--"
        diff = row["one_run_diff_winpct"]
        tag = "Clutch" if diff >= 0.05 else "Cold" if diff <= -0.05 else "Even"
        one_run_rec = f"{int(row['one_run_w'])}-{int(row['one_run_l'])}"
        one_run_pct = f"{row['one_run_winpct']:.3f}"
        overall_pct = f"{row['overall_winpct']:.3f}"
        diff_txt = f"{diff:+.3f}"
        lines.append(
            f"{name:<20} {conf_div:<4} {tag:<8} {one_run_rec:>10} "
            f"{one_run_pct:>6} {overall_pct:>8} {diff_txt:>7}"
        )
    lines.append("")
    lines.append("Key:")
    lines.append("  Clutch -> One-run win% exceeds overall by 0.050+")
    lines.append("  Even   -> Within +/- 0.049 of overall win%")
    lines.append("  Cold   -> One-run win% trails overall by 0.050+")
    return "\n".join(lines)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ABL one-run record report.")
    parser.add_argument("--base", type=str, default=".", help="Base directory for CSVs.")
    parser.add_argument("--logs", type=str, help="Explicit path to team log CSV.")
    parser.add_argument(
        "--out",
        type=str,
        default="out/csv_out/z_ABL_One_Run_Record.csv",
        help="Output CSV (default: out/csv_out/z_ABL_One_Run_Record.csv).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    base_dir = Path(args.base).resolve()
    logs_override = Path(args.logs).resolve() if args.logs else None

    log_df, source_path = autodetect_logs(base_dir, logs_override)
    meta = build_team_meta(base_dir)
    team_limits = load_team_game_limits(base_dir)
    report_df = build_report(log_df, meta, team_limits)

    output_path = (base_dir / args.out).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text_filename = output_path.with_suffix(".txt").name
    if output_path.parent.name.lower() in {'csv_out'}:
        text_dir = output_path.parent.parent / "txt_out"
    else:
        text_dir = output_path.parent
    text_dir.mkdir(parents=True, exist_ok=True)
    text_path = text_dir / text_filename

    report_df.to_csv(output_path, index=False)
    if report_df.empty:
        text_path.write_text(stamp_text_block("No qualifying games found."), encoding="utf-8")
        print("No qualifying teams found; CSV is empty.")
        return

    text_path.write_text(stamp_text_block(build_text_report(report_df)), encoding="utf-8")

    preview = report_df.head(12)
    print("One-run performance (top 12):")
    print(preview.to_string(index=False))
    print(
        f"\nWrote {len(report_df)} rows to {output_path} "
        f"and summary to {text_path} (source: {source_path})."
    )


if __name__ == "__main__":
    main()

