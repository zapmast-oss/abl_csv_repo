"""ABL Momentum Windows: compare last-N vs prior-N win percentages per team."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from abl_config import stamp_text_block

CANDIDATE_LOGS = [
    "team_game_log.csv",
    "teams_game_log.csv",
    "game_log_team.csv",
    "team_log.csv",
    "schedule_results.csv",
    "games.csv",
]
TEAM_MIN, TEAM_MAX = 1, 24


def pick_column(df: pd.DataFrame, *names: str) -> Optional[str]:
    lowered = {col.lower(): col for col in df.columns}
    for name in names:
        key = name.lower()
        if key in lowered:
            return lowered[key]
    return None


def load_game_log(base: Path, override: Optional[Path]) -> Tuple[pd.DataFrame, Path]:
    if override:
        if not override.exists():
            raise FileNotFoundError(f"Specified log file not found: {override}")
        return pd.read_csv(override), override
    for candidate in CANDIDATE_LOGS:
        candidate_path = base / candidate
        if candidate_path.exists():
            return pd.read_csv(candidate_path), candidate_path
    raise FileNotFoundError(
        f"Unable to find a game log in {base}. Looked for: {', '.join(CANDIDATE_LOGS)}"
    )


def expand_home_away_frame(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    home_team_col = pick_column(df, "home_team", "hometeam", "home_id", "team1")
    away_team_col = pick_column(df, "away_team", "awayteam", "away_id", "team0")
    date_col = pick_column(df, "game_date", "date", "gamedate", "GameDate")
    if not (home_team_col and away_team_col and date_col):
        return None
    home_runs_col = pick_column(
        df, "home_runs", "runs_home", "home_score", "score1", "runs1"
    )
    away_runs_col = pick_column(
        df, "away_runs", "runs_away", "away_score", "score0", "runs0"
    )
    if not (home_runs_col and away_runs_col):
        return None
    home_name_col = pick_column(df, "home_team_name", "home_team_display", "home_name")
    away_name_col = pick_column(df, "away_team_name", "away_team_display", "away_name")
    played_col = pick_column(df, "played", "completed")

    records: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        date_val = row[date_col]
        home_team = row[home_team_col]
        away_team = row[away_team_col]
        home_runs = row[home_runs_col]
        away_runs = row[away_runs_col]
        home_disp = row[home_name_col] if home_name_col else None
        away_disp = row[away_name_col] if away_name_col else None
        played_val = row[played_col] if played_col else None

        for is_home in (True, False):
            team_id = home_team if is_home else away_team
            opp_id = away_team if is_home else home_team
            runs_for = home_runs if is_home else away_runs
            runs_against = away_runs if is_home else home_runs
            team_display = home_disp if is_home else away_disp

            if pd.isna(team_id) or pd.isna(date_val):
                continue

            result = None
            if pd.notna(runs_for) and pd.notna(runs_against):
                if runs_for > runs_against:
                    result = "W"
                elif runs_for < runs_against:
                    result = "L"
                else:
                    result = "T"

            records.append(
                {
                    "team_id": team_id,
                    "team_display": team_display,
                    "game_date": date_val,
                    "runs_for": runs_for,
                    "runs_against": runs_against,
                    "result": result,
                    "opp_id": opp_id,
                    "played": played_val,
                }
            )
    if not records:
        return None
    return pd.DataFrame.from_records(records)


def ensure_team_log(df: pd.DataFrame) -> pd.DataFrame:
    team_col = pick_column(df, "team_id", "teamid", "teamID", "TeamID")
    date_col = pick_column(df, "game_date", "date", "gamedate", "GameDate")
    if team_col and date_col:
        result = df.copy()
    else:
        expanded = expand_home_away_frame(df)
        if expanded is not None:
            result = expanded
        else:
            raise ValueError(
                "Could not locate per-team columns (team_id/game_date) or convertible home/away columns."
            )
    played_col = pick_column(result, "played", "completed")
    if played_col:
        mask = pd.to_numeric(result[played_col], errors="coerce").fillna(0) != 0
        result = result[mask]
    return result
    raise ValueError(
        "Could not locate per-team columns (team_id/game_date) or convertible home/away columns."
    )


def compute_win_mask(
    df: pd.DataFrame,
    result_col: Optional[str],
    rf_col: Optional[str],
    ra_col: Optional[str],
) -> pd.Series:
    if result_col:
        return df[result_col].astype(str).str.upper().str.startswith("W")
    if rf_col and ra_col:
        rf = pd.to_numeric(df[rf_col], errors="coerce")
        ra = pd.to_numeric(df[ra_col], errors="coerce")
        return rf > ra
    raise ValueError(
        "Need a result column (values starting with W/L) or both runs_for and runs_against columns."
    )


def summarize_window(
    group: pd.DataFrame, index_slice: pd.Index, win_mask: pd.Series
) -> Tuple[Optional[int], Optional[int], Optional[float]]:
    games = len(index_slice)
    if games == 0:
        return None, None, None
    wins = int(win_mask.loc[index_slice].sum())
    losses = games - wins
    winpct = round(wins / games, 3) if games > 0 else None
    return wins, losses, winpct


def format_float(value: Optional[float]) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    return round(float(value), 3)


def safe_int_value(value: Optional[float]) -> int:
    if value is None or pd.isna(value):
        return 0
    return int(value)


def compute_current_streak(win_flags: pd.Series) -> Tuple[str, int]:
    if win_flags is None or win_flags.empty:
        return "", 0
    last_result = bool(win_flags.iloc[-1])
    streak = 0
    for val in reversed(win_flags.tolist()):
        if pd.isna(val):
            break
        if bool(val) == last_result:
            streak += 1
        else:
            break
    label = "W" if last_result else "L"
    return label, streak


def build_momentum(df: pd.DataFrame, window: int) -> pd.DataFrame:
    team_col = pick_column(df, "team_id", "teamid", "teamID", "TeamID")
    date_col = pick_column(df, "game_date", "date", "gamedate", "GameDate")
    result_col = pick_column(df, "result")
    runs_for_col = pick_column(df, "runs_for", "runsfor", "r", "rf", "RunsFor")
    runs_against_col = pick_column(df, "runs_against", "runsagainst", "ra", "RunsAgainst")
    display_col = pick_column(df, "team_display", "team_name", "name", "TeamName")

    if not team_col or not date_col:
        raise ValueError("team_id and game_date (or aliases) are required in the log.")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    df["team_id_work"] = pd.to_numeric(df[team_col], errors="coerce").astype("Int64")
    df = df[(df["team_id_work"] >= TEAM_MIN) & (df["team_id_work"] <= TEAM_MAX)]
    df = df.sort_values([team_col, date_col])

    if df.empty:
        return pd.DataFrame(
            columns=[
                "team_id",
                "team_display",
                "g_recent",
                "last10_w",
                "last10_l",
                "last10_winpct",
                "prior10_w",
                "prior10_l",
                "prior10_winpct",
                "delta_winpct",
                "streak",
                "description",
            ]
        )

    win_mask = compute_win_mask(df, result_col, runs_for_col, runs_against_col)

    records: List[Dict[str, Optional[float]]] = []
    for team_id, group in df.groupby(team_col):
        group = group.sort_values(date_col)
        recent = group.tail(window * 2)
        g_recent = len(recent)
        if g_recent == 0:
            continue

        # Keep original index positions so win_mask lookup stays aligned
        recent = recent.reset_index()
        last_slice = recent.tail(window)
        prior_slice = recent.iloc[max(0, len(recent) - window * 2) : max(0, len(recent) - window)]

        last_idx = pd.Index(last_slice["index"])
        prior_idx = pd.Index(prior_slice["index"]) if not prior_slice.empty else pd.Index([])

        last_w, last_l, last_pct = summarize_window(df, last_idx, win_mask)
        prior_w, prior_l, prior_pct = summarize_window(df, prior_idx, win_mask)

        delta = None
        if last_pct is not None and prior_pct is not None:
            delta = round(last_pct - prior_pct, 3)

        display_value = ""
        if display_col:
            latest_val = group[display_col].iloc[-1]
            display_value = str(latest_val) if pd.notna(latest_val) else ""

        win_flags = win_mask.loc[group.index]
        streak_label, streak_len = compute_current_streak(win_flags)
        streak_str = f"{streak_label}{streak_len}" if streak_len else ""

        records.append(
            {
                "team_id": int(team_id),
                "team_display": display_value,
                "g_recent": g_recent,
                "last10_w": last_w,
                "last10_l": last_l,
                "last10_winpct": last_pct,
                "prior10_w": prior_w,
                "prior10_l": prior_l,
                "prior10_winpct": prior_pct,
                "delta_winpct": delta,
                "streak": streak_str,
            }
        )

    result_df = pd.DataFrame(records)
    if result_df.empty:
        return result_df

    for col in ["last10_winpct", "prior10_winpct", "delta_winpct"]:
        result_df[col] = result_df[col].apply(format_float)

    result_df = result_df.sort_values(
        by=["delta_winpct", "last10_winpct"],
        ascending=[False, False],
        na_position="last",
    )
    result_df["description"] = result_df.apply(describe_momentum, axis=1)
    return result_df


def load_team_meta(base: Path) -> Tuple[Dict[int, str], Dict[int, str]]:
    candidates = [
        base / "team_info.csv",
        base / "team_record.csv",
        base / "team_records.csv",
        base / "teams.csv",
        base / "standings.csv",
    ]
    display_map: Dict[int, str] = {}
    conf_map: Dict[int, str] = {}
    for path in candidates:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        team_col = pick_column(df, "team_id", "teamid", "TeamID", "tid")
        if not team_col:
            continue
        display_col = pick_column(df, "team_display", "team_name", "name", "nickname")
        city_col = pick_column(df, "city", "city_name")
        nickname_col = pick_column(df, "nickname")
        sub_col = pick_column(df, "sub_league_id", "subleague_id", "sub_league", "conference_id")
        div_col = pick_column(df, "division_id", "division")
        conf_lookup = {0: "N", 1: "A"}
        div_lookup = {0: "E", 1: "C", 2: "W"}
        for _, row in df.iterrows():
            tid = row.get(team_col)
            if pd.isna(tid):
                continue
            tid_int = int(tid)
            if tid_int not in display_map:
                if display_col and pd.notna(row.get(display_col)):
                    display_map[tid_int] = str(row.get(display_col))
                elif city_col and nickname_col and pd.notna(row.get(city_col)) and pd.notna(row.get(nickname_col)):
                    display_map[tid_int] = f"{row.get(city_col)} {row.get(nickname_col)}"
                elif city_col and pd.notna(row.get(city_col)):
                    display_map[tid_int] = str(row.get(city_col))
            if tid_int not in conf_map and sub_col and div_col:
                sub_val = row.get(sub_col)
                div_val = row.get(div_col)
                if pd.notna(sub_val) and pd.notna(div_val):
                    try:
                        sub_key = int(sub_val)
                    except (ValueError, TypeError):
                        sub_key = None
                    try:
                        div_key = int(div_val)
                    except (ValueError, TypeError):
                        div_key = None
                    sub = conf_lookup.get(sub_key, str(sub_val)[0].upper())
                    div = div_lookup.get(div_key, str(div_val)[0].upper())
                    conf_map[tid_int] = f"{sub}-{div}"
    return display_map, conf_map


def describe_momentum(row: pd.Series) -> str:
    delta = row.get("delta_winpct")
    last_w = safe_int_value(row.get("last10_w"))
    last_l = safe_int_value(row.get("last10_l"))
    prior_w = safe_int_value(row.get("prior10_w"))
    prior_l = safe_int_value(row.get("prior10_l"))
    if delta is None or pd.isna(delta):
        return "Insufficient recent games for momentum read."
    tag = "OK"
    if delta >= 0.2:
        tag = "HOT"
    elif delta <= -0.2:
        tag = "COLD"
    trend = "gained" if delta >= 0 else "lost"
    streak = row.get("streak")
    streak_text = f" Current streak {streak}." if streak else ""
    return (
        f"{tag}: {trend} {abs(delta):.3f} (last10 {last_w}-{last_l} vs prior10 {prior_w}-{prior_l})."
        f"{streak_text}"
    )


def build_text_report(df: pd.DataFrame, window: int, limit: int = 24) -> str:
    lines = [
        "ABL Momentum Windows",
        "=" * 24,
        f"Compares each team's last {window} games vs the previous {window} to show real momentum swings.",
        "Great for broadcast color: highlight surges, slides, and steady clubs at a glance.",
        "",
    ]
    header = (
        f"{'Team':<26} {'dWin%':>7} {'Last':>9} {'Prior':>10} {'Streak':>8}  Note"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for _, row in df.head(limit).iterrows():
        name = row["team_display"] or f"Team {int(row['team_id'])}"
        combo = row.get("conf_div", "")
        header = f"{name} ({combo})" if combo else name
        delta = row["delta_winpct"]
        delta_txt = f"{delta:+.3f}" if delta is not None and not pd.isna(delta) else "N/A"
        last_record = f"{safe_int_value(row['last10_w'])}-{safe_int_value(row['last10_l'])}"
        prior_record = f"{safe_int_value(row['prior10_w'])}-{safe_int_value(row['prior10_l'])}"
        streak = row.get("streak") or "-"
        note = row.get("description", "")
        lines.append(
            f"{header:<26} {delta_txt:>7} {last_record:>9} {prior_record:>10} {streak:>8}  {note}"
        )
    lines.append("")
    lines.append("Key:")
    lines.append("  HOT  -> delta_winpct >= +0.200 (surging).")
    lines.append("  COLD -> delta_winpct <= -0.200 (sliding).")
    lines.append("  OK   -> between those bands (steady).")
    lines.append("")
    lines.append("Definitions:")
    lines.append("  Last10/Prior10: win/loss records for the latest vs previous window.")
    lines.append("  delta_winpct: difference between those win percentages.")
    lines.append("  Streak: active run of consecutive wins/losses (sign denotes W/L).")
    lines.append("  Conf-Div: conference/division combo (e.g., N-E).")
    return "\n".join(lines)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute ABL momentum windows (last-N vs prior-N win percentages)."
    )
    parser.add_argument(
        "--base",
        type=str,
        default=".",
        help="Base directory to search for logs (default: current directory).",
    )
    parser.add_argument(
        "--logs",
        type=str,
        help="Explicit path to a log CSV (overrides autodetect).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="Window size for comparison (default: 10).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="out/csv_out/z_ABL_Momentum_Windows.csv",
        help="Output CSV path (default: out/csv_out/z_ABL_Momentum_Windows.csv).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv or sys.argv[1:])

    base_dir = Path(args.base).resolve()
    override_path = Path(args.logs).resolve() if args.logs else None

    raw_df, source_path = load_game_log(base_dir, override_path)
    log_df = ensure_team_log(raw_df)
    result_df = build_momentum(log_df, args.window)

    display_map, conf_map = load_team_meta(base_dir)
    if not result_df.empty:
        if display_map:
            result_df["team_display"] = result_df.apply(
                lambda row: row["team_display"]
                if row["team_display"]
                else display_map.get(int(row["team_id"]), ""),
                axis=1,
            )
        result_df["conf_div"] = result_df["team_id"].map(conf_map).fillna("")

    output_path = (base_dir / args.out).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text_filename = output_path.with_suffix(".txt").name
    if output_path.parent.name.lower() in {'csv_out'}:
        text_dir = output_path.parent.parent / "text_out"
    else:
        text_dir = output_path.parent
    text_dir.mkdir(parents=True, exist_ok=True)
    text_path = text_dir / text_filename

    if not result_df.empty:
        ordered_cols = [
            "team_id",
            "team_display",
            "conf_div",
            "g_recent",
            "last10_w",
            "last10_l",
            "last10_winpct",
            "prior10_w",
            "prior10_l",
            "prior10_winpct",
            "delta_winpct",
            "streak",
            "description",
        ]
        result_df = result_df.reindex(columns=ordered_cols)
        text_path.write_text(stamp_text_block(build_text_report(result_df, window=args.window)), encoding="utf-8")
    else:
        text_path.write_text(stamp_text_block("No qualifying games found."), encoding="utf-8")

    result_df.to_csv(output_path, index=False)

    if result_df.empty:
        print("No qualifying games found; output CSV is empty.")
        return

    preview = result_df.head(12)
    print("Top momentum shifts:")
    print(preview.to_string(index=False))
    print(f"\nWrote {len(result_df)} rows to {output_path} (source: {source_path}).")


if __name__ == "__main__":
    main()

