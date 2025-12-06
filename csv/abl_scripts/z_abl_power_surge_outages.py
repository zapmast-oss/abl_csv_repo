"""ABL Power Surge & Outage tracker."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from abl_config import stamp_text_block

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
                if isinstance(value, (int, float, np.number)):
                    if pd.isna(value):
                        text_value = "NA"
                    else:
                        text_value = format(value, fmt) if fmt else str(value)
                else:
                    text_value = str(value)
                fmt_str = f"{{:>{width}}}" if align_right else f"{{:<{width}}}"
                parts.append(fmt_str.format(text_value[:width]))
            lines.append(" ".join(parts))
    lines.append("")
    if key_lines:
        lines.append("Key:")
        for entry in key_lines:
            lines.append(f"  {entry}")
        lines.append("")
    if def_lines:
        lines.append("Definitions:")
        for entry in def_lines:
            lines.append(f"  {entry}")
    return "\n".join(lines).rstrip()


TABLE_COLUMNS: Sequence[Tuple[str, str, int, bool, str]] = [
    ("Team", "team_display", 18, False, ""),
    ("Games", "games_current", 6, True, ".0f"),
    ("HR", "HR_current", 4, True, ".0f"),
    ("PA", "PA_current", 6, True, ".0f"),
    ("HR/PA", "HR_per_PA_current", 8, True, ".3f"),
    ("ΔHR/PA", "delta_HR_per_PA", 8, True, ".3f"),
]

CSV_COLUMNS = [
    "team_id",
    "team_display",
    "week_start",
    "week_end",
    "games_current",
    "HR_current",
    "PA_current",
    "HR_per_PA_current",
    "games_prev",
    "HR_prev",
    "PA_prev",
    "HR_per_PA_prev",
    "delta_HR_per_PA",
    "pct_change",
    "surge_flag",
    "rating",
]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track weekly HR surge/outage trends.")
    parser.add_argument("--base", type=str, default=".", help="Base directory for CSV exports.")
    parser.add_argument("--logs", type=str, help="Override team game log CSV.")
    parser.add_argument("--boxes", type=str, help="Override team box log CSV when logs missing.")
    parser.add_argument("--games", type=str, help="Override schedule/games CSV for park info.")
    parser.add_argument("--teams", type=str, help="Override team info CSV.")
    parser.add_argument("--parks", type=str, help="Override park info CSV.")
    parser.add_argument("--week-end", type=str, dest="week_end", help="Force week end date (YYYY-MM-DD).")
    parser.add_argument("--limit", type=int, default=10, help="Max entries per section in the text report.")
    parser.add_argument(
        "--out",
        type=str,
        default="out/csv_out/z_ABL_Power_Surge_Outages.csv",
        help="Output CSV path (defaults to out/csv_out/...).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def empty_logs_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["team_id", "game_date", "park_id", "HR", "PA"])


def classify_delta(delta: float) -> str:
    if pd.isna(delta):
        return ""
    if delta >= 0.015:
        return "Firestorm"
    if delta >= 0.008:
        return "Heating Up"
    if delta <= -0.015:
        return "Blackout"
    if delta <= -0.008:
        return "Cooling Off"
    return ""


def build_team_trends(
    logs: pd.DataFrame,
    team_names: Dict[int, str],
    week_end: Optional[str],
) -> Tuple[pd.DataFrame, Optional[Tuple[Tuple[pd.Timestamp, pd.Timestamp], Tuple[pd.Timestamp, pd.Timestamp]]]]:
    if logs.empty or logs["game_date"].dropna().empty:
        return pd.DataFrame(columns=CSV_COLUMNS), None
    current_window, prior_window = determine_weeks(logs["game_date"], week_end)
    current = aggregate_week(logs, ["team_id"], current_window[0], current_window[1])
    prior = aggregate_week(logs, ["team_id"], prior_window[0], prior_window[1])
    if current.empty:
        return pd.DataFrame(columns=CSV_COLUMNS), (current_window, prior_window)
    merged = merge_weeks(current, prior, ["team_id"])
    merged = merged.rename(
        columns={
            "games": "games_current",
            "HR": "HR_current",
            "PA": "PA_current",
            "HR_per_PA": "HR_per_PA_current",
        }
    )
    merged["team_display"] = merged["team_id"].apply(lambda tid: team_names.get(tid, f"Team {int(tid)}"))
    for col in ["games_prev", "HR_prev", "PA_prev"]:
        merged[col] = merged[col].fillna(0)
    merged["HR_per_PA_prev"] = merged["HR_per_PA_prev"]
    merged["week_start"] = merged["week_start"].dt.strftime("%Y-%m-%d")
    merged["week_end"] = merged["week_end"].dt.strftime("%Y-%m-%d")
    merged["delta_HR_per_PA"] = merged["delta_HR_per_PA"].round(4)
    merged["pct_change"] = merged["pct_change"].round(4)
    merged["HR_per_PA_current"] = merged["HR_per_PA_current"].round(4)
    merged["HR_per_PA_prev"] = merged["HR_per_PA_prev"].round(4)
    merged["rating"] = merged["delta_HR_per_PA"].apply(classify_delta)
    merged = merged.sort_values(by="delta_HR_per_PA", ascending=False, na_position="last").reset_index(drop=True)
    return merged, (current_window, prior_window)


def render_report_text(
    surges: pd.DataFrame,
    outages: pd.DataFrame,
    windows: Optional[Tuple[Tuple[pd.Timestamp, pd.Timestamp], Tuple[pd.Timestamp, pd.Timestamp]]],
    limit: int,
) -> str:
    header_lines = [
        "ABL Power Surge & Outage Tracker",
        "=================================",
        "Tracks week-over-week changes in team HR/PA to surface the hottest and coldest power bats.",
        "Why it matters: flags lineup or park-influenced power trends for storylines, broadcasts, and matchup prep.",
    ]
    if windows:
        (curr_start, curr_end), (prev_start, prev_end) = windows
        header = (
            f"Window: {curr_start:%Y-%m-%d} to {curr_end:%Y-%m-%d} "
            f"(compared to {prev_start:%Y-%m-%d} to {prev_end:%Y-%m-%d})"
        )
    else:
        header = "Window: No valid date range detected."
    header_block = "\n".join(header_lines + [header])
    surges_section = text_table(
        surges.head(limit),
        TABLE_COLUMNS,
        "Power Surges",
        "Largest increases in HR per PA week-over-week",
        [
            "Surge flag triggers when HR/PA jumps by >= 0.005 over the prior week.",
        ],
        [
            "HR/PA = home runs divided by plate appearances.",
            "HR/PA compares current seven-day window to the previous seven-day window.",
        ],
    )
    outages_section = text_table(
        outages.head(limit),
        TABLE_COLUMNS,
        "Power Outages",
        "Largest drops in HR per PA week-over-week",
        [
            "Outage flag triggers when HR/PA falls by <= -0.005 over the prior week.",
        ],
        [
            "Games/HR/PA columns reference the current seven-day window.",
            "Prior metrics are included in the CSV for deeper dives.",
        ],
    )
    return "\n\n".join([header_block, surges_section, outages_section])

def write_report(
    base_dir: Path,
    csv_path_value: str,
    report_df: pd.DataFrame,
    text_payload: str,
) -> None:
    out_path = (base_dir / csv_path_value).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_df = report_df.reindex(columns=CSV_COLUMNS)
    csv_df.to_csv(out_path, index=False)
    text_filename = out_path.with_suffix(".txt").name
    if out_path.parent.name.lower() == "csv_out":
        text_dir = out_path.parent.parent / "text_out"
    else:
        text_dir = out_path.parent
    text_dir.mkdir(parents=True, exist_ok=True)
    (text_dir / text_filename).write_text(stamp_text_block(text_payload), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    base_dir = Path(args.base).resolve()
    logs = empty_logs_frame()
    try:
        logs = load_logs(
            base_dir,
            resolve_path(base_dir, args.logs),
            resolve_path(base_dir, args.boxes),
            resolve_path(base_dir, args.games),
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Warning: {exc}")
    team_names = load_team_names(base_dir, resolve_path(base_dir, args.teams))
    report_df, windows = build_team_trends(logs, team_names, args.week_end)
    if report_df.empty:
        text_payload = (
            "ABL Power Surge & Outage Tracker\n"
            "Tracks week-over-week changes in team HR/PA to surface the hottest and coldest power bats.\n"
            "Data unavailable for the requested window; ensure batting logs are exported before running this report."
        )
        write_report(base_dir, args.out, report_df, text_payload)
        print("No power surge/outage data available; emitted placeholder outputs.")
        return
    surges = report_df[report_df["surge_flag"] == "SURGE"].sort_values(
        by="delta_HR_per_PA", ascending=False, na_position="last"
    )
    outages = report_df[report_df["surge_flag"] == "OUTAGE"].sort_values(
        by="delta_HR_per_PA", ascending=True, na_position="last"
    )
    text_payload = render_report_text(surges, outages, windows, args.limit)
    write_report(base_dir, args.out, report_df, text_payload)
    print(
        f"Power surge/outage report created with {len(report_df)} teams: "
        f"{surges['team_id'].nunique()} surges, {outages['team_id'].nunique()} outages."
    )


if __name__ == "__main__":
    main()
