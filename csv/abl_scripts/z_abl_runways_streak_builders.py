"""ABL Runways/Streak Builders report."""

from __future__ import annotations

import argparse
import numbers
from datetime import timedelta
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from abl_config import stamp_text_block

TEAM_MIN, TEAM_MAX = 1, 24

SCHEDULE_CANDIDATES = [
    "schedule.csv",
    "games.csv",
    "team_game_log.csv",
    "game_results_by_team.csv",
]
RECORD_CANDIDATES = [
    "team_record.csv",
    "standings.csv",
    "teams.csv",
]
TEAM_INFO_CANDIDATES = [
    "team_info.csv",
    "teams.csv",
    "team_record.csv",
]
PARK_CANDIDATES = [
    "parks.csv",
    "park_info.csv",
    "park_factors.csv",
]


def pick_column(df: pd.DataFrame, *names: str) -> Optional[str]:
    lowered = {c.lower(): c for c in df.columns}
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


def load_schedule(base: Path, override: Optional[Path]) -> pd.DataFrame:
    df = read_first(base, override, SCHEDULE_CANDIDATES)
    if df is None:
        raise FileNotFoundError("Unable to locate schedule/games data.")
    date_col = pick_column(df, "game_date", "date")
    home_col = pick_column(df, "home_team_id", "home_team", "home", "team0")
    away_col = pick_column(df, "away_team_id", "away_team", "away", "team1")
    park_col = pick_column(df, "park_id", "park")
    game_id_col = pick_column(df, "game_id", "game_key")
    played_col = pick_column(df, "played", "is_final")
    result_col = pick_column(df, "result")
    runs_home_col = pick_column(df, "runs0", "home_score")
    runs_away_col = pick_column(df, "runs1", "away_score")
    if not date_col or not home_col or not away_col:
        raise ValueError("Schedule missing key columns.")
    data = df.copy()
    data["game_date"] = pd.to_datetime(data[date_col])
    data["home_team_id"] = pd.to_numeric(data[home_col], errors="coerce").astype("Int64")
    data["away_team_id"] = pd.to_numeric(data[away_col], errors="coerce").astype("Int64")
    if park_col:
        park_series = data[park_col].astype(str)
        park_series = park_series.where(~park_series.str.lower().isin(["nan", "none", ""]))
        data["park_id"] = park_series.fillna("")
    else:
        data["park_id"] = ""
    if played_col:
        data["played"] = data[played_col].astype(str).str.lower().isin(["1", "true", "t", "y", "yes"])
    elif result_col:
        data["played"] = data[result_col].notna()
    elif runs_home_col and runs_away_col:
        data["played"] = df[runs_home_col].notna() & df[runs_away_col].notna()
    else:
        data["played"] = False
    data["game_id"] = data[game_id_col] if game_id_col else np.nan
    filtered = data[
        (
            data["home_team_id"].between(TEAM_MIN, TEAM_MAX)
            | data["away_team_id"].between(TEAM_MIN, TEAM_MAX)
        )
    ].copy()
    return filtered[["game_id", "game_date", "home_team_id", "away_team_id", "park_id", "played"]]


def load_team_records(base: Path, override: Optional[Path]) -> pd.DataFrame:
    df = read_first(base, override, RECORD_CANDIDATES)
    if df is None:
        raise FileNotFoundError("Unable to locate team records.")
    team_col = pick_column(df, "team_id", "teamid")
    w_col = pick_column(df, "w", "wins")
    l_col = pick_column(df, "l", "losses")
    r_col = pick_column(df, "r", "runs_scored")
    ra_col = pick_column(df, "ra", "runs_allowed")
    g_col = pick_column(df, "g", "games")
    if not team_col or not w_col or not l_col:
        raise ValueError("Records missing wins/losses.")
    data = df.copy()
    data["team_id"] = pd.to_numeric(data[team_col], errors="coerce").astype("Int64")
    data = data.dropna(subset=["team_id"])
    data = data[(data["team_id"] >= TEAM_MIN) & (data["team_id"] <= TEAM_MAX)]
    data["W"] = pd.to_numeric(data[w_col], errors="coerce").fillna(0)
    data["L"] = pd.to_numeric(data[l_col], errors="coerce").fillna(0)
    data["G"] = pd.to_numeric(data[g_col], errors="coerce").fillna(data["W"] + data["L"])
    data["R"] = pd.to_numeric(data[r_col], errors="coerce") if r_col else np.nan
    data["RA"] = pd.to_numeric(data[ra_col], errors="coerce") if ra_col else np.nan
    data["Wpct"] = data["W"] / (data["W"] + data["L"]).replace(0, np.nan)
    data["R_pg"] = data["R"] / data["G"].replace(0, np.nan)
    data["RA_pg"] = data["RA"] / data["G"].replace(0, np.nan)
    return data[["team_id", "Wpct", "R_pg", "RA_pg"]]


def load_teams(base: Path, override: Optional[Path]) -> Tuple[Dict[int, str], Dict[int, str]]:
    df = read_first(base, override, TEAM_INFO_CANDIDATES)
    if df is None:
        return {}, {}
    team_col = pick_column(df, "team_id", "teamid", "TeamID")
    abbr_col = pick_column(df, "abbr", "team_abbr", "short_name")
    name_col = pick_column(df, "team_display", "team_name", "name", "TeamName")
    sub_col = pick_column(df, "sub_league_id", "subleague_id", "league_id")
    div_col = pick_column(df, "division_id", "division")
    team_display: Dict[int, str] = {}
    conf_map: Dict[int, str] = {}
    conf_lookup = {0: "N", 1: "A"}
    div_lookup = {0: "E", 1: "C", 2: "W"}
    if not team_col:
        return team_display, conf_map
    for _, row in df.iterrows():
        tid_val = row.get(team_col)
        if pd.isna(tid_val):
            continue
        try:
            tid = int(tid_val)
        except (TypeError, ValueError):
            continue
        if not (TEAM_MIN <= tid <= TEAM_MAX):
            continue
        display = None
        if abbr_col and pd.notna(row.get(abbr_col)) and str(row.get(abbr_col)).strip():
            display = str(row.get(abbr_col)).strip()
        elif name_col and pd.notna(row.get(name_col)):
            display = str(row.get(name_col)).strip()
        team_display[tid] = display or str(tid)
        if (tid not in conf_map) and sub_col and div_col:
            sub_val = row.get(sub_col)
            div_val = row.get(div_col)
            if pd.notna(sub_val) and pd.notna(div_val):
                try:
                    sub_key = int(sub_val)
                except (TypeError, ValueError):
                    sub_key = None
                try:
                    div_key = int(div_val)
                except (TypeError, ValueError):
                    div_key = None
                conf_map[tid] = f"{conf_lookup.get(sub_key, str(sub_val)[0].upper())}-{div_lookup.get(div_key, str(div_val)[0].upper())}"
    return team_display, conf_map


def load_team_home_parks(base: Path, override: Optional[Path]) -> Dict[int, str]:
    df = read_first(base, override, TEAM_INFO_CANDIDATES)
    if df is None:
        return {}
    team_col = pick_column(df, "team_id", "teamid")
    park_col = pick_column(df, "park_id", "home_park_id")
    mapping = {}
    if not team_col or not park_col:
        return mapping
    for _, row in df.iterrows():
        tid_val = row.get(team_col)
        if pd.isna(tid_val):
            continue
        try:
            tid = int(tid_val)
        except (TypeError, ValueError):
            continue
        if TEAM_MIN <= tid <= TEAM_MAX and pd.notna(row.get(park_col)):
            mapping[tid] = str(row.get(park_col))
    return mapping


def load_park_factors(base: Path, override: Optional[Path]) -> Dict[str, float]:
    df = read_first(base, override, PARK_CANDIDATES)
    if df is None:
        return {}
    park_col = pick_column(df, "park_id", "ParkID", "park")
    run_col = pick_column(df, "run_factor", "pf_runs", "runsfactor")
    if not park_col or not run_col:
        return {}
    mapping = {}
    for _, row in df.iterrows():
        pid = row.get(park_col)
        val = row.get(run_col)
        if pd.notna(pid) and pd.notna(val):
            mapping[str(pid)] = float(val)
    return mapping


def determine_today(schedule: pd.DataFrame, today_str: Optional[str]) -> pd.Timestamp:
    if today_str:
        return pd.to_datetime(today_str)
    played_dates = schedule.loc[schedule["played"], "game_date"]
    if not played_dates.empty:
        return played_dates.max()
    return schedule["game_date"].min() - timedelta(days=1)


def upcoming_games(schedule: pd.DataFrame, today: pd.Timestamp, window: int) -> pd.DataFrame:
    future = schedule[~schedule["played"]].copy()
    if future.empty:
        future = schedule[schedule["game_date"] > today].copy()
    rows = []
    for team_id in range(TEAM_MIN, TEAM_MAX + 1):
        team_games = future[
            (future["home_team_id"] == team_id) | (future["away_team_id"] == team_id)
        ].copy()
        team_games["is_home"] = team_games["home_team_id"] == team_id
        team_games["opp_id"] = team_games.apply(
            lambda r: r["away_team_id"] if r["is_home"] else r["home_team_id"],
            axis=1,
        )
        team_games = team_games.sort_values("game_date").head(window)
        team_games["game_no"] = range(1, len(team_games) + 1)
        team_games["team_id"] = team_id
        rows.append(team_games)
    return pd.concat(rows, ignore_index=True)


def safe_div(numer: float, denom: float) -> float:
    if pd.isna(numer) or pd.isna(denom) or denom == 0:
        return np.nan
    return numer / denom


def runway_rating(score: float) -> str:
    if pd.isna(score):
        return ""
    if score >= 0.65:
        return "Clear Skies"
    if score >= 0.55:
        return "Smooth Runway"
    if score >= 0.45:
        return "Turbulent"
    return "Stormy"


def text_table(
    df: pd.DataFrame,
    columns: Sequence[Tuple[str, str, int, bool, str]],
    title: str,
    threshold_line: str,
    key_lines: Sequence[str],
    def_lines: Sequence[str],
) -> str:
    lines = [title, "=" * len(title), ""]
    header = " ".join(
        f"{label:<{width}}" if not align_right else f"{label:>{width}}"
        for label, _, width, align_right, _ in columns
    )
    lines.append(header)
    lines.append("-" * len(header))
    if df.empty:
        lines.append("(No teams qualified for this view.)")
    for _, row in df.iterrows():
        parts = []
        for _, col_name, width, align_right, fmt in columns:
            value = row.get(col_name, "")
            if isinstance(value, numbers.Number):
                if pd.isna(value):
                    display = "NA"
                else:
                    display = format(value, fmt) if fmt else str(value)
            else:
                display = str(value)
            fmt_str = f"{{:>{width}}}" if align_right else f"{{:<{width}}}"
            parts.append(fmt_str.format(display[:width]))
        lines.append(" ".join(parts))
    lines.append("")
    lines.append(threshold_line)
    lines.append("")
    lines.append("Key:")
    for line in key_lines:
        lines.append(f"  {line}")
    lines.append("")
    lines.append("Definitions:")
    for line in def_lines:
        lines.append(f"  {line}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="ABL Runways/Streak Builders.")
    parser.add_argument("--base", type=str, default=".", help="Base directory.")
    parser.add_argument("--schedule", type=str, help="Override schedule file.")
    parser.add_argument("--records", type=str, help="Override team records file.")
    parser.add_argument("--parks", type=str, help="Override park factors.")
    parser.add_argument("--teams", type=str, help="Override team info.")
    parser.add_argument("--out_next10", type=str, default="out/csv_out/z_ABL_Runways_Next10_Games.csv", help="Next games CSV.")
    parser.add_argument("--out_summary", type=str, default="out/csv_out/z_ABL_Runways_Summary.csv", help="Summary CSV.")
    parser.add_argument("--today", type=str, help="Override today date (YYYY-MM-DD).")
    parser.add_argument("--window", type=int, default=10, help="Number of games to scan.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    base_dir = Path(args.base).resolve()
    schedule = load_schedule(base_dir, resolve_path(base_dir, args.schedule))
    records = load_team_records(base_dir, resolve_path(base_dir, args.records))
    team_parks = load_team_home_parks(base_dir, resolve_path(base_dir, args.teams))
    park_factors = load_park_factors(base_dir, resolve_path(base_dir, args.parks))
    team_names, conf_map = load_teams(base_dir, resolve_path(base_dir, args.teams))

    today = determine_today(schedule, args.today)
    future_games = upcoming_games(schedule, today, args.window)

    lg_r = records["R_pg"].mean(skipna=True)
    lg_ra = records["RA_pg"].mean(skipna=True)

    merged = future_games.merge(
        records.rename(columns={"team_id": "opp_id"}),
        on="opp_id",
        how="left",
        suffixes=("", "_opp"),
    )

    def resolve_park(row: pd.Series) -> str:
        raw = row.get("park_id", "")
        if isinstance(raw, str) and raw:
            return raw
        home_tid = row.get("home_team_id")
        if pd.notna(home_tid):
            return str(team_parks.get(int(home_tid), ""))
        return ""

    merged["park_id"] = merged.apply(resolve_park, axis=1).fillna("")
    merged["PRF"] = merged["park_id"].map(park_factors).fillna(1.0)
    merged["opp_Wpct"] = merged["Wpct"]
    merged["opp_R_pg"] = merged["R_pg"]
    merged["opp_RA_pg"] = merged["RA_pg"]

    merged["opp_def_factor"] = merged.apply(
        lambda r: safe_div(r["opp_RA_pg"], lg_ra) if pd.notna(r["opp_RA_pg"]) and pd.notna(lg_ra) else 1.0,
        axis=1,
    )
    merged["opp_off_factor"] = merged.apply(
        lambda r: safe_div(r["opp_R_pg"], lg_r) if pd.notna(r["opp_R_pg"]) and pd.notna(lg_r) else 1.0,
        axis=1,
    )

    merged["PRE_for"] = merged["PRF"] * merged["opp_def_factor"]
    merged["PRE_against"] = merged["PRF"] * merged["opp_off_factor"]
    merged["ease_component"] = (1.0 - merged["opp_Wpct"].fillna(0.5)).clip(0.0, 1.0)
    merged["is_home_flag"] = np.where(merged["is_home"], "Y", "")
    merged["team_display"] = merged["team_id"].map(team_names).fillna(merged["team_id"].astype(str))
    merged["conf_div"] = merged["team_id"].map(conf_map).fillna("")

    next10_cols = [
        "team_id",
        "game_no",
        "game_date",
        "is_home_flag",
        "opp_id",
        "opp_Wpct",
        "park_id",
        "PRF",
        "PRE_for",
        "PRE_against",
        "ease_component",
    ]
    next10 = merged[next10_cols].copy()
    next10["opp_Wpct"] = next10["opp_Wpct"].round(3)
    for col in ["PRF", "PRE_for", "PRE_against", "ease_component"]:
        next10[col] = next10[col].round(3)
    if not next10.empty:
        next10["game_date"] = pd.to_datetime(next10["game_date"]).dt.strftime("%Y-%m-%d")
    next10 = next10.sort_values(["team_id", "game_no"]).reset_index(drop=True)
    next10_text_df = merged.copy()
    next10_text_df["game_date_str"] = pd.to_datetime(next10_text_df["game_date"]).dt.strftime("%Y-%m-%d")
    next10_text_df["ha"] = np.where(next10_text_df["is_home"], "H", "A")
    next10_text_df["opp_display"] = next10_text_df["opp_id"].map(team_names).fillna(next10_text_df["opp_id"].astype(str))
    for col in ["opp_Wpct", "PRF", "PRE_for", "PRE_against", "ease_component"]:
        next10_text_df[col] = next10_text_df[col].round(3)
    next10_text_df = next10_text_df.sort_values(["team_id", "game_no"]).reset_index(drop=True)

    summary = merged.groupby("team_id", as_index=False).agg(
        games_count=("game_no", "count"),
        avg_opp_Wpct=("opp_Wpct", "mean"),
        avg_PRF=("PRF", "mean"),
        avg_PRE_for=("PRE_for", "mean"),
        avg_PRE_against=("PRE_against", "mean"),
        ease=("ease_component", "mean"),
    )
    all_teams = pd.DataFrame({"team_id": list(range(TEAM_MIN, TEAM_MAX + 1))})
    summary = all_teams.merge(summary, on="team_id", how="left")
    summary["games_count"] = summary["games_count"].fillna(0).astype(int)
    summary["env_diff"] = summary["avg_PRE_for"] - summary["avg_PRE_against"]
    summary["env_diff_clamped"] = summary["env_diff"].clip(-0.5, 0.5)
    summary["RunwayScore"] = (
        0.7 * summary["ease"]
        + 0.3 * (summary["env_diff_clamped"] + 0.5)
    )
    summary["RunwayScore"] = summary["RunwayScore"].fillna(summary["ease"])
    summary["soft_opponents"] = np.where(summary["avg_opp_Wpct"] <= 0.480, "Y", "")
    summary["hitter_friendly"] = np.where(summary["avg_PRE_for"] >= 1.05, "Y", "")
    summary["pitcher_warning"] = np.where(summary["avg_PRE_against"] >= 1.05, "Y", "")
    summary["team_display"] = summary["team_id"].map(team_names).fillna(summary["team_id"].astype(str))
    summary["conf_div"] = summary["team_id"].map(conf_map).fillna("")
    summary["flag_summary"] = summary.apply(
        lambda r: ",".join(
            flag
            for flag, cond in [
                ("SOF", r["soft_opponents"] == "Y"),
                ("HIT", r["hitter_friendly"] == "Y"),
                ("ARM", r["pitcher_warning"] == "Y"),
            ]
            if cond
        ),
        axis=1,
    )
    summary["runway_rating"] = summary["RunwayScore"].apply(runway_rating)
    summary = summary.sort_values(
        by=["RunwayScore", "avg_opp_Wpct", "env_diff"],
        ascending=[False, True, False],
        na_position="last",
    )

    next10_path = Path(args.out_next10)
    if not next10_path.is_absolute():
        next10_path = base_dir / next10_path
    next10_path.parent.mkdir(parents=True, exist_ok=True)
    next10.rename(columns={"is_home_flag": "is_home"}, inplace=True)
    next10.to_csv(next10_path, index=False)
    next10_text_columns = [
        ("Team", "team_display", 8, False, ""),
        ("Conf", "conf_div", 6, False, ""),
        ("No", "game_no", 3, True, ".0f"),
        ("Date", "game_date_str", 10, False, ""),
        ("H/A", "ha", 3, False, ""),
        ("Opponent", "opp_display", 9, False, ""),
        ("Opp%", "opp_Wpct", 6, True, ".3f"),
        ("PRF", "PRF", 5, True, ".3f"),
        ("For", "PRE_for", 5, True, ".3f"),
        ("Ag", "PRE_against", 5, True, ".3f"),
        ("Ease", "ease_component", 5, True, ".3f"),
    ]
    next10_text_view = next10_text_df.copy()
    next10_text_view["team_display"] = next10_text_view["team_display"].fillna(next10_text_view["team_id"].astype(str))
    next10_text_view["conf_div"] = next10_text_view["conf_div"].fillna("")
    next10_text_output = text_table(
        next10_text_view.head(60),
        next10_text_columns,
        "Upcoming Runway - Next Games",
        f"Showing first 60 matchups across each team's next {args.window} games.",
        [
            "H/A indicates home (H) or road (A) for the listed team.",
            "Ease = 1 - opponent W% (capped 0-1). Higher PRF/For favor hitters; higher Against warns of tougher pitching environments.",
        ],
        [
            "PRF = park run factor (1.00 = neutral). PRE For/Ag blend park + opponent scoring rates.",
            "Opp% is the opponent winning percentage from team records.",
        ],
    )
    txt_dir = base_dir / "out" / "text_out"
    txt_dir.mkdir(parents=True, exist_ok=True)
    next10_txt_path = txt_dir / next10_path.with_suffix(".txt").name
    next10_txt_path.write_text(stamp_text_block(next10_text_output), encoding="utf-8")

    summary_path = Path(args.out_summary)
    if not summary_path.is_absolute():
        summary_path = base_dir / summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_out = summary[
        [
            "team_id",
            "team_display",
            "conf_div",
            "games_count",
            "avg_opp_Wpct",
            "avg_PRF",
            "avg_PRE_for",
            "avg_PRE_against",
            "RunwayScore",
            "runway_rating",
            "soft_opponents",
            "hitter_friendly",
            "pitcher_warning",
        ]
    ].copy()
    for col in ["avg_opp_Wpct", "avg_PRF", "avg_PRE_for", "avg_PRE_against", "RunwayScore"]:
        summary_out[col] = summary_out[col].round(3)
    summary_out.to_csv(summary_path, index=False)

    easiest = summary.head(10)
    toughest = summary.sort_values(by="RunwayScore", ascending=True).head(10)
    columns = [
        ("Team", "team_display", 8, False, ""),
        ("Conf", "conf_div", 6, False, ""),
        ("Gms", "games_count", 4, True, ".0f"),
        ("Runway", "RunwayScore", 7, True, ".3f"),
        ("Rating", "runway_rating", 13, False, ""),
        ("Opp%", "avg_opp_Wpct", 6, True, ".3f"),
        ("EnvDiff", "env_diff", 7, True, ".3f"),
        ("Flags", "flag_summary", 10, False, ""),
    ]
    key_lines = [
        "Ratings: Clear Skies (>=0.65), Smooth Runway (0.55-0.64), Turbulent (0.45-0.54), Stormy (<0.45).",
        "Flags: SOF=soft opponents (<=.480 opp%), HIT=hitter-friendly environment (PRE_for>=1.05), ARM=pitcher caution (PRE_against>=1.05).",
    ]
    def_lines = [
        "RunwayScore blends opponent ease (70%) with run-environment edge (30%).",
        "EnvDiff = avg_PRE_for - avg_PRE_against (positive favors hitters).",
        f"Window covers each team's next {args.window} scheduled games (played flag = 0).",
    ]
    easiest_table = text_table(
        easiest,
        columns,
        "Top Runways (Easiest)",
        f"Most favorable upcoming {args.window}-game windows",
        key_lines,
        def_lines,
    )
    toughest_table = text_table(
        toughest,
        columns,
        "Tough Runways",
        f"Most challenging upcoming {args.window}-game windows",
        key_lines,
        def_lines,
    )
    text_output = f"{easiest_table}\n\n{toughest_table}"
    text_path = txt_dir / summary_path.with_suffix(".txt").name
    text_path.write_text(stamp_text_block(text_output), encoding="utf-8")
    print(text_output)


if __name__ == "__main__":
    main()
