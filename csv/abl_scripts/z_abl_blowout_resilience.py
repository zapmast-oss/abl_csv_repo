"""ABL blowout resilience report (regular-season only)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

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
    type_col = pick_column(df, "game_type", "type", "schedule_type")
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
        type_val = row.get(type_col) if type_col else None
        base_home = {
            "team_id": int(home_id),
            "runs_for": home_runs,
            "runs_against": away_runs,
            "result": "W" if home_runs > away_runs else "L" if home_runs < away_runs else "T",
            "game_date": date_val,
        }
        base_away = {
            "team_id": int(away_id),
            "runs_for": away_runs,
            "runs_against": home_runs,
            "result": "W" if away_runs > home_runs else "L" if away_runs < home_runs else "T",
            "game_date": date_val,
        }
        if type_col:
            base_home["game_type"] = type_val
            base_away["game_type"] = type_val
        records.append(base_home)
        records.append(base_away)
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


def determine_win_and_margin(
    df: pd.DataFrame,
    result_col: Optional[str],
    runs_for_col: Optional[str],
    runs_against_col: Optional[str],
) -> pd.DataFrame:
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
        inferred = pd.Series(pd.NA, index=df.index, dtype="Float64")
        inferred[mask] = runs_for[mask] > runs_against[mask]
        win_flag = inferred
    else:
        missing_mask = win_flag.isna()
        mask = missing_mask & runs_for.notna() & runs_against.notna()
        win_flag = pd.Series(win_flag, dtype="Float64")
        win_flag[mask] = runs_for[mask] > runs_against[mask]

    margin = pd.Series(pd.NA, index=df.index, dtype="Float64")
    played_mask = runs_for.notna() & runs_against.notna()
    zero_mask = (runs_for == 0) & (runs_against == 0)
    mask_margin = played_mask & ~zero_mask
    margin[mask_margin] = runs_for[mask_margin] - runs_against[mask_margin]
    win_flag[~mask_margin] = pd.NA

    df["win_flag"] = win_flag
    df["runs_for"] = runs_for
    df["runs_against"] = runs_against
    df["run_diff"] = margin
    return df


def build_report(df: pd.DataFrame, meta: Dict[int, dict]) -> pd.DataFrame:
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
    type_col = pick_column(work, "game_type", "type", "schedule_type")
    if type_col:
        numeric_types = pd.to_numeric(work[type_col], errors="coerce")
        numeric_mask = numeric_types == 0
        if numeric_mask.any():
            work = work[numeric_mask]
        else:
            text_mask = (
                work[type_col]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin({"regular", "regular season", "reg", "season"})
            )
            if text_mask.any():
                work = work[text_mask]
    playoff_col = pick_column(work, "is_playoff", "playoff", "postseason", "is_postseason")
    if playoff_col:
        playoff_mask = (
            work[playoff_col]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin({"1", "true", "yes", "y", "post", "playoff", "ps"})
        )
        if playoff_mask.any():
            work = work[~playoff_mask]
    if date_col:
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        valid_dates = work[date_col].dropna()
        if not valid_dates.empty:
            latest_year = valid_dates.dt.year.max()
            work = work[work[date_col].dt.year == latest_year]
        work = work.sort_values(date_col)

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
                "blowout_g",
                "blowout_w",
                "blowout_l",
                "blowout_winpct",
                "blowout_avg_margin",
                "blowout_share",
            ]
        )

    overall_stats = (
        overall_df.groupby("team_id")["win_flag"]
        .agg(overall_w="sum", overall_g="count")
        .reset_index()
    )
    overall_stats["overall_l"] = overall_stats["overall_g"] - overall_stats["overall_w"]
    overall_stats["overall_winpct"] = overall_stats["overall_w"] / overall_stats["overall_g"]

    blowout_df = overall_df.dropna(subset=["run_diff"])
    blowout_df = blowout_df[blowout_df["run_diff"].abs() >= 5]
    blowout_stats = (
        blowout_df.groupby("team_id")
        .agg(
            blowout_w=("win_flag", "sum"),
            blowout_g=("win_flag", "count"),
            blowout_avg_margin=("run_diff", "mean"),
        )
        .reset_index()
    )
    blowout_stats["blowout_l"] = blowout_stats["blowout_g"] - blowout_stats["blowout_w"]
    blowout_stats["blowout_winpct"] = blowout_stats["blowout_w"] / blowout_stats["blowout_g"]

    result = overall_stats.merge(blowout_stats, on="team_id", how="left")
    result["blowout_share"] = result["blowout_g"] / result["overall_g"]

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

    int_cols = ["overall_g", "overall_w", "overall_l", "blowout_g", "blowout_w", "blowout_l"]
    for col in int_cols:
        if col in result.columns:
            result[col] = result[col].fillna(0).astype(int)

    result["overall_winpct"] = result["overall_winpct"].round(3)
    result["blowout_winpct"] = result["blowout_winpct"].round(3)
    result["blowout_avg_margin"] = result["blowout_avg_margin"].round(2)
    result["blowout_share"] = result["blowout_share"].round(3)

    result = result.sort_values(
        by=["blowout_winpct", "blowout_g", "blowout_avg_margin"],
        ascending=[False, False, False],
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
        "blowout_g",
        "blowout_w",
        "blowout_l",
        "blowout_winpct",
        "blowout_avg_margin",
        "blowout_share",
    ]
    return result[column_order]


def build_text_report(df: pd.DataFrame, limit: int = 24) -> str:
    lines = [
        "ABL Blowout Resilience",
        "=" * 26,
        "Regular-season only: shows how often each club thrives in 5+ run decisions and how big those wins typically are.",
        "Meaning: higher win% and positive margin in blowouts signal resiliency when games get lopsided.",
        "",
    ]
    header = (
        f"{'Team':<22} {'CD':<4} {'Tag':<9} "
        f"{'Blowouts':>12} {'Win%':>7} {'AvgDiff':>8} {'Share':>7} {'Overall':>12}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for _, row in df.head(limit).iterrows():
        name = row["team_display"] or f"Team {int(row['team_id'])}"
        conf_div = row.get("conf_div") or "--"
        blowout_rate = row["blowout_winpct"]
        tag = "Dominant" if blowout_rate >= 0.7 else "Tough" if blowout_rate >= 0.5 else "Fragile"
        blowout_record = f"{int(row['blowout_w'])}-{int(row['blowout_l'])}"
        overall_record = f"{int(row['overall_w'])}-{int(row['overall_l'])}"
        lines.append(
            f"{name:<22} {conf_div:<4} {tag:<9} "
            f"{blowout_record:>12} {blowout_rate:>7.3f} {row['blowout_avg_margin']:>8.2f} "
            f"{row['blowout_share']:>7.3f} {overall_record:>12}"
        )
    lines.append("")
    lines.append("Key:")
    lines.append("  Dominant -> blowout win% >= 0.700")
    lines.append("  Tough    -> blowout win% between 0.500 and 0.699")
    lines.append("  Fragile  -> blowout win% < 0.500")
    lines.append("")
    lines.append("Definitions:")
    lines.append("  Blowout game: decided by 5+ runs (abs(runs_for - runs_against) >= 5).")
    lines.append("  Share = blowout games / total games; AvgDiff = mean run differential in blowouts.")
    return "\n".join(lines)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ABL blowout resilience report.")
    parser.add_argument("--base", type=str, default=".", help="Base directory for CSVs.")
    parser.add_argument("--logs", type=str, help="Explicit path to team log CSV.")
    parser.add_argument(
        "--out",
        type=str,
        default="out/z_ABL_Blowout_Resilience.csv",
        help="Output CSV (default: out/z_ABL_Blowout_Resilience.csv).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    base_dir = Path(args.base).resolve()
    logs_override = Path(args.logs).resolve() if args.logs else None

    log_df, source_path = autodetect_logs(base_dir, logs_override)
    meta = build_team_meta(base_dir)
    report_df = build_report(log_df, meta)

    output_path = (base_dir / args.out).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text_dir = base_dir / "out" / "txt_out"
    text_dir.mkdir(parents=True, exist_ok=True)
    text_path = text_dir / output_path.name.replace(".csv", ".txt")

    report_df.to_csv(output_path, index=False)
    if report_df.empty:
        text_path.write_text("No qualifying games found.", encoding="utf-8")
        print("No qualifying teams found; CSV is empty.")
        return

    text_path.write_text(build_text_report(report_df), encoding="utf-8")
    preview = report_df.head(12)
    print("Blowout resilience (top 12):")
    print(preview.to_string(index=False))
    print(
        f"\nWrote {len(report_df)} rows to {output_path} "
        f"and summary to {text_path} (source: {source_path})."
    )


if __name__ == "__main__":
    main()
