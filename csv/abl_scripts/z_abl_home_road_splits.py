"""ABL home/road splits report."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from abl_config import stamp_text_block

TEAM_MIN, TEAM_MAX = 1, 24
SEASON_CANDIDATES = [
    "team_season.csv",
    "team_totals.csv",
    "team_record.csv",
    "teams_season.csv",
    "standings.csv",
]
LOG_CANDIDATES = [
    "team_game_log.csv",
    "teams_game_log.csv",
    "game_log_team.csv",
    "team_log.csv",
    "schedule_results.csv",
    "games.csv",
]
PARK_CANDIDATES = ["park_factors.csv", "parks.csv"]


def pick_column(df: pd.DataFrame, *names: str) -> Optional[str]:
    lowered = {col.lower(): col for col in df.columns}
    for name in names:
        key = name.lower()
        if key in lowered:
            return lowered[key]
    return None


def load_team_meta(base: Path) -> Dict[int, dict]:
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
    park_col = pick_column(df, "park_id", "home_park_id", "stadium_id")
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
        meta[tid_int] = {
            "name": name_value,
            "park_id": row.get(park_col) if park_col else None,
        }
    return meta


def autodetect_season(base: Path, override: Optional[Path]) -> Optional[pd.DataFrame]:
    candidates = [Path(override)] if override else [base / name for name in SEASON_CANDIDATES]
    for path in candidates:
        if path and path.exists():
            return pd.read_csv(path)
    return None


def autodetect_logs(base: Path, override: Optional[Path]) -> Optional[pd.DataFrame]:
    candidates = [Path(override)] if override else [base / name for name in LOG_CANDIDATES]
    for path in candidates:
        if path and path.exists():
            return pd.read_csv(path)
    return None


def expand_games_to_team_rows(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    away_col = pick_column(df, "away_team_id", "away_team", "team0", "visteam")
    home_col = pick_column(df, "home_team_id", "home_team", "team1", "hometeam")
    runs_away_col = pick_column(df, "away_runs", "runs_away", "score0", "runs0", "r0", "away_score")
    runs_home_col = pick_column(df, "home_runs", "runs_home", "score1", "runs1", "r1", "home_score")
    if not all([away_col, home_col, runs_away_col, runs_home_col]):
        return None
    date_col = pick_column(df, "game_date", "date", "gamedate", "GameDate")
    records = []
    for _, row in df.iterrows():
        away_id = pd.to_numeric(row.get(away_col), errors="coerce")
        home_id = pd.to_numeric(row.get(home_col), errors="coerce")
        if pd.isna(away_id) or pd.isna(home_id):
            continue
        away_runs = pd.to_numeric(row.get(runs_away_col), errors="coerce")
        home_runs = pd.to_numeric(row.get(runs_home_col), errors="coerce")
        date_val = pd.to_datetime(row.get(date_col), errors="coerce") if date_col else pd.NaT
        for is_home, team_id, rf, ra in [
            (False, int(away_id), away_runs, home_runs),
            (True, int(home_id), home_runs, away_runs),
        ]:
            result = pd.NA
            if pd.notna(rf) and pd.notna(ra):
                result = "W" if rf > ra else "L" if rf < ra else "T"
            records.append(
                {
                    "team_id": team_id,
                    "runs_for": rf,
                    "runs_against": ra,
                    "is_home": is_home,
                    "result": result,
                    "game_date": date_val,
                }
            )
    return pd.DataFrame(records)


def autodetect_parks(base: Path, override: Optional[Path]) -> Optional[pd.DataFrame]:
    candidates = [Path(override)] if override else [base / name for name in PARK_CANDIDATES]
    for path in candidates:
        if path and path.exists():
            return pd.read_csv(path)
    return None


def parse_home_road_from_season(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    team_col = pick_column(df, "team_id", "teamid", "teamID", "TeamID")
    if not team_col:
        return None
    cols = {
        "home_w": pick_column(df, "home_w", "homewins", "hw"),
        "home_l": pick_column(df, "home_l", "homelosses", "hl"),
        "road_w": pick_column(df, "road_w", "roadwins", "rw"),
        "road_l": pick_column(df, "road_l", "roadlosses", "rl"),
        "home_rs": pick_column(df, "home_rs", "home_runs_scored", "hrs"),
        "home_ra": pick_column(df, "home_ra", "home_runs_against", "hra"),
        "road_rs": pick_column(df, "road_rs", "road_runs_scored", "rrs"),
        "road_ra": pick_column(df, "road_ra", "road_runs_against", "rra"),
        "team_display": pick_column(df, "team_display", "team_name", "name", "TeamName"),
    }
    if not all([cols["home_w"], cols["home_l"], cols["road_w"], cols["road_l"]]):
        return None
    data_rows = []
    for _, row in df.iterrows():
        team_id = pd.to_numeric(row[team_col], errors="coerce")
        if pd.isna(team_id):
            continue
        team_id = int(team_id)
        if team_id < TEAM_MIN or team_id > TEAM_MAX:
            continue
        home_w = pd.to_numeric(row[cols["home_w"]], errors="coerce")
        home_l = pd.to_numeric(row[cols["home_l"]], errors="coerce")
        road_w = pd.to_numeric(row[cols["road_w"]], errors="coerce")
        road_l = pd.to_numeric(row[cols["road_l"]], errors="coerce")
        home_rs = pd.to_numeric(row[cols["home_rs"]], errors="coerce") if cols["home_rs"] else pd.NA
        home_ra = pd.to_numeric(row[cols["home_ra"]], errors="coerce") if cols["home_ra"] else pd.NA
        road_rs = pd.to_numeric(row[cols["road_rs"]], errors="coerce") if cols["road_rs"] else pd.NA
        road_ra = pd.to_numeric(row[cols["road_ra"]], errors="coerce") if cols["road_ra"] else pd.NA
        data_rows.append(
            {
                "team_id": team_id,
                "team_display": row.get(cols["team_display"]) if cols["team_display"] else "",
                "home_w": home_w,
                "home_l": home_l,
                "road_w": road_w,
                "road_l": road_l,
                "home_rs": home_rs,
                "home_ra": home_ra,
                "road_rs": road_rs,
                "road_ra": road_ra,
            }
        )
    return pd.DataFrame(data_rows)


def parse_home_road_from_logs(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    df = df.copy()
    team_col = pick_column(df, "team_id", "teamid", "teamID", "TeamID")
    if not team_col:
        expanded = expand_games_to_team_rows(df)
        if expanded is None or expanded.empty:
            return None
        df = expanded
        team_col = "team_id"
        result_col = pick_column(df, "result")
        runs_for_col = pick_column(df, "runs_for")
        runs_against_col = pick_column(df, "runs_against")
        home_flag_col = "is_home"
        home_team_col = None
    else:
        result_col = pick_column(df, "result")
        runs_for_col = pick_column(df, "runs_scored", "runs_for", "rs", "r")
        runs_against_col = pick_column(df, "runs_against", "ra")
        home_flag_col = pick_column(df, "home_away", "venue_flag", "is_home", "homeflag")
        home_team_col = pick_column(df, "home_team_id", "home_team", "team1", "hometeam")
    df["team_id"] = pd.to_numeric(df[team_col], errors="coerce").astype("Int64")
    df = df[(df["team_id"] >= TEAM_MIN) & (df["team_id"] <= TEAM_MAX)].copy()
    if df.empty:
        return None

    df.loc[:, "runs_for"] = pd.to_numeric(df[runs_for_col], errors="coerce") if runs_for_col else pd.NA
    df.loc[:, "runs_against"] = pd.to_numeric(df[runs_against_col], errors="coerce") if runs_against_col else pd.NA

    if result_col:
        df.loc[:, "win_flag"] = df[result_col].astype(str).str.upper().str.startswith("W")
    else:
        df.loc[:, "win_flag"] = pd.NA
    mask_scores = df["runs_for"].notna() & df["runs_against"].notna()
    df.loc[df["win_flag"].isna() & mask_scores, "win_flag"] = df["runs_for"] > df["runs_against"]
    df.loc[:, "win_flag"] = pd.to_numeric(df["win_flag"], errors="coerce")

    if home_flag_col:
        if df[home_flag_col].dropna().isin([True, False]).all():
            df.loc[:, "is_home"] = df[home_flag_col].astype(bool)
        else:
            flag = df[home_flag_col].astype(str).str.upper()
            home_values = {"H", "HOME", "1", "TRUE", "T"}
            df.loc[:, "is_home"] = flag.isin(home_values)
    elif home_team_col:
        df.loc[:, "is_home"] = df["team_id"] == pd.to_numeric(df[home_team_col], errors="coerce").astype("Int64")
    else:
        df.loc[:, "is_home"] = False

    home_df = df[df["is_home"]]
    road_df = df[~df["is_home"]]
    if home_df.empty and road_df.empty:
        return None

    def aggregate(group_df: pd.DataFrame) -> Tuple[int, int, int, float, float]:
        valid = group_df["win_flag"].notna()
        g = int(valid.sum())
        wins = int(group_df.loc[valid, "win_flag"].sum())
        losses = g - wins
        rs = group_df.loc[valid, "runs_for"].sum(skipna=True)
        ra = group_df.loc[valid, "runs_against"].sum(skipna=True)
        return g, wins, losses, rs, ra

    records = []
    for team_id, team_group in df.groupby("team_id"):
        home_group = team_group[team_group["is_home"]]
        road_group = team_group[~team_group["is_home"]]
        home_stats = aggregate(home_group) if not home_group.empty else (0, 0, 0, pd.NA, pd.NA)
        road_stats = aggregate(road_group) if not road_group.empty else (0, 0, 0, pd.NA, pd.NA)
        records.append(
            {
                "team_id": int(team_id),
                "team_display": "",
                "home_w": home_stats[1],
                "home_l": home_stats[2],
                "home_g": home_stats[0],
                "home_rs": home_stats[3],
                "home_ra": home_stats[4],
                "road_w": road_stats[1],
                "road_l": road_stats[2],
                "road_g": road_stats[0],
                "road_rs": road_stats[3],
                "road_ra": road_stats[4],
            }
        )
    return pd.DataFrame(records)


def load_park_factors(base: Path, override: Optional[Path]) -> Dict[str, float]:
    df = autodetect_parks(base, override)
    if df is None:
        return {}
    park_col = pick_column(df, "park_id", "stadium_id", "park")
    run_col = pick_column(df, "run_factor", "pf_runs", "runs_factor")
    if not park_col or not run_col:
        return {}
    factors = {}
    for _, row in df.iterrows():
        park_id = row.get(park_col)
        if pd.isna(park_id):
            continue
        factors[str(park_id)] = float(row.get(run_col))
    return factors


def compute_metrics(df: pd.DataFrame, meta: Dict[int, dict], park_factors: Dict[str, float]) -> pd.DataFrame:
    df = df.copy()
    df["home_g"] = df.get("home_g")
    df["road_g"] = df.get("road_g")
    if "home_g" not in df.columns or df["home_g"].isna().all():
        df["home_g"] = df["home_w"] + df["home_l"]
    if "road_g" not in df.columns or df["road_g"].isna().all():
        df["road_g"] = df["road_w"] + df["road_l"]

    df = df[(df["home_g"] > 0) & (df["road_g"] > 0)]
    df["home_winpct"] = df["home_w"] / df["home_g"]
    df["road_winpct"] = df["road_w"] / df["road_g"]
    df["split_diff_winpct"] = df["home_winpct"] - df["road_winpct"]

    df["home_runs_per_g"] = (df["home_rs"] + df["home_ra"]) / df["home_g"]
    df["road_runs_per_g"] = (df["road_rs"] + df["road_ra"]) / df["road_g"]
    df.loc[df["home_g"] == 0, "home_runs_per_g"] = pd.NA
    df.loc[df["road_g"] == 0, "road_runs_per_g"] = pd.NA
    df["run_env_ratio"] = df["home_runs_per_g"] / df["road_runs_per_g"]
    df.loc[df["road_runs_per_g"].isna() | (df["road_runs_per_g"] == 0), "run_env_ratio"] = pd.NA

    df["team_display"] = df["team_display"].fillna("")
    df["park_run_factor"] = pd.NA
    for idx, row in df.iterrows():
        tid = int(row["team_id"])
        info = meta.get(tid, {})
        if not df.at[idx, "team_display"]:
            df.at[idx, "team_display"] = info.get("name", "")
        park_id = info.get("park_id")
        if park_id is not None:
            df.at[idx, "park_run_factor"] = park_factors.get(str(park_id), pd.NA)

    int_cols = ["home_g", "home_w", "home_l", "road_g", "road_w", "road_l"]
    for col in int_cols:
        df[col] = df[col].astype("Int64")

    df["home_winpct"] = df["home_winpct"].round(3)
    df["road_winpct"] = df["road_winpct"].round(3)
    df["split_diff_winpct"] = df["split_diff_winpct"].round(3)
    df["home_runs_per_g"] = df["home_runs_per_g"].round(2)
    df["road_runs_per_g"] = df["road_runs_per_g"].round(2)
    df["run_env_ratio"] = df["run_env_ratio"].round(3)
    if "park_run_factor" in df.columns:
        df["park_run_factor"] = pd.to_numeric(df["park_run_factor"], errors="coerce").round(1)
    return df


def build_text_report(df: pd.DataFrame, limit: int = 24) -> str:
    lines = [
        "ABL Home/Road Splits",
        "=" * 26,
        "Compares each club's home versus road win percentage and scoring profile to expose venue edges.",
        "Useful for matchup prep: target teams that feast in their park or wilt when they leave it.",
        "",
    ]
    header = f"{'Team':<20} {'Profile':<12} {'Home W-L':>10} {'Pct':>6} {'Road W-L':>10} {'Pct':>6} {'dPct':>7} {'Runs H/R':>15}"
    lines.append(header)
    lines.append("-" * len(header))
    for _, row in df.head(limit).iterrows():
        name = row["team_display"] or f"Team {int(row['team_id'])}"
        diff = row["split_diff_winpct"]
        tag = "Home heavy" if diff >= 0.05 else "Road warriors" if diff <= -0.05 else "Balanced"
        home_w = row.get("home_w", pd.NA)
        home_l = row.get("home_l", pd.NA)
        road_w = row.get("road_w", pd.NA)
        road_l = row.get("road_l", pd.NA)
        home_rec = f"{int(home_w)}-{int(home_l)}" if pd.notna(home_w) and pd.notna(home_l) else "NA"
        road_rec = f"{int(road_w)}-{int(road_l)}" if pd.notna(road_w) and pd.notna(road_l) else "NA"
        home_pct = f"{row['home_winpct']:.3f}" if pd.notna(row["home_winpct"]) else " NA "
        road_pct = f"{row['road_winpct']:.3f}" if pd.notna(row["road_winpct"]) else " NA "
        diff_txt = f"{diff:+.3f}" if pd.notna(diff) else " NA "
        runs_txt = (
            f"{row['home_runs_per_g']:.2f}/{row['road_runs_per_g']:.2f}"
            if pd.notna(row["home_runs_per_g"]) and pd.notna(row["road_runs_per_g"])
            else "NA/NA"
        )
        lines.append(
            f"{name:<20} {tag:<12} {home_rec:>10} {home_pct:>6} {road_rec:>10} {road_pct:>6} {diff_txt:>7} {runs_txt:>15}"
        )
    lines.append("")
    lines.append("Key:")
    lines.append("  Home heavy    -> home W% exceeds road by 0.050 or more.")
    lines.append("  Balanced      -> difference within +/-0.049.")
    lines.append("  Road warriors -> road W% exceeds home by 0.050 or more.")
    lines.append("  Runs H/R      -> average runs per game home/road for context.")
    return "\n".join(lines)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ABL home/road splits report.")
    parser.add_argument("--base", type=str, default=".", help="Base directory for CSVs.")
    parser.add_argument("--season", type=str, help="Explicit season totals CSV.")
    parser.add_argument("--logs", type=str, help="Explicit per-game logs CSV.")
    parser.add_argument("--parks", type=str, help="Explicit park factors CSV.")
    parser.add_argument(
        "--out",
        type=str,
        default="out/csv_out/z_ABL_Home_Road_Splits.csv",
        help="Output CSV (default: out/csv_out/z_ABL_Home_Road_Splits.csv).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    base_dir = Path(args.base).resolve()
    meta = load_team_meta(base_dir)
    park_factors = load_park_factors(base_dir, Path(args.parks) if args.parks else None)

    season_df = autodetect_season(base_dir, Path(args.season) if args.season else None)
    parsed_season = parse_home_road_from_season(season_df) if season_df is not None else None

    if parsed_season is None or parsed_season.empty:
        logs_path = Path(args.logs) if args.logs else None
        logs_df = autodetect_logs(base_dir, logs_path)
        parsed_logs = parse_home_road_from_logs(logs_df) if logs_df is not None else None
        data_df = parsed_logs
        data_source = str(logs_path) if logs_path else "per-game logs"
    else:
        data_df = parsed_season
        data_source = str(Path(args.season)) if args.season else "season totals"

    if data_df is None or data_df.empty:
        raise FileNotFoundError("No usable season totals or per-game logs found.")

    report_df = compute_metrics(data_df, meta, park_factors)
    if report_df.empty:
        raise ValueError("No valid data after filtering teams or computing metrics.")

    column_order = [
        "team_id",
        "team_display",
        "home_g",
        "home_w",
        "home_l",
        "home_winpct",
        "road_g",
        "road_w",
        "road_l",
        "road_winpct",
        "split_diff_winpct",
        "home_runs_per_g",
        "road_runs_per_g",
        "run_env_ratio",
        "park_run_factor",
    ]
    for col in column_order:
        if col not in report_df.columns:
            report_df[col] = pd.NA
    report_df = report_df[column_order]

    report_df["abs_split_diff"] = report_df["split_diff_winpct"].abs()
    report_df = report_df.sort_values(
        by=["abs_split_diff", "home_winpct"],
        ascending=[False, False],
        na_position="last",
    ).drop(columns="abs_split_diff")

    output_path = (base_dir / args.out).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_path, index=False)
    text_filename = output_path.with_suffix(".txt").name
    if output_path.parent.name.lower() in {'csv_out'}:
        text_dir = output_path.parent.parent / "text_out"
    else:
        text_dir = output_path.parent
    text_dir.mkdir(parents=True, exist_ok=True)
    text_path = text_dir / text_filename
    text_path.write_text(stamp_text_block(build_text_report(report_df)), encoding="utf-8")

    preview = report_df.head(12)
    print("Home/Road splits (top 12):")
    print(preview.to_string(index=False))
    print(f"\nWrote {len(report_df)} rows to {output_path} and summary to {text_path}.")


if __name__ == "__main__":
    main()

