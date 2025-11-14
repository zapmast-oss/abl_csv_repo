"""ABL SOS Last 14 days report."""

from __future__ import annotations

import argparse
import math
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
]


def pick_column(df: pd.DataFrame, *names: str) -> Optional[str]:
    lowered = {col.lower(): col for col in df.columns}
    for name in names:
        key = name.lower()
        if key in lowered:
            return lowered[key]
    return None


def load_team_meta(base: Path) -> Dict[int, str]:
    teams_path = base / "teams.csv"
    if not teams_path.exists():
        return {}
    df = pd.read_csv(teams_path)
    team_col = pick_column(df, "team_id", "teamid", "teamID", "TeamID")
    name_col = pick_column(df, "team_display", "team_name", "name", "nickname")
    city_col = pick_column(df, "city", "city_name")
    nickname_col = pick_column(df, "nickname")
    abbr_col = pick_column(df, "abbr")
    meta: Dict[int, str] = {}
    for _, row in df.iterrows():
        tid = row.get(team_col)
        if pd.isna(tid):
            continue
        tid = int(tid)
        if tid in meta:
            continue
        if name_col and pd.notna(row.get(name_col)):
            meta[tid] = str(row.get(name_col))
        elif city_col and nickname_col and pd.notna(row.get(city_col)) and pd.notna(row.get(nickname_col)):
            meta[tid] = f"{row.get(city_col)} {row.get(nickname_col)}"
        elif city_col and pd.notna(row.get(city_col)):
            meta[tid] = str(row.get(city_col))
        elif nickname_col and pd.notna(row.get(nickname_col)):
            meta[tid] = str(row.get(nickname_col))
        elif abbr_col and pd.notna(row.get(abbr_col)):
            meta[tid] = str(row.get(abbr_col))
        else:
            meta[tid] = ""
    return meta


def autodetect_logs(base: Path, override: Optional[Path]) -> Optional[pd.DataFrame]:
    candidates = [Path(override)] if override else [base / name for name in LOG_CANDIDATES]
    for path in candidates:
        if path and path.exists():
            return pd.read_csv(path)
    return None


def expand_games_to_team_rows(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    away_col = pick_column(df, "away_team_id", "away_team", "team0", "visitor_team_id", "visteam")
    home_col = pick_column(df, "home_team_id", "home_team", "team1", "hometeam")
    runs_away_col = pick_column(df, "away_runs", "runs_away", "score0", "runs0", "r0")
    runs_home_col = pick_column(df, "home_runs", "runs_home", "score1", "runs1", "r1")
    date_col = pick_column(df, "game_date", "date", "gamedate", "GameDate")
    if not all([away_col, home_col, runs_away_col, runs_home_col]):
        return None
    rows = []
    for _, row in df.iterrows():
        away_id = pd.to_numeric(row.get(away_col), errors="coerce")
        home_id = pd.to_numeric(row.get(home_col), errors="coerce")
        if pd.isna(away_id) or pd.isna(home_id):
            continue
        away_runs = pd.to_numeric(row.get(runs_away_col), errors="coerce")
        home_runs = pd.to_numeric(row.get(runs_home_col), errors="coerce")
        date_val = pd.to_datetime(row.get(date_col), errors="coerce") if date_col else pd.NaT
        for team_id, opp_id, rf, ra in [
            (int(away_id), int(home_id), away_runs, home_runs),
            (int(home_id), int(away_id), home_runs, away_runs),
        ]:
            result = pd.NA
            if pd.notna(rf) and pd.notna(ra):
                result = "W" if rf > ra else "L" if rf < ra else "T"
            rows.append(
                {
                    "team_id": team_id,
                    "opponent_id": opp_id,
                    "runs_for": rf,
                    "runs_against": ra,
                    "result": result,
                    "game_date": date_val,
                }
            )
    return pd.DataFrame(rows)


def prepare_logs(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    df = df.copy()
    team_col = pick_column(df, "team_id", "teamid", "teamID", "TeamID")
    if not team_col:
        expanded = expand_games_to_team_rows(df)
        if expanded is None:
            return None
        df = expanded
        team_col = "team_id"
        opponent_col = "opponent_id"
        result_col = pick_column(df, "result")
        runs_for_col = pick_column(df, "runs_for")
        runs_against_col = pick_column(df, "runs_against")
        date_col = pick_column(df, "game_date", "date")
    else:
        opponent_col = pick_column(df, "opponent_id", "opp_id", "opponent")
        result_col = pick_column(df, "result")
        runs_for_col = pick_column(df, "runs_for", "runs_scored", "rs", "r")
        runs_against_col = pick_column(df, "runs_against", "ra")
        date_col = pick_column(df, "game_date", "date", "gamedate", "GameDate")

    df["team_id"] = pd.to_numeric(df[team_col], errors="coerce").astype("Int64")
    df = df[(df["team_id"] >= TEAM_MIN) & (df["team_id"] <= TEAM_MAX)].copy()
    if df.empty:
        return None

    if opponent_col:
        df["opponent_id"] = pd.to_numeric(df[opponent_col], errors="coerce").astype("Int64")
    else:
        home_col = pick_column(df, "home_team_id", "home_team", "team1", "hometeam")
        away_col = pick_column(df, "away_team_id", "away_team", "team0", "visitor_team_id", "visteam")
        df["opponent_id"] = pd.NA
        if home_col and away_col:
            home_ids = pd.to_numeric(df[home_col], errors="coerce").astype("Int64")
            away_ids = pd.to_numeric(df[away_col], errors="coerce").astype("Int64")
            df.loc[df["team_id"] == home_ids, "opponent_id"] = away_ids
            df.loc[df["team_id"] == away_ids, "opponent_id"] = home_ids

    if runs_for_col and runs_against_col:
        df["runs_for"] = pd.to_numeric(df[runs_for_col], errors="coerce")
        df["runs_against"] = pd.to_numeric(df[runs_against_col], errors="coerce")
    else:
        df["runs_for"] = pd.NA
        df["runs_against"] = pd.NA

    if result_col:
        df["win_flag"] = df[result_col].astype(str).str.upper().str.startswith("W")
    else:
        df["win_flag"] = pd.NA
    mask_scores = df["runs_for"].notna() & df["runs_against"].notna()
    df.loc[df["win_flag"].isna() & mask_scores, "win_flag"] = df["runs_for"] > df["runs_against"]
    df["win_flag"] = pd.to_numeric(df["win_flag"], errors="coerce").astype("Float64")
    df.loc[mask_scores & (df["runs_for"] == df["runs_against"]), "win_flag"] = pd.NA

    df["game_date"] = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT
    df = df.dropna(subset=["game_date", "opponent_id"])
    return df


def compute_last14_window(df: pd.DataFrame) -> pd.DataFrame:
    valid_dates = df.loc[df["win_flag"].notna(), "game_date"].dropna()
    if valid_dates.empty:
        anchor_date = df["game_date"].max()
    else:
        anchor_date = valid_dates.max()
    start_date = anchor_date - pd.Timedelta(days=13)
    return df[(df["game_date"] >= start_date) & (df["game_date"] <= anchor_date)].copy()


def aggregate_last14(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.dropna(subset=["win_flag"])
        .groupby("team_id")["win_flag"]
        .agg(last14_w="sum", last14_g="count")
        .reset_index()
    )
    agg["last14_l"] = agg["last14_g"] - agg["last14_w"]
    agg["last14_winpct"] = agg["last14_w"] / agg["last14_g"]
    return agg


def compute_sos(df: pd.DataFrame, stats: pd.DataFrame, min_games: int) -> pd.DataFrame:
    stats_lookup = stats.set_index("team_id")
    league_mask = stats["last14_g"] > 0
    league_avg = stats.loc[league_mask, "last14_winpct"].mean() if league_mask.any() else pd.NA
    df = df.copy()
    df = df.merge(
        stats_lookup[["last14_winpct", "last14_g"]],
        left_on="opponent_id",
        right_index=True,
        how="left",
    )
    df = df.rename(
        columns={
            "last14_winpct": "last14_winpct_opp",
            "last14_g": "last14_g_opp",
        }
    )
    df_valid = df[df["last14_g_opp"] >= min_games]
    sos = (
        df_valid.groupby("team_id")["last14_winpct_opp"]
        .agg(sos14_avg_opp_winpct="mean", sos14_games_used="count")
        .reset_index()
    )
    sos["league_last14_winpct"] = league_avg
    sos["sos14_diff_vs_league"] = sos["sos14_avg_opp_winpct"] - league_avg
    return sos


def build_text_report(df: pd.DataFrame, limit: int = 24) -> str:
    lines = ["ABL SOS Last 14", "=" * 18, ""]
    for _, row in df.head(limit).iterrows():
        name = row["team_display"] or f"Team {int(row['team_id'])}"
        tag = "Gauntlet" if pd.notna(row["sos14_diff_vs_league"]) and row["sos14_diff_vs_league"] >= 0.02 else "Soft" if pd.notna(row["sos14_diff_vs_league"]) and row["sos14_diff_vs_league"] <= -0.02 else "Neutral"
        lines.append(
            f"{name:<20} {tag:<8} | SOS {row['sos14_avg_opp_winpct']:.3f} "
            f"| League {row['league_last14_winpct']:.3f} | Î” {row['sos14_diff_vs_league']:+.3f} "
            f"| Record {int(row['last14_w'])}-{int(row['last14_l'])}"
        )
    lines.append("")
    lines.append("Key:")
    lines.append("  SOS = average opponent win% over same 14-day window.")
    lines.append("  Gauntlet -> facing tougher slate (Î” >= +0.020).")
    lines.append("  Soft     -> easier slate (Î” <= -0.020).")
    lines.append("  Neutral  -> within +/-0.019 of league average.")
    return "\n".join(lines)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ABL strength of schedule (last 14 days).")
    parser.add_argument("--base", type=str, default=".", help="Base directory for CSVs.")
    parser.add_argument("--logs", type=str, help="Explicit per-game logs CSV.")
    parser.add_argument(
        "--min_sos_games",
        type=int,
        default=3,
        help="Minimum opponent games required to include in SOS average (default: 3).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="out/csv_out/z_ABL_SOS_Last14.csv",
        help="Output CSV (default: out/csv_out/z_ABL_SOS_Last14.csv).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    base_dir = Path(args.base).resolve()
    logs_path = Path(args.logs) if args.logs else None
    logs_df = autodetect_logs(base_dir, logs_path)
    prepared = prepare_logs(logs_df) if logs_df is not None else None
    if prepared is None or prepared.empty:
        raise FileNotFoundError("No usable team logs found for SOS calculation.")

    window_df = compute_last14_window(prepared)
    if window_df.empty:
        raise ValueError("No games within the last-14 window.")

    stats = aggregate_last14(window_df)
    if stats.empty:
        raise ValueError("No team records computed for last-14 window.")

    sos = compute_sos(window_df, stats, args.min_sos_games)
    meta = load_team_meta(base_dir)
    report = stats.merge(sos, on="team_id", how="left")
    report["team_display"] = report["team_id"].apply(lambda tid: meta.get(tid, ""))
    report["last14_l"] = report["last14_g"] - report["last14_w"]

    for col in ["sos14_avg_opp_winpct", "league_last14_winpct", "sos14_diff_vs_league", "last14_winpct"]:
        report[col] = report[col].round(3)
    report["sos14_games_used"] = report["sos14_games_used"].fillna(0).astype(int)
    mask_no_sos = report["sos14_games_used"] < args.min_sos_games
    report.loc[mask_no_sos, ["sos14_avg_opp_winpct", "sos14_diff_vs_league"]] = pd.NA

    report = report[
        [
            "team_id",
            "team_display",
            "last14_g",
            "last14_w",
            "last14_l",
            "last14_winpct",
            "sos14_games_used",
            "sos14_avg_opp_winpct",
            "league_last14_winpct",
            "sos14_diff_vs_league",
        ]
    ]
    report = report.sort_values(
        by=["sos14_avg_opp_winpct", "last14_winpct"],
        ascending=[False, False],
        na_position="last",
    )

    output_path = (base_dir / args.out).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    txt_dir = base_dir / "out" / "txt_out"
    txt_dir.mkdir(parents=True, exist_ok=True)
    text_path = txt_dir / output_path.with_suffix(".txt").name
    text_path.write_text(stamp_text_block(build_text_report(report)), encoding="utf-8")

    preview = report.head(12)
    print("SOS last 14 days (top 12):")
    print(preview.to_string(index=False))
    print(
        f"\nWrote {len(report)} rows to {output_path} and summary to {text_path} "
        f"(source: {logs_path if logs_path else 'autodetected logs'})."
    )


if __name__ == "__main__":
    main()
