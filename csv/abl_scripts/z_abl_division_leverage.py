"""ABL Division Leverage report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from abl_config import stamp_text_block

TEAM_MIN, TEAM_MAX = 1, 24
TEAM_META_CANDIDATES = [
    "team_info.csv",
    "teams.csv",
    "league_structure.csv",
]
LOG_CANDIDATES = [
    "team_game_log.csv",
    "teams_game_log.csv",
    "game_log_team.csv",
    "team_log.csv",
    "schedule_results.csv",
    "games.csv",
]
SCHEDULE_CANDIDATES = [
    "master_schedule.csv",
    "schedule.csv",
    "fixtures.csv",
    "games.csv",
]


def pick_column(df: pd.DataFrame, *names: str) -> Optional[str]:
    lowered = {col.lower(): col for col in df.columns}
    for name in names:
        key = name.lower()
        if key in lowered:
            return lowered[key]
    return None


def load_team_meta(base: Path, override: Optional[Path]) -> pd.DataFrame:
    candidates = [Path(override)] if override else [base / name for name in TEAM_META_CANDIDATES]
    for path in candidates:
        if not path or not path.exists():
            continue
        df = pd.read_csv(path)
        team_col = pick_column(df, "team_id", "teamid", "teamID", "TeamID")
        div_col = pick_column(df, "division_id", "divisionid", "div_id")
        sub_col = pick_column(df, "sub_league_id", "subleague_id", "sub_id", "subleague")
        if not team_col or not div_col:
            continue
        name_col = pick_column(df, "team_display", "team_name", "name", "TeamName")
        selected_cols = [team_col, div_col] + ([sub_col] if sub_col else []) + ([name_col] if name_col else [])
        meta = df[selected_cols].copy()
        rename_map = {team_col: "team_id", div_col: "division_id"}
        if sub_col:
            rename_map[sub_col] = "sub_league_id"
        if name_col:
            rename_map[name_col] = "team_display"
        meta.rename(columns=rename_map, inplace=True)
        meta["team_display"] = meta.get("team_display", "").fillna("")
        meta["team_id"] = pd.to_numeric(meta["team_id"], errors="coerce").astype("Int64")
        meta["division_id"] = pd.to_numeric(meta["division_id"], errors="coerce").astype("Int64")
        if "sub_league_id" in meta.columns:
            meta["sub_league_id"] = pd.to_numeric(meta["sub_league_id"], errors="coerce").astype("Int64")
        meta = meta[(meta["team_id"] >= TEAM_MIN) & (meta["team_id"] <= TEAM_MAX)]
        if not meta.empty:
            return meta
    raise FileNotFoundError("No usable team metadata file (team_id + division_id) found.")


def autodetect_logs(base: Path, override: Optional[Path]) -> pd.DataFrame:
    candidates = [Path(override)] if override else [base / name for name in LOG_CANDIDATES]
    for path in candidates:
        if path and path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError("No played-game log file found.")


def autodetect_schedule(base: Path, override: Optional[Path]) -> Optional[pd.DataFrame]:
    candidates = [Path(override)] if override else [base / name for name in SCHEDULE_CANDIDATES]
    for path in candidates:
        if path and path.exists():
            return pd.read_csv(path)
    return None


def ensure_opponent(df: pd.DataFrame) -> pd.DataFrame:
    opponent_col = pick_column(df, "opponent_id", "opp_id", "opponent", "opponentteamid")
    if opponent_col:
        df["opponent_id"] = pd.to_numeric(df[opponent_col], errors="coerce").astype("Int64")
        return df
    home_col = pick_column(df, "home_team_id", "home_team", "team1", "hometeam")
    away_col = pick_column(df, "away_team_id", "away_team", "team0", "visteam")
    if not home_col or not away_col:
        return df
    df["opponent_id"] = pd.NA
    home_ids = pd.to_numeric(df[home_col], errors="coerce").astype("Int64")
    away_ids = pd.to_numeric(df[away_col], errors="coerce").astype("Int64")
    team_col = pick_column(df, "team_id", "teamid", "teamID", "TeamID")
    if not team_col:
        # Expand games.csv style to team-centric
        records = []
        core_cols = set(df.columns)
        for _, row in df.iterrows():
            home_id = pd.to_numeric(row.get(home_col), errors="coerce")
            away_id = pd.to_numeric(row.get(away_col), errors="coerce")
            if pd.isna(home_id) or pd.isna(away_id):
                continue
            for team_id, opp_id, suffix in [(home_id, away_id, "_home"), (away_id, home_id, "_away")]:
                new_row = row.copy()
                new_row["team_id"] = team_id
                new_row["opponent_id"] = opp_id
                home_runs_val = None
                away_runs_val = None
                for candidate in ["runs_home", "home_runs", "score1", "runs1", "r1"]:
                    if candidate in core_cols:
                        home_runs_val = new_row.get(candidate)
                        break
                for candidate in ["runs_away", "away_runs", "score0", "runs0", "r0"]:
                    if candidate in core_cols:
                        away_runs_val = new_row.get(candidate)
                        break
                if home_runs_val is not None and away_runs_val is not None:
                    new_row["runs_for"] = home_runs_val if suffix == "_home" else away_runs_val
                    new_row["runs_against"] = away_runs_val if suffix == "_home" else home_runs_val
                records.append(new_row)
        return pd.DataFrame(records)
    team_ids = pd.to_numeric(df[team_col], errors="coerce").astype("Int64")
    df.loc[team_ids == home_ids, "opponent_id"] = away_ids
    df.loc[team_ids == away_ids, "opponent_id"] = home_ids
    return df


def prepare_played_logs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = ensure_opponent(df)
    team_col = pick_column(df, "team_id", "teamid", "teamID", "TeamID")
    if not team_col:
        raise ValueError("team_id column missing from logs even after expansion.")
    result_col = pick_column(df, "result")
    runs_for_col = pick_column(df, "runs_for", "runs_scored", "rs", "r")
    runs_against_col = pick_column(df, "runs_against", "ra")
    date_col = pick_column(df, "game_date", "date", "gamedate", "GameDate")

    df["team_id"] = pd.to_numeric(df[team_col], errors="coerce").astype("Int64")
    df = df[(df["team_id"] >= TEAM_MIN) & (df["team_id"] <= TEAM_MAX)].copy()
    df["opponent_id"] = pd.to_numeric(df["opponent_id"], errors="coerce").astype("Int64")
    df["game_date"] = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT
    df = df.dropna(subset=["team_id", "opponent_id", "game_date"])

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
    played_col = pick_column(df, "played", "completed", "is_played")
    if played_col:
        df["played_mask"] = pd.to_numeric(df[played_col], errors="coerce").fillna(0).astype(int) != 0
    else:
        df["played_mask"] = False
    mask_scores = df["runs_for"].notna() & df["runs_against"].notna()
    scored_nonzero = mask_scores & ((df["runs_for"] + df["runs_against"]) > 0)
    df.loc[scored_nonzero, "played_mask"] = True
    if result_col:
        df.loc[df[result_col].notna(), "played_mask"] = True

    df["inferred_win"] = df["runs_for"] > df["runs_against"]
    df.loc[df["win_flag"].isna() & scored_nonzero, "win_flag"] = df.loc[
        df["win_flag"].isna() & scored_nonzero, "inferred_win"
    ]
    df["win_flag"] = pd.to_numeric(df["win_flag"], errors="coerce")
    df = df.drop(columns=["inferred_win"])
    df = df[df["played_mask"]].dropna(subset=["win_flag"])
    return df


def aggregate_played(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(meta[["team_id", "division_id", "team_display"]], on="team_id", how="left", suffixes=("", "_team"))
    df = df.merge(
        meta[["team_id", "division_id"]].rename(columns={"team_id": "opponent_id", "division_id": "opp_division_id"}),
        on="opponent_id",
        how="left",
    )
    df["is_division_game"] = df["division_id"] == df["opp_division_id"]
    overall = (
        df.groupby("team_id")["win_flag"]
        .agg(overall_w="sum", overall_g="count")
        .reset_index()
    )
    overall["overall_l"] = overall["overall_g"] - overall["overall_w"]
    overall["overall_winpct"] = overall["overall_w"] / overall["overall_g"]

    div_df = df[df["is_division_game"]]
    div_stats = (
        div_df.groupby("team_id")["win_flag"]
        .agg(div_w="sum", div_g="count")
        .reset_index()
    )
    div_stats["div_l"] = div_stats["div_g"] - div_stats["div_w"]
    div_stats["div_winpct"] = div_stats["div_w"] / div_stats["div_g"]

    report = overall.merge(div_stats, on="team_id", how="left")
    report["team_display"] = report["team_id"].map(meta.set_index("team_id")["team_display"].fillna(""))
    report["div_diff_winpct"] = report["div_winpct"] - report["overall_winpct"]
    return report


def expand_schedule_games(df: pd.DataFrame) -> pd.DataFrame:
    home_col = pick_column(df, "home_team_id", "home_team", "team1", "hometeam")
    away_col = pick_column(df, "away_team_id", "away_team", "team0", "visteam")
    if not home_col or not away_col:
        return pd.DataFrame()
    date_col = pick_column(df, "game_date", "date", "gamedate", "GameDate")
    played_col = pick_column(df, "played", "completed", "is_played")
    result_col = pick_column(df, "result")
    runs_home_col = pick_column(df, "home_runs", "runs_home", "score1", "runs1", "r1", "home_score")
    runs_away_col = pick_column(df, "away_runs", "runs_away", "score0", "runs0", "r0", "away_score")

    records: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        home_id = pd.to_numeric(row.get(home_col), errors="coerce")
        away_id = pd.to_numeric(row.get(away_col), errors="coerce")
        if pd.isna(home_id) or pd.isna(away_id):
            continue
        date_val = pd.to_datetime(row.get(date_col), errors="coerce") if date_col else pd.NaT
        played_flag = pd.NA
        if played_col:
            played_flag = pd.to_numeric(row.get(played_col), errors="coerce")
        elif result_col and pd.notna(row.get(result_col)):
            played_flag = 1
        elif runs_home_col and runs_away_col and pd.notna(row.get(runs_home_col)) and pd.notna(row.get(runs_away_col)):
            played_flag = 1
        else:
            played_flag = 0
        if pd.isna(played_flag):
            played_flag = 0
        played_flag = 1 if float(played_flag) > 0 else 0
        for team_id, opp_id in [(home_id, away_id), (away_id, home_id)]:
            records.append(
                {
                    "team_id": int(team_id),
                    "opponent_id": int(opp_id),
                    "game_date": date_val,
                    "played_flag": played_flag,
                }
            )
    return pd.DataFrame(records)


def compute_remaining(df: pd.DataFrame, anchor_date: pd.Timestamp, meta: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df = df.merge(meta[["team_id", "division_id"]], on="team_id", how="left")
    df = df.merge(
        meta[["team_id", "division_id"]].rename(columns={"team_id": "opponent_id", "division_id": "opp_division_id"}),
        on="opponent_id",
        how="left",
    )
    df["played_flag"] = pd.to_numeric(df["played_flag"], errors="coerce")
    df["is_future"] = False
    df.loc[df["played_flag"].isna(), "is_future"] = True
    df.loc[df["played_flag"] == 0, "is_future"] = True
    df.loc[df["played_flag"] > 0, "is_future"] = False
    if df["game_date"].notna().any():
        missing_mask = df["played_flag"].isna()
        df.loc[missing_mask & (df["game_date"] > anchor_date), "is_future"] = True
        df.loc[missing_mask & (df["game_date"] <= anchor_date), "is_future"] = False
        df.loc[df["played_flag"] == 0, "is_future"] = True
    df = df[df["is_future"]]
    if df.empty:
        return pd.DataFrame()
    df = df.dropna(subset=["division_id", "opp_division_id"])
    if df.empty:
        return pd.DataFrame()
    df["is_division_match"] = df["division_id"] == df["opp_division_id"]
    agg = (
        df.groupby("team_id")
        .agg(
            remaining_g=("team_id", "size"),
            remaining_div_g=("is_division_match", "sum"),
        )
        .reset_index()
    )
    agg["remaining_div_share"] = agg["remaining_div_g"] / agg["remaining_g"]
    return agg


def leverage_tag(share: Optional[float]) -> str:
    if share is None or pd.isna(share):
        return "Finished"
    if share >= 0.35:
        return "Div-heavy"
    if share >= 0.2:
        return "Balanced"
    if share > 0:
        return "Light"
    return "Finished"


def build_text_report(df: pd.DataFrame, limit: int = 24) -> str:
    lines = [
        "ABL Division Leverage",
        "=" * 24,
        "Shows each club's divisional record plus the share of remaining intra-division games to judge leverage.",
        "",
        f"{'Team':<22} {'CD':<4} {'Profile':<10} {'Div Record':<23} {'Remaining Div Games':<32}",
        "-" * 100,
    ]
    def safe_int(value: object, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    for _, row in df.head(limit).iterrows():
        name = row["team_display"] or f"Team {int(row['team_id'])}"
        share_value = row.get("remaining_div_share")
        tag = leverage_tag(share_value)
        div_winpct = f"{row['div_winpct']:.3f}" if pd.notna(row["div_winpct"]) else "NA"
        share = f"{share_value:.3f}" if pd.notna(share_value) else "NA"
        conf_div = row.get("conf_div", "")
        div_w = row.get("div_w")
        div_l = row.get("div_l")
        rem_div = row.get("remaining_div_g")
        rem_total = row.get("remaining_g")
        div_record = (
            f"Div {safe_int(div_w):2d}-{safe_int(div_l):2d} ({div_winpct})"
            if pd.notna(div_w) and pd.notna(div_l)
            else "Div NA"
        )
        if pd.notna(rem_total) and pd.notna(rem_div) and safe_int(rem_total) > 0:
            remain_record = f"{safe_int(rem_div):2d}/{safe_int(rem_total):2d} div ({share})"
        else:
            remain_record = f"{safe_int(rem_div):2d}/{safe_int(rem_total):2d} div (NA)"
        lines.append(
            f"{name:<22} {conf_div:<4} {tag:<10} {div_record:<23} {remain_record:<32}"
        )
    lines.append("")
    lines.extend(
        [
            "Key:",
            "  Div-heavy -> >=35% of remaining games are intra-division (high leverage).",
            "  Balanced  -> 20-34% of remaining games inside the division.",
            "  Light     -> <20% of remaining games vs division foes.",
            "  Finished  -> No intra-division games left on the slate.",
            "",
            "Definition:",
            "  Division leverage measures how much control a team still has over its division race,",
            "  combining current divisional record with the share of remaining games against divisional opponents.",
        ]
    )
    return "\n".join(lines)


def build_conf_div_map(meta: pd.DataFrame) -> Dict[int, str]:
    conf_map = {0: "N", 1: "A"}
    div_map = {0: "E", 1: "C", 2: "W"}
    mapping: Dict[int, str] = {}
    for _, row in meta.iterrows():
        team_id = row.get("team_id")
        if pd.isna(team_id):
            continue
        sub_val = row.get("sub_league_id")
        div_val = row.get("division_id")
        conf = conf_map.get(int(sub_val)) if pd.notna(sub_val) else ""
        div = div_map.get(int(div_val)) if pd.notna(div_val) else ""
        label = "-".join(filter(None, [conf, div]))
        if label:
            mapping[int(team_id)] = label
    return mapping


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ABL division leverage report.")
    parser.add_argument("--base", type=str, default=".", help="Base directory for CSVs.")
    parser.add_argument("--logs", type=str, help="Explicit played logs CSV.")
    parser.add_argument("--teams", type=str, help="Explicit team metadata CSV.")
    parser.add_argument("--sched", type=str, help="Explicit future schedule CSV.")
    parser.add_argument(
        "--out",
        type=str,
        default="out/csv_out/z_ABL_Division_Leverage.csv",
        help="Output CSV (default: out/csv_out/z_ABL_Division_Leverage.csv).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    base_dir = Path(args.base).resolve()
    meta = load_team_meta(base_dir, Path(args.teams) if args.teams else None)
    conf_div_map = build_conf_div_map(meta)

    logs_df = autodetect_logs(base_dir, Path(args.logs) if args.logs else None)
    played = prepare_played_logs(logs_df)
    report = aggregate_played(played, meta)
    if report.empty:
        raise ValueError("No played games after filtering teams.")
    anchor_date = played["game_date"].max()
    if pd.isna(anchor_date):
        anchor_date = pd.Timestamp.today()

    sched_df = autodetect_schedule(base_dir, Path(args.sched) if args.sched else None)
    if sched_df is not None:
        remaining_raw = expand_schedule_games(sched_df)
        remaining = compute_remaining(remaining_raw, anchor_date, meta)
    else:
        remaining = pd.DataFrame()

    if remaining.empty:
        report["remaining_g"] = 0
        report["remaining_div_g"] = 0
        report["remaining_div_share"] = pd.NA
    else:
        report = report.merge(remaining, on="team_id", how="left")
        report["remaining_g"] = report["remaining_g"].fillna(0).astype(int)
        report["remaining_div_g"] = report["remaining_div_g"].fillna(0).astype(int)
        report.loc[report["remaining_g"] == 0, "remaining_div_share"] = pd.NA
        mask_share = report["remaining_div_share"].notna()
        report.loc[mask_share, "remaining_div_share"] = report.loc[mask_share, "remaining_div_share"].astype(float).round(3)

    report["overall_winpct"] = report["overall_winpct"].round(3)
    report["div_winpct"] = report["div_winpct"].round(3)
    report["div_diff_winpct"] = report["div_diff_winpct"].round(3)

    column_order = [
        "team_id",
        "team_display",
        "overall_g",
        "overall_w",
        "overall_l",
        "overall_winpct",
        "div_g",
        "div_w",
        "div_l",
        "div_winpct",
        "div_diff_winpct",
        "remaining_g",
        "remaining_div_g",
        "remaining_div_share",
    ]
    report = report[column_order]
    report = report.sort_values(
        by=["remaining_div_g", "div_winpct"],
        ascending=[False, False],
        na_position="last",
    )

    output_path = (base_dir / args.out).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    text_filename = output_path.with_suffix(".txt").name
    if output_path.parent.name.lower() in {'csv_out'}:
        text_dir = output_path.parent.parent / "txt_out"
    else:
        text_dir = output_path.parent
    text_dir.mkdir(parents=True, exist_ok=True)
    text_path = text_dir / text_filename
    text_df = report.copy()
    text_df["conf_div"] = text_df["team_id"].map(conf_div_map).fillna("")
    text_path.write_text(stamp_text_block(build_text_report(text_df)), encoding="utf-8")

    preview = report.head(12)
    print("Division leverage (top 12):")
    print(preview.to_string(index=False))
    print(f"\nWrote {len(report)} rows to {output_path} and summary to {text_path}.")


if __name__ == "__main__":
    main()

