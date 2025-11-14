"""ABL Bullpen Stress Index: workload/fatigue metrics over last-14/7 days."""

from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from abl_config import stamp_text_block

TEAM_MIN, TEAM_MAX = 1, 24
APPEARANCE_CANDIDATES = [
    "pitcher_game_log.csv",
    "pitching_game_log.csv",
    "pitching_appearances.csv",
    "players_pitching_gamelog.csv",
    "players_game_pitching_stats.csv",
]
TEAM_LOG_CANDIDATES = [
    "team_game_log.csv",
    "teams_game_log.csv",
    "game_log_team.csv",
    "team_log.csv",
]
TEAM_INFO_CANDIDATES = [
    "team_info.csv",
    "teams.csv",
    "standings.csv",
    "team_record.csv",
]
PITCHING_SORTABLE_FILE = "abl_statistics_team_statistics___info_-_sortable_stats_pitching_2.csv"


def pick_column(df: pd.DataFrame, *names: str) -> Optional[str]:
    lowered = {col.lower(): col for col in df.columns}
    for name in names:
        key = name.lower()
        if key in lowered:
            return lowered[key]
    return None


def read_first(base: Path, override: Optional[Path], candidates: Sequence[str]) -> Optional[pd.DataFrame]:
    paths: List[Path] = []
    if override:
        override_path = Path(override)
        if not override_path.exists():
            raise FileNotFoundError(f"Specified file not found: {override_path}")
        paths = [override_path]
    else:
        paths = [base / name for name in candidates]
    for path in paths:
        if path.exists():
            return pd.read_csv(path)
    return None


def load_game_dates(base: Path) -> pd.Series:
    games_path = base / "games.csv"
    if not games_path.exists():
        raise FileNotFoundError("games.csv required to map game_id to dates.")
    games_df = pd.read_csv(games_path, usecols=["game_id", "date"])
    games_df["game_date"] = pd.to_datetime(games_df["date"], errors="coerce")
    return games_df.set_index("game_id")["game_date"]


def load_appearances(base: Path, override: Optional[Path]) -> pd.DataFrame:
    df = read_first(base, override, APPEARANCE_CANDIDATES)
    if df is None:
        raise FileNotFoundError("Unable to locate pitcher appearance logs.")
    team_col = pick_column(df, "team_id", "teamid", "teamID", "TeamID")
    player_col = pick_column(df, "player_id", "playerid", "PlayerID")
    date_col = pick_column(df, "game_date", "date", "GameDate")
    game_id_col = pick_column(df, "game_id", "GameID", "gameid")
    if not team_col or not player_col:
        raise ValueError("Appearance logs require team_id and player_id.")
    if not date_col:
        if not game_id_col:
            raise ValueError("Appearance logs require game_date or game_id for mapping.")
        game_dates = load_game_dates(base)

    df = df.copy()
    df["team_id"] = pd.to_numeric(df[team_col], errors="coerce").astype("Int64")
    df["player_id"] = pd.to_numeric(df[player_col], errors="coerce").astype("Int64")
    if date_col:
        df["game_date"] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        df["game_date"] = df[game_id_col].map(game_dates)
    df = df[
        df["team_id"].between(TEAM_MIN, TEAM_MAX)
        & df["player_id"].notna()
        & df["game_date"].notna()
    ].copy()

    ip_col = pick_column(df, "ip")
    ip_outs_col = pick_column(df, "ip_outs", "outs")
    if ip_col:
        df["ip_val"] = pd.to_numeric(df[ip_col], errors="coerce")
    else:
        df["ip_val"] = np.nan
    if ip_outs_col:
        ip_outs = pd.to_numeric(df[ip_outs_col], errors="coerce")
        df.loc[df["ip_val"].isna() & ip_outs.notna(), "ip_val"] = ip_outs / 3.0
    df = df[df["ip_val"].notna()].copy()

    relief_col = pick_column(df, "relief_flag", "is_relief", "rp_flag")
    gs_col = pick_column(df, "gs", "game_started", "start_flag")
    if relief_col:
        df["is_relief"] = pd.to_numeric(df[relief_col], errors="coerce").fillna(0).astype(int) == 1
    elif gs_col:
        df["is_relief"] = pd.to_numeric(df[gs_col], errors="coerce").fillna(0).astype(int) == 0
    else:
        df["is_relief"] = True
    df = df[df["is_relief"]].copy()

    li_col = pick_column(df, "leverage_index", "li")
    sv_col = pick_column(df, "sv")
    hld_col = pick_column(df, "hld", "hold")
    gf_col = pick_column(df, "gf")
    svsit_col = pick_column(df, "sv_sit", "save_situation")

    hi = pd.Series(np.nan, index=df.index)
    hi_available = False
    if li_col:
        li_vals = pd.to_numeric(df[li_col], errors="coerce")
        hi = li_vals >= 1.5
        hi_available = li_vals.notna().any()
    proxies = []
    for col in [sv_col, hld_col, gf_col, svsit_col]:
        if col:
            proxies.append(pd.to_numeric(df[col], errors="coerce").fillna(0))
    if not hi_available and proxies:
        proxy_flag = np.zeros(len(df), dtype=bool)
        for series in proxies:
            proxy_flag |= series > 0
        hi = proxy_flag
        hi_available = True
    if not hi_available:
        hi = pd.Series([np.nan] * len(df), index=df.index)
    df["hi_flag"] = hi

    return df[["team_id", "player_id", "game_date", "ip_val", "hi_flag"]]


def load_team_logs(base: Path, override: Optional[Path]) -> Optional[pd.DataFrame]:
    df = read_first(base, override, TEAM_LOG_CANDIDATES)
    if df is None:
        return None
    team_col = pick_column(df, "team_id", "teamid", "teamID", "TeamID")
    date_col = pick_column(df, "game_date", "date", "GameDate")
    if not team_col or not date_col:
        return None
    df = df.copy()
    df["team_id"] = pd.to_numeric(df[team_col], errors="coerce").astype("Int64")
    df["game_date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[
        df["team_id"].between(TEAM_MIN, TEAM_MAX) & df["game_date"].notna()
    ].copy()
    return df[["team_id", "game_date"]]


def load_team_names(base: Path, override: Optional[Path]) -> Tuple[Dict[int, str], Dict[int, str]]:
    df = read_first(base, override, TEAM_INFO_CANDIDATES)
    if df is None:
        return {}, {}
    team_col = pick_column(df, "team_id", "teamid", "teamID", "TeamID")
    name_col = pick_column(df, "team_display", "team_name", "name", "TeamName")
    div_col = pick_column(df, "division_id", "divisionid", "div_id")
    sub_col = pick_column(df, "sub_league_id", "subleague_id", "sub_id", "subleague")
    if not team_col or not name_col:
        return {}, {}
    meta = pd.DataFrame()
    meta["team_id"] = pd.to_numeric(df[team_col], errors="coerce").astype("Int64")
    meta["team_display"] = df[name_col].fillna("")
    if div_col:
        meta["division_id"] = pd.to_numeric(df[div_col], errors="coerce").astype("Int64")
    else:
        meta["division_id"] = pd.NA
    if sub_col:
        meta["sub_league_id"] = pd.to_numeric(df[sub_col], errors="coerce").astype("Int64")
    else:
        meta["sub_league_id"] = pd.NA
    meta = meta[(meta["team_id"] >= TEAM_MIN) & (meta["team_id"] <= TEAM_MAX)]
    names = meta.set_index("team_id")["team_display"].to_dict()
    conf_map = {0: "N", 1: "A"}
    div_map = {0: "E", 1: "C", 2: "W"}
    tags: Dict[int, str] = {}
    for _, row in meta.iterrows():
        tid = int(row["team_id"])
        conf = conf_map.get(int(row["sub_league_id"])) if pd.notna(row["sub_league_id"]) else ""
        div = div_map.get(int(row["division_id"])) if pd.notna(row["division_id"]) else ""
        label = "-".join(filter(None, [conf, div]))
        tags[tid] = label
    return names, tags


def load_inherited_runs(base: Path) -> pd.DataFrame:
    path = base / PITCHING_SORTABLE_FILE
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    id_col = pick_column(df, "team_id", "teamid", "teamID", "ID")
    if not id_col:
        return pd.DataFrame()
    df = df.copy()
    df["team_id"] = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")
    df = df[df["team_id"].between(TEAM_MIN, TEAM_MAX)]
    ir_col = pick_column(df, "IR")
    irs_col = pick_column(df, "IRS")
    irs_pct_col = pick_column(df, "IRS%", "IRS_pct")
    lob_col = pick_column(df, "LOB%", "LOB_pct")
    data = pd.DataFrame()
    data["team_id"] = df["team_id"]
    if ir_col:
        data["season_ir"] = pd.to_numeric(df[ir_col], errors="coerce")
    if irs_col:
        data["season_irs"] = pd.to_numeric(df[irs_col], errors="coerce")
    if irs_pct_col:
        data["season_irs_pct"] = pd.to_numeric(df[irs_pct_col], errors="coerce")
    if lob_col:
        data["season_lob_pct"] = pd.to_numeric(df[lob_col], errors="coerce")
    return data


def derive_games(df: pd.DataFrame, team_logs: Optional[pd.DataFrame], start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    if team_logs is not None:
        mask = (team_logs["game_date"] >= start) & (team_logs["game_date"] <= end)
        counts = (
            team_logs.loc[mask]
            .groupby("team_id")["game_date"]
            .nunique()
        )
    else:
        mask = (df["game_date"] >= start) & (df["game_date"] <= end)
        counts = (
            df.loc[mask, ["team_id", "game_date"]]
            .drop_duplicates()
            .groupby("team_id")["game_date"]
            .nunique()
        )
    return counts


def compute_b2b(window_df: pd.DataFrame) -> pd.Series:
    if window_df.empty:
        return pd.Series(dtype="float64")
    totals: Dict[int, int] = {}
    for (team_id, _), group in window_df.groupby(["team_id", "player_id"]):
        dates = (
            group["game_date"]
            .drop_duplicates()
            .sort_values()
            .to_numpy(dtype="datetime64[D]")
        )
        if dates.size == 0:
            continue
        diffs = np.diff(dates).astype(int)
        count = int((diffs == 1).sum())
        totals[team_id] = totals.get(team_id, 0) + count
    return pd.Series(totals)


def summarize_window(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    team_logs: Optional[pd.DataFrame],
) -> pd.DataFrame:
    window_df = df[(df["game_date"] >= start) & (df["game_date"] <= end)].copy()
    team_games = derive_games(df, team_logs, start, end)
    ip_sum = window_df.groupby("team_id")["ip_val"].sum()
    apps = window_df.groupby("team_id").size()
    relievers = window_df.groupby("team_id")["player_id"].nunique()
    b2b_counts = compute_b2b(window_df)

    hi_counts = None
    if window_df["hi_flag"].notna().any():
        hi_counts = window_df.groupby("team_id")["hi_flag"].sum(min_count=1)

    idx = pd.Index(range(TEAM_MIN, TEAM_MAX + 1), name="team_id")
    result = pd.DataFrame(index=idx)
    result["team_games"] = team_games.reindex(idx)
    result["ip"] = ip_sum.reindex(idx)
    result["apps"] = apps.reindex(idx)
    result["relievers"] = relievers.reindex(idx)
    result["b2b"] = b2b_counts.reindex(idx)

    result["ip_per_game"] = result["ip"] / result["team_games"]
    result["apps_per_game"] = result["apps"] / result["team_games"]
    result["b2b_rate"] = result["b2b"] / result["apps"]

    if hi_counts is not None:
        result["hi"] = hi_counts.reindex(idx)
        result["hi_share"] = result["hi"] / result["apps"]
    else:
        result["hi"] = np.nan
        result["hi_share"] = np.nan

    result["stress_index"] = (
        result["ip_per_game"]
        + 0.5 * result["b2b_rate"]
        + 0.5 * result["hi_share"]
    )
    return result.reset_index()


def round_column(df: pd.DataFrame, col: str, digits: int) -> None:
    if col in df.columns:
        df[col] = df[col].round(digits)


def classify_stress(value: float) -> str:
    if pd.isna(value):
        return "Unknown"
    if value >= 3.0:
        return "Critical"
    if value >= 2.5:
        return "High"
    if value >= 2.0:
        return "Moderate"
    return "Manageable"


def build_text_report(df: pd.DataFrame, limit: int = 24) -> str:
    lines = [
        "ABL Bullpen Stress Index",
        "=" * 27,
        "Last-14/7-day bullpen workload snapshot: innings per game, back-to-back usage, and high-leverage share.",
        "Meaning: higher stress scores flag pens carrying heavier loads and potential fatigue risk; inherited runner data adds season-long context.",
        "",
    ]
    header = (
        f"{'Team':<22} {'CD':<4} {'Rating':<10} "
        f"{'Stress14':>8} {'Stress7':>8} {'IP/G14':>8} {'B2B14':>8} {'HiLev14':>8} {'IR':>5} {'IRS%':>6}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for _, row in df.head(limit).iterrows():
        tag = row.get("conf_div", "")
        name = row["team_display"] or f"Team {int(row['team_id'])}"
        display = name
        tag_display = tag or ""
        stress14 = f"{row['stress_index_14']:.3f}" if pd.notna(row["stress_index_14"]) else "NA"
        stress7 = f"{row['stress_index_7']:.3f}" if pd.notna(row["stress_index_7"]) else "NA"
        ip_pg = f"{row['ip_per_game_14']:.2f}" if pd.notna(row["ip_per_game_14"]) else "NA"
        b2b = f"{row['b2b_rate_14']:.3f}" if pd.notna(row["b2b_rate_14"]) else "NA"
        hi = f"{row['hi_share_14']:.3f}" if pd.notna(row["hi_share_14"]) else "NA"
        level = classify_stress(row["stress_index_14"])
        ir = f"{int(row['season_ir'])}" if pd.notna(row.get("season_ir")) else "NA"
        irs_pct = (
            f"{row['season_irs_pct']:.3f}" if pd.notna(row.get("season_irs_pct")) else "NA"
        )
        lines.append(
            f"{display:<22} {tag_display:<4} {level:<10} "
            f"{stress14:>8} {stress7:>8} {ip_pg:>8} {b2b:>8} {hi:>8} {ir:>5} {irs_pct:>6}"
        )
    lines.append("")
    lines.append("Key (Ratings):")
    lines.append("  Critical >=3.0 | High 2.5-2.99 | Moderate 2.0-2.49 | Manageable <2.0 (Unknown if data gaps).")
    lines.append("")
    lines.append("Definitions:")
    lines.append("  Stress14/7 = composite bullpen strain over last 14 / last 7 days.")
    lines.append("  IP/G14 (IP/G7) = bullpen innings per team game across last 14 (7) days.")
    lines.append("  B2B14 (B2B7) = share of relief appearances made on back-to-back days in last 14 (7) days.")
    lines.append("  HiLev14 (HiLev7) = share of outings with leverage index >=1.5 or marked SV/HLD/GF/save-sit.")
    lines.append("  IR/IRS% = season-long inherited runners and their scoring rate from the sortable pitching report.")
    lines.append("  Stress Index = IP/G + 0.5*B2B Rate + 0.5*High-Leverage Share (higher = more strain).")
    lines.append("  Windows always reference the anchor_date-based last-14 and last-7 day periods.")
    return "\n".join(lines)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ABL Bullpen Stress Index.")
    parser.add_argument("--base", type=str, default=".", help="Base directory for CSVs.")
    parser.add_argument("--apps", type=str, help="Override pitcher appearance log.")
    parser.add_argument("--teamlogs", type=str, help="Override team game logs.")
    parser.add_argument("--teams", type=str, help="Override team info file.")
    parser.add_argument(
        "--out",
        type=str,
        default="out/csv_out/z_ABL_Bullpen_Stress_Index.csv",
        help="Output CSV path.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    base_dir = Path(args.base).resolve()

    apps_df = load_appearances(base_dir, Path(args.apps) if args.apps else None)
    if apps_df.empty:
        raise ValueError("No relief appearances after filtering.")
    team_logs = load_team_logs(base_dir, Path(args.teamlogs) if args.teamlogs else None)
    names, conf_div_tags = load_team_names(base_dir, Path(args.teams) if args.teams else None)

    anchor_date = apps_df["game_date"].max()
    if pd.isna(anchor_date):
        raise ValueError("Unable to determine anchor date from appearances.")
    last14_start = anchor_date - timedelta(days=13)
    last7_start = anchor_date - timedelta(days=6)

    summary14 = summarize_window(apps_df, last14_start, anchor_date, team_logs)
    summary7 = summarize_window(apps_df, last7_start, anchor_date, team_logs)

    suffix_map = {
        "team_games": "team_games",
        "relievers": "relievers_used",
        "ip": "ip",
        "ip_per_game": "ip_per_game",
        "apps": "apps",
        "apps_per_game": "apps_per_game",
        "b2b": "b2b",
        "b2b_rate": "b2b_rate",
        "hi": "hi",
        "hi_share": "hi_share",
        "stress_index": "stress_index",
    }
    summary14 = summary14.rename(columns={k: f"{v}_14" for k, v in suffix_map.items()})
    summary7 = summary7.rename(columns={k: f"{v}_7" for k, v in suffix_map.items()})

    report = summary14.merge(summary7, on="team_id", how="outer")
    report["team_display"] = report["team_id"].map(names).fillna("")
    report["conf_div"] = report["team_id"].map(conf_div_tags).fillna("")

    inherited = load_inherited_runs(base_dir)
    if not inherited.empty:
        report = report.merge(inherited, on="team_id", how="left")

    round_specs = {
        "ip_per_game_14": 2,
        "apps_per_game_14": 2,
        "b2b_rate_14": 3,
        "hi_share_14": 3,
        "stress_index_14": 3,
        "ip_per_game_7": 2,
        "apps_per_game_7": 2,
        "b2b_rate_7": 3,
        "hi_share_7": 3,
        "stress_index_7": 3,
        "season_irs_pct": 3,
        "season_lob_pct": 3,
    }
    for col, digits in round_specs.items():
        round_column(report, col, digits)

    # Ensure inherited-run columns exist even if the sortable export lacked them.
    inherited_cols = ["season_ir", "season_irs", "season_irs_pct", "season_lob_pct"]
    for col in inherited_cols:
        if col not in report.columns:
            report[col] = np.nan

    text_df = report.copy()
    column_order = [
        "team_id",
        "team_display",
        "team_games_14",
        "relievers_used_14",
        "ip_14",
        "ip_per_game_14",
        "apps_14",
        "apps_per_game_14",
        "b2b_14",
        "b2b_rate_14",
        "hi_14",
        "hi_share_14",
        "stress_index_14",
        "team_games_7",
        "relievers_used_7",
        "ip_7",
        "ip_per_game_7",
        "apps_7",
        "apps_per_game_7",
        "b2b_7",
        "b2b_rate_7",
        "hi_7",
        "hi_share_7",
        "stress_index_7",
        "season_ir",
        "season_irs",
        "season_irs_pct",
        "season_lob_pct",
    ]
    report = report[column_order]

    report = report.sort_values(
        by=["stress_index_14", "stress_index_7"],
        ascending=[False, False],
        na_position="last",
    )

    out_path = (base_dir / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_path, index=False)

    text_dir = base_dir / "out" / "txt_out"
    text_dir.mkdir(parents=True, exist_ok=True)
    text_path = text_dir / out_path.name.replace(".csv", ".txt")
    text_path.write_text(stamp_text_block(build_text_report(text_df)), encoding="utf-8")

    preview = report.head(12)
    print("Bullpen Stress Index (top 12):")
    print(preview.to_string(index=False))
    print(f"\nWrote {len(report)} rows to {out_path} and summary to {text_path}.")


if __name__ == "__main__":
    main()
