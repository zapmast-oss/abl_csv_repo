"""ABL Rotation Stability: quantify how consistent each rotation has been."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from abl_config import stamp_text_block

PITCH_LOG_CANDIDATES = [
    "pitcher_game_log.csv",
    "pitching_game_log.csv",
    "players_pitching_gamelog.csv",
    "pitching_appearances.csv",
    "players_game_pitching_stats.csv",
]
INJURY_CANDIDATES = [
    "injuries.csv",
    "disabled_list.csv",
    "transactions.csv",
    "player_status.csv",
]
TEAM_INFO_CANDIDATES = [
    "team_info.csv",
    "teams.csv",
    "standings.csv",
    "team_record.csv",
]
TEAM_MIN, TEAM_MAX = 1, 24


def pick_column(df: pd.DataFrame, *names: str) -> Optional[str]:
    lowered = {col.lower(): col for col in df.columns}
    for name in names:
        key = name.lower()
        if key in lowered:
            return lowered[key]
    return None


def read_first(base: Path, override: Optional[Path], candidates: Sequence[str]) -> Optional[pd.DataFrame]:
    if override:
        if not override.exists():
            raise FileNotFoundError(f"Specified file not found: {override}")
        return pd.read_csv(override)
    for name in candidates:
        path = base / name
        if path.exists():
            return pd.read_csv(path)
    return None


def resolve_text_path(csv_path: Path) -> Path:
    txt_name = csv_path.with_suffix(".txt").name
    parent = csv_path.parent
    if parent.name.lower() in {'csv_out'}:
        txt_dir = parent.parent / "txt_out"
    else:
        txt_dir = parent
    txt_dir.mkdir(parents=True, exist_ok=True)
    return txt_dir / txt_name


def load_team_names(base: Path, override: Optional[Path]) -> Tuple[Dict[int, str], Dict[int, str]]:
    df = read_first(base, override, TEAM_INFO_CANDIDATES)
    if df is None:
        return {}, {}
    team_col = pick_column(df, "team_id", "teamid", "teamID", "TeamID")
    display_col = pick_column(df, "team_display", "team_name", "name", "TeamName")
    sub_col = pick_column(df, "sub_league_id", "subleague_id", "conference_id")
    div_col = pick_column(df, "division_id", "division")
    if not team_col:
        return {}, {}
    display_map: Dict[int, str] = {}
    conf_map: Dict[int, str] = {}
    conf_lookup = {0: "N", 1: "A"}
    div_lookup = {0: "E", 1: "C", 2: "W"}
    for _, row in df.iterrows():
        tid = row.get(team_col)
        if pd.isna(tid):
            continue
        tid_int = int(tid)
        if not (TEAM_MIN <= tid_int <= TEAM_MAX):
            continue
        if display_col and pd.notna(row.get(display_col)):
            display_map[tid_int] = str(row.get(display_col))
        if tid_int not in conf_map and sub_col and div_col:
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
                sub = conf_lookup.get(sub_key, str(sub_val)[0].upper())
                div = div_lookup.get(div_key, str(div_val)[0].upper())
                conf_map[tid_int] = f"{sub}-{div}"
    return display_map, conf_map


def load_game_dates(base: Path) -> Optional[pd.DataFrame]:
    path = base / "games.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, usecols=["game_id", "date"])
    df["game_id"] = df["game_id"].astype(str)
    df["game_date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["game_date"])


def load_pitch_logs(base: Path, override: Optional[Path]) -> pd.DataFrame:
    df = read_first(base, override, PITCH_LOG_CANDIDATES)
    if df is None:
        raise FileNotFoundError("Unable to locate pitcher game logs or appearances.")
    df = df.copy()
    team_col = pick_column(df, "team_id", "teamid", "teamID")
    player_col = pick_column(df, "player_id", "playerID")
    date_col = pick_column(df, "game_date", "date", "GameDate")
    game_col = pick_column(df, "game_id", "gameid")
    if not (team_col and player_col):
        raise ValueError("Pitching logs require team_id and player_id columns.")
    df["team_id"] = pd.to_numeric(df[team_col], errors="coerce").astype("Int64")
    df["player_id"] = pd.to_numeric(df[player_col], errors="coerce").astype("Int64")
    if date_col:
        df["game_date"] = pd.to_datetime(df[date_col], errors="coerce")
    elif game_col:
        df["game_id"] = df[game_col].astype(str)
        game_dates = load_game_dates(base)
        if game_dates is None:
            raise ValueError("Pitching logs missing game_date; games.csv required for mapping.")
        df = df.merge(game_dates, on="game_id", how="left")
        df["game_date"] = df["game_date"]
    else:
        raise ValueError("Pitching logs require either game_date or game_id for date mapping.")
    df = df.dropna(subset=["team_id", "player_id", "game_date"])
    df = df[(df["team_id"] >= TEAM_MIN) & (df["team_id"] <= TEAM_MAX)]
    ip_col = pick_column(df, "ip")
    outs_col = pick_column(df, "ip_outs", "outs")
    if ip_col:
        df["IP"] = pd.to_numeric(df[ip_col], errors="coerce")
    else:
        df["IP"] = np.nan
    if outs_col:
        outs = pd.to_numeric(df[outs_col], errors="coerce")
        df.loc[df["IP"].isna() & outs.notna(), "IP"] = outs / 3.0
    er_col = pick_column(df, "er", "ER")
    df["ER"] = pd.to_numeric(df[er_col], errors="coerce") if er_col else np.nan
    gs_col = pick_column(df, "gs", "GS", "games_started_flag")
    if gs_col:
        df["GS_flag"] = pd.to_numeric(df[gs_col], errors="coerce").fillna(0).astype(int)
    else:
        df["GS_flag"] = np.nan
    role_col = pick_column(df, "role", "pitcher_role", "position", "pos")
    if role_col:
        df["role_flag"] = (
            df[role_col].astype(str).str.upper().isin({"SP", "STARTER"})
        )
    else:
        df["role_flag"] = False
    df["team_display"] = ""
    disp_col = pick_column(df, "team_display", "team_name", "name", "TeamName")
    if disp_col:
        df["team_display"] = df[disp_col].fillna("")
    if not game_col:
        game_col = pick_column(df, "game_id", "gameid")
    df["game_id"] = df[game_col].astype(str) if game_col else ""
    return df[
        [
            "team_id",
            "player_id",
            "game_date",
            "IP",
            "ER",
            "GS_flag",
            "role_flag",
            "team_display",
            "game_id",
        ]
    ]


def identify_starts(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    if data["GS_flag"].notna().any():
        data["is_start"] = data["GS_flag"] == 1
    else:
        data["is_start"] = data["role_flag"]
        if not data["is_start"].any():
            data = data.sort_values(
                ["team_id", "game_date", "game_id", "player_id"]
            )
            first_mask = (
                data.groupby(["team_id", "game_date"]).cumcount() == 0
            )
            data["is_start"] = first_mask
    data = data[data["is_start"] & data["IP"].notna()]
    return data


def compute_rotation_metrics(starts: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Timestamp]:
    if starts.empty:
        return pd.DataFrame(), pd.NaT
    starts["QS_flag"] = np.where(
        starts["ER"].notna(), (starts["IP"] >= 6.0) & (starts["ER"] <= 3), np.nan
    )
    starts["ER_clean"] = starts["ER"].fillna(0)
    team_totals = starts.groupby("team_id")["is_start"].size().rename("total_starts")
    unique_starters = starts.groupby("team_id")["player_id"].nunique().rename("unique_starters")
    ip_sum = starts.groupby("team_id")["IP"].sum().rename("ip_sum")
    qs_count = (
        starts[starts["QS_flag"] == True]
        .groupby("team_id")["QS_flag"]
        .size()
        .rename("qs_count")
    )
    qs_denom = (
        starts[starts["QS_flag"].notna()]
        .groupby("team_id")["QS_flag"]
        .size()
        .rename("qs_denom")
    )
    per_starter = (
        starts.groupby(["team_id", "player_id"])
        .agg(
            starts=("is_start", "size"),
            ip_sum=("IP", "sum"),
            er_sum=("ER_clean", "sum"),
        )
        .reset_index()
    )
    def compute_top_share(group: pd.DataFrame) -> Tuple[float, float]:
        total = group["starts"].sum()
        ordered = group.sort_values(
            ["starts", "ip_sum", "er_sum"],
            ascending=[False, False, True],
        )
        top5_starts = ordered.head(5)["starts"].sum()
        share = top5_starts / total if total > 0 else np.nan
        return top5_starts, share
    top_rows = []
    for team_id, group in per_starter.groupby("team_id"):
        top5, share = compute_top_share(group)
        top_rows.append({"team_id": team_id, "top5_starts": top5, "top5_share": share})
    top_df = pd.DataFrame(top_rows)
    summary = (
        pd.DataFrame({"team_id": team_totals.index})
        .merge(team_totals.reset_index(), on="team_id", how="left")
        .merge(unique_starters.reset_index(), on="team_id", how="left")
        .merge(ip_sum.reset_index(), on="team_id", how="left")
        .merge(qs_count.reset_index(), on="team_id", how="left")
        .merge(qs_denom.reset_index(), on="team_id", how="left")
        .merge(top_df, on="team_id", how="left")
    )
    summary["avg_ip_per_start"] = summary["ip_sum"] / summary["total_starts"]
    summary["qs_pct"] = summary["qs_count"] / summary["qs_denom"]
    anchor_date = starts["game_date"].max()
    return summary, anchor_date


def load_injuries(base: Path, override: Optional[Path]) -> Optional[pd.DataFrame]:
    df = read_first(base, override, INJURY_CANDIDATES)
    if df is None:
        return None
    df = df.copy()
    pid_col = pick_column(df, "player_id", "playerID")
    team_col = pick_column(df, "team_id", "teamID")
    if not pid_col:
        return None
    df["player_id"] = pd.to_numeric(df[pid_col], errors="coerce").astype("Int64")
    if team_col:
        df["team_id"] = pd.to_numeric(df[team_col], errors="coerce").astype("Int64")
    else:
        df["team_id"] = pd.NA
    on_il_col = pick_column(df, "on_il", "on_dl", "is_injured", "current_il")
    if on_il_col:
        df["on_il_flag"] = pd.to_numeric(df[on_il_col], errors="coerce").fillna(0).astype(int)
    else:
        df["on_il_flag"] = np.nan
    il_days_col = pick_column(df, "il_days", "days_on_il", "injury_days_to_date")
    if il_days_col:
        df["il_days"] = pd.to_numeric(df[il_days_col], errors="coerce")
    else:
        df["il_days"] = np.nan
    start_col = pick_column(df, "start_date", "injury_start", "dl_start", "date_start")
    end_col = pick_column(df, "end_date", "injury_end", "dl_end", "date_end")
    df["start_date"] = pd.to_datetime(df[start_col], errors="coerce") if start_col else pd.NaT
    df["end_date"] = pd.to_datetime(df[end_col], errors="coerce") if end_col else pd.NaT
    return df


def compute_injury_metrics(
    injuries: Optional[pd.DataFrame],
    starter_table: pd.DataFrame,
    anchor_date: pd.Timestamp,
) -> Tuple[Dict[int, float], Dict[int, int]]:
    if injuries is None or injuries.empty or starter_table.empty or pd.isna(anchor_date):
        return {}, {}
    starter_players = starter_table["player_id"].unique().tolist()
    starter_set = set(starter_players)
    player_team_map = starter_table.set_index("player_id")["team_id"].to_dict()
    df = injuries[injuries["player_id"].isin(starter_set)].copy()
    if df.empty:
        return {}, {}
    if df["il_days"].isna().all():
        effective_end = df["end_date"].fillna(anchor_date)
        effective_end = effective_end.clip(upper=anchor_date)
        duration = (effective_end - df["start_date"]).dt.days + 1
        duration = duration.where(duration > 0, 0)
        df["il_days_calc"] = duration
    else:
        df["il_days_calc"] = df["il_days"].fillna(0)
    player_days = df.groupby("player_id")["il_days_calc"].sum()
    if "on_il_flag" in df.columns:
        player_il_flag = (
            df.groupby("player_id")["on_il_flag"].max().replace({np.nan: 0}).astype(int)
        )
    else:
        player_il_flag = pd.Series(dtype=int)
    starter_il_days: Dict[int, float] = {}
    starter_current_il: Dict[int, int] = {}
    for _, row in starter_table.iterrows():
        team_id = int(row["team_id"])
        pid = int(row["player_id"])
        days = float(player_days.get(pid, 0))
        if days:
            starter_il_days[team_id] = starter_il_days.get(team_id, 0.0) + days
        flag = int(player_il_flag.get(pid, 0)) if not player_il_flag.empty else 0
        if flag:
            starter_current_il[team_id] = starter_current_il.get(team_id, 0) + 1
    return starter_il_days, starter_current_il


def assign_rating(row: pd.Series) -> str:
    share = row.get("top5_share")
    avg_ip = row.get("avg_ip_per_start")
    qs_pct = row.get("qs_pct")
    if pd.isna(share) or pd.isna(avg_ip):
        return "NA"
    qs_val = 0.0 if pd.isna(qs_pct) else float(qs_pct)
    if share >= 0.80 and avg_ip >= 5.0 and qs_val >= 0.30:
        return "Elite"
    if share >= 0.75 and avg_ip >= 4.8:
        return "Stable"
    if share >= 0.65:
        return "Mixed"
    return "Volatile"


def build_final_table(
    summary: pd.DataFrame,
    team_names: Dict[int, str],
    conf_map: Dict[int, str],
    starter_table: pd.DataFrame,
    starter_il_days: Dict[int, float],
    starter_current_il: Dict[int, int],
) -> pd.DataFrame:
    if summary.empty:
        return summary
    summary["team_display"] = summary["team_id"].map(team_names).fillna("")
    summary["conf_div"] = summary["team_id"].map(conf_map).fillna("")
    summary["starter_il_days"] = summary["team_id"].map(starter_il_days)
    summary["starters_current_il"] = summary["team_id"].map(starter_current_il)
    summary["top5_share"] = summary["top5_share"].round(3)
    summary["avg_ip_per_start"] = summary["avg_ip_per_start"].round(2)
    summary["qs_pct"] = summary["qs_pct"].round(3)
    summary["starter_il_days"] = summary["starter_il_days"].round(0)
    def _format_int_like(value: object) -> str:
        if value in ("", None) or (isinstance(value, float) and pd.isna(value)):
            return ""
        try:
            return str(int(float(value)))
        except (TypeError, ValueError):
            return ""

    summary["starter_il_days"] = summary["starter_il_days"].apply(_format_int_like)
    summary["starters_current_il"] = summary["starters_current_il"].apply(_format_int_like)
    summary["rotation_rating"] = summary.apply(assign_rating, axis=1)
    summary = summary[
        [
            "team_id",
            "team_display",
            "conf_div",
            "total_starts",
            "unique_starters",
            "top5_starts",
            "top5_share",
            "avg_ip_per_start",
            "qs_count",
            "qs_denom",
            "qs_pct",
            "starter_il_days",
            "starters_current_il",
            "rotation_rating",
        ]
    ]
    summary = summary.sort_values(
        by=["top5_share", "avg_ip_per_start", "qs_pct"],
        ascending=[False, False, False],
        na_position="last",
    )
    return summary


def build_text_report(df: pd.DataFrame, limit: Optional[int] = None) -> str:
    lines = [
        "ABL Rotation Stability",
        "=" * 28,
        "",
        "Measures how heavily each club leans on its top five starters, how deep they work, and whether injuries are forcing churn.",
        "Helps flag playoff-ready rotations versus staffs still scrambling for healthy innings.",
        "",
    ]
    subset = df if limit is None else df.head(limit)
    header = (
        f"{'Team':<28} {'Rating':<10} {'Top5':>7} {'AvgIP':>7} {'QS%':>7} "
        f"{'IL Days':>9} {'IL Arms':>8}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    if subset.empty:
        lines.append("(No starter data available.)")
    for _, row in subset.iterrows():
        rating = row.get("rotation_rating", "NA")
        qs_text = f"{row['qs_pct']:.3f}" if pd.notna(row["qs_pct"]) else "NA"
        combo = row.get("conf_div", "")
        name = row["team_display"]
        header = f"{name} ({combo})" if combo else name
        top5 = f"{row['top5_share']:.3f}" if pd.notna(row["top5_share"]) else "NA"
        avg_ip = f"{row['avg_ip_per_start']:.2f}" if pd.notna(row["avg_ip_per_start"]) else "NA"
        il_days_val = pd.to_numeric(row["starter_il_days"], errors="coerce")
        il_heads_val = pd.to_numeric(row["starters_current_il"], errors="coerce")
        il_days = f"{int(il_days_val)}" if pd.notna(il_days_val) else "NA"
        il_heads = f"{int(il_heads_val)}" if pd.notna(il_heads_val) else "NA"
        lines.append(
            f"{header:<28} {rating:<10} {top5:>7} {avg_ip:>7} {qs_text:>7} {il_days:>9} {il_heads:>8}"
        )
    lines.append("")
    lines.append("Key:")
    lines.append("  Elite  -> top5_share >= 0.80, Avg IP >= 5.0, QS% >= 0.300.")
    lines.append("  Stable -> top5_share >= 0.75 and Avg IP >= 4.8.")
    lines.append("  Mixed  -> top5_share >= 0.65 (heavier rotation mix).")
    lines.append("  Volatile -> top5_share < 0.65 (frequent shuffling).")
    lines.append("")
    lines.append("Definitions:")
    lines.append("  Top5 share: starts handled by the five busiest starters.")
    lines.append("  AvgIP: average innings per start.")
    lines.append("  QS%: quality-start rate (>=6 IP, <=3 ER when ER available).")
    lines.append("  IL metrics: starter IL days total and current IL headcount (blank if unavailable).")
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute ABL rotation stability metrics.")
    parser.add_argument("--base", type=str, default=".", help="Base directory for CSVs.")
    parser.add_argument("--apps", type=str, help="Override pitcher appearances/logs file.")
    parser.add_argument("--inj", type=str, help="Override injuries/status file.")
    parser.add_argument("--teams", type=str, help="Override team info file.")
    parser.add_argument(
        "--out",
        type=str,
        default="out/csv_out/z_ABL_Rotation_Stability.csv",
        help="Output CSV path.",
    )
    return parser.parse_args(argv if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    base_dir = Path(args.base).resolve()
    apps_path = Path(args.apps).resolve() if args.apps else None
    inj_path = Path(args.inj).resolve() if args.inj else None
    team_path = Path(args.teams).resolve() if args.teams else None

    logs = load_pitch_logs(base_dir, apps_path)
    starts = identify_starts(logs)
    summary, anchor_date = compute_rotation_metrics(starts)
    if summary.empty:
        team_names, conf_map = load_team_names(base_dir, team_path)
        empty_df = pd.DataFrame(
            columns=[
                "team_id",
                "team_display",
                "total_starts",
                "unique_starters",
                "top5_starts",
                "top5_share",
                "avg_ip_per_start",
                "qs_count",
                "qs_denom",
                "qs_pct",
                "starter_il_days",
                "starters_current_il",
            ]
        )
        out_path = (base_dir / args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        empty_df.to_csv(out_path, index=False)
        text_path = out_path.with_suffix(".txt")
        text_path.write_text(stamp_text_block("No starter data available."), encoding="utf-8")
        print("No starter data available; CSV written with headers only.")
        return

    starter_table = (
        starts.groupby(["team_id", "player_id"])
        .agg(starts=("is_start", "size"))
        .reset_index()
    )
    injuries_df = load_injuries(base_dir, inj_path)
    starter_il_days, starters_current_il = compute_injury_metrics(
        injuries_df, starter_table, anchor_date
    )
    team_names, conf_map = load_team_names(base_dir, team_path)
    final_df = build_final_table(
        summary,
        team_names,
        conf_map,
        starter_table,
        starter_il_days,
        starters_current_il,
    )

    out_path = (base_dir / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_path, index=False)
    text_path = resolve_text_path(out_path)
    text_path.write_text(stamp_text_block(build_text_report(final_df)), encoding="utf-8")

    print("Rotation stability (top 12):")
    print(final_df.head(12).to_string(index=False))
    print(f"\nWrote {len(final_df)} rows to {out_path}.")


if __name__ == "__main__":
    main()

