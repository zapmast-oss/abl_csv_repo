"""ABL Manager Tendencies: small-ball, hook speed, platoon usage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from abl_config import stamp_text_block

TEAM_MIN, TEAM_MAX = 1, 24

BATTING_CANDIDATES = [
    "team_batting.csv",
    "teams_batting.csv",
    "batting_team_totals.csv",
    "team_batting_stats.csv",
    "teams_batting_stats.csv",
]
BASERUN_CANDIDATES = [
    "baserunning.csv",
    "team_baserunning.csv",
]
APPEARANCE_CANDIDATES = [
    "pitcher_game_log.csv",
    "pitching_game_log.csv",
    "players_pitching_gamelog.csv",
    "pitching_appearances.csv",
    "players_game_pitching_stats.csv",
]
SPLITS_CANDIDATES = [
    "batting_splits_vs_hand.csv",
    "team_batting_splits_vs_hand.csv",
]
TEAM_INFO_CANDIDATES = [
    "team_info.csv",
    "teams.csv",
    "standings.csv",
    "team_record.csv",
]
LINEUP_FILE = "players_game_batting.csv"
COACH_FILE = "coaches.csv"
PLAYERS_FILE = "players.csv"

SMALLBALL_TIERS = [
    (1.30, "Bunt & Run"),
    (1.10, "Pressure"),
    (0.90, "Balanced"),
    (0.00, "Swing Away"),
]
HOOK_TIERS = [
    (1.15, "Quick Hook"),
    (1.00, "Short Leash"),
    (0.85, "Standard"),
    (0.00, "Patient"),
]
PLATOON_TIERS = [
    (1.50, "Heavy Platoon"),
    (1.15, "Targeted"),
    (0.85, "Balanced"),
    (0.00, "Static"),
]
MANAGER_TIERS = [
    (1.40, "Elite"),
    (1.15, "Aggressive"),
    (0.90, "Balanced"),
    (0.70, "Conservative"),
    (0.00, "Passive"),
]


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


def load_team_names(base: Path, override: Optional[Path]) -> Tuple[Dict[int, str], Dict[int, str], Dict[int, str]]:
    df = read_first(base, override, TEAM_INFO_CANDIDATES)
    if df is None:
        return {}, {}, {}
    team_col = pick_column(df, "team_id", "teamid", "teamID", "TeamID")
    name_col = pick_column(df, "team_display", "team_name", "name", "TeamName")
    div_col = pick_column(df, "division_id", "divisionid", "div_id")
    sub_col = pick_column(df, "sub_league_id", "subleague_id", "sub_id", "subleague")
    abbr_col = pick_column(df, "abbr", "team_abbr")
    if not team_col or not name_col:
        return {}, {}, {}
    meta = pd.DataFrame()
    meta["team_id"] = pd.to_numeric(df[team_col], errors="coerce").astype("Int64")
    meta["team_display"] = df[name_col].fillna("")
    if abbr_col:
        meta["team_abbr"] = df[abbr_col].fillna("")
    else:
        meta["team_abbr"] = ""
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
    meta["conf_div"] = (
        meta["sub_league_id"].map(conf_map).fillna("")
        + "-"
        + meta["division_id"].map(div_map).fillna("")
    ).str.strip("-")
    conf_div = meta.set_index("team_id")["conf_div"].to_dict()
    abbrs = meta.set_index("team_id")["team_abbr"].to_dict()
    return names, conf_div, abbrs


def load_manager_names(base: Path) -> Dict[int, str]:
    path = base / COACH_FILE
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    team_col = pick_column(df, "team_id", "teamid", "teamID", "TeamID")
    pos_col = pick_column(df, "position")
    occ_col = pick_column(df, "occupation")
    if not team_col:
        return {}
    df["team_id"] = pd.to_numeric(df[team_col], errors="coerce").astype("Int64")
    df = df[(df["team_id"] >= TEAM_MIN) & (df["team_id"] <= TEAM_MAX)]
    manager_df = df
    pos_vals = pd.to_numeric(df[pos_col], errors="coerce") if pos_col else None
    occ_vals = pd.to_numeric(df[occ_col], errors="coerce") if occ_col else None
    if occ_vals is not None:
        primary = df[occ_vals == 2]
        if not primary.empty:
            manager_df = primary
        elif pos_vals is not None:
            fallback = df[pos_vals == 1]
            if fallback.empty:
                fallback = df[pos_vals == 3]
            if not fallback.empty:
                manager_df = fallback
    elif pos_vals is not None:
        manager_df = df[pos_vals == 1]
        if manager_df.empty:
            manager_df = df[pos_vals == 3]
        if manager_df.empty:
            manager_df = df.copy()
    df = manager_df
    df["manager_name"] = (df["first_name"].fillna("") + " " + df["last_name"].fillna("")).str.strip()
    df = df[df["manager_name"].str.len() > 0]
    return (
        df.sort_values(["team_id", "manager_name"])
        .drop_duplicates("team_id")
        .set_index("team_id")["manager_name"]
        .to_dict()
    )


def load_player_hands(base: Path) -> Tuple[Dict[int, str], Dict[int, str]]:
    path = base / PLAYERS_FILE
    if not path.exists():
        return {}, {}
    df = pd.read_csv(path, usecols=["player_id", "bats", "throws"])
    bats_map = {1: "R", 2: "L", 3: "S"}
    throws_map = {1: "R", 2: "L", 3: "S"}
    df["bats_hand"] = pd.to_numeric(df["bats"], errors="coerce").map(bats_map)
    df["throws_hand"] = pd.to_numeric(df["throws"], errors="coerce").map(throws_map)
    bat_dict = df.set_index("player_id")["bats_hand"].to_dict()
    throw_dict = df.set_index("player_id")["throws_hand"].to_dict()
    return bat_dict, throw_dict


def load_lineups(base: Path, bats_map: Dict[int, str], override: Optional[Path] = None) -> Optional[pd.DataFrame]:
    path = Path(override) if override else base / LINEUP_FILE
    if not path.exists():
        return None
    df = pd.read_csv(path, usecols=["player_id", "team_id", "game_id", "gs"])
    df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce").astype("Int64")
    df = df[(df["team_id"] >= TEAM_MIN) & (df["team_id"] <= TEAM_MAX)]
    df["gs"] = pd.to_numeric(df["gs"], errors="coerce").fillna(0)
    df = df[df["gs"] > 0].copy()
    df["bats_hand"] = df["player_id"].map(bats_map)
    df["game_id"] = df["game_id"].astype(str)
    return df[["team_id", "game_id", "bats_hand"]]


def load_batting(base: Path, override: Optional[Path]) -> pd.DataFrame:
    df = read_first(base, override, BATTING_CANDIDATES)
    if df is None:
        raise FileNotFoundError("Unable to find team batting totals.")
    team_col = pick_column(df, "team_id", "teamid", "teamID", "TeamID")
    name_col = pick_column(df, "team_display", "team_name", "name", "TeamName")
    sh_col = pick_column(df, "sh", "sac_bunt", "sac")
    sb_col = pick_column(df, "sb")
    cs_col = pick_column(df, "cs")
    pa_col = pick_column(df, "pa")
    if not team_col:
        raise ValueError("team_id column missing in batting totals.")
    data = pd.DataFrame()
    data["team_id"] = pd.to_numeric(df[team_col], errors="coerce").astype("Int64")
    data = data[(data["team_id"] >= TEAM_MIN) & (data["team_id"] <= TEAM_MAX)].copy()
    data["team_display"] = df[name_col].fillna("") if name_col else ""
    data["SH"] = pd.to_numeric(df[sh_col], errors="coerce") if sh_col else np.nan
    data["SB"] = pd.to_numeric(df[sb_col], errors="coerce") if sb_col else np.nan
    data["CS"] = pd.to_numeric(df[cs_col], errors="coerce") if cs_col else np.nan
    data["PA"] = pd.to_numeric(df[pa_col], errors="coerce") if pa_col else np.nan
    return data


def load_apps(base: Path, override: Optional[Path]) -> pd.DataFrame:
    df = read_first(base, override, APPEARANCE_CANDIDATES)
    if df is None:
        raise FileNotFoundError("Unable to find pitcher appearance logs.")
    team_col = pick_column(df, "team_id", "teamid", "teamID", "TeamID")
    player_col = pick_column(df, "player_id", "playerid", "PlayerID")
    date_col = pick_column(df, "game_date", "date", "GameDate")
    game_col = pick_column(df, "game_id", "gameid")
    ip_col = pick_column(df, "ip")
    outs_col = pick_column(df, "ip_outs", "outs")
    gs_col = pick_column(df, "gs", "start_flag")
    er_col = pick_column(df, "er")
    if not team_col or not player_col:
        raise ValueError("Appearance logs require team_id and player_id.")
    df = df.copy()
    df["team_id"] = pd.to_numeric(df[team_col], errors="coerce").astype("Int64")
    df["player_id"] = pd.to_numeric(df[player_col], errors="coerce").astype("Int64")
    if date_col:
        df["game_date"] = pd.to_datetime(df[date_col], errors="coerce")
    elif game_col:
        games = read_first(base, None, ["games.csv"])
        if games is None:
            raise ValueError("Need game_date or games.csv to map appearances.")
        games_map = games.set_index("game_id")["date"]
        df["game_date"] = pd.to_datetime(df[game_col].map(games_map), errors="coerce")
    else:
        raise ValueError("Appearance logs need game_date or game_id mapping.")
    df = df[
        df["team_id"].between(TEAM_MIN, TEAM_MAX)
        & df["player_id"].notna()
        & df["game_date"].notna()
    ].copy()
    if game_col:
        df["game_id"] = df[game_col].astype(str)
    else:
        df["game_id"] = df["game_date"].astype(str)
    if ip_col:
        df["ip_val"] = pd.to_numeric(df[ip_col], errors="coerce")
    else:
        df["ip_val"] = np.nan
    if outs_col:
        outs = pd.to_numeric(df[outs_col], errors="coerce")
        df.loc[df["ip_val"].isna() & outs.notna(), "ip_val"] = outs / 3.0
    df["ip_val"] = df["ip_val"].fillna(0)
    df["ER"] = pd.to_numeric(df[er_col], errors="coerce") if er_col else np.nan
    df["started"] = False
    if gs_col:
        df["started"] = pd.to_numeric(df[gs_col], errors="coerce").fillna(0).astype(int) == 1
    else:
        # assume first appearance per team/date is starter
        starter_idx = (
            df.sort_values(["team_id", "game_date", "player_id"])
            .groupby(["team_id", "game_date"])
            .head(1)
            .index
        )
        df.loc[starter_idx, "started"] = True
    return df


def load_splits(base: Path, override: Optional[Path]) -> Optional[pd.DataFrame]:
    df = read_first(base, override, SPLITS_CANDIDATES)
    if df is None:
        return None
    team_col = pick_column(df, "team_id", "teamid", "teamID", "TeamID")
    par_col = pick_column(df, "pa_vr", "pavr", "pa_vs_r", "pa_r", "pa_vs_rhp")
    pal_col = pick_column(df, "pa_vl", "pavl", "pa_vs_l", "pa_l", "pa_vs_lhp")
    if not team_col or not par_col or not pal_col:
        return None
    splits = pd.DataFrame()
    splits["team_id"] = pd.to_numeric(df[team_col], errors="coerce").astype("Int64")
    splits = splits[(splits["team_id"] >= TEAM_MIN) & (splits["team_id"] <= TEAM_MAX)]
    splits["PA_vR"] = pd.to_numeric(df[par_col], errors="coerce")
    splits["PA_vL"] = pd.to_numeric(df[pal_col], errors="coerce")
    adv_col = pick_column(df, "pa_adv", "platoon_adv_pa")
    splits["PA_adv"] = pd.to_numeric(df[adv_col], errors="coerce") if adv_col else np.nan
    return splits


def compute_smallball(batting: pd.DataFrame, g_est: pd.Series) -> pd.DataFrame:
    df = batting[["team_id", "team_display", "SH", "SB", "CS"]].copy()
    df["g_est"] = df["team_id"].map(g_est)
    df["sb_attempts"] = df["SB"].fillna(0) + df["CS"].fillna(0)
    df.loc[df["SB"].isna() & df["CS"].isna(), "sb_attempts"] = np.nan
    df["sb_attempts_pg"] = df["sb_attempts"] / df["g_est"]
    df["sh_pg"] = df["SH"] / df["g_est"]
    return df


def compute_hook(apps: pd.DataFrame) -> pd.DataFrame:
    starters = apps[apps["started"]].copy()
    if starters.empty:
        return pd.DataFrame()
    grouped = starters.groupby("team_id")
    starts_total = grouped.size()
    quick_mask = (starters["ip_val"] < 5.0) & (starters["ER"].fillna(0) <= 3)
    quick_counts = quick_mask.groupby(starters["team_id"]).sum()
    avg_ip = grouped["ip_val"].sum() / starts_total
    result = pd.DataFrame(
        {
            "team_id": starts_total.index,
            "starts_total": starts_total.values,
            "quick_hook_count": quick_counts.reindex(starts_total.index, fill_value=0).values,
            "quick_hook_rate": (quick_counts / starts_total).values,
            "avg_ip_start": avg_ip.values,
        }
    )
    return result


def compute_platoon_from_lineups(lineups: Optional[pd.DataFrame], apps: pd.DataFrame) -> pd.DataFrame:
    if lineups is None or lineups.empty or apps.empty:
        return pd.DataFrame(columns=["team_id", "platoon_usage_rate"])
    starts = apps[(apps["started"]) & apps["game_id"].notna()].copy()
    starts = starts.dropna(subset=["throws_hand"])
    if starts.empty:
        return pd.DataFrame(columns=["team_id", "platoon_usage_rate"])
    starts_small = starts[["game_id", "team_id", "throws_hand"]]
    opp = starts_small.merge(starts_small, on="game_id", suffixes=("", "_opp"))
    opp = opp[opp["team_id"] != opp["team_id_opp"]]
    opp = opp[["game_id", "team_id", "throws_hand_opp"]].drop_duplicates()
    opp = opp.rename(columns={"throws_hand_opp": "opp_hand"})
    lineup = lineups.merge(opp, on=["team_id", "game_id"], how="inner")
    lineup = lineup[lineup["bats_hand"].isin(["R", "L", "S"]) & lineup["opp_hand"].isin(["R", "L"])]
    if lineup.empty:
        return pd.DataFrame(columns=["team_id", "platoon_usage_rate"])
    lineup["advantage"] = np.where(
        ((lineup["opp_hand"] == "R") & lineup["bats_hand"].isin(["L", "S"]))
        | ((lineup["opp_hand"] == "L") & lineup["bats_hand"].isin(["R", "S"])),
        1,
        0,
    )
    per_game = (
        lineup.groupby(["team_id", "game_id", "opp_hand"])["advantage"]
        .mean()
        .reset_index(name="adv_share")
    )
    per_hand = (
        per_game.groupby(["team_id", "opp_hand"])
        .agg(avg=("adv_share", "mean"), games=("adv_share", "size"))
        .reset_index()
    )
    rows = []
    for team_id, group in per_hand.groupby("team_id"):
        total_games = group["games"].sum()
        if total_games == 0:
            rate = np.nan
        else:
            rate = (group["games"] * (group["avg"].sub(0.5).abs())).sum() / total_games
        rows.append({"team_id": team_id, "platoon_usage_rate": rate})
    return pd.DataFrame(rows)


def compute_platoon(
    splits: Optional[pd.DataFrame],
    lineups: Optional[pd.DataFrame],
    apps: pd.DataFrame,
) -> pd.DataFrame:
    if splits is not None and not splits.empty:
        df = splits.copy()
        total_pa = df["PA_vR"] + df["PA_vL"]
        df["platoon_usage_rate"] = (df["PA_vR"] - df["PA_vL"]).abs() / total_pa
        df.loc[total_pa == 0, "platoon_usage_rate"] = np.nan
        return df[["team_id", "platoon_usage_rate"]]
    return compute_platoon_from_lineups(lineups, apps)


def compute_g_est(apps: pd.DataFrame, batting: pd.DataFrame) -> pd.Series:
    starter_mask = apps["started"]
    starts = (
        apps[starter_mask]
        .groupby("team_id")
        .size()
        .rename("starts_total")
        .astype(float)
    )
    games_from_apps = (
        apps.groupby("team_id")["game_date"]
        .nunique()
        .rename("game_dates")
        .astype(float)
    )
    g_est = starts.combine_first(games_from_apps)
    batting_ids = batting["team_id"].unique()
    g_est = g_est.reindex(batting_ids)
    return g_est


def compute_league_plus(series: pd.Series) -> Tuple[pd.Series, float]:
    valid = series.replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index), np.nan
    league_avg = valid.mean()
    plus = series / league_avg
    return plus, league_avg


def classify_tier(value: float, tiers: Sequence[Tuple[float, str]]) -> str:
    if pd.isna(value):
        return "NA"
    for threshold, label in tiers:
        if value >= threshold:
            return label
    return tiers[-1][1]


def build_text_report(df: pd.DataFrame, limit: int = 24) -> str:
    lines = [
        "ABL Manager Tendencies",
        "=" * 26,
        "Summarizes each skipper's small-ball usage, hook speed, and platoon appetite versus league norms.",
        "Useful for scouting playoff matchups?know who bunts, who yanks starters fast, and who trusts platoons.",
        "",
    ]

    def fmt_val(val: float) -> str:
        return f"{val:.3f}" if pd.notna(val) else "NA"

    rows = []
    for _, row in df.head(limit).iterrows():
        team_name = row["team_display"] or f"Team {int(row['team_id'])}"
        team_abbr = row.get("team_abbr", "").strip()
        if not team_abbr:
            team_abbr = (team_name[:3] if team_name else "").upper()
        if not team_abbr:
            team_abbr = f"T{int(row['team_id']):02d}"
        manager = row.get("manager_name", "")
        tag = row.get("conf_div", "")
        combo = f"{team_abbr} {tag}".strip()
        row_header = f"{manager} ({combo})" if manager else combo
        rows.append(
            (
                row_header,
                row.get("manager_rating", "NA"),
                fmt_val(row.get("manager_index")),
                row.get("smallball_rating", "NA"),
                fmt_val(row.get("smallball_index")),
                row.get("hook_rating", "NA"),
                fmt_val(row.get("hook_index")),
                row.get("platoon_rating", "NA"),
                fmt_val(row.get("platoon_index")),
            )
        )

    headers = [
        "Manager / Team",
        "Rating",
        "Idx",
        "Small",
        "Idx",
        "Hook",
        "Idx",
        "Platoon",
        "Idx",
    ]
    col_widths = [len(h) for h in headers]
    for r in rows:
        for i, val in enumerate(r):
            col_widths[i] = max(col_widths[i], len(str(val)))

    def fmt_row(vals):
        return (
            f"{vals[0]:<{col_widths[0]}} | "
            f"{vals[1]:<{col_widths[1]}} {vals[2]:>{col_widths[2]}} | "
            f"{vals[3]:<{col_widths[3]}} {vals[4]:>{col_widths[4]}} | "
            f"{vals[5]:<{col_widths[5]}} {vals[6]:>{col_widths[6]}} | "
            f"{vals[7]:<{col_widths[7]}} {vals[8]:>{col_widths[8]}}"
        )

    header_line = fmt_row(headers)
    lines.append(header_line)
    lines.append("-" * len(header_line))
    for r in rows:
        lines.append(fmt_row(r))
    lines.append("")

    lines.append("Key:")
    lines.append("  Small-ball ratings: Bunt & Run (>=1.30), Pressure (1.10-1.29), Balanced (0.90-1.09), Swing Away (<0.90).")
    lines.append("  Hook ratings: Quick Hook (>=1.15), Short Leash (1.00-1.14), Standard (0.85-0.99), Patient (<0.85).")
    lines.append("  Platoon ratings: Heavy (>=1.50), Targeted (1.15-1.49), Balanced (0.85-1.14), Static (<0.85).")
    lines.append("  Manager ratings: Elite (>=1.40), Aggressive (1.15-1.39), Balanced (0.90-1.14), Conservative (0.70-0.89), Passive (<0.70).")
    lines.append("")
    lines.append("Definitions:")
    lines.append("  smallball_index = 0.6*steal_plus + 0.4*bunt_plus (plus metrics use league avg as 1.00).")
    lines.append("  sb_attempts_pg = (SB + CS) per estimated game (starter count fallback).")
    lines.append("  sh_pg = sac bunts per estimated game.")
    lines.append("  quick_hook_rate = share of starts <5 IP with <=3 ER; hook index compares to league rate.")
    lines.append("  avg_ip_start = innings per start (context for hook decisions).")
    lines.append("  platoon_usage_rate = |PA vs RHP - PA vs LHP| / total PA (lineup-derived when splits absent).")
    lines.append("  manager_index = average of the three indices above (available components only).")
    return "\n".join(lines)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ABL Manager Tendencies.")
    parser.add_argument("--base", type=str, default=".", help="Base directory for CSVs.")
    parser.add_argument("--batting", type=str, help="Override batting totals file.")
    parser.add_argument("--baserun", type=str, help="Override baserunning detail file.")
    parser.add_argument("--apps", type=str, help="Override pitcher appearance logs.")
    parser.add_argument("--splits", type=str, help="Override platoon splits file.")
    parser.add_argument("--lineups", type=str, help="Override lineup file (not used if splits found).")
    parser.add_argument("--teams", type=str, help="Override team info file.")
    parser.add_argument(
        "--out",
        type=str,
        default="out/csv_out/z_ABL_Manager_Tendencies.csv",
        help="Output CSV path (default inside out/csv_out).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    base_dir = Path(args.base).resolve()

    bats_map, throws_map = load_player_hands(base_dir)
    batting = load_batting(base_dir, Path(args.batting) if args.batting else None)
    apps = load_apps(base_dir, Path(args.apps) if args.apps else None)
    apps["throws_hand"] = apps["player_id"].map(throws_map)
    splits = load_splits(base_dir, Path(args.splits) if args.splits else None)
    lineups = load_lineups(base_dir, bats_map, Path(args.lineups) if args.lineups else None)
    names_map, conf_div_map, abbr_map = load_team_names(base_dir, Path(args.teams) if args.teams else None)
    manager_map = load_manager_names(base_dir)

    g_est = compute_g_est(apps, batting)
    smallball = compute_smallball(batting, g_est)

    hook = compute_hook(apps)
    platoon = compute_platoon(splits, lineups, apps)

    report = smallball.merge(hook, on="team_id", how="left")
    report = report.merge(platoon, on="team_id", how="left")
    report["team_display"] = report["team_display"].where(
        report["team_display"].astype(str).str.len() > 0,
        report["team_id"].map(names_map).fillna(""),
    )
    report["conf_div"] = report["team_id"].map(conf_div_map).fillna("")
    report["manager_name"] = report["team_id"].map(manager_map).fillna("")
    report["team_abbr"] = report["team_id"].map(abbr_map).fillna("").astype(str)
    missing_abbr = report["team_abbr"].str.strip() == ""
    report.loc[missing_abbr, "team_abbr"] = (
        report.loc[missing_abbr, "team_display"].fillna("").str[:3].str.upper()
    )
    still_missing = report["team_abbr"].str.strip() == ""
    report.loc[still_missing, "team_abbr"] = report.loc[still_missing, "team_id"].apply(
        lambda tid: f"T{int(tid):02d}"
    )

    # plus metrics
    steal_plus, lg_sb = compute_league_plus(report["sb_attempts_pg"])
    bunt_plus, lg_bunt = compute_league_plus(report["sh_pg"])
    hook_plus, _ = compute_league_plus(report["quick_hook_rate"])
    avg_ip_plus, _ = compute_league_plus(report["avg_ip_start"])
    platoon_plus, _ = compute_league_plus(report["platoon_usage_rate"])

    report["steal_plus"] = steal_plus
    report["bunt_plus"] = bunt_plus
    report["hook_plus"] = hook_plus
    report["platoon_plus"] = platoon_plus
    report["smallball_index"] = 0.6 * report["steal_plus"] + 0.4 * report["bunt_plus"]
    report["hook_index"] = report["hook_plus"]
    report["platoon_index"] = report["platoon_plus"]
    report["smallball_rating"] = report["smallball_index"].apply(lambda v: classify_tier(v, SMALLBALL_TIERS))
    report["hook_rating"] = report["hook_index"].apply(lambda v: classify_tier(v, HOOK_TIERS))
    report["platoon_rating"] = report["platoon_index"].apply(lambda v: classify_tier(v, PLATOON_TIERS))
    report["manager_index"] = report[["smallball_index", "hook_index", "platoon_index"]].mean(axis=1, skipna=True)
    report["manager_rating"] = report["manager_index"].apply(lambda v: classify_tier(v, MANAGER_TIERS))

    round_cols = {
        "sb_attempts_pg": 3,
        "steal_plus": 3,
        "sh_pg": 3,
        "bunt_plus": 3,
        "quick_hook_rate": 3,
        "avg_ip_start": 2,
        "hook_plus": 3,
        "platoon_usage_rate": 3,
        "platoon_plus": 3,
        "smallball_index": 3,
        "hook_index": 3,
        "platoon_index": 3,
        "manager_index": 3,
    }
    for col, digits in round_cols.items():
        if col in report.columns:
            report[col] = pd.to_numeric(report[col], errors="coerce").round(digits)

    column_order = [
        "team_id",
        "team_display",
        "team_abbr",
        "manager_name",
        "manager_rating",
        "manager_index",
        "conf_div",
        "g_est",
        "sb_attempts",
        "sb_attempts_pg",
        "steal_plus",
        "SH",
        "sh_pg",
        "bunt_plus",
        "starts_total",
        "quick_hook_count",
        "quick_hook_rate",
        "avg_ip_start",
        "hook_plus",
        "platoon_usage_rate",
        "platoon_plus",
        "smallball_index",
        "smallball_rating",
        "hook_index",
        "hook_rating",
        "platoon_index",
        "platoon_rating",
    ]
    report = report[column_order]

    report = report.sort_values(
        by=["smallball_index", "hook_index", "platoon_index"],
        ascending=[False, False, False],
        na_position="last",
    )

    out_path = (base_dir / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_path, index=False)

    text_filename = out_path.with_suffix(".txt").name
    if out_path.parent.name.lower() in {'csv_out'}:
        text_dir = out_path.parent.parent / "text_out"
    else:
        text_dir = out_path.parent
    text_dir.mkdir(parents=True, exist_ok=True)
    text_path = text_dir / text_filename
    text_path.write_text(stamp_text_block(build_text_report(report)), encoding="utf-8")

    preview = report.head(12)
    print("Manager Tendencies (top 12):")
    print(preview.to_string(index=False))
    print(f"\nWrote {len(report)} rows to {out_path} and summary to {text_path}.")


if __name__ == "__main__":
    main()

