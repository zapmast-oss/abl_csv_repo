"""ABL Catcher Battery Value report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from abl_config import stamp_text_block

TEAM_MIN, TEAM_MAX = 1, 24

FIELDING_CANDIDATES = [
    "players_career_fielding_stats.csv",
    "players_fielding.csv",
    "fielding.csv",
]
PLAYER_FIELD_STATS_FILE = "abl_statistics_player_statistics_-_sortable_stats_player_field_stats.csv"
ROSTER_CANDIDATES = [
    "players.csv",
    "player_register.csv",
    "rosters.csv",
]
TEAM_INFO_CANDIDATES = [
    "team_info.csv",
    "teams.csv",
    "standings.csv",
    "team_record.csv",
]
BATTERY_CANDIDATES = [
    "pitching_by_catcher.csv",
    "pitcher_splits_by_catcher.csv",
    "battery_splits.csv",
]
GAMELOG_CANDIDATES = [
    "players_game_pitching_stats.csv",
    "pitching_gamelogs.csv",
    "player_game_pitching.csv",
]
LINEUP_CANDIDATES = [
    "lineup_by_game.csv",
    "defense_by_game.csv",
    "game_starters.csv",
    "boxscore_fielding.csv",
]
C_FIELDING_TEAM_FILE = "abl_statistics_team_statistics___info_-_sortable_stats_c_fielding_1.csv"


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


def load_team_abbr_lookup(base: Path) -> Dict[str, int]:
    path = base / "teams.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    id_col = pick_column(df, "team_id", "teamid", "teamID")
    abbr_col = pick_column(df, "abbr", "Abbr")
    if not id_col or not abbr_col:
        return {}
    data = pd.DataFrame()
    data["team_id"] = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")
    data["abbr"] = df[abbr_col].astype(str).str.strip().str.upper()
    data = data[data["team_id"].between(TEAM_MIN, TEAM_MAX)]
    return {row["abbr"]: int(row["team_id"]) for _, row in data.iterrows() if row["abbr"]}


def load_fielding(base: Path, override: Optional[Path]) -> pd.DataFrame:
    if not override:
        player_file = base / PLAYER_FIELD_STATS_FILE
        if player_file.exists():
            lookup = load_team_abbr_lookup(base)
            if lookup:
                df = pd.read_csv(player_file)
                id_col = pick_column(df, "ID", "player_id")
                team_col = pick_column(df, "TM", "team", "Abbr")
                pos_col = pick_column(df, "POS.1", "Position")
                ip_col = pick_column(df, "IP")
                sba_col = pick_column(df, "SBA")
                rto_col = pick_column(df, "RTO")
                pb_col = pick_column(df, "PB")
                if id_col and team_col and pos_col and ip_col:
                    data = df.copy()
                    data["catcher_id"] = pd.to_numeric(data[id_col], errors="coerce").astype("Int64")
                    data = data.dropna(subset=["catcher_id"])
                    data["pos"] = data[pos_col].astype(str).str.strip().str.upper()
                    data = data[data["pos"] == "C"]
                    data["team_abbr"] = data[team_col].astype(str).str.strip().str.upper()
                    data["team_id"] = data["team_abbr"].map(lookup)
                    data = data.dropna(subset=["team_id"])
                    data["team_id"] = data["team_id"].astype(int)
                    data = data[
                        (data["team_id"] >= TEAM_MIN)
                        & (data["team_id"] <= TEAM_MAX)
                    ]
                    data["INN"] = pd.to_numeric(data[ip_col], errors="coerce")
                    if sba_col:
                        data["SBA"] = pd.to_numeric(data[sba_col], errors="coerce")
                    else:
                        data["SBA"] = np.nan
                    if rto_col:
                        data["RTO"] = pd.to_numeric(data[rto_col], errors="coerce")
                    else:
                        data["RTO"] = np.nan
                    if pb_col:
                        data["PB_val"] = pd.to_numeric(data[pb_col], errors="coerce")
                    else:
                        data["PB_val"] = np.nan
                    data["SB"] = np.where(
                        data["SBA"].notna() & data["RTO"].notna(),
                        data["SBA"] - data["RTO"],
                        np.nan,
                    )
                    grouped = (
                        data.groupby(["team_id", "catcher_id"], as_index=False)[
                            ["INN", "SB", "RTO", "PB_val"]
                        ]
                        .sum(numeric_only=True)
                        .rename(
                            columns={
                                "INN": "INN_caught",
                                "RTO": "CS",
                                "PB_val": "PB",
                            }
                        )
                    )
                    grouped["WP"] = np.nan
                    return grouped
    df = read_first(base, override, FIELDING_CANDIDATES)
    if df is None:
        raise FileNotFoundError("Unable to locate catcher fielding totals.")
    id_col = pick_column(df, "player_id", "playerid", "PlayerID")
    team_col = pick_column(df, "team_id", "teamid", "TeamID")
    pos_col = pick_column(df, "position", "pos")
    inn_col = pick_column(df, "inn", "ip", "innings")
    sb_col = pick_column(df, "sb_allowed", "sb")
    cs_col = pick_column(df, "cs")
    pb_col = pick_column(df, "pb", "passed_balls")
    wp_col = pick_column(df, "wp", "wild_pitches")
    year_col = pick_column(df, "year", "season")
    if not id_col or not team_col or not pos_col or not inn_col:
        raise ValueError("Fielding file missing key columns.")
    data = df.copy()
    if year_col:
        max_year = pd.to_numeric(data[year_col], errors="coerce").max()
        data = data[pd.to_numeric(data[year_col], errors="coerce") == max_year]
    data["player_id"] = pd.to_numeric(data[id_col], errors="coerce").astype("Int64")
    data["team_id"] = pd.to_numeric(data[team_col], errors="coerce").astype("Int64")
    data = data.dropna(subset=["player_id", "team_id"])
    data = data[(data["team_id"] >= TEAM_MIN) & (data["team_id"] <= TEAM_MAX)]
    pos_map = {"2": "C", "C": "C"}
    data["pos"] = data[pos_col].astype(str).str.strip().str.upper().map(pos_map)
    data = data[data["pos"] == "C"]
    out = pd.DataFrame()
    out["player_id"] = data["player_id"]
    out["team_id"] = data["team_id"]
    out["INN"] = pd.to_numeric(data[inn_col], errors="coerce")
    out["SB"] = pd.to_numeric(data[sb_col], errors="coerce") if sb_col else np.nan
    out["CS"] = pd.to_numeric(data[cs_col], errors="coerce") if cs_col else np.nan
    out["PB"] = pd.to_numeric(data[pb_col], errors="coerce") if pb_col else np.nan
    out["WP"] = pd.to_numeric(data[wp_col], errors="coerce") if wp_col else np.nan
    out = out.groupby(["team_id", "player_id"], as_index=False).sum(numeric_only=True)
    return out


def load_roster(base: Path, override: Optional[Path]) -> pd.DataFrame:
    df = read_first(base, override, ROSTER_CANDIDATES)
    if df is None:
        return pd.DataFrame(columns=["player_id", "player_name"])
    id_col = pick_column(df, "player_id", "playerid", "PlayerID")
    first_col = pick_column(df, "first_name", "firstname")
    last_col = pick_column(df, "last_name", "lastname")
    full_col = pick_column(df, "name_full", "name", "player_name")
    if not id_col:
        return pd.DataFrame(columns=["player_id", "player_name"])
    data = df.copy()
    data["player_id"] = pd.to_numeric(data[id_col], errors="coerce").astype("Int64")
    data = data.dropna(subset=["player_id"])
    out = pd.DataFrame()
    out["player_id"] = data["player_id"]
    if first_col and last_col:
        out["player_name"] = (
            data[first_col].fillna("").astype(str).str.strip()
            + " "
            + data[last_col].fillna("").astype(str).str.strip()
        ).str.strip()
    elif full_col:
        out["player_name"] = data[full_col].fillna("").astype(str)
    else:
        out["player_name"] = out["player_id"].astype(str)
    return out


def load_team_info(base: Path, override: Optional[Path]) -> Tuple[Dict[int, str], Dict[int, str]]:
    df = read_first(base, override, TEAM_INFO_CANDIDATES)
    if df is None:
        return {}, {}
    team_col = pick_column(df, "team_id", "teamid", "TeamID")
    name_col = pick_column(df, "abbr", "team_abbr", "team_display", "team_name", "name")
    sub_col = pick_column(df, "sub_league_id", "subleague_id", "conference_id")
    div_col = pick_column(df, "division_id", "division")
    names: Dict[int, str] = {}
    conf_map: Dict[int, str] = {}
    conf_lookup = {0: "N", 1: "A"}
    div_lookup = {0: "E", 1: "C", 2: "W"}
    if not team_col:
        return names, conf_map
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
        if name_col and pd.notna(row.get(name_col)):
            names[tid] = str(row.get(name_col))
        if tid in conf_map or not sub_col or not div_col:
            continue
        sub_val = row.get(sub_col)
        div_val = row.get(div_col)
        if pd.isna(sub_val) or pd.isna(div_val):
            continue
        try:
            sub_key = int(sub_val)
        except (TypeError, ValueError):
            sub_key = None
        try:
            div_key = int(div_val)
        except (TypeError, ValueError):
            div_key = None
        conf_map[tid] = f"{conf_lookup.get(sub_key, str(sub_val)[0].upper())}-{div_lookup.get(div_key, str(div_val)[0].upper())}"
    return names, conf_map


def load_battery_splits(base: Path, override: Optional[Path]) -> pd.DataFrame:
    df = read_first(base, override, BATTERY_CANDIDATES)
    if df is None:
        return pd.DataFrame(columns=["catcher_id", "team_id", "IP_with_c", "ER_with_c"])
    catcher_col = pick_column(df, "catcher_id", "player_id")
    team_col = pick_column(df, "team_id", "teamid", "TeamID")
    ip_col = pick_column(df, "ip", "IP", "ip_caught")
    ip_outs_col = pick_column(df, "ip_outs", "outs")
    er_col = pick_column(df, "er", "ER", "earned_runs")
    if not catcher_col or not team_col or not ip_col and not ip_outs_col:
        return pd.DataFrame(columns=["catcher_id", "team_id", "IP_with_c", "ER_with_c"])
    data = df.copy()
    data["catcher_id"] = pd.to_numeric(data[catcher_col], errors="coerce").astype("Int64")
    data["team_id"] = pd.to_numeric(data[team_col], errors="coerce").astype("Int64")
    data = data.dropna(subset=["catcher_id", "team_id"])
    data = data[(data["team_id"] >= TEAM_MIN) & (data["team_id"] <= TEAM_MAX)]
    if ip_outs_col and ip_outs_col in data:
        ip = pd.to_numeric(data[ip_outs_col], errors="coerce") / 3.0
    else:
        ip = pd.to_numeric(data[ip_col], errors="coerce") if ip_col else np.nan
    out = pd.DataFrame()
    out["catcher_id"] = data["catcher_id"]
    out["team_id"] = data["team_id"]
    out["IP_with_c"] = ip
    out["ER_with_c"] = pd.to_numeric(data[er_col], errors="coerce") if er_col else np.nan
    out = out.groupby(["team_id", "catcher_id"], as_index=False).sum(numeric_only=True)
    return out


def load_gamelogs(base: Path, override: Optional[Path]) -> pd.DataFrame:
    df = read_first(base, override, GAMELOG_CANDIDATES)
    if df is None:
        return pd.DataFrame()
    team_col = pick_column(df, "team_id", "teamid")
    game_col = pick_column(df, "game_id", "game_key")
    ip_col = pick_column(df, "ip", "IP")
    ip_outs_col = pick_column(df, "ip_outs", "outs")
    er_col = pick_column(df, "er", "ER")
    if not team_col or not game_col or (not ip_col and not ip_outs_col):
        return pd.DataFrame()
    data = df.copy()
    data["team_id"] = pd.to_numeric(data[team_col], errors="coerce").astype("Int64")
    data = data.dropna(subset=["team_id"])
    data = data[(data["team_id"] >= TEAM_MIN) & (data["team_id"] <= TEAM_MAX)]
    data["game_id"] = data[game_col]
    if ip_outs_col and ip_outs_col in data:
        data["IP"] = pd.to_numeric(data[ip_outs_col], errors="coerce") / 3.0
    else:
        data["IP"] = pd.to_numeric(data[ip_col], errors="coerce")
    data["ER"] = pd.to_numeric(data[er_col], errors="coerce") if er_col else np.nan
    return data[["team_id", "game_id", "IP", "ER"]]


def load_lineups(base: Path, override: Optional[Path]) -> pd.DataFrame:
    df = read_first(base, override, LINEUP_CANDIDATES)
    if df is None:
        return pd.DataFrame()
    team_col = pick_column(df, "team_id", "teamid")
    game_col = pick_column(df, "game_id", "game_key")
    player_col = pick_column(df, "player_id", "catcher_id")
    pos_col = pick_column(df, "position", "pos")
    if not team_col or not game_col or not player_col or not pos_col:
        return pd.DataFrame()
    data = df.copy()
    data["team_id"] = pd.to_numeric(data[team_col], errors="coerce").astype("Int64")
    data = data.dropna(subset=["team_id"])
    data = data[(data["team_id"] >= TEAM_MIN) & (data["team_id"] <= TEAM_MAX)]
    data["game_id"] = data[game_col]
    data["player_id"] = pd.to_numeric(data[player_col], errors="coerce").astype("Int64")
    data["pos"] = data[pos_col].astype(str).str.strip().str.upper()
    data = data[data["pos"].isin(["2", "C"])]
    starters = data.groupby(["team_id", "game_id"], as_index=False).first()[["team_id", "game_id", "player_id"]]
    return starters.rename(columns={"player_id": "catcher_id"})


def derive_battery_from_logs(gamelogs: pd.DataFrame, lineups: pd.DataFrame) -> pd.DataFrame:
    if gamelogs.empty or lineups.empty:
        return pd.DataFrame(columns=["team_id", "catcher_id", "IP_with_c", "ER_with_c"])
    team_game_totals = gamelogs.groupby(["team_id", "game_id"], as_index=False).agg(
        IP_with_c=("IP", "sum"),
        ER_with_c=("ER", "sum"),
    )
    merged = team_game_totals.merge(lineups, on=["team_id", "game_id"], how="inner")
    agg = merged.groupby(["team_id", "catcher_id"], as_index=False).agg(
        IP_with_c=("IP_with_c", "sum"),
        ER_with_c=("ER_with_c", "sum"),
    )
    return agg


def load_team_c_fielding(base: Path) -> pd.DataFrame:
    path = base / C_FIELDING_TEAM_FILE
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    id_col = pick_column(df, "team_id", "teamid", "teamID", "ID")
    if not id_col:
        return pd.DataFrame()
    df = df.copy()
    df["team_id"] = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")
    df = df[df["team_id"].between(TEAM_MIN, TEAM_MAX)]
    ip_col = pick_column(df, "IP")
    sba_col = pick_column(df, "SBA")
    rto_col = pick_column(df, "RTO")
    pb_col = pick_column(df, "PB")
    data = pd.DataFrame()
    data["team_id"] = df["team_id"]
    data["team_c_ip"] = pd.to_numeric(df[ip_col], errors="coerce") if ip_col else np.nan
    if sba_col:
        data["team_sba"] = pd.to_numeric(df[sba_col], errors="coerce")
    if rto_col:
        data["team_rto"] = pd.to_numeric(df[rto_col], errors="coerce")
    if pb_col:
        data["team_pb"] = pd.to_numeric(df[pb_col], errors="coerce")
    return data


def safe_div(numer: float, denom: float) -> float:
    if pd.isna(numer) or pd.isna(denom) or denom == 0:
        return np.nan
    return numer / denom


def classify_rating(runs_saved: float) -> str:
    if pd.isna(runs_saved):
        return "Unknown"
    if runs_saved >= 10:
        return "Battery Anchor"
    if runs_saved >= 5:
        return "Security Blanket"
    if runs_saved >= 0:
        return "Solid Receiver"
    if runs_saved >= -5:
        return "Work In Progress"
    return "Hazardous Signal"


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
        lines.append("(No catchers met the qualification thresholds.)")
    else:
        for _, row in df.iterrows():
            parts = []
            for _, col_name, width, align_right, fmt in columns:
                value = row.get(col_name, "")
                if isinstance(value, (int, float, np.number)):
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
    lines.append("Key:")
    for line in key_lines:
        lines.append(f"  {line}")
    lines.append("")
    lines.append("Definitions:")
    for line in def_lines:
        lines.append(f"  {line}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="ABL Catcher Battery Value report.")
    parser.add_argument("--base", type=str, default=".", help="Base directory.")
    parser.add_argument("--fielding", type=str, help="Override catcher fielding file.")
    parser.add_argument("--battery", type=str, help="Override pitching-by-catcher file.")
    parser.add_argument("--gamelogs", type=str, help="Override pitching gamelog file.")
    parser.add_argument("--lineups", type=str, help="Override lineup/catcher of record file.")
    parser.add_argument("--teams", type=str, help="Override team info file.")
    parser.add_argument("--roster", type=str, help="Override roster file.")
    parser.add_argument("--out", type=str, default="out/csv_out/z_ABL_Catcher_Battery_Value.csv", help="Output CSV path.")
    parser.add_argument("--min_inn_c", type=float, default=150.0, help="Minimum innings caught for ERA stability.")
    parser.add_argument("--min_sbcs", type=int, default=15, help="Minimum SB+CS for CS% stability.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    base_dir = Path(args.base).resolve()
    fielding = load_fielding(base_dir, resolve_path(base_dir, args.fielding))
    roster = load_roster(base_dir, resolve_path(base_dir, args.roster))
    team_display, conf_map = load_team_info(base_dir, resolve_path(base_dir, args.teams))
    team_c_totals = load_team_c_fielding(base_dir)

    battery = load_battery_splits(base_dir, resolve_path(base_dir, args.battery))
    if battery.empty:
        gamelog_df = load_gamelogs(base_dir, resolve_path(base_dir, args.gamelogs))
        lineup_df = load_lineups(base_dir, resolve_path(base_dir, args.lineups))
        battery = derive_battery_from_logs(gamelog_df, lineup_df)

    df = fielding.rename(columns={"player_id": "catcher_id", "INN": "INN_caught"})
    if not roster.empty:
        df = df.merge(roster.rename(columns={"player_id": "catcher_id", "player_name": "catcher_name"}), on="catcher_id", how="left")
    else:
        df["catcher_name"] = df["catcher_id"].astype(str)
    df["catcher_name"] = df["catcher_name"].fillna(df["catcher_id"].astype(str))

    if not team_c_totals.empty:
        df = df.merge(team_c_totals, on="team_id", how="left")
        share = np.where(
            df.get("team_c_ip").notna() & (df.get("team_c_ip") > 0) & df["INN_caught"].notna(),
            df["INN_caught"] / df["team_c_ip"],
            np.nan,
        )
        for metric, team_metric in [("SB", "team_sba"), ("CS", "team_rto"), ("PB", "team_pb")]:
            if team_metric in df.columns:
                mask = df[metric].isna() & df[team_metric].notna() & pd.notna(share)
                df.loc[mask, metric] = share[mask] * df.loc[mask, team_metric]
        df = df.drop(columns=[col for col in ["team_c_ip", "team_sba", "team_rto", "team_pb"] if col in df.columns])

    if not battery.empty:
        df = df.merge(battery, on=["team_id", "catcher_id"], how="left")
    else:
        df["IP_with_c"] = np.nan
        df["ER_with_c"] = np.nan

    df["team_display"] = df["team_id"].map(team_display)
    df["team_display"] = df.apply(
        lambda r: r["team_display"]
        if pd.notna(r["team_display"])
        else (f"T{int(r['team_id'])}" if pd.notna(r["team_id"]) else ""),
        axis=1,
    )
    df["conf_div"] = df["team_id"].map(conf_map).fillna("")

    df["SB_att"] = df["SB"].fillna(0) + df["CS"].fillna(0)
    df["CS_pct"] = df.apply(lambda r: safe_div(r["CS"], r["SB_att"]), axis=1)
    df["stable_cs"] = np.where(df["SB_att"] >= args.min_sbcs, "Y", "")

    team_totals = df.groupby("team_id", as_index=False)[["IP_with_c", "ER_with_c"]].sum(min_count=1)
    team_totals.rename(columns={"IP_with_c": "team_IP_total", "ER_with_c": "team_ER_total"}, inplace=True)
    df = df.merge(team_totals, on="team_id", how="left")
    df["ERA_with"] = df.apply(lambda r: safe_div(r["ER_with_c"] * 9, r["IP_with_c"]) if r["IP_with_c"] > 0 else np.nan, axis=1)
    df["IP_other"] = df["team_IP_total"] - df["IP_with_c"]
    df["ER_other"] = df["team_ER_total"] - df["ER_with_c"]
    df["ERA_other"] = df.apply(lambda r: safe_div(r["ER_other"] * 9, r["IP_other"]) if r["IP_other"] > 0 else np.nan, axis=1)
    df["ERA_delta"] = df["ERA_other"] - df["ERA_with"]
    df["runs_saved"] = df.apply(lambda r: safe_div(r["ERA_delta"] * r["IP_with_c"], 9), axis=1)
    df["runs_saved_per_150"] = df.apply(
        lambda r: r["runs_saved"] * (150.0 / r["IP_with_c"]) if pd.notna(r["runs_saved"]) and r["IP_with_c"] > 0 else np.nan,
        axis=1,
    )
    df["stable_era"] = np.where(df["IP_with_c"] >= args.min_inn_c, "Y", "")

    lg_cs_pct = df.loc[df["stable_cs"] == "Y", "CS_pct"].mean(skipna=True)
    lg_era_with = df.loc[df["stable_era"] == "Y", "ERA_with"].mean(skipna=True)

    df["CS_plus"] = df["CS_pct"] / lg_cs_pct if lg_cs_pct and not np.isnan(lg_cs_pct) else np.nan
    df["ERA_plus"] = df.apply(
        lambda r: safe_div(r["ERA_other"], r["ERA_with"]) if pd.notna(r["ERA_other"]) and pd.notna(r["ERA_with"]) and r["ERA_with"] != 0 else np.nan,
        axis=1,
    )

    df["rating"] = df["runs_saved"].apply(classify_rating)

    csv_columns = [
        "team_id",
        "team_display",
        "conf_div",
        "catcher_id",
        "catcher_name",
        "INN_caught",
        "SB",
        "CS",
        "SB_att",
        "CS_pct",
        "PB",
        "WP",
        "IP_with_c",
        "ER_with_c",
        "ERA_with",
        "IP_other",
        "ERA_other",
        "ERA_delta",
        "runs_saved",
        "runs_saved_per_150",
        "CS_plus",
        "ERA_plus",
        "stable_cs",
        "stable_era",
        "rating",
    ]

    csv_df = df[csv_columns].copy()
    csv_df["INN_caught"] = csv_df["INN_caught"].round(1)
    for col in ["CS_pct", "IP_with_c", "IP_other", "runs_saved", "runs_saved_per_150"]:
        csv_df[col] = csv_df[col].round(1)
    csv_df["ER_with_c"] = csv_df["ER_with_c"].round(2)
    csv_df["ERA_with"] = csv_df["ERA_with"].round(2)
    csv_df["ERA_other"] = csv_df["ERA_other"].round(2)
    csv_df["ERA_delta"] = csv_df["ERA_delta"].round(2)
    csv_df["CS_plus"] = csv_df["CS_plus"].round(3)
    csv_df["ERA_plus"] = csv_df["ERA_plus"].round(3)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = base_dir / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_df.to_csv(out_path, index=False)

    display_df = df.sort_values(
        by=["runs_saved", "ERA_delta", "CS_pct"],
        ascending=[False, False, False],
        na_position="last",
    ).head(25)
    text_columns = [
        ("Catcher", "catcher_name", 22, False, ""),
        ("Tm", "team_display", 5, False, ""),
        ("Conf", "conf_div", 6, False, ""),
        ("INN", "INN_caught", 6, True, ".1f"),
        ("CS%", "CS_pct", 6, True, ".3f"),
        ("ERA w/", "ERA_with", 7, True, ".2f"),
        ("ERA oth", "ERA_other", 8, True, ".2f"),
        ("dERA", "ERA_delta", 6, True, ".2f"),
        ("Runs", "runs_saved", 6, True, ".1f"),
        ("Rating", "rating", 16, False, ""),
    ]
    cs_str = f"{lg_cs_pct:.3f}" if pd.notna(lg_cs_pct) else "NA"
    era_str = f"{lg_era_with:.2f}" if pd.notna(lg_era_with) else "NA"
    subtitle = (
        "Battery impact leaderboard: quantifies how each catcher influences steals and pitcher ERA when he is behind the plate.\n"
        f"League CS% {cs_str}, Avg ERA w/ catcher {era_str}"
    )
    text_output = text_table(
        display_df,
        text_columns,
        "ABL Catcher Battery Value",
        subtitle,
        [
            "Ratings: Battery Anchor (>=+10 runs), Security Blanket (+5 to +9.9), Solid Receiver (0 to +4.9), Work In Progress (-5 to -0.1), Hazardous Signal (<-5).",
        ],
        [
            "CS% calculated when SB+CS >= min_sbcs; ERA impact when innings caught >= min_inn_c.",
            "Runs saved = (ERA_other - ERA_with) * IP/9; positive favors the catcher.",
            "Plus metrics compare to league averages when stable data exists.",
        ],
    )
    text_dir = base_dir / "out" / "text_out"
    text_dir.mkdir(parents=True, exist_ok=True)
    text_path = text_dir / out_path.name.replace(".csv", ".txt")
    text_path.write_text(stamp_text_block(text_output), encoding="utf-8")
    print(text_output)


if __name__ == "__main__":
    main()
