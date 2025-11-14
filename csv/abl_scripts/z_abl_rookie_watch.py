"""ABL Rookie Watch: highlight rookie hitters and pitchers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from abl_config import stamp_text_block

TEAM_MIN, TEAM_MAX = 1, 24

ROSTER_CANDIDATES = [
    "players.csv",
    "player_register.csv",
    "rosters.csv",
]
BAT_CANDIDATES = [
    "players_career_batting_stats.csv",
    "player_batting_totals.csv",
    "players_batting.csv",
    "batting_players.csv",
]
FIELDING_CANDIDATES = [
    "players_career_fielding_stats.csv",
    "fielding.csv",
    "players_fielding.csv",
    "fielding_totals.csv",
]
PITCH_CANDIDATES = [
    "players_career_pitching_stats.csv",
    "player_pitching_totals.csv",
    "players_pitching.csv",
    "pitching_players.csv",
]
TEAM_INFO_CANDIDATES = [
    "team_record.csv",
    "team_info.csv",
    "teams.csv",
    "standings.csv",
]
PARK_CANDIDATES = [
    "park_factors.csv",
    "parks.csv",
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
        if not override.exists():
            raise FileNotFoundError(f"Specified file not found: {override}")
        return pd.read_csv(override)
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


def _read_rookie_flags(path: Path, yes_tokens: set[str], no_tokens: set[str]) -> Dict[int, float]:
    try:
        df = pd.read_csv(path, usecols=["ID", "ROOK"])
    except (ValueError, FileNotFoundError):
        return {}
    flags: Dict[int, float] = {}
    for pid_raw, rook_raw in zip(df["ID"], df["ROOK"]):
        try:
            pid = int(pid_raw)
        except (TypeError, ValueError):
            continue
        token = str(rook_raw).strip().upper()
        if not token:
            continue
        if token in yes_tokens:
            flags[pid] = 1.0
        elif token in no_tokens:
            flags[pid] = 0.0
    return flags


def load_external_rookie_map(base: Path) -> Dict[int, float]:
    rook_map: Dict[int, float] = {}
    yes_tokens = {"YES", "Y", "TRUE", "1", "ROOK", "ROOKIE"}
    no_tokens = {"NO", "N", "FALSE", "0"}
    preferred = base / "abl_statistics_player_statistics_-_sortable_stats_player_indicative_2.csv"
    if preferred.exists():
        rook_map.update(_read_rookie_flags(preferred, yes_tokens, no_tokens))
    for path in base.glob("abl_statistics_player_statistics_-_*.csv"):
        if preferred.exists() and path.resolve() == preferred.resolve():
            continue
        rookies = _read_rookie_flags(path, yes_tokens, no_tokens)
        rook_map.update(rookies)
    return rook_map


def load_extra_war_map(base: Path, filename: str) -> Dict[int, float]:
    path = base / filename
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path, usecols=["ID", "WAR"])
    except (ValueError, FileNotFoundError):
        return {}
    war_map: Dict[int, float] = {}
    for pid_raw, war in zip(df["ID"], df["WAR"]):
        try:
            pid = int(pid_raw)
        except (TypeError, ValueError):
            continue
        try:
            war_val = float(war)
        except (TypeError, ValueError):
            continue
        if pd.notna(war_val):
            war_map[pid] = war_val
    return war_map


def resolve_text_path(csv_path: Path) -> Path:
    text_name = csv_path.with_suffix(".txt").name
    parent = csv_path.parent
    if parent.name.lower() in {'csv_out'}:
        text_dir = parent.parent / "txt_out"
    else:
        text_dir = parent
    text_dir.mkdir(parents=True, exist_ok=True)
    return text_dir / text_name


def load_teams(base: Path, override: Optional[Path]) -> Tuple[Dict[int, str], Dict[int, str], Dict[int, str], float]:
    df = read_first(base, override, TEAM_INFO_CANDIDATES)
    if df is None:
        return {}, {}, {}, np.nan
    team_col = pick_column(df, "team_id", "teamid", "TeamID")
    name_col = pick_column(df, "team_display", "team_name", "name", "TeamName")
    abbr_col = pick_column(df, "abbr", "abbreviation")
    conf_col = pick_column(df, "sub_league_id", "conference_id", "subleague_id")
    div_col = pick_column(df, "division_id", "division")
    w_col = pick_column(df, "wins", "w", "Wins")
    l_col = pick_column(df, "losses", "l", "Losses")
    g_col = pick_column(df, "g", "games")
    display_map: Dict[int, str] = {}
    abbr_map: Dict[int, str] = {}
    conf_map: Dict[int, str] = {}
    games = []
    conf_lookup = {0: "N", 1: "A"}
    div_lookup = {0: "E", 1: "C", 2: "W"}
    if not team_col:
        return display_map, abbr_map, conf_map, np.nan
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
            display_map[tid] = str(row[name_col])
        if abbr_col and pd.notna(row.get(abbr_col)):
            abbr_map[tid] = str(row[abbr_col]).upper()
        if tid not in conf_map and conf_col and div_col:
            sub_val = row.get(conf_col)
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
        if g_col and pd.notna(row.get(g_col)):
            try:
                games.append(float(row[g_col]))
            except (TypeError, ValueError):
                pass
        elif w_col and l_col and pd.notna(row.get(w_col)) and pd.notna(row.get(l_col)):
            try:
                games.append(float(row[w_col]) + float(row[l_col]))
            except (TypeError, ValueError):
                continue
    lg_games = float(np.mean(games)) if games else np.nan
    return display_map, abbr_map, conf_map, lg_games


def enrich_team_maps_from_teams_file(
    base: Path,
    display_map: Dict[int, str],
    abbr_map: Dict[int, str],
    conf_map: Dict[int, str],
) -> None:
    teams_path = base / "teams.csv"
    if not teams_path.exists():
        return
    df = pd.read_csv(teams_path)
    team_col = pick_column(df, "team_id", "teamid", "teamID")
    name_col = pick_column(df, "team_display", "team_name", "name", "nickname")
    abbr_col = pick_column(df, "abbr", "abbreviation")
    sub_col = pick_column(df, "sub_league_id", "subleague_id")
    div_col = pick_column(df, "division_id", "division")
    division_map = {0: "E", 1: "C", 2: "W"}
    conf_lookup = {0: "N", 1: "A"}
    if not team_col:
        return
    for _, row in df.iterrows():
        tid_val = row.get(team_col)
        if pd.isna(tid_val):
            continue
        try:
            tid = int(tid_val)
        except (TypeError, ValueError):
            continue
        if name_col and tid not in display_map and pd.notna(row.get(name_col)):
            display_map[tid] = str(row[name_col])
        if abbr_col and tid not in abbr_map and pd.notna(row.get(abbr_col)):
            abbr_map[tid] = str(row[abbr_col]).upper()
        if tid not in conf_map and sub_col and div_col:
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
                conf_map[tid] = f"{conf_lookup.get(sub_key, str(sub_val)[0].upper())}-{division_map.get(div_key, str(div_val)[0].upper())}"


def load_parks(base: Path, override: Optional[Path]) -> Dict[int, float]:
    df = read_first(base, override, PARK_CANDIDATES)
    if df is None:
        return {}
    team_col = pick_column(df, "team_id", "teamid", "TeamID")
    pf_col = pick_column(df, "run_factor", "pf_runs", "pf", "ParkFactor")
    park_map: Dict[int, float] = {}
    if team_col and pf_col:
        for _, row in df.iterrows():
            tid = row.get(team_col)
            if pd.isna(tid):
                continue
            try:
                tid = int(tid)
            except (TypeError, ValueError):
                continue
            factor = pd.to_numeric(row.get(pf_col), errors="coerce")
            if pd.isna(factor):
                continue
            park_map[tid] = float(factor)
    return park_map


def load_roster(base: Path, override: Optional[Path]) -> pd.DataFrame:
    df = read_first(base, override, ROSTER_CANDIDATES)
    if df is None:
        raise FileNotFoundError("Unable to locate roster/player master file.")
    id_col = pick_column(df, "player_id", "playerid", "PlayerID")
    team_col = pick_column(df, "team_id", "teamid", "TeamID")
    first_col = pick_column(df, "first_name", "firstname")
    last_col = pick_column(df, "last_name", "lastname")
    name_col = pick_column(df, "name_full", "name", "player_name")
    age_col = pick_column(df, "age", "Age")
    bats_col = pick_column(df, "bats", "bat_hand", "Bats")
    throws_col = pick_column(df, "throws", "throw_hand", "Throws")
    pos_col = pick_column(df, "pos", "position")
    rookie_cols = [pick_column(df, "rook", "rookie", "rookie_flag", "is_rookie")]
    service_years_col = pick_column(df, "service_time_years", "ml_service_years")
    service_days_col = pick_column(df, "service_days", "service_time_days")
    acquired_col = pick_column(df, "acquired")
    acquired_date_col = pick_column(df, "acquired_date")
    exp_col = pick_column(df, "experience")
    if not id_col or not team_col:
        raise ValueError("Roster file missing player_id/team_id columns.")
    roster = pd.DataFrame()
    roster["player_id"] = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")
    roster["team_id"] = pd.to_numeric(df[team_col], errors="coerce").astype("Int64")
    roster = roster.dropna(subset=["player_id", "team_id"])
    roster = roster[(roster["team_id"] >= TEAM_MIN) & (roster["team_id"] <= TEAM_MAX)]
    if first_col and last_col:
        names = (
            df[first_col].fillna("").astype(str).str.strip()
            + " "
            + df[last_col].fillna("").astype(str).str.strip()
        ).str.strip()
        roster["player_name"] = names.loc[roster.index]
    elif name_col:
        roster["player_name"] = df[name_col].fillna("").astype(str)
    else:
        roster["player_name"] = roster["player_id"].astype(str)
    roster["age"] = pd.to_numeric(df[age_col], errors="coerce") if age_col else np.nan
    roster["bats"] = df[bats_col].fillna("").astype(str) if bats_col else ""
    roster["throws"] = df[throws_col].fillna("").astype(str) if throws_col else ""
    roster["pos"] = df[pos_col].fillna("").astype(str).str.upper() if pos_col else ""
    if rookie_cols[0]:
        rookie_raw = df[rookie_cols[0]]
        if rookie_raw.dtype == object or str(rookie_raw.dtype).startswith(("O", "U")):
            cleaned = rookie_raw.astype(str).str.strip().str.upper()
            yes_values = {"YES", "Y", "TRUE", "1", "ROOK", "ROOKIE"}
            no_values = {"NO", "N", "FALSE", "0"}
            rookie_flag = pd.Series(np.nan, index=cleaned.index, dtype="float64")
            rookie_flag[cleaned.isin(yes_values)] = 1.0
            rookie_flag[cleaned.isin(no_values)] = 0.0
            if rookie_flag.notna().any():
                roster["rookie_flag"] = rookie_flag
            else:
                roster["rookie_flag"] = pd.to_numeric(rookie_raw, errors="coerce")
        else:
            roster["rookie_flag"] = pd.to_numeric(rookie_raw, errors="coerce")
    else:
        roster["rookie_flag"] = np.nan
    roster["service_years"] = pd.to_numeric(df[service_years_col], errors="coerce") if service_years_col else np.nan
    roster["service_days"] = pd.to_numeric(df[service_days_col], errors="coerce") if service_days_col else np.nan
    roster["acquired"] = df[acquired_col].fillna("").astype(str) if acquired_col else ""
    roster["acquired_date"] = (
        pd.to_datetime(df[acquired_date_col], errors="coerce") if acquired_date_col else pd.NaT
    )
    roster["experience"] = pd.to_numeric(df[exp_col], errors="coerce") if exp_col else np.nan

    ext_rookie_map = load_external_rookie_map(base)
    if ext_rookie_map:
        ext_series = roster["player_id"].map(ext_rookie_map)
        roster["rookie_flag"] = ext_series.fillna(0.0)

    return roster


def is_rookie(row: pd.Series, player_type: str, season_start: pd.Timestamp) -> bool:
    if pd.notna(row.get("rookie_flag")):
        return row["rookie_flag"] == 1
    if pd.notna(row.get("service_years")):
        return row["service_years"] <= 1
    days = row.get("service_days")
    if pd.notna(days) and days <= 45:
        if player_type == "hit":
            ab = row.get("career_ab", row.get("AB"))
            if pd.notna(ab) and ab < 130:
                return True
        else:
            ip = row.get("career_ip", row.get("IP"))
            if pd.notna(ip) and ip < 50:
                return True
    acquired_date = row.get("acquired_date")
    acquired_text = str(row.get("acquired", "")).lower()
    experience = row.get("experience")
    recent_acq = pd.notna(acquired_date) and pd.notna(season_start) and acquired_date >= season_start
    if recent_acq:
        if any(keyword in acquired_text for keyword in ["draft", "signed", "international", "rule 5"]):
            return True
        if pd.notna(experience) and experience <= 1:
            return True
        if player_type == "hit":
            ab = row.get("career_ab")
            if pd.notna(ab) and ab < 130:
                return True
        else:
            ip = row.get("career_ip")
            if pd.notna(ip) and ip < 50:
                return True
    return False


def load_batting(base: Path, override: Optional[Path]) -> pd.DataFrame:
    df = read_first(base, override, BAT_CANDIDATES)
    if df is None:
        raise FileNotFoundError("Unable to locate batting totals.")
    id_col = pick_column(df, "player_id", "playerid", "PlayerID")
    team_col = pick_column(df, "team_id", "teamid", "TeamID")
    pa_col = pick_column(df, "pa", "PA")
    ab_col = pick_column(df, "ab", "AB")
    h_col = pick_column(df, "h", "H")
    bb_col = pick_column(df, "bb", "BB")
    hbp_col = pick_column(df, "hbp", "HBP")
    sf_col = pick_column(df, "sf", "SF")
    double_col = pick_column(df, "doubles", "2b", "d")
    triple_col = pick_column(df, "triples", "3b", "t")
    hr_col = pick_column(df, "hr", "HR")
    tb_col = pick_column(df, "tb", "TB")
    war_col = pick_column(df, "WAR", "war", "oWAR", "BatWAR")
    obp_col = pick_column(df, "obp", "OBP")
    slg_col = pick_column(df, "slg", "SLG")
    ops_col = pick_column(df, "ops", "OPS")
    split_col = pick_column(df, "split_id", "split")
    year_col = pick_column(df, "year", "season")
    if not id_col or not team_col or not pa_col or not ab_col or not h_col:
        raise ValueError("Batting totals missing essential columns.")
    df = df.copy()
    if split_col:
        df = df[df[split_col] == df[split_col].min()]
    if year_col:
        df = df[df[year_col] == df[year_col].max()]
    df["player_id"] = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")
    df["team_id"] = pd.to_numeric(df[team_col], errors="coerce").astype("Int64")
    df = df.dropna(subset=["player_id", "team_id"])
    df = df[(df["team_id"] >= TEAM_MIN) & (df["team_id"] <= TEAM_MAX)]
    stat_cols = {
        "PA": pa_col,
        "AB": ab_col,
        "H": h_col,
        "BB": bb_col,
        "HBP": hbp_col,
        "SF": sf_col,
        "Doubles": double_col,
        "Triples": triple_col,
        "HR": hr_col,
        "TB_raw": tb_col,
    }
    for key, col in stat_cols.items():
        if col:
            df[key] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[key] = 0.0
    df["WAR"] = pd.to_numeric(df[war_col], errors="coerce") if war_col else np.nan
    grouped = (
        df.groupby(["player_id", "team_id"], as_index=False)[
            ["PA", "AB", "H", "BB", "HBP", "SF", "Doubles", "Triples", "HR", "TB_raw", "WAR"]
        ].sum()
    )
    extra_war = load_extra_war_map(base, "abl_statistics_player_statistics_-_sortable_stats_player_bat_stats.csv")
    if extra_war:
        grouped["WAR"] = grouped["player_id"].map(extra_war).combine_first(grouped["WAR"])
    grouped["singles"] = grouped["H"] - grouped["Doubles"] - grouped["Triples"] - grouped["HR"]
    grouped["singles"] = grouped["singles"].clip(lower=0.0)
    grouped["TB"] = grouped["TB_raw"]
    missing_tb = grouped["TB"] == 0
    grouped.loc[missing_tb, "TB"] = (
        grouped.loc[missing_tb, "singles"]
        + 2 * grouped.loc[missing_tb, "Doubles"]
        + 3 * grouped.loc[missing_tb, "Triples"]
        + 4 * grouped.loc[missing_tb, "HR"]
    )
    if ops_col and ops_col in df.columns:
        ops_series = (
            df.groupby(["player_id", "team_id"], as_index=False)[ops_col]
            .mean()
            .rename(columns={ops_col: "OPS_direct"})
        )
        grouped = grouped.merge(ops_series, on=["player_id", "team_id"], how="left")
    else:
        grouped["OPS_direct"] = np.nan
    if obp_col and obp_col in df.columns:
        obp_series = (
            df.groupby(["player_id", "team_id"], as_index=False)[obp_col]
            .mean()
            .rename(columns={obp_col: "OBP_direct"})
        )
        grouped = grouped.merge(obp_series, on=["player_id", "team_id"], how="left")
    else:
        grouped["OBP_direct"] = np.nan
    if slg_col and slg_col in df.columns:
        slg_series = (
            df.groupby(["player_id", "team_id"], as_index=False)[slg_col]
            .mean()
            .rename(columns={slg_col: "SLG_direct"})
        )
        grouped = grouped.merge(slg_series, on=["player_id", "team_id"], how="left")
    else:
        grouped["SLG_direct"] = np.nan
    grouped["OBP_calc"] = grouped.apply(
        lambda r: (r["H"] + r["BB"] + r["HBP"]) / (r["AB"] + r["BB"] + r["HBP"] + r["SF"])
        if (r["AB"] + r["BB"] + r["HBP"] + r["SF"]) > 0
        else np.nan,
        axis=1,
    )
    grouped["SLG_calc"] = grouped.apply(lambda r: r["TB"] / r["AB"] if r["AB"] > 0 else np.nan, axis=1)
    grouped["OBP"] = grouped["OBP_direct"].combine_first(grouped["OBP_calc"])
    grouped["SLG"] = grouped["SLG_direct"].combine_first(grouped["SLG_calc"])
    grouped["OPS"] = grouped["OPS_direct"].combine_first(grouped["OBP"] + grouped["SLG"])
    grouped["career_ab"] = grouped["AB"]
    return grouped[
        [
            "player_id",
            "team_id",
            "PA",
            "HR",
            "OBP",
            "SLG",
            "OPS",
            "WAR",
            "career_ab",
        ]
    ]


def load_fielding(base: Path, override: Optional[Path]) -> pd.DataFrame:
    df = read_first(base, override, FIELDING_CANDIDATES)
    if df is None:
        return pd.DataFrame(columns=["player_id", "def_runs"])
    id_col = pick_column(df, "player_id", "playerid", "PlayerID")
    team_col = pick_column(df, "team_id", "teamid", "TeamID")
    pos_col = pick_column(df, "pos", "position")
    year_col = pick_column(df, "year", "season")
    split_col = pick_column(df, "split_id", "split")
    drs_col = pick_column(df, "drs", "DRS", "uzr", "UZR", "defense")
    zr_col = pick_column(df, "zr", "zone_rating", "ZR")
    if not id_col:
        return pd.DataFrame(columns=["player_id", "def_runs"])
    df = df.copy()
    if split_col:
        df = df[df[split_col] == df[split_col].min()]
    if year_col:
        df = df[df[year_col] == df[year_col].max()]
    data = pd.DataFrame()
    data["player_id"] = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")
    data["team_id"] = pd.to_numeric(df[team_col], errors="coerce").astype("Int64") if team_col else np.nan
    data["pos"] = df[pos_col].fillna("").astype(str).str.upper() if pos_col else ""
    if drs_col:
        data["def_val"] = pd.to_numeric(df[drs_col], errors="coerce")
    elif zr_col:
        data["def_val"] = pd.to_numeric(df[zr_col], errors="coerce")
    else:
        return pd.DataFrame(columns=["player_id", "def_runs"])
    data = data.dropna(subset=["player_id"])
    data = data[(data["team_id"].between(TEAM_MIN, TEAM_MAX, inclusive="both")) | data["team_id"].isna()]
    data = data[~data["pos"].isin(["P", "SP", "RP", "DH"])]
    grouped = data.groupby("player_id", as_index=False)["def_val"].sum()
    grouped = grouped.rename(columns={"def_val": "def_runs"})
    return grouped


def load_pitching(base: Path, override: Optional[Path]) -> pd.DataFrame:
    df = read_first(base, override, PITCH_CANDIDATES)
    if df is None:
        raise FileNotFoundError("Unable to locate pitching totals.")
    id_col = pick_column(df, "player_id", "playerid", "PlayerID")
    team_col = pick_column(df, "team_id", "teamid", "TeamID")
    ip_col = pick_column(df, "ip", "IP")
    ip_outs_col = pick_column(df, "ip_outs", "IPouts")
    er_col = pick_column(df, "er", "ER")
    so_col = pick_column(df, "so", "SO", "k", "K")
    bb_col = pick_column(df, "bb", "BB")
    hr_col = pick_column(df, "hr", "HR")
    bf_col = pick_column(df, "bf", "BF", "batters_faced")
    era_col = pick_column(df, "era", "ERA")
    war_col = pick_column(df, "WAR", "war", "pWAR")
    ab_col = pick_column(df, "ab", "AB")
    hbp_col = pick_column(df, "hp", "HBP")
    sf_col = pick_column(df, "sf", "SF")
    split_col = pick_column(df, "split_id", "split")
    year_col = pick_column(df, "year", "season")
    if not id_col or not team_col or not (ip_col or ip_outs_col):
        raise ValueError("Pitching totals missing essential columns.")
    df = df.copy()
    if split_col:
        df = df[df[split_col] == df[split_col].min()]
    if year_col:
        df = df[df[year_col] == df[year_col].max()]
    df["player_id"] = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")
    df["team_id"] = pd.to_numeric(df[team_col], errors="coerce").astype("Int64")
    df = df.dropna(subset=["player_id", "team_id"])
    df = df[(df["team_id"] >= TEAM_MIN) & (df["team_id"] <= TEAM_MAX)]
    if ip_col:
        df["IP_raw"] = pd.to_numeric(df[ip_col], errors="coerce").fillna(0.0)
    else:
        df["IP_raw"] = pd.to_numeric(df[ip_outs_col], errors="coerce").fillna(0.0) / 3.0
    df["ER_raw"] = pd.to_numeric(df[er_col], errors="coerce").fillna(0.0) if er_col else 0.0
    df["SO_raw"] = pd.to_numeric(df[so_col], errors="coerce").fillna(0.0) if so_col else 0.0
    df["BB_raw"] = pd.to_numeric(df[bb_col], errors="coerce").fillna(0.0) if bb_col else 0.0
    df["HR_raw"] = pd.to_numeric(df[hr_col], errors="coerce").fillna(0.0) if hr_col else 0.0
    df["BF_raw"] = pd.to_numeric(df[bf_col], errors="coerce").fillna(0.0) if bf_col else 0.0
    df["AB_raw"] = pd.to_numeric(df[ab_col], errors="coerce").fillna(0.0) if ab_col else 0.0
    df["HBP_raw"] = pd.to_numeric(df[hbp_col], errors="coerce").fillna(0.0) if hbp_col else 0.0
    df["SF_raw"] = pd.to_numeric(df[sf_col], errors="coerce").fillna(0.0) if sf_col else 0.0
    df["ERA_direct"] = pd.to_numeric(df[era_col], errors="coerce") if era_col else np.nan
    df["WAR"] = pd.to_numeric(df[war_col], errors="coerce") if war_col else np.nan
    grouped = (
        df.groupby(["player_id", "team_id"], as_index=False)[
            [
                "IP_raw",
                "ER_raw",
                "SO_raw",
                "BB_raw",
                "HR_raw",
                "BF_raw",
                "AB_raw",
                "HBP_raw",
                "SF_raw",
                "ERA_direct",
                "WAR",
            ]
        ].sum()
    )
    extra_war = load_extra_war_map(base, "abl_statistics_player_statistics_-_sortable_stats_player_pitch_stats_2.csv")
    if extra_war:
        grouped["WAR"] = grouped["player_id"].map(extra_war).combine_first(grouped["WAR"])
    grouped = grouped.rename(
        columns={
            "IP_raw": "IP",
            "ER_raw": "ER",
            "SO_raw": "SO",
            "BB_raw": "BB",
            "HR_raw": "HR",
            "BF_raw": "BF",
            "AB_raw": "AB",
            "HBP_raw": "HBP",
            "SF_raw": "SF",
        }
    )
    grouped["career_ip"] = grouped["IP"]
    grouped["ERA"] = grouped["ERA_direct"]
    return grouped


def load_anchor_date(base: Path) -> pd.Timestamp:
    games_path = base / "games.csv"
    if not games_path.exists():
        return pd.NaT
    games = pd.read_csv(games_path)
    date_col = pick_column(games, "game_date", "date")
    type_col = pick_column(games, "game_type", "type")
    played_col = pick_column(games, "played")
    if not date_col:
        return pd.NaT
    games["game_date"] = pd.to_datetime(games[date_col], errors="coerce")
    if type_col:
        games = games[(games[type_col].fillna(0) == 0)]
    if played_col:
        games = games[games[played_col] == 1]
    return games["game_date"].max()


def compute_league_ops(batting: pd.DataFrame) -> float:
    subset = batting.dropna(subset=["OPS", "PA"])
    if subset.empty:
        return np.nan
    return float((subset["OPS"] * subset["PA"]).sum() / subset["PA"].sum())


def compute_age_cohorts(batting: pd.DataFrame, roster: pd.DataFrame) -> Dict[int, float]:
    merged = batting.merge(roster[["player_id", "age"]], on="player_id", how="left")
    merged = merged.dropna(subset=["OPS", "PA", "age"])
    if merged.empty:
        return {}
    merged["age_int"] = merged["age"].astype(int)
    cohorts = {}
    for age, group in merged.groupby("age_int"):
        pa_sum = group["PA"].sum()
        if pa_sum > 0:
            cohorts[age] = float((group["OPS"] * group["PA"]).sum() / pa_sum)
    return cohorts


def compute_def_runs(fielding: pd.DataFrame) -> Dict[int, float]:
    if fielding.empty:
        return {}
    return dict(zip(fielding["player_id"], fielding["def_runs"]))


def compute_fip_constant(pitching: pd.DataFrame) -> Tuple[float, float, float, float, float]:
    subset = pitching.dropna(subset=["IP", "ER", "HR", "BB", "SO"])
    subset = subset[subset["IP"] > 0]
    if subset.empty:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    lg_ip = subset["IP"].sum()
    lg_er = subset["ER"].sum()
    lg_hr = subset["HR"].sum()
    lg_bb = subset["BB"].sum()
    lg_so = subset["SO"].sum()
    lg_era = (lg_er * 9.0) / lg_ip if lg_ip > 0 else np.nan
    fip_const = lg_era - ((13 * lg_hr + 3 * lg_bb - 2 * lg_so) / lg_ip)
    return float(fip_const), lg_ip, lg_hr, lg_bb, lg_so


def calc_k_pct(row: pd.Series) -> float:
    if pd.notna(row.get("BF")) and row["BF"] > 0:
        return row["SO"] / row["BF"]
    ab = row.get("AB", 0) or 0
    bb = row.get("BB", 0) or 0
    hbp = row.get("HBP", 0) or 0
    sf = row.get("SF", 0) or 0
    alt = ab + bb + hbp + sf
    if alt > 0:
        return row["SO"] / alt
    ip = row.get("IP")
    if pd.notna(ip) and ip > 0:
        return row["SO"] / (ip * 3.0)
    return np.nan


def calc_bb_pct(row: pd.Series) -> float:
    if pd.notna(row.get("BF")) and row["BF"] > 0:
        return row["BB"] / row["BF"]
    ab = row.get("AB", 0) or 0
    bb = row.get("BB", 0) or 0
    hbp = row.get("HBP", 0) or 0
    sf = row.get("SF", 0) or 0
    alt = ab + bb + hbp + sf
    if alt > 0:
        return row["BB"] / alt
    ip = row.get("IP")
    if pd.notna(ip) and ip > 0:
        return row["BB"] / (ip * 3.0)
    return np.nan


def rate_hitter(value: float) -> str:
    if pd.isna(value):
        return "Unknown"
    if value >= 150:
        return "Meteoric"
    if value >= 120:
        return "Impact"
    if value >= 100:
        return "Steady"
    if value >= 80:
        return "Learning"
    return "Finding Feet"


def rate_pitcher(pace: float) -> str:
    if pd.isna(pace):
        return "Unknown"
    if pace >= 4.0:
        return "Ace Track"
    if pace >= 2.5:
        return "Rotation Ready"
    if pace >= 1.0:
        return "Contributor"
    if pace >= 0.0:
        return "Apprentice"
    return "Rebuild"


def build_text_table(
    df: pd.DataFrame,
    columns: Sequence[Tuple[str, str, int, bool]],
    title: str,
    description_lines: Optional[Sequence[str]],
    thresholds: str,
    key_lines: Sequence[str],
    definition_lines: Sequence[str],
    format_map: Optional[Dict[str, str]] = None,
) -> str:
    lines = [title, "=" * len(title), ""]
    if description_lines:
        for line in description_lines:
            lines.append(line)
        lines.append("")
    header = " ".join(
        f"{label:<{width}}" if not align_right else f"{label:>{width}}"
        for label, _, width, align_right in columns
    )
    lines.append(header)
    lines.append("-" * len(header))
    if df.empty:
        lines.append("(No rookies met the qualification thresholds.)")
    for _, row in df.iterrows():
        parts = []
        for _, col_name, width, align_right in columns:
            value = row.get(col_name, "")
            fmt_str = format_map.get(col_name) if format_map else None
            if isinstance(value, (int, float)) and pd.isna(value):
                display = "NA"
            elif fmt_str and isinstance(value, (int, float)):
                try:
                    display = fmt_str.format(value)
                except Exception:
                    display = str(value)
            else:
                display = str(value)
            fmt = f"{{:>{width}}}" if align_right else f"{{:<{width}}}"
            parts.append(fmt.format(display[:width]))
        lines.append(" ".join(parts))
    lines.append("")
    lines.append(thresholds)
    lines.append("")
    lines.append("Key:")
    for line in key_lines:
        lines.append(f"  {line}")
    lines.append("")
    lines.append("Definitions:")
    for line in definition_lines:
        lines.append(f"  {line}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="ABL Rookie Watch report.")
    parser.add_argument("--base", type=str, default=".", help="Base directory.")
    parser.add_argument("--roster", type=str, help="Override roster file.")
    parser.add_argument("--batting", type=str, help="Override batting totals.")
    parser.add_argument("--fielding", type=str, help="Override fielding totals.")
    parser.add_argument("--pitching", type=str, help="Override pitching totals.")
    parser.add_argument("--teams", type=str, help="Override team info file.")
    parser.add_argument("--parks", type=str, help="Override park factors file.")
    parser.add_argument(
        "--out_hit",
        type=str,
        default="out/csv_out/z_ABL_Rookie_Watch_Hitters.csv",
        help="Output CSV for hitters.",
    )
    parser.add_argument(
        "--out_pit",
        type=str,
        default="out/csv_out/z_ABL_Rookie_Watch_Pitchers.csv",
        help="Output CSV for pitchers.",
    )
    parser.add_argument("--min_pa", type=int, default=50, help="Minimum PA for hitter ranking.")
    parser.add_argument("--min_ip", type=int, default=15, help="Minimum IP for pitcher ranking.")
    parser.add_argument("--show_all", action="store_true", help="Include non-qualifiers.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    base_dir = Path(args.base).resolve()
    roster_df = load_roster(base_dir, resolve_path(base_dir, args.roster))
    batting_df = load_batting(base_dir, resolve_path(base_dir, args.batting))
    fielding_df = load_fielding(base_dir, resolve_path(base_dir, args.fielding))
    pitching_df = load_pitching(base_dir, resolve_path(base_dir, args.pitching))
    team_map, abbr_map, conf_map, lg_team_games = load_teams(base_dir, resolve_path(base_dir, args.teams))
    enrich_team_maps_from_teams_file(base_dir, team_map, abbr_map, conf_map)
    park_map = load_parks(base_dir, resolve_path(base_dir, args.parks))
    anchor_date = load_anchor_date(base_dir)
    if pd.isna(anchor_date):
        season_start = pd.Timestamp(year=pd.Timestamp.today().year, month=1, day=1)
    else:
        season_start = pd.Timestamp(year=anchor_date.year, month=1, day=1)

    def_runs_map = compute_def_runs(fielding_df)
    lg_ops = compute_league_ops(batting_df)
    age_cohorts = compute_age_cohorts(batting_df, roster_df)

    hitters = batting_df.merge(roster_df, on=["player_id", "team_id"], how="left", suffixes=("", "_roster"))
    hitters = hitters[hitters["PA"].fillna(0) > 0]
    hitters["def_runs"] = hitters["player_id"].map(def_runs_map).fillna(0.0)
    hitters["park_factor"] = hitters["team_id"].map(park_map).fillna(100.0)
    hitters["OPS_adj"] = hitters["OPS"] / (hitters["park_factor"] / 100.0)
    hitters["OPS_plus"] = 100 * (hitters["OPS_adj"] / lg_ops) if pd.notna(lg_ops) else np.nan
    hitters["age_int"] = hitters["age"].astype("Float64")
    def compute_age_adj(row: pd.Series) -> float:
        if pd.isna(row["OPS_adj"]):
            return np.nan
        age_val = row.get("age_int")
        if pd.notna(age_val):
            cohort_ops = age_cohorts.get(int(age_val))
            if cohort_ops:
                return 100 * (row["OPS_adj"] / cohort_ops)
        return row["OPS_plus"]

    hitters["AgeAdj_OPS_plus"] = hitters.apply(compute_age_adj, axis=1)
    hitters["is_rookie"] = hitters.apply(lambda r: is_rookie(r, "hit", season_start), axis=1)
    hitters["team_display"] = hitters["team_id"].map(team_map).fillna("")
    hitters["conf_div"] = hitters["team_id"].map(conf_map).fillna("")
    hitters["team_abbr"] = hitters["team_id"].map(abbr_map).fillna("")
    if pd.notna(lg_team_games):
        hitters["WAR_pace_162"] = hitters["WAR"] * (162 / lg_team_games)
    else:
        hitters["WAR_pace_162"] = np.nan
    hitters["rating"] = hitters["AgeAdj_OPS_plus"].apply(rate_hitter)
    hitters = hitters[hitters["is_rookie"]]
    hitters = hitters if args.show_all else hitters[hitters["PA"] >= args.min_pa]
    hitters = hitters.sort_values(
        by=["AgeAdj_OPS_plus", "OPS_plus", "PA"],
        ascending=[False, False, False],
        na_position="last",
    )

    fip_const, _, _, _, _ = compute_fip_constant(pitching_df)
    pitchers = pitching_df.merge(roster_df, on=["player_id", "team_id"], how="left", suffixes=("", "_roster"))
    pitchers = pitchers[pitchers["IP"].fillna(0) > 0]
    pitchers["is_rookie"] = pitchers.apply(lambda r: is_rookie(r, "pit", season_start), axis=1)
    pitchers["team_display"] = pitchers["team_id"].map(team_map).fillna("")
    pitchers["conf_div"] = pitchers["team_id"].map(conf_map).fillna("")
    pitchers["team_abbr"] = pitchers["team_id"].map(abbr_map).fillna("")
    pitchers["ERA_calc"] = (pitchers["ER"] * 9.0) / pitchers["IP"]
    pitchers["ERA_final"] = pitchers["ERA"].combine_first(pitchers["ERA_calc"])
    if pd.notna(fip_const):
        pitchers["FIP"] = ((13 * pitchers["HR"] + 3 * pitchers["BB"] - 2 * pitchers["SO"]) / pitchers["IP"]) + fip_const
    else:
        pitchers["FIP"] = np.nan
    pitchers["K_pct"] = pitchers.apply(calc_k_pct, axis=1)
    pitchers["BB_pct"] = pitchers.apply(calc_bb_pct, axis=1)
    if pd.notna(lg_team_games):
        pitchers["WAR_pace_162"] = pitchers["WAR"] * (162 / lg_team_games)
    else:
        pitchers["WAR_pace_162"] = np.nan
    pitchers["rating"] = pitchers["WAR_pace_162"].apply(rate_pitcher)
    pitchers = pitchers[pitchers["is_rookie"]]
    pitchers = pitchers if args.show_all else pitchers[pitchers["IP"] >= args.min_ip]
    pitchers = pitchers.sort_values(
        by=["WAR_pace_162", "IP", "ERA_final"],
        ascending=[False, False, True],
        na_position="last",
    )

    out_hit = resolve_path(base_dir, args.out_hit) or Path(args.out_hit)
    if not out_hit.is_absolute():
        out_hit = base_dir / out_hit
    out_hit.parent.mkdir(parents=True, exist_ok=True)
    hit_csv = hitters[
        [
            "team_id",
            "team_display",
            "player_id",
            "player_name",
            "age",
            "PA",
            "HR",
            "OPS",
            "OPS_plus",
            "AgeAdj_OPS_plus",
            "def_runs",
            "WAR",
            "WAR_pace_162",
            "conf_div",
            "team_abbr",
            "rating",
        ]
    ].copy()
    for col in ["OPS", "OPS_plus", "AgeAdj_OPS_plus", "def_runs", "WAR", "WAR_pace_162"]:
        hit_csv[col] = pd.to_numeric(hit_csv[col], errors="coerce")
    hit_csv["OPS"] = hit_csv["OPS"].round(3)
    hit_csv["OPS_plus"] = hit_csv["OPS_plus"].round(0)
    hit_csv["AgeAdj_OPS_plus"] = hit_csv["AgeAdj_OPS_plus"].round(0)
    hit_csv["def_runs"] = hit_csv["def_runs"].round(1)
    hit_csv["WAR"] = hit_csv["WAR"].round(1)
    hit_csv["WAR_pace_162"] = hit_csv["WAR_pace_162"].round(1)
    def format_team(row: pd.Series) -> str:
        base_name = row["team_display"] or row["team_abbr"] or f"T{int(row['team_id'])}"
        if row["conf_div"]:
            return f"{base_name} ({row['conf_div']})"
        return base_name

    hit_csv["team_label"] = hit_csv.apply(format_team, axis=1)
    csv_cols_hit = [
        "team_id",
        "team_display",
        "player_id",
        "player_name",
        "age",
        "PA",
        "HR",
        "OPS",
        "OPS_plus",
        "AgeAdj_OPS_plus",
        "def_runs",
        "WAR",
        "WAR_pace_162",
    ]
    hit_csv[csv_cols_hit].to_csv(out_hit, index=False)

    out_pit = resolve_path(base_dir, args.out_pit) or Path(args.out_pit)
    if not out_pit.is_absolute():
        out_pit = base_dir / out_pit
    out_pit.parent.mkdir(parents=True, exist_ok=True)
    pit_csv = pitchers[
        [
            "team_id",
            "team_display",
            "player_id",
            "player_name",
            "age",
            "IP",
            "ERA_final",
            "FIP",
            "K_pct",
            "BB_pct",
            "WAR",
            "WAR_pace_162",
            "conf_div",
            "team_abbr",
            "rating",
        ]
    ].copy()
    pit_csv = pit_csv.rename(columns={"ERA_final": "ERA"})
    for col in ["IP", "ERA", "FIP", "K_pct", "BB_pct", "WAR", "WAR_pace_162"]:
        pit_csv[col] = pd.to_numeric(pit_csv[col], errors="coerce")
    pit_csv["IP"] = pit_csv["IP"].round(1)
    pit_csv["ERA"] = pit_csv["ERA"].round(2)
    pit_csv["FIP"] = pit_csv["FIP"].round(2)
    pit_csv["K_pct"] = pit_csv["K_pct"].round(3)
    pit_csv["BB_pct"] = pit_csv["BB_pct"].round(3)
    pit_csv["WAR"] = pit_csv["WAR"].round(1)
    pit_csv["WAR_pace_162"] = pit_csv["WAR_pace_162"].round(1)
    pit_csv["team_label"] = pit_csv.apply(format_team, axis=1)
    csv_cols_pit = [
        "team_id",
        "team_display",
        "player_id",
        "player_name",
        "age",
        "IP",
        "ERA",
        "FIP",
        "K_pct",
        "BB_pct",
        "WAR",
        "WAR_pace_162",
    ]
    pit_csv[csv_cols_pit].to_csv(out_pit, index=False)

    hit_text_cols = [
        ("Player", "player_name", 24, False),
        ("Team", "team_label", 20, False),
        ("Rating", "rating", 12, False),
        ("OPS", "OPS", 7, True),
        ("OPS+", "OPS_plus", 6, True),
        ("Age+", "AgeAdj_OPS_plus", 6, True),
        ("Def", "def_runs", 6, True),
        ("WAR162", "WAR_pace_162", 7, True),
    ]
    hit_text = build_text_table(
        hit_csv.head(25),
        hit_text_cols,
        "ABL Rookie Watch (Hitters)",
        [
            "Spotlights rookie-eligible bats clearing the PA floor and ranks them by age-adjusted OPS pace.",
            "Great for quickly seeing which first-year hitters are already tilting lineups versus those still ramping up.",
        ],
        f"Threshold: PA >= {args.min_pa} (unless show_all).",
        [
            "Meteoric >=150 AgeAdj OPS+, Impact 120-149, Steady 100-119, Learning 80-99, Finding Feet <80.",
        ],
        [
            "OPS+ numbers adjust for park and league average (100 = league average).",
            "AgeAdj OPS+ compares to same-age peers when available.",
            "WAR pace scaled to 162 games using league-average games played.",
        ],
        format_map={
            "OPS": "{:.3f}",
            "OPS_plus": "{:.0f}",
            "AgeAdj_OPS_plus": "{:.0f}",
            "def_runs": "{:.1f}",
            "WAR_pace_162": "{:.1f}",
        },
    )
    hit_txt_path = resolve_text_path(out_hit)
    hit_txt_path.write_text(stamp_text_block(hit_text), encoding="utf-8")

    pit_text_cols = [
        ("Player", "player_name", 24, False),
        ("Team", "team_label", 20, False),
        ("Rating", "rating", 14, False),
        ("IP", "IP", 6, True),
        ("ERA", "ERA", 6, True),
        ("FIP", "FIP", 6, True),
        ("K%", "K_pct", 7, True),
        ("BB%", "BB_pct", 7, True),
        ("WAR162", "WAR_pace_162", 7, True),
    ]
    pit_text = build_text_table(
        pit_csv.head(25),
        pit_text_cols,
        "ABL Rookie Watch (Pitchers)",
        [
            "Highlights rookie arms meeting the IP floor, layering WAR pace with run-prevention and strikeout/command signals.",
            "Helps flag which first-year pitchers are rotation-ready contributors versus developmental depth.",
        ],
        f"Threshold: IP >= {args.min_ip} (unless show_all).",
        [
            "Ace Track pace >=4.0 WAR, Rotation Ready 2.5-3.9, Contributor 1.0-2.4, Apprentice 0-0.9, Rebuild <0.",
        ],
        [
            "ERA uses provided figure or ER-based calc; FIP uses league constant when inputs exist.",
            "K%/BB% prefer BF when available, otherwise use plate-appearance approximations.",
            "WAR pace scaled to 162 games using league-average schedule length.",
        ],
        format_map={
            "IP": "{:.1f}",
            "ERA": "{:.2f}",
            "FIP": "{:.2f}",
            "K_pct": "{:.3f}",
            "BB_pct": "{:.3f}",
            "WAR_pace_162": "{:.1f}",
        },
    )
    pit_txt_path = resolve_text_path(out_pit)
    pit_txt_path.write_text(stamp_text_block(pit_text), encoding="utf-8")

    print(hit_text)
    print("")
    print(pit_text)


if __name__ == "__main__":
    main()

