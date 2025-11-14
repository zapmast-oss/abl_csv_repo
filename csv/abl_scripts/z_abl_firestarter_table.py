"""ABL Firestarter Table: quantify leadoff spark and first-inning punch."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from abl_config import stamp_text_block

TEAM_MIN, TEAM_MAX = 1, 24

SPOT_CANDIDATES = [
    "batting_splits_by_lineup_spot.csv",
    "batting_lineup_spot_splits.csv",
    "team_batting_lineup_spot.csv",
]
INNING_CANDIDATES = [
    "team_splits_by_inning.csv",
    "team_inning_splits.csv",
    "team_runs_by_inning.csv",
]
LINESCORE_CANDIDATES = [
    "game_linescore.csv",
    "linescores.csv",
    "games_linescore.csv",
    "games_score.csv",
]
RECORD_CANDIDATES = [
    "team_record.csv",
    "team_season.csv",
    "team_totals.csv",
    "standings.csv",
]
LOG_CANDIDATES = [
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
TEAM_BATTING_TOTALS = [
    "team_batting_stats.csv",
    "team_batting.csv",
    "batting_team_totals.csv",
]
GAMES_FILE = "games.csv"
ATBAT_FILE = "players_at_bat_batting_stats.csv"
PLAYER_GAME_BAT_FILE = "players_game_batting.csv"


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


def resolve_optional_path(base: Path, value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    cand = Path(value)
    if not cand.is_absolute():
        cand = base / cand
    return cand


def load_team_info(base: Path, override: Optional[Path]) -> Tuple[Dict[int, str], Dict[int, str]]:
    df = read_first(base, override, TEAM_INFO_CANDIDATES)
    if df is None:
        return {}, {}
    team_col = pick_column(df, "team_id", "teamid", "teamID")
    display_col = pick_column(df, "team_display", "team_name", "name", "TeamName")
    sub_col = pick_column(df, "sub_league_id", "subleague_id", "conference_id")
    div_col = pick_column(df, "division_id", "division")
    display_map: Dict[int, str] = {}
    conf_map: Dict[int, str] = {}
    if not team_col:
        return display_map, conf_map
    conf_lookup = {0: "N", 1: "A"}
    div_lookup = {0: "E", 1: "C", 2: "W"}
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
        if display_col and pd.notna(row.get(display_col)):
            display_map[tid] = str(row.get(display_col))
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
        sub = conf_lookup.get(sub_key, str(sub_val)[0].upper())
        div = div_lookup.get(div_key, str(div_val)[0].upper())
        conf_map[tid] = f"{sub}-{div}"
    return display_map, conf_map


def load_games(base: Path) -> pd.DataFrame:
    path = base / GAMES_FILE
    if not path.exists():
        raise FileNotFoundError("games.csv is required to map home/away teams.")
    df = pd.read_csv(path)
    gid_col = pick_column(df, "game_id", "GameID")
    home_col = pick_column(df, "home_team", "home_team_id")
    away_col = pick_column(df, "away_team", "away_team_id")
    date_col = pick_column(df, "date", "game_date")
    if not gid_col or not home_col or not away_col:
        raise ValueError("games.csv must include game_id, home_team, and away_team columns.")
    data = pd.DataFrame()
    data["game_id"] = pd.to_numeric(df[gid_col], errors="coerce").astype("Int64")
    data["home_team"] = pd.to_numeric(df[home_col], errors="coerce").astype("Int64")
    data["away_team"] = pd.to_numeric(df[away_col], errors="coerce").astype("Int64")
    data["game_date"] = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT
    return data.dropna(subset=["game_id"])


def load_team_batting_totals(base: Path) -> Optional[pd.DataFrame]:
    df = read_first(base, None, TEAM_BATTING_TOTALS)
    if df is None:
        return None
    team_col = pick_column(df, "team_id", "teamid", "teamID")
    if not team_col:
        return None
    df = df.copy()
    df["team_id"] = pd.to_numeric(df[team_col], errors="coerce").astype("Int64")
    df = df[(df["team_id"] >= TEAM_MIN) & (df["team_id"] <= TEAM_MAX)]
    split_col = pick_column(df, "split_id", "split")
    if split_col and split_col in df.columns:
        df = df[df[split_col] == df[split_col].min()]
    year_col = pick_column(df, "year", "season")
    if year_col and year_col in df.columns:
        latest = df[year_col].max()
        df = df[df[year_col] == latest]
    agg_cols = {}
    for col_name in ["g", "G"]:
        if col_name in df.columns:
            agg_cols["g"] = col_name
            break
    run_col = pick_column(df, "r", "runs", "runs_scored", "RS")
    if "g" not in agg_cols and not run_col:
        return None
    data = pd.DataFrame()
    data["team_id"] = df["team_id"]
    if "g" in agg_cols:
        data["g_bat"] = pd.to_numeric(df[agg_cols["g"]], errors="coerce")
    else:
        data["g_bat"] = np.nan
    if run_col:
        data["runs_bat"] = pd.to_numeric(df[run_col], errors="coerce")
    else:
        data["runs_bat"] = np.nan
    return data


def load_record_totals(base: Path, override: Optional[Path]) -> Optional[pd.DataFrame]:
    df = read_first(base, override, RECORD_CANDIDATES)
    if df is None:
        return None
    team_col = pick_column(df, "team_id", "teamid", "teamID")
    if not team_col:
        return None
    g_col = pick_column(df, "g", "games")
    w_col = pick_column(df, "w", "wins")
    l_col = pick_column(df, "l", "losses")
    runs_col = pick_column(df, "runs_scored", "rs", "r")
    data = pd.DataFrame()
    data["team_id"] = pd.to_numeric(df[team_col], errors="coerce").astype("Int64")
    data = data[(data["team_id"] >= TEAM_MIN) & (data["team_id"] <= TEAM_MAX)]
    if g_col:
        data["g_rec"] = pd.to_numeric(df.loc[data.index, g_col], errors="coerce")
    elif w_col and l_col:
        wins = pd.to_numeric(df.loc[data.index, w_col], errors="coerce")
        losses = pd.to_numeric(df.loc[data.index, l_col], errors="coerce")
        data["g_rec"] = wins + losses
    else:
        data["g_rec"] = np.nan
    if runs_col:
        data["runs_rec"] = pd.to_numeric(df.loc[data.index, runs_col], errors="coerce")
    else:
        data["runs_rec"] = np.nan
    return data


def load_leadoff_data(base: Path, override: Optional[Path]) -> pd.DataFrame:
    df = read_first(base, override, SPOT_CANDIDATES)
    if df is not None:
        team_col = pick_column(df, "team_id", "teamid", "teamID")
        spot_col = pick_column(df, "spot", "lineup_spot")
        if team_col and spot_col:
            copy = df.copy()
            copy["team_id"] = pd.to_numeric(copy[team_col], errors="coerce").astype("Int64")
            copy = copy[(copy["team_id"] >= TEAM_MIN) & (copy["team_id"] <= TEAM_MAX)]
            copy = copy[copy[spot_col] == 1]
            if not copy.empty:
                pa_col = pick_column(copy, "pa", "PA")
                h_col = pick_column(copy, "h", "H")
                bb_col = pick_column(copy, "bb", "BB")
                hbp_col = pick_column(copy, "hbp", "HBP")
                ab_col = pick_column(copy, "ab", "AB")
                sf_col = pick_column(copy, "sf", "SF")
                obp_col = pick_column(copy, "obp", "OBP")
                group = copy.groupby("team_id")
                result = []
                for tid, grp in group:
                    pa = pd.to_numeric(grp[pa_col], errors="coerce").sum() if pa_col else np.nan
                    if all(col is not None for col in (h_col, bb_col, hbp_col, ab_col)):
                        h = pd.to_numeric(grp[h_col], errors="coerce").sum()
                        bb = pd.to_numeric(grp[bb_col], errors="coerce").sum()
                        hbp = pd.to_numeric(grp[hbp_col], errors="coerce").sum()
                        ab = pd.to_numeric(grp[ab_col], errors="coerce").sum()
                        sf = pd.to_numeric(grp[sf_col], errors="coerce").sum() if sf_col else 0.0
                        denom = ab + bb + hbp + sf
                        leadoff_obp = (h + bb + hbp) / denom if denom > 0 else np.nan
                    elif obp_col and pa_col:
                        weighted = pd.to_numeric(grp[obp_col], errors="coerce") * pd.to_numeric(grp[pa_col], errors="coerce")
                        total_pa = pd.to_numeric(grp[pa_col], errors="coerce").sum()
                        leadoff_obp = weighted.sum() / total_pa if total_pa > 0 else np.nan
                    elif obp_col:
                        leadoff_obp = pd.to_numeric(grp[obp_col], errors="coerce").mean()
                    else:
                        leadoff_obp = np.nan
                    result.append({"team_id": tid, "leadoff_PA": pa, "leadoff_OBP": leadoff_obp})
                if result:
                    return pd.DataFrame(result)
    # fallback to reconstructing from play-by-play at-bats
    return load_leadoff_from_atbats(base)


def load_leadoff_from_atbats(base: Path) -> pd.DataFrame:
    atbat_path = base / ATBAT_FILE
    gamebat_path = base / PLAYER_GAME_BAT_FILE
    if not atbat_path.exists() or not gamebat_path.exists():
        return pd.DataFrame({"team_id": [], "leadoff_PA": [], "leadoff_OBP": []})
    usecols_at = ["player_id", "team_id", "game_id", "spot"]
    try:
        at_df = pd.read_csv(atbat_path, usecols=usecols_at)
    except ValueError:
        at_df = pd.read_csv(atbat_path)
    for col in ["player_id", "team_id", "game_id", "spot"]:
        if col not in at_df.columns:
            return pd.DataFrame({"team_id": [], "leadoff_PA": [], "leadoff_OBP": []})
    at_df = at_df.dropna(subset=["player_id", "team_id", "game_id", "spot"])
    at_df["team_id"] = pd.to_numeric(at_df["team_id"], errors="coerce").astype("Int64")
    at_df["player_id"] = pd.to_numeric(at_df["player_id"], errors="coerce").astype("Int64")
    at_df["game_id"] = pd.to_numeric(at_df["game_id"], errors="coerce").astype("Int64")
    at_df["spot"] = pd.to_numeric(at_df["spot"], errors="coerce")
    at_df = at_df[(at_df["team_id"] >= TEAM_MIN) & (at_df["team_id"] <= TEAM_MAX)]
    at_df = at_df[at_df["spot"] == 1]
    if at_df.empty:
        return pd.DataFrame({"team_id": [], "leadoff_PA": [], "leadoff_OBP": []})
    at_df = at_df.drop_duplicates(subset=["player_id", "team_id", "game_id"])

    usecols_game = ["player_id", "team_id", "game_id", "ab", "h", "bb", "hp", "sf", "pa"]
    try:
        bat_df = pd.read_csv(gamebat_path, usecols=usecols_game)
    except ValueError:
        bat_df = pd.read_csv(gamebat_path)
    required = {"player_id", "team_id", "game_id", "ab", "h", "bb", "pa"}
    if not required.issubset(set(bat_df.columns)):
        return pd.DataFrame({"team_id": [], "leadoff_PA": [], "leadoff_OBP": []})
    bat_df = bat_df.dropna(subset=["player_id", "team_id", "game_id"])
    bat_df["team_id"] = pd.to_numeric(bat_df["team_id"], errors="coerce").astype("Int64")
    bat_df["player_id"] = pd.to_numeric(bat_df["player_id"], errors="coerce").astype("Int64")
    bat_df["game_id"] = pd.to_numeric(bat_df["game_id"], errors="coerce").astype("Int64")
    bat_df = bat_df[(bat_df["team_id"] >= TEAM_MIN) & (bat_df["team_id"] <= TEAM_MAX)]

    merged = at_df.merge(
        bat_df[
            [
                "player_id",
                "team_id",
                "game_id",
                "ab",
                "h",
                "bb",
                "hp",
                "sf",
                "pa",
            ]
        ],
        on=["player_id", "team_id", "game_id"],
        how="left",
    )
    merged = merged.dropna(subset=["pa"])
    for col in ["ab", "h", "bb", "hp", "sf", "pa"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
        else:
            merged[col] = 0.0
    agg = merged.groupby("team_id").agg(
        leadoff_PA=("pa", "sum"),
        leadoff_AB=("ab", "sum"),
        leadoff_H=("h", "sum"),
        leadoff_BB=("bb", "sum"),
        leadoff_HBP=("hp", "sum"),
        leadoff_SF=("sf", "sum"),
    )
    agg = agg.reset_index()
    agg["leadoff_OBP"] = agg.apply(
        lambda row: (row["leadoff_H"] + row["leadoff_BB"] + row["leadoff_HBP"])
        / (row["leadoff_AB"] + row["leadoff_BB"] + row["leadoff_HBP"] + row["leadoff_SF"])
        if (row["leadoff_AB"] + row["leadoff_BB"] + row["leadoff_HBP"] + row["leadoff_SF"]) > 0
        else np.nan,
        axis=1,
    )
    return agg[["team_id", "leadoff_PA", "leadoff_OBP"]]


def load_inning_splits(base: Path, override: Optional[Path]) -> Optional[pd.DataFrame]:
    df = read_first(base, override, INNING_CANDIDATES)
    if df is None:
        return None
    team_col = pick_column(df, "team_id", "teamid", "teamID")
    if not team_col:
        return None
    r1_col = pick_column(df, "r1", "runs_inn1", "runs_1st", "runs_first", "runs_1")
    ops_col = None
    for candidate in ("ops1", "ops_1", "ops_first", "ops_1st", "OPS1", "OPS_1st"):
        col = pick_column(df, candidate)
        if col:
            ops_col = col
            break
    data = pd.DataFrame()
    data["team_id"] = pd.to_numeric(df[team_col], errors="coerce").astype("Int64")
    data = data[(data["team_id"] >= TEAM_MIN) & (data["team_id"] <= TEAM_MAX)]
    if r1_col:
        data["R1"] = pd.to_numeric(df.loc[data.index, r1_col], errors="coerce")
    else:
        data["R1"] = np.nan
    if ops_col:
        data["ops_1st"] = pd.to_numeric(df.loc[data.index, ops_col], errors="coerce")
    else:
        data["ops_1st"] = np.nan
    return data


def _normalize_team_flag(value) -> Optional[int]:
    if pd.isna(value):
        return None
    if isinstance(value, str):
        val = value.strip().lower()
        if val in {"home", "h", "1", "bottom"}:
            return 1
        if val in {"away", "a", "0", "top"}:
            return 0
    try:
        num = int(value)
        if num in (0, 1):
            return num
    except (TypeError, ValueError):
        return None
    return None


def normalize_linescore(df: pd.DataFrame) -> pd.DataFrame:
    gid_col = pick_column(df, "game_id", "GameID")
    if not gid_col:
        raise ValueError("Linescore data requires a game_id column.")
    team_col = pick_column(df, "team", "side", "team_flag")
    inning_col = pick_column(df, "inning", "inn", "inning_num")
    score_col = pick_column(df, "score", "runs")
    if team_col and inning_col and score_col:
        data = df[[gid_col, team_col, inning_col, score_col]].copy()
        data.columns = ["game_id", "team_flag_raw", "inning", "runs"]
        data["game_id"] = pd.to_numeric(data["game_id"], errors="coerce").astype("Int64")
        data["team_flag"] = data["team_flag_raw"].apply(_normalize_team_flag)
        data["inning"] = pd.to_numeric(data["inning"], errors="coerce").astype("Int64")
        data["runs"] = pd.to_numeric(data["runs"], errors="coerce").fillna(0.0)
        data = data.dropna(subset=["team_flag", "inning"])
        data["team_flag"] = data["team_flag"].astype(int)
        return data[["game_id", "team_flag", "inning", "runs"]]
    return _normalize_wide_linescore(df, gid_col)


def _normalize_wide_linescore(df: pd.DataFrame, gid_col: str) -> pd.DataFrame:
    records = []
    home_patterns = [
        re.compile(r"h(\d+)$", re.IGNORECASE),
        re.compile(r"home(\d+)$", re.IGNORECASE),
        re.compile(r"inn(\d+)_home", re.IGNORECASE),
    ]
    away_patterns = [
        re.compile(r"a(\d+)$", re.IGNORECASE),
        re.compile(r"away(\d+)$", re.IGNORECASE),
        re.compile(r"inn(\d+)_away", re.IGNORECASE),
    ]
    for _, row in df.iterrows():
        gid = row.get(gid_col)
        if pd.isna(gid):
            continue
        try:
            gid_int = int(gid)
        except (TypeError, ValueError):
            continue
        for col in df.columns:
            if col == gid_col:
                continue
            value = row.get(col)
            if pd.isna(value):
                continue
            flag = None
            inning = None
            col_lower = col.lower()
            for pattern in away_patterns:
                match = pattern.match(col_lower)
                if match:
                    flag = 0
                    inning = int(match.group(1))
                    break
            if inning is None:
                for pattern in home_patterns:
                    match = pattern.match(col_lower)
                    if match:
                        flag = 1
                        inning = int(match.group(1))
                        break
            if inning is None or flag is None:
                continue
            runs = pd.to_numeric(value, errors="coerce")
            if pd.isna(runs):
                runs = 0.0
            records.append({"game_id": gid_int, "team_flag": flag, "inning": inning, "runs": float(runs)})
    if not records:
        raise ValueError("Unable to parse linescore file for inning data.")
    return pd.DataFrame(records)


def load_linescore(base: Path, override: Optional[Path]) -> pd.DataFrame:
    df = read_first(base, override, LINESCORE_CANDIDATES)
    if df is None:
        raise FileNotFoundError("Unable to locate linescore data.")
    return normalize_linescore(df)


def calc_r1_from_linescore(linescore: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    if linescore.empty:
        raise RuntimeError("Linescore data empty.")
    scores = linescore[linescore["inning"] == 1]
    if scores.empty:
        return pd.DataFrame({"team_id": [], "R1": []})
    grouped = scores.groupby(["game_id", "team_flag"], as_index=False)["runs"].sum()
    home_map = games.set_index("game_id")["home_team"].to_dict()
    away_map = games.set_index("game_id")["away_team"].to_dict()
    records = []
    for _, row in grouped.iterrows():
        gid = int(row["game_id"])
        flag = int(row["team_flag"])
        team_id = home_map.get(gid) if flag == 1 else away_map.get(gid)
        if pd.isna(team_id):
            continue
        team_id = int(team_id)
        if not (TEAM_MIN <= team_id <= TEAM_MAX):
            continue
        records.append({"team_id": team_id, "R1": row["runs"]})
    if not records:
        return pd.DataFrame({"team_id": [], "R1": []})
    return pd.DataFrame(records).groupby("team_id", as_index=False)["R1"].sum()


def combine_games_runs(record_df: Optional[pd.DataFrame], batting_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    base = pd.DataFrame({"team_id": list(range(TEAM_MIN, TEAM_MAX + 1))})
    if record_df is not None:
        base = base.merge(record_df, on="team_id", how="left")
    else:
        base["g_rec"] = np.nan
        base["runs_rec"] = np.nan
    if batting_df is not None:
        base = base.merge(batting_df, on="team_id", how="left")
    else:
        base["g_bat"] = np.nan
        base["runs_bat"] = np.nan
    base["g"] = base.apply(
        lambda row: row["g_rec"]
        if not pd.isna(row["g_rec"])
        else (row["g_bat"] if not pd.isna(row["g_bat"]) else np.nan),
        axis=1,
    )
    base["runs_scored"] = base.apply(
        lambda row: row["runs_rec"]
        if not pd.isna(row["runs_rec"])
        else (row["runs_bat"] if not pd.isna(row["runs_bat"]) else np.nan),
        axis=1,
    )
    return base[["team_id", "g", "runs_scored"]]


def load_first_inning_runs(
    base: Path,
    inning_override: Optional[Path],
    linescore_override: Optional[Path],
    games: pd.DataFrame,
) -> pd.DataFrame:
    inning_df = load_inning_splits(base, inning_override)
    if inning_df is not None and inning_df["R1"].notna().any():
        return inning_df[["team_id", "R1", "ops_1st"]]
    linescore_df = load_linescore(base, linescore_override)
    r1 = calc_r1_from_linescore(linescore_df, games)
    r1["ops_1st"] = np.nan
    return r1


def load_logs_runs(base: Path, override: Optional[Path]) -> Optional[pd.DataFrame]:
    df = read_first(base, override, LOG_CANDIDATES)
    if df is None:
        return None
    team_col = pick_column(df, "team_id", "teamid", "teamID")
    date_col = pick_column(df, "game_date", "date")
    runs_col = pick_column(df, "runs_for", "r", "runs_scored")
    if not team_col or not date_col or not runs_col:
        return None
    data = pd.DataFrame()
    data["team_id"] = pd.to_numeric(df[team_col], errors="coerce").astype("Int64")
    data = data[(data["team_id"] >= TEAM_MIN) & (data["team_id"] <= TEAM_MAX)]
    data["game_date"] = pd.to_datetime(df[date_col], errors="coerce")
    data["runs_for"] = pd.to_numeric(df[runs_col], errors="coerce")
    grouped = data.groupby("team_id").agg(
        g_logs=("game_date", "nunique"),
        runs_logs=("runs_for", "sum"),
    )
    grouped = grouped.reset_index()
    return grouped


def enrich_with_logs(base: pd.DataFrame, logs_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if logs_df is None:
        return base
    merged = base.merge(logs_df, on="team_id", how="left")
    merged["g"] = merged.apply(
        lambda row: row["g"] if not pd.isna(row["g"]) else row["g_logs"],
        axis=1,
    )
    merged["runs_scored"] = merged.apply(
        lambda row: row["runs_scored"] if not pd.isna(row["runs_scored"]) else row["runs_logs"],
        axis=1,
    )
    return merged.drop(columns=["g_logs", "runs_logs"])


def rate_firestarter(index: float) -> str:
    if pd.isna(index):
        return "Unknown"
    if index >= 0.52:
        return "Inferno"
    if index >= 0.44:
        return "Blazing"
    if index >= 0.37:
        return "Charged"
    if index >= 0.32:
        return "Warm"
    return "Cold"


def build_text_report(df: pd.DataFrame) -> str:
    lines = [
        "ABL Firestarter Table",
        "=" * 25,
        "Shows how each clubâ€™s leadoff OBP and first-inning production fuel early offense,",
        "spotlighting teams that seize momentum by scoring before opponents settle in.",
        "",
    ]
    header = f"{'Team':<24} {'CD':<4} {'Rating':<12} {'OBP':>7} {'R1/G':>8} {'R1%':>8} {'OPS1st':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for _, row in df.iterrows():
        team_label = row["team_display"]
        conf = row["conf_div"] or "--"
        obp_txt = f"{row['leadoff_OBP']:.3f}" if not pd.isna(row["leadoff_OBP"]) else "NA "
        r1g_txt = f"{row['R1_per_g']:.3f}" if not pd.isna(row["R1_per_g"]) else "NA "
        r1share_txt = f"{row['R1_share']:.3f}" if not pd.isna(row["R1_share"]) else "NA "
        ops_txt = f"{row['ops_1st']:.3f}" if not pd.isna(row["ops_1st"]) else "NA "
        lines.append(
            f"{team_label:<24} {conf:<4} {row['fire_rating']:<12} {obp_txt:>7} {r1g_txt:>8} {r1share_txt:>8} {ops_txt:>8}"
        )
    lines.append("")
    lines.append("Key:")
    lines.append("  Inferno >=0.52, Blazing 0.44-0.51, Charged 0.37-0.43, Warm 0.32-0.36, Cold <0.32 (spark index).")
    lines.append("")
    lines.append("Definitions:")
    lines.append("  Leadoff OBP = OBP for lineup spot #1 (uses H/BB/HBP/AB/SF aggregates).")
    lines.append("  R1/G = total first-inning runs divided by games played.")
    lines.append("  R1% = share of total runs scored that arrive in the first inning.")
    lines.append("  OPS1st = first-inning OPS when inning split data provides it (NA otherwise).")
    lines.append("  Spark Index = average of leadoff OBP and R1/G (when both exist) used for ratings.")
    return "\n".join(lines)


def print_top_table(df: pd.DataFrame) -> None:
    subset = df[
        [
            "team_display",
            "conf_div",
            "fire_rating",
            "leadoff_OBP",
            "R1_per_g",
            "R1_share",
            "ops_1st",
        ]
    ].head(12)
    display_df = subset.copy()
    display_df = display_df.rename(
        columns={
            "team_display": "Team",
            "conf_div": "ConfDiv",
            "fire_rating": "Rating",
            "leadoff_OBP": "OBP",
            "R1_per_g": "R1/G",
            "R1_share": "R1%",
            "ops_1st": "OPS1st",
        }
    )
    for col in ["OBP", "R1/G", "R1%", "OPS1st"]:
        display_df[col] = display_df[col].map(lambda v: f"{v:.3f}" if not pd.isna(v) else "NA ")
    print(display_df.to_string(index=False))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ABL Firestarter Table report.")
    parser.add_argument("--base", type=str, default=".", help="Base directory for CSV files.")
    parser.add_argument("--spot", type=str, help="Override lineup-spot splits file.")
    parser.add_argument("--inning", type=str, help="Override inning splits file.")
    parser.add_argument("--linescore", type=str, help="Override linescore file.")
    parser.add_argument("--record", type=str, help="Override season record file.")
    parser.add_argument("--logs", type=str, help="Override team logs file.")
    parser.add_argument("--teams", type=str, help="Override team info file.")
    parser.add_argument(
        "--out",
        type=str,
        default="out/csv_out/z_ABL_Firestarter_Table.csv",
        help="Output CSV path (default inside out/csv_out).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    base_dir = Path(args.base).resolve()
    spot_override = resolve_optional_path(base_dir, args.spot)
    inning_override = resolve_optional_path(base_dir, args.inning)
    linescore_override = resolve_optional_path(base_dir, args.linescore)
    record_override = resolve_optional_path(base_dir, args.record)
    logs_override = resolve_optional_path(base_dir, args.logs)
    teams_override = resolve_optional_path(base_dir, args.teams)

    display_map, conf_map = load_team_info(base_dir, teams_override)
    lineup_df = load_leadoff_data(base_dir, spot_override)
    batting_df = load_team_batting_totals(base_dir)
    record_df = load_record_totals(base_dir, record_override)
    games_runs = combine_games_runs(record_df, batting_df)
    logs_df = load_logs_runs(base_dir, logs_override)
    games_runs = enrich_with_logs(games_runs, logs_df)

    games_df = load_games(base_dir)
    first_inning_df = load_first_inning_runs(base_dir, inning_override, linescore_override, games_df)

    df = pd.DataFrame({"team_id": list(range(TEAM_MIN, TEAM_MAX + 1))})
    df = df.merge(games_runs, on="team_id", how="left")
    df = df.merge(lineup_df, on="team_id", how="left")
    df = df.merge(first_inning_df, on="team_id", how="left")

    df["R1_per_g"] = df.apply(
        lambda row: (row["R1"] / row["g"]) if (not pd.isna(row["R1"]) and not pd.isna(row["g"]) and row["g"] > 0) else np.nan,
        axis=1,
    )
    df["R1_share"] = df.apply(
        lambda row: (row["R1"] / row["runs_scored"])
        if (not pd.isna(row["R1"]) and not pd.isna(row["runs_scored"]) and row["runs_scored"] > 0)
        else np.nan,
        axis=1,
    )
    df["spark_index"] = df.apply(
        lambda row: np.nanmean([val for val in (row["leadoff_OBP"], row["R1_per_g"]) if not pd.isna(val)])
        if not (pd.isna(row["leadoff_OBP"]) and pd.isna(row["R1_per_g"]))
        else np.nan,
        axis=1,
    )
    df["fire_rating"] = df["spark_index"].apply(rate_firestarter)
    df["team_display"] = df["team_id"].apply(lambda tid: display_map.get(tid, f"Team {tid}"))
    df["conf_div"] = df["team_id"].apply(lambda tid: conf_map.get(tid, ""))

    df = df.sort_values(
        by=["leadoff_OBP", "R1_per_g"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = base_dir / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    csv_df = df.copy()
    for col in ["leadoff_OBP", "R1_per_g", "R1_share"]:
        csv_df[col] = csv_df[col].round(3)
    csv_columns = [
        "team_id",
        "team_display",
        "conf_div",
        "g",
        "leadoff_PA",
        "leadoff_OBP",
        "runs_scored",
        "R1",
        "R1_per_g",
        "R1_share",
        "ops_1st",
        "fire_rating",
    ]
    csv_df[csv_columns].to_csv(out_path, index=False)

    text_report = build_text_report(df)
    text_filename = out_path.with_suffix(".txt").name
    if out_path.parent.name.lower() in {'csv_out'}:
        text_dir = out_path.parent.parent / "txt_out"
    else:
        text_dir = out_path.parent
    text_dir.mkdir(parents=True, exist_ok=True)
    (text_dir / text_filename).write_text(stamp_text_block(text_report), encoding="utf-8")

    print_top_table(df)


if __name__ == "__main__":
    main()

