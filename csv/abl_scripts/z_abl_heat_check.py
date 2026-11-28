"""ABL Heat Check: spotlight 7-day OPS heaters with prior-week contrast."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from abl_config import stamp_text_block

TEAM_MIN, TEAM_MAX = 1, 24

GAMELOG_CANDIDATES = [
    "players_game_batting.csv",
    "players_batting_gamelog.csv",
    "batting_gamelogs.csv",
]
GAMES_FILE = "games.csv"
TOTALS_CANDIDATES = [
    "players_batting.csv",
    "player_batting_totals.csv",
    "batting_players.csv",
]
ROSTER_CANDIDATES = [
    "players.csv",
    "rosters.csv",
]
TEAM_INFO_CANDIDATES = [
    "team_info.csv",
    "teams.csv",
    "standings.csv",
    "team_record.csv",
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


def resolve_optional_path(base: Path, value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path


def load_team_info(base: Path, override: Optional[Path]) -> Tuple[Dict[int, str], Dict[int, str], Dict[int, str]]:
    df = read_first(base, override, TEAM_INFO_CANDIDATES)
    if df is None:
        return {}, {}, {}
    team_col = pick_column(df, "team_id", "teamid", "TeamID")
    name_col = pick_column(df, "team_display", "team_name", "name", "TeamName")
    abbr_col = pick_column(df, "abbr", "abbreviation")
    sub_col = pick_column(df, "sub_league_id", "subleague_id", "conference_id")
    div_col = pick_column(df, "division_id", "division")
    names: Dict[int, str] = {}
    abbrs: Dict[int, str] = {}
    confs: Dict[int, str] = {}
    conf_lookup = {0: "N", 1: "A"}
    div_lookup = {0: "E", 1: "C", 2: "W"}
    if not team_col:
        return names, confs
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
        if abbr_col and pd.notna(row.get(abbr_col)):
            abbrs[tid] = str(row.get(abbr_col)).upper()
        if tid in confs or not sub_col or not div_col:
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
        confs[tid] = f"{conf_lookup.get(sub_key, str(sub_val)[0].upper())}-{div_lookup.get(div_key, str(div_val)[0].upper())}"
    return names, abbrs, confs


def load_roster_info(base: Path, override: Optional[Path]) -> Tuple[Dict[int, str], Dict[int, str]]:
    df = read_first(base, override, ROSTER_CANDIDATES)
    if df is None:
        return {}, {}
    id_col = pick_column(df, "player_id", "playerid", "PlayerID")
    first_col = pick_column(df, "first_name", "firstname")
    last_col = pick_column(df, "last_name", "lastname")
    name_col = pick_column(df, "name_full", "name", "player_name")
    pos_col = pick_column(df, "pos", "position")
    names: Dict[int, str] = {}
    positions: Dict[int, str] = {}
    if not id_col:
        return names, positions
    df["player_id"] = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")
    for _, row in df.dropna(subset=["player_id"]).iterrows():
        pid = int(row["player_id"])
        if first_col and last_col and pd.notna(row.get(first_col)) and pd.notna(row.get(last_col)):
            names[pid] = f"{row[first_col]} {row[last_col]}".strip()
        elif name_col and pd.notna(row.get(name_col)):
            names[pid] = str(row[name_col]).strip()
        if pos_col and pd.notna(row.get(pos_col)):
            positions[pid] = str(row[pos_col]).strip().upper()
    return names, positions


def load_totals(base: Path, override: Optional[Path]) -> pd.DataFrame:
    df = read_first(base, override, TOTALS_CANDIDATES)
    if df is None:
        return pd.DataFrame(columns=["player_id", "team_id", "season_PA", "season_OPS"])
    id_col = pick_column(df, "player_id", "playerid", "PlayerID")
    team_col = pick_column(df, "team_id", "teamid", "TeamID")
    pa_col = pick_column(df, "pa", "PA")
    obp_col = pick_column(df, "obp", "OBP")
    slg_col = pick_column(df, "slg", "SLG")
    ops_col = pick_column(df, "ops", "OPS")
    if not id_col or not team_col or not pa_col:
        return pd.DataFrame(columns=["player_id", "team_id", "season_PA", "season_OPS"])
    totals = df.copy()
    totals["player_id"] = pd.to_numeric(totals[id_col], errors="coerce").astype("Int64")
    totals["team_id"] = pd.to_numeric(totals[team_col], errors="coerce").astype("Int64")
    totals["season_PA"] = pd.to_numeric(totals[pa_col], errors="coerce")
    totals["season_OPS"] = pd.to_numeric(totals[ops_col], errors="coerce") if ops_col else np.nan
    if totals["season_OPS"].isna().all() and obp_col and slg_col:
        totals["season_OPS"] = pd.to_numeric(totals[obp_col], errors="coerce") + pd.to_numeric(
            totals[slg_col], errors="coerce"
        )
    return totals[["player_id", "team_id", "season_PA", "season_OPS"]]


def load_games(base: Path) -> pd.DataFrame:
    path = base / GAMES_FILE
    if not path.exists():
        raise FileNotFoundError("games.csv is required to derive game dates.")
    df = pd.read_csv(path)
    gid_col = pick_column(df, "game_id", "GameID")
    date_col = pick_column(df, "date", "game_date", "GameDate")
    type_col = pick_column(df, "game_type", "type")
    played_col = pick_column(df, "played")
    games = pd.DataFrame()
    games["game_id"] = pd.to_numeric(df[gid_col], errors="coerce").astype("Int64")
    games["game_date"] = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT
    if type_col:
        games["game_type"] = pd.to_numeric(df[type_col], errors="coerce")
    else:
        games["game_type"] = 0
    if played_col:
        games["played"] = pd.to_numeric(df[played_col], errors="coerce").fillna(0)
    else:
        games["played"] = 1
    games = games.dropna(subset=["game_id"])
    regular_mask = games["game_type"].fillna(0) == 0
    games = games[regular_mask & (games["played"] == 1)]
    return games[["game_id", "game_date"]]


def load_gamelogs(base: Path, override: Optional[Path]) -> pd.DataFrame:
    df = read_first(base, override, GAMELOG_CANDIDATES)
    if df is None:
        raise FileNotFoundError("Unable to locate batting game logs.")
    id_col = pick_column(df, "player_id", "playerid", "PlayerID")
    team_col = pick_column(df, "team_id", "teamid", "TeamID")
    date_col = pick_column(df, "game_date", "date", "GameDate")
    game_id_col = pick_column(df, "game_id", "GameID")
    pa_col = pick_column(df, "pa", "PA")
    ab_col = pick_column(df, "ab", "AB")
    h_col = pick_column(df, "h", "H")
    bb_col = pick_column(df, "bb", "BB")
    hbp_col = pick_column(df, "hbp", "HBP")
    sf_col = pick_column(df, "sf", "SF")
    tb_col = pick_column(df, "tb", "TB", "pitches_seen")
    double_col = pick_column(df, "2b", "d", "doubles")
    triple_col = pick_column(df, "3b", "t", "triples")
    hr_col = pick_column(df, "hr", "HR")
    if not all([id_col, team_col, game_id_col, pa_col, ab_col, h_col]):
        raise ValueError("Game logs missing required columns.")
    games = load_games(base)
    logs = df.copy()
    logs["player_id"] = pd.to_numeric(logs[id_col], errors="coerce").astype("Int64")
    logs["team_id"] = pd.to_numeric(logs[team_col], errors="coerce").astype("Int64")
    logs["game_id"] = pd.to_numeric(logs[game_id_col], errors="coerce").astype("Int64") if game_id_col else pd.NA
    logs = logs.merge(games, on="game_id", how="left")
    logs = logs.dropna(subset=["game_date"])
    logs["game_date"] = logs["game_date"].fillna(pd.Timestamp("1970-01-01"))
    logs = logs[(logs["team_id"] >= TEAM_MIN) & (logs["team_id"] <= TEAM_MAX)]
    logs["PA"] = pd.to_numeric(logs[pa_col], errors="coerce").fillna(0.0)
    logs["AB"] = pd.to_numeric(logs[ab_col], errors="coerce").fillna(0.0)
    logs["H"] = pd.to_numeric(logs[h_col], errors="coerce").fillna(0.0)
    logs["BB"] = pd.to_numeric(logs[bb_col], errors="coerce").fillna(0.0)
    logs["HBP"] = pd.to_numeric(logs[hbp_col], errors="coerce").fillna(0.0) if hbp_col else 0.0
    logs["SF"] = pd.to_numeric(logs[sf_col], errors="coerce").fillna(0.0) if sf_col else 0.0
    logs["TB"] = pd.to_numeric(logs[tb_col], errors="coerce") if tb_col else np.nan
    logs["Doubles"] = pd.to_numeric(logs[double_col], errors="coerce").fillna(0.0) if double_col else 0.0
    logs["Triples"] = pd.to_numeric(logs[triple_col], errors="coerce").fillna(0.0) if triple_col else 0.0
    logs["HR"] = pd.to_numeric(logs[hr_col], errors="coerce").fillna(0.0) if hr_col else 0.0
    return logs[
        [
            "player_id",
            "team_id",
            "game_date",
            "PA",
            "AB",
            "H",
            "BB",
            "HBP",
            "SF",
            "TB",
            "Doubles",
            "Triples",
            "HR",
        ]
    ]


def compute_tb(row: pd.Series) -> float:
    if not pd.isna(row["TB"]):
        return row["TB"]
    singles = row["H"] - row["Doubles"] - row["Triples"] - row["HR"]
    singles = max(singles, 0.0)
    return singles + 2 * row["Doubles"] + 3 * row["Triples"] + 4 * row["HR"]


def agg_window(logs: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    window = logs[(logs["game_date"] >= start) & (logs["game_date"] <= end)].copy()
    if window.empty:
        return pd.DataFrame()
    window["TB_calc"] = window.apply(compute_tb, axis=1)
    grouped = window.groupby(["player_id", "team_id"], as_index=False).agg(
        PA=("PA", "sum"),
        AB=("AB", "sum"),
        H=("H", "sum"),
        BB=("BB", "sum"),
        HBP=("HBP", "sum"),
        SF=("SF", "sum"),
        TB=("TB_calc", "sum"),
    )
    grouped["OBP"] = grouped.apply(
        lambda r: (r["H"] + r["BB"] + r["HBP"]) / (r["AB"] + r["BB"] + r["HBP"] + r["SF"])
        if (r["AB"] + r["BB"] + r["HBP"] + r["SF"]) > 0
        else np.nan,
        axis=1,
    )
    grouped["SLG"] = grouped.apply(lambda r: r["TB"] / r["AB"] if r["AB"] > 0 else np.nan, axis=1)
    grouped["OPS"] = grouped["OBP"] + grouped["SLG"]
    grouped = grouped.rename(columns={"PA": "window_PA", "OBP": "window_OBP", "SLG": "window_SLG", "OPS": "window_OPS"})
    return grouped[["player_id", "team_id", "window_PA", "window_OBP", "window_SLG", "window_OPS"]]


def rate_delta(delta: float) -> str:
    if pd.isna(delta):
        return "Unknown"
    if delta >= 0.300:
        return "Inferno"
    if delta >= 0.180:
        return "Scorching"
    if delta >= 0.060:
        return "Hot"
    if delta >= -0.060:
        return "Steady"
    if delta >= -0.180:
        return "Cooling"
    return "Ice Cold"


def build_text_report(df: pd.DataFrame, min_pa_last7: int, min_pa_prior7: int) -> str:
    lines = [
        "ABL Heat Check",
        "=" * 15,
        "Tracks the hottest 7-day OPS runs versus the previous week to spotlight true lineup infernos.",
        "Useful for identifying short-term matchup boosts or warning signs of regression before it hits the box score.",
        "",
    ]
    header = f"{'Player':<28} {'Tm':<4} {'CD':<4} {'Rating':<11} {'OPS7':>7} {'OPSpr7':>8} {'dOPS':>7} {'PA7':>5}"
    lines.append(header)
    lines.append("-" * len(header))
    if df.empty:
        lines.append("(No hitters met the last-7 PA threshold.)")
    for _, row in df.iterrows():
        player = row["player_name"]
        team = row["team_abbr"] or "--"
        conf = row["conf_div"] or "--"
        last_ops = f"{row['last7_OPS']:.3f}" if not pd.isna(row["last7_OPS"]) else "NA "
        prior_ops = f"{row['prior7_OPS']:.3f}" if not pd.isna(row["prior7_OPS"]) else "NA "
        delta = f"{row['delta_OPS']:.3f}" if not pd.isna(row["delta_OPS"]) else "NA "
        lines.append(
            f"{player:<28} {team:<4} {conf:<4} {row['heat_rating']:<11} {last_ops:>7} {prior_ops:>8} {delta:>7} {int(row['last7_PA']):>5}"
        )
    lines.append("")
    lines.append(f"Thresholds: last7 PA >= {min_pa_last7}; prior7 PA >= {min_pa_prior7} to compare.")
    lines.append("")
    lines.append("Key:")
    lines.append("  Inferno dOPS >= +0.300, Scorching 0.180-0.299, Hot 0.060-0.179, Steady +/-0.059,")
    lines.append("  Cooling -0.179 to -0.060, Ice Cold <= -0.180.")
    lines.append("")
    lines.append("Definitions:")
    lines.append("  last7 = most recent 7-day OPS window; prior7 = the 7-day span immediately before that.")
    lines.append("  dOPS = last7 OPS minus prior7 OPS (blank when prior7 PA below threshold).")
    lines.append("  Season OPS/PA provide macro context for each hitter's year-long baseline.")
    return "\n".join(lines)


def print_top_table(df: pd.DataFrame) -> None:
    subset = df[
        [
            "player_name",
            "team_abbr",
            "conf_div",
            "heat_rating",
            "last7_OPS",
            "prior7_OPS",
            "delta_OPS",
            "last7_PA",
        ]
    ].head(25)
    display_df = subset.copy()
    display_df = display_df.rename(
        columns={
            "player_name": "Player",
            "team_abbr": "Team",
            "conf_div": "ConfDiv",
            "heat_rating": "Rating",
            "last7_OPS": "OPS7",
            "prior7_OPS": "OPSpr7",
            "delta_OPS": "dOPS",
            "last7_PA": "PA7",
        }
    )
    for col in ["OPS7", "OPSpr7", "dOPS"]:
        display_df[col] = display_df[col].map(lambda v: f"{v:.3f}" if not pd.isna(v) else "NA ")
    print(display_df.to_string(index=False))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ABL Heat Check report.")
    parser.add_argument("--base", type=str, default=".", help="Base directory for CSV files.")
    parser.add_argument("--gamelogs", type=str, help="Override batting game logs file.")
    parser.add_argument("--totals", type=str, help="Override season totals file.")
    parser.add_argument("--roster", type=str, help="Override roster/positions file.")
    parser.add_argument("--teams", type=str, help="Override team info file.")
    parser.add_argument("--min_pa_last7", type=int, default=20, help="Minimum PA in last7 window.")
    parser.add_argument("--min_pa_prior7", type=int, default=10, help="Minimum PA in prior7 window.")
    parser.add_argument(
        "--out",
        type=str,
        default="out/csv_out/z_ABL_Heat_Check.csv",
        help="Output CSV path (default inside out/csv_out).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    base_dir = Path(args.base).resolve()
    gamelog_override = resolve_optional_path(base_dir, args.gamelogs)
    totals_override = resolve_optional_path(base_dir, args.totals)
    roster_override = resolve_optional_path(base_dir, args.roster)
    teams_override = resolve_optional_path(base_dir, args.teams)

    logs = load_gamelogs(base_dir, gamelog_override)
    if logs.empty:
        raise RuntimeError("No game logs available.")
    anchor_date = logs["game_date"].max()
    last7_start = anchor_date - pd.Timedelta(days=6)
    prior7_start = anchor_date - pd.Timedelta(days=13)
    prior7_end = anchor_date - pd.Timedelta(days=7)

    last7 = agg_window(logs, last7_start, anchor_date)
    prior7 = agg_window(logs, prior7_start, prior7_end)

    names_map, pos_map = load_roster_info(base_dir, roster_override)
    totals = load_totals(base_dir, totals_override)
    team_map, abbr_map, conf_map = load_team_info(base_dir, teams_override)

    merged = last7.merge(prior7, on=["player_id", "team_id"], how="left", suffixes=("_last7", "_prior7"))
    merged = merged.merge(totals, on=["player_id", "team_id"], how="left")
    merged["player_name"] = merged["player_id"].apply(lambda pid: names_map.get(int(pid), f"Player {int(pid)}"))
    merged["team_display"] = merged["team_id"].apply(lambda tid: team_map.get(tid, ""))
    merged["team_abbr"] = merged["team_id"].apply(lambda tid: abbr_map.get(tid, ""))
    merged["conf_div"] = merged["team_id"].apply(lambda tid: conf_map.get(tid, ""))

    merged["last7_PA"] = merged["window_PA_last7"]
    merged["last7_OBP"] = merged["window_OBP_last7"]
    merged["last7_SLG"] = merged["window_SLG_last7"]
    merged["last7_OPS"] = merged["window_OPS_last7"]
    merged["prior7_PA"] = merged["window_PA_prior7"]
    merged["prior7_OPS"] = merged["window_OPS_prior7"]
    merged["delta_OPS"] = merged["last7_OPS"] - merged["prior7_OPS"]
    merged.loc[merged["prior7_PA"] < args.min_pa_prior7, ["prior7_OPS", "delta_OPS"]] = np.nan

    merged["heat_rating"] = merged["delta_OPS"].apply(rate_delta)

    merged = merged[merged["last7_PA"] >= args.min_pa_last7]
    merged = merged.sort_values(
        by=["delta_OPS", "last7_OPS", "last7_PA"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = base_dir / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    csv_df = merged.copy()
    for col in ["last7_OBP", "last7_SLG", "last7_OPS", "prior7_OPS", "delta_OPS", "season_OPS"]:
        csv_df[col] = pd.to_numeric(csv_df[col], errors="coerce").round(3)
    csv_columns = [
        "team_id",
        "team_display",
        "conf_div",
        "player_id",
        "player_name",
        "last7_PA",
        "last7_OBP",
        "last7_SLG",
        "last7_OPS",
        "prior7_PA",
        "prior7_OPS",
        "delta_OPS",
        "season_PA",
        "season_OPS",
        "heat_rating",
    ]
    csv_df[csv_columns].to_csv(out_path, index=False)

    text_report = build_text_report(merged.head(25), args.min_pa_last7, args.min_pa_prior7)
    text_filename = out_path.with_suffix(".txt").name
    if out_path.parent.name.lower() in {'csv_out'}:
        text_dir = out_path.parent.parent / "text_out"
    else:
        text_dir = out_path.parent
    text_dir.mkdir(parents=True, exist_ok=True)
    (text_dir / text_filename).write_text(stamp_text_block(text_report), encoding="utf-8")

    if merged.empty:
        print("No hitters met the last-7 PA threshold.")
    else:
        print_top_table(merged)


if __name__ == "__main__":
    main()

