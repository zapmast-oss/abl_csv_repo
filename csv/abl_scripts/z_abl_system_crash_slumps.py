"""ABL System Crash Slumps report."""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

TEAM_MIN, TEAM_MAX = 1, 24

LOG_CANDIDATES = [
    "players_batting_gamelog.csv",
    "batting_gamelog_players.csv",
    "player_game_batting.csv",
    "players_game_batting.csv",
]
TOTAL_CANDIDATES = [
    "players_batting.csv",
    "player_batting_totals.csv",
    "batting_players.csv",
]
TEAM_INFO_CANDIDATES = [
    "team_info.csv",
    "teams.csv",
    "standings.csv",
    "team_record.csv",
]


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


def load_game_dates(base: Path) -> pd.Series:
    games_path = base / "games.csv"
    if not games_path.exists():
        raise FileNotFoundError("games.csv required to map game ids to dates.")
    games_df = pd.read_csv(games_path, usecols=["game_id", "date"])
    games_df["game_date"] = pd.to_datetime(games_df["date"], errors="coerce")
    return games_df.set_index("game_id")["game_date"]


def load_teams(base: Path, override: Optional[Path]) -> Tuple[Dict[int, str], Dict[int, str]]:
    df = read_first(base, override, TEAM_INFO_CANDIDATES)
    if df is None:
        return {}, {}
    team_col = pick_column(df, "team_id", "teamid", "TeamID")
    name_col = pick_column(df, "team_display", "team_name", "name", "abbr")
    sub_col = pick_column(df, "sub_league_id", "conference_id")
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


def load_totals(base: Path, override: Optional[Path], logs_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    df = read_first(base, override, TOTAL_CANDIDATES)
    if df is None:
        return aggregate_totals_from_logs(logs_df)
    team_col = pick_column(df, "team_id", "teamid")
    player_col = pick_column(df, "player_id", "playerid")
    pa_col = pick_column(df, "pa", "PA")
    obp_col = pick_column(df, "obp")
    slg_col = pick_column(df, "slg")
    ops_col = pick_column(df, "ops")
    hr_col = pick_column(df, "hr")
    ab_col = pick_column(df, "ab")
    h_col = pick_column(df, "h")
    bb_col = pick_column(df, "bb")
    hbp_col = pick_column(df, "hbp", "hp")
    sf_col = pick_column(df, "sf")
    tb_col = pick_column(df, "tb")
    if not team_col or not player_col or not pa_col:
        return aggregate_totals_from_logs(logs_df)
    data = df.copy()
    data["team_id"] = pd.to_numeric(data[team_col], errors="coerce").astype("Int64")
    data["player_id"] = pd.to_numeric(data[player_col], errors="coerce").astype("Int64")
    data = data.dropna(subset=["team_id", "player_id"])
    data = data[(data["team_id"] >= TEAM_MIN) & (data["team_id"] <= TEAM_MAX)]
    data["season_PA"] = pd.to_numeric(data[pa_col], errors="coerce").fillna(0)
    if ops_col:
        data["OPS_season"] = pd.to_numeric(data[ops_col], errors="coerce")
    else:
        OBP = pd.to_numeric(data[obp_col], errors="coerce") if obp_col else np.nan
        SLG = pd.to_numeric(data[slg_col], errors="coerce") if slg_col else np.nan
        if tb_col and ab_col and h_col:
            SLG = pd.to_numeric(data[tb_col], errors="coerce") / pd.to_numeric(data[ab_col], errors="coerce").replace(0, np.nan)
        if obp_col is None:
            h = pd.to_numeric(data[h_col], errors="coerce")
            bb = pd.to_numeric(data[bb_col], errors="coerce")
            hbp = pd.to_numeric(data[hbp_col], errors="coerce") if hbp_col else 0
            sf = pd.to_numeric(data[sf_col], errors="coerce") if sf_col else 0
            ab = pd.to_numeric(data[ab_col], errors="coerce")
            denom = ab + bb + hbp + sf
            OBP = (h + bb + hbp) / denom.replace(0, np.nan)
        data["OPS_season"] = OBP + SLG
    return data[["team_id", "player_id", "season_PA", "OPS_season"]]


def load_logs(base: Path, override: Optional[Path]) -> pd.DataFrame:
    df = read_first(base, override, LOG_CANDIDATES)
    if df is None:
        raise FileNotFoundError("Unable to locate batting gamelogs.")
    name_field = pick_column(df, "player_name", "name_full")
    first_col = pick_column(df, "first_name")
    last_col = pick_column(df, "last_name")
    required_cols = {
        "team_id": pick_column(df, "team_id", "teamid"),
        "player_id": pick_column(df, "player_id", "playerid"),
        "date": pick_column(df, "game_date", "date"),
        "game_id": pick_column(df, "game_id", "gameid"),
        "pa": pick_column(df, "pa", "PA"),
        "ab": pick_column(df, "ab", "AB"),
        "h": pick_column(df, "h", "H"),
        "bb": pick_column(df, "bb", "BB"),
        "hbp": pick_column(df, "hbp", "HP"),
        "sf": pick_column(df, "sf"),
        "sh": pick_column(df, "sh"),
        "tb": pick_column(df, "tb"),
        "doubles": pick_column(df, "2b", "doubles"),
        "triples": pick_column(df, "3b", "triples"),
        "hr": pick_column(df, "hr"),
    }
    has_ids = required_cols["team_id"] and required_cols["player_id"]
    has_date = required_cols["date"] is not None
    has_game_id = required_cols["game_id"] is not None
    if not has_ids or (not has_date and not has_game_id):
        raise ValueError("Logs missing identifiers.")
    data = df.copy()
    data["team_id"] = pd.to_numeric(data[required_cols["team_id"]], errors="coerce").astype("Int64")
    data["player_id"] = pd.to_numeric(data[required_cols["player_id"]], errors="coerce").astype("Int64")
    data = data.dropna(subset=["team_id", "player_id"])
    data = data[(data["team_id"] >= TEAM_MIN) & (data["team_id"] <= TEAM_MAX)]
    if has_date:
        data["game_date"] = pd.to_datetime(data[required_cols["date"]], errors="coerce")
    else:
        game_dates = load_game_dates(base)
        data["game_date"] = data[required_cols["game_id"]].map(game_dates)
    data = data.dropna(subset=["game_date"])
    for key in ["pa", "ab", "h", "bb", "hbp", "sf", "sh", "tb", "doubles", "triples", "hr"]:
        if required_cols[key]:
            data[key.upper()] = pd.to_numeric(data[required_cols[key]], errors="coerce").fillna(0)
        else:
            data[key.upper()] = 0
    if not required_cols["pa"]:
        data["PA"] = (
            data["AB"] + data["BB"] + data["HBP"] + data["SF"] + data["SH"]
        )
    if not required_cols["tb"]:
        singles = data["H"] - data["DOUBLES"] - data["TRIPLES"] - data["HR"]
        singles = singles.clip(lower=0)
        data["TB"] = singles + 2 * data["DOUBLES"] + 3 * data["TRIPLES"] + 4 * data["HR"]
    if name_field:
        data["player_name"] = data[name_field].astype(str)
    elif first_col and last_col:
        data["player_name"] = (
            data[first_col].fillna("").astype(str).str.strip()
            + " "
            + data[last_col].fillna("").astype(str).str.strip()
        ).str.strip()
    else:
        data["player_name"] = data["player_id"].astype(str)
    return data[["team_id", "player_id", "player_name", "game_date", "PA", "AB", "H", "BB", "HBP", "SF", "TB"]]


def compute_ops(hits: float, walks: float, hbp: float, ab: float, sf: float, tb: float) -> float:
    ob_denom = ab + walks + hbp + sf
    obp = (hits + walks + hbp) / ob_denom if ob_denom > 0 else np.nan
    slg = tb / ab if ab > 0 else np.nan
    if pd.notna(obp) and pd.notna(slg):
        return obp + slg
    return np.nan


def aggregate_totals_from_logs(logs: Optional[pd.DataFrame]) -> pd.DataFrame:
    if logs is None or logs.empty:
        raise FileNotFoundError("Unable to build batting totals from logs; provide a totals export.")
    agg = (
        logs.groupby(["team_id", "player_id"], as_index=False)[["PA", "AB", "H", "BB", "HBP", "SF", "TB"]]
        .sum(min_count=1)
    )
    agg["season_PA"] = agg["PA"]
    agg["OPS_season"] = agg.apply(
        lambda row: compute_ops(row["H"], row["BB"], row["HBP"], row["AB"], row["SF"], row["TB"]),
        axis=1,
    )
    return agg[["team_id", "player_id", "season_PA", "OPS_season"]]


def render_text_table(
    df: pd.DataFrame,
    columns: Sequence[Tuple[str, str, int, bool, str]],
    title: str,
    subtitle: str,
    key_lines: Sequence[str],
    notes: Sequence[str],
) -> str:
    lines = [title, "=" * len(title)]
    if subtitle:
        lines.append(subtitle)
    lines.append("")
    header_parts = []
    for name, _, width, _, _ in columns:
        header_parts.append(name.ljust(width))
    header = " ".join(header_parts)
    lines.append(header)
    lines.append("-" * len(header))
    for _, row in df.iterrows():
        row_parts = []
        for _, field, width, is_numeric, fmt in columns:
            value = row.get(field)
            if is_numeric:
                if pd.isna(value):
                    text = "NA"
                else:
                    fmt_spec = fmt if fmt else ""
                    text = f"{value:{fmt_spec}}"
                row_parts.append(text.rjust(width))
            else:
                text = str(value) if value not in (None, np.nan) else ""
                row_parts.append(text[:width].ljust(width))
        lines.append(" ".join(row_parts))
    if key_lines:
        lines.append("")
        lines.extend(key_lines)
    if notes:
        lines.append("")
        lines.extend(notes)
    return "\n".join(lines)


def rolling_windows(group: pd.DataFrame, window_pa: int, min_ratio: float = 0.8) -> pd.DataFrame:
    group = group.sort_values("game_date").copy()
    group["games"] = 1
    agg = group.groupby("game_date", as_index=False).sum()
    stats = ["PA", "AB", "H", "BB", "HBP", "SF", "TB", "games"]
    for stat in stats:
        agg[f"c_{stat}"] = agg[stat].cumsum()
    windows = []
    for i in range(len(agg)):
        start = 0
        best_start = 0
        while start <= i:
            pa_window = agg.loc[i, "c_PA"] - (agg.loc[start - 1, "c_PA"] if start > 0 else 0)
            if pa_window >= window_pa:
                best_start = start
                start += 1
            else:
                break
        start_idx = best_start
        pa_window = agg.loc[i, "c_PA"] - (agg.loc[start_idx - 1, "c_PA"] if start_idx > 0 else 0)
        if pa_window < min_ratio * window_pa:
            continue
        window_stats = {
            stat: agg.loc[i, f"c_{stat}"] - (agg.loc[start_idx - 1, f"c_{stat}"] if start_idx > 0 else 0)
            for stat in stats
        }
        ops = compute_ops(
            window_stats["H"],
            window_stats["BB"],
            window_stats["HBP"],
            window_stats["AB"],
            window_stats["SF"],
            window_stats["TB"],
        )
        if pd.isna(ops):
            continue
        windows.append(
            {
                "player_id": group["player_id"].iloc[0],
                "team_id": group["team_id"].iloc[0],
                "player_name": group["player_name"].iloc[0],
                "date_start": agg.loc[start_idx, "game_date"],
                "date_end": agg.loc[i, "game_date"],
                "PA_window": window_stats["PA"],
                "games_in_window": window_stats["games"],
                "days_in_window": (agg.loc[i, "game_date"] - agg.loc[start_idx, "game_date"]).days + 1,
                "OPS_window": ops,
            }
        )
    return pd.DataFrame(windows)


def text_summary(current: pd.DataFrame, team_counts: pd.DataFrame) -> str:
    columns = [
        ("Player", "player_name", 22, False, ""),
        ("Team", "team_display", 8, False, ""),
        ("Conf", "conf_div", 6, False, ""),
        ("OPS", "OPS_window", 6, True, ".3f"),
        ("Season", "OPS_season", 6, True, ".3f"),
        ("dOPS", "delta_OPS", 6, True, ".3f"),
        ("PA", "PA_window", 5, True, ".0f"),
    ]
    table = render_text_table(
        current.sort_values(by=["delta_OPS", "OPS_window", "PA_window"], ascending=[True, True, False]).head(25),
        columns,
        "ABL System Crash Slumps",
        "Top 25 active slumps (worst delta OPS)",
        [
            "Ratings: delta <= -0.300 (Critical Failure), -0.300 < delta <= -0.200 (System Crash), delta > -0.200 (Soft Warning).",
        ],
        [
            "Windows require >= 80% of PA threshold; delta compares rolling OPS to season OPS.",
        ],
    )
    lines = [table, "", "Team Slump Counts", "------------------"]
    for _, row in team_counts.iterrows():
        lines.append(f"{row['team_display']:<12} ({row['conf_div']}) : {int(row['slumping_players'])} crashers")
    return "\n".join(lines)
def classify_rating(delta: float) -> str:
    if pd.isna(delta):
        return "Unknown"
    if delta <= -0.30:
        return "Critical Failure"
    if delta <= -0.20:
        return "System Crash"
    return "Soft Warning"


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="ABL System Crash Slumps.")
    parser.add_argument("--base", type=str, default=".", help="Base directory.")
    parser.add_argument("--gamelogs", type=str, help="Override gamelog path.")
    parser.add_argument("--totals", type=str, help="Override totals path.")
    parser.add_argument("--teams", type=str, help="Override team info path.")
    parser.add_argument("--out_current", type=str, default="out/z_ABL_System_Crash_Slumps_Current.csv", help="Current slumps CSV.")
    parser.add_argument("--out_history", type=str, default="out/z_ABL_System_Crash_Slumps_History.csv", help="History CSV.")
    parser.add_argument("--window_pa", type=int, default=50, help="Rolling PA target.")
    parser.add_argument("--delta_thresh", type=float, default=-0.200, help="OPS delta threshold (<= value flags slump).")
    parser.add_argument("--min_pa_season", type=int, default=100, help="Minimum season PA to consider.")
    parser.add_argument("--lookback_days", type=int, default=28, help="History lookback in days (<=0 for all).")
    args = parser.parse_args(list(argv) if argv is not None else None)

    base_dir = Path(args.base).resolve()
    logs = load_logs(base_dir, resolve_path(base_dir, args.gamelogs))
    totals = load_totals(base_dir, resolve_path(base_dir, args.totals), logs)
    team_names, conf_map = load_teams(base_dir, resolve_path(base_dir, args.teams))

    window_frames = []
    for _, group in logs.groupby(["team_id", "player_id", "player_name"]):
        win_df = rolling_windows(group, args.window_pa)
        if not win_df.empty:
            window_frames.append(win_df)
    if not window_frames:
        print("No rolling windows met the criteria.")
        return
    windows = pd.concat(window_frames, ignore_index=True)
    windows = windows.merge(totals, on=["team_id", "player_id"], how="left")
    windows["team_display"] = windows["team_id"].map(team_names).fillna(windows["team_id"].astype(str))
    windows["conf_div"] = windows["team_id"].map(conf_map).fillna("")
    windows = windows.dropna(subset=["OPS_season"])
    windows = windows[windows["season_PA"] >= args.min_pa_season]
    windows["delta_OPS"] = windows["OPS_window"] - windows["OPS_season"]
    windows["slump_flag"] = np.where(windows["delta_OPS"] <= args.delta_thresh, "SYSTEM_CRASH", "")
    windows["rating"] = windows["delta_OPS"].apply(classify_rating)

    slumps = windows[windows["slump_flag"] == "SYSTEM_CRASH"].copy()
    if slumps.empty:
        print("No current slumps detected.")
        return

    latest = (
        slumps.sort_values("date_end")
        .groupby(["team_id", "player_id"], as_index=False)
        .tail(1)
        .sort_values(by=["delta_OPS", "OPS_window", "PA_window"], ascending=[True, True, False])
    )

    current_path = Path(args.out_current)
    if not current_path.is_absolute():
        current_path = base_dir / current_path
    current_path.parent.mkdir(parents=True, exist_ok=True)
    current_out = latest[
        [
            "team_id",
            "team_display",
            "conf_div",
            "player_id",
            "player_name",
            "season_PA",
            "OPS_season",
            "date_start",
            "date_end",
            "PA_window",
            "games_in_window",
            "days_in_window",
            "OPS_window",
            "delta_OPS",
            "slump_flag",
            "rating",
        ]
    ].copy()
    for col in ["OPS_season", "OPS_window", "delta_OPS"]:
        current_out[col] = current_out[col].round(3)
    current_out.to_csv(current_path, index=False)

    history_path = Path(args.out_history)
    if not history_path.is_absolute():
        history_path = base_dir / history_path
    history_path.parent.mkdir(parents=True, exist_ok=True)
    max_date = slumps["date_end"].max()
    if args.lookback_days and args.lookback_days > 0:
        cutoff = max_date - timedelta(days=args.lookback_days)
        history = slumps[slumps["date_end"] >= cutoff].copy()
    else:
        history = slumps.copy()
    history_out = history[
        [
            "team_id",
            "team_display",
            "conf_div",
            "player_id",
            "player_name",
            "season_PA",
            "OPS_season",
            "date_start",
            "date_end",
            "PA_window",
            "games_in_window",
            "days_in_window",
            "OPS_window",
            "delta_OPS",
        ]
    ].copy()
    for col in ["OPS_season", "OPS_window", "delta_OPS"]:
        history_out[col] = history_out[col].round(3)
    history_out.to_csv(history_path, index=False)

    team_counts = (
        latest.groupby("team_id")
        .agg(slumping_players=("player_id", "nunique"))
        .reset_index()
        .sort_values(by="slumping_players", ascending=False)
    )
    team_counts["team_display"] = team_counts["team_id"].map(team_names).fillna(team_counts["team_id"].astype(str))
    team_counts["conf_div"] = team_counts["team_id"].map(conf_map).fillna("")

    print(text_summary(current_out, team_counts))


if __name__ == "__main__":
    main()

