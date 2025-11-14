"""ABL Basepath Pressure: evaluate SB aggressiveness and base advancement pressure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

TEAM_MIN, TEAM_MAX = 1, 24

TEAM_INFO_CANDIDATES = [
    "team_info.csv",
    "teams.csv",
    "standings.csv",
    "team_record.csv",
]
BATTING_CANDIDATES = [
    "team_batting.csv",
    "teams_batting.csv",
    "batting_team_totals.csv",
    "team_baserunning.csv",
    "team_batting_stats.csv",
]
USER_BASERUNNING_FILE = "abl_team_bat_baserunning.csv"
BATTING_SORTABLE_FILE = "abl_statistics_team_statistics___info_-_sortable_stats_batting_stats.csv"
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


def load_team_info(base: Path, override: Optional[Path]) -> Tuple[Dict[int, str], Dict[int, str]]:
    df = read_first(base, override, TEAM_INFO_CANDIDATES)
    if df is None:
        return {}, {}
    team_col = pick_column(df, "team_id", "teamid", "teamID")
    display_col = pick_column(df, "team_display", "team_name", "name", "TeamName")
    sub_col = pick_column(df, "sub_league_id", "subleague_id", "conference_id")
    div_col = pick_column(df, "division_id", "division")
    names: Dict[int, str] = {}
    conf_map: Dict[int, str] = {}
    conf_lookup = {0: "N", 1: "A"}
    div_lookup = {0: "E", 1: "C", 2: "W"}
    if not team_col:
        return names, conf_map
    for _, row in df.iterrows():
        tid = row.get(team_col)
        if pd.isna(tid):
            continue
        tid_int = int(tid)
        if not (TEAM_MIN <= tid_int <= TEAM_MAX):
            continue
        if display_col and pd.notna(row.get(display_col)):
            names[tid_int] = str(row.get(display_col))
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
    return names, conf_map


def load_batting_totals(base: Path, override: Optional[Path]) -> pd.DataFrame:
    df = read_first(base, override, BATTING_CANDIDATES)
    if df is None:
        raise FileNotFoundError("Unable to locate team batting/baserunning totals.")
    team_col = pick_column(df, "team_id", "teamid", "teamID")
    if not team_col:
        raise ValueError("team_id column missing in batting totals.")
    df = df.copy()
    df["team_id"] = pd.to_numeric(df[team_col], errors="coerce").astype("Int64")
    df = df[(df["team_id"] >= TEAM_MIN) & (df["team_id"] <= TEAM_MAX)]
    def numeric(names: Sequence[str], default: float = np.nan) -> pd.Series:
        col = pick_column(df, *names)
        if col:
            return pd.to_numeric(df[col], errors="coerce")
        return pd.Series(default, index=df.index)
    totals = pd.DataFrame()
    totals["team_id"] = df["team_id"]
    disp_col = pick_column(df, "team_display", "team_name", "name", "TeamName")
    if disp_col:
        totals["team_display"] = df[disp_col].fillna("")
    else:
        totals["team_display"] = ""
    totals["SB"] = numeric(["sb", "SB"]).fillna(0)
    totals["CS"] = numeric(["cs", "CS"]).fillna(0)
    totals["OOB"] = numeric(["oob", "outs_on_base", "OutsOnBase"])
    return totals


def load_user_baserunning(
    base: Path,
    override: Optional[Path],
    team_name_lookup: Dict[int, str],
) -> pd.DataFrame:
    candidate_paths: Sequence[Path] = []
    if override:
        candidate_paths = [override]
    else:
        candidate_paths = [
            base / USER_BASERUNNING_FILE,
            base / BATTING_SORTABLE_FILE,
        ]
    source_path: Optional[Path] = None
    for path in candidate_paths:
        if path and path.exists():
            source_path = path
            break
    if source_path is None:
        return pd.DataFrame()
    df = pd.read_csv(source_path)
    df = df.copy()
    team_col = pick_column(df, "team_id", "teamid", "teamID", "ID")
    if team_col:
        df["team_id"] = pd.to_numeric(df[team_col], errors="coerce").astype("Int64")
    else:
        name_col = pick_column(df, "Team Name", "team_name", "TeamName")
        if not name_col:
            raise ValueError("Baserunning file missing team identifiers.")
        reverse_lookup = {
            str(name).strip().lower(): tid for tid, name in team_name_lookup.items()
        }
        df["team_id"] = (
            df[name_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(reverse_lookup)
            .astype("Int64")
        )
    df = df[(df["team_id"] >= TEAM_MIN) & (df["team_id"] <= TEAM_MAX)]
    ubr_col = pick_column(df, "UBR")
    if not ubr_col:
        raise ValueError("Baserunning file missing UBR column.")
    data = pd.DataFrame()
    data["team_id"] = df["team_id"]
    data["UBR"] = pd.to_numeric(df[ubr_col], errors="coerce")
    bsr_col = pick_column(df, "BsR")
    if bsr_col:
        data["BsR"] = pd.to_numeric(df[bsr_col], errors="coerce")
    return data


def load_games_played(base: Path, record_override: Optional[Path], logs_override: Optional[Path]) -> pd.Series:
    record_df = read_first(base, record_override, RECORD_CANDIDATES)
    if record_df is not None:
        team_col = pick_column(record_df, "team_id", "teamid", "teamID")
        wins_col = pick_column(record_df, "wins", "w")
        losses_col = pick_column(record_df, "losses", "l")
        if team_col and wins_col and losses_col:
            rec = record_df.copy()
            rec["team_id"] = pd.to_numeric(rec[team_col], errors="coerce").astype("Int64")
            rec = rec[(rec["team_id"] >= TEAM_MIN) & (rec["team_id"] <= TEAM_MAX)]
            rec["wins"] = pd.to_numeric(rec[wins_col], errors="coerce").fillna(0)
            rec["losses"] = pd.to_numeric(rec[losses_col], errors="coerce").fillna(0)
            rec["games"] = rec["wins"] + rec["losses"]
            if not rec.empty:
                return rec.set_index("team_id")["games"]
    logs_df = read_first(base, logs_override, LOG_CANDIDATES)
    if logs_df is None:
        raise FileNotFoundError("Need either record file or team logs to determine games played.")
    team_col = pick_column(logs_df, "team_id", "teamid", "teamID")
    date_col = pick_column(logs_df, "game_date", "date", "GameDate")
    if not (team_col and date_col):
        raise ValueError("Logs file missing team_id or game_date.")
    logs_df = logs_df.copy()
    logs_df["team_id"] = pd.to_numeric(logs_df[team_col], errors="coerce").astype("Int64")
    logs_df = logs_df[(logs_df["team_id"] >= TEAM_MIN) & (logs_df["team_id"] <= TEAM_MAX)]
    logs_df["game_date"] = pd.to_datetime(logs_df[date_col], errors="coerce")
    logs_df = logs_df.dropna(subset=["team_id", "game_date"])
    games = logs_df.drop_duplicates(["team_id", "game_date"]).groupby("team_id").size()
    return games.rename("games").astype(float)


def calc_rates(
    totals: pd.DataFrame,
    user_baserunning: pd.DataFrame,
    games: pd.Series,
    team_names: Dict[int, str],
    conf_map: Dict[int, str],
) -> pd.DataFrame:
    if user_baserunning.empty:
        user_baserunning = pd.DataFrame({"team_id": totals["team_id"], "UBR": np.nan})
    df = totals.merge(user_baserunning, on="team_id", how="left")
    df["team_display"] = df["team_id"].map(team_names).fillna(df.get("team_display", ""))
    df["conf_div"] = df["team_id"].map(conf_map).fillna("")
    df["g"] = df["team_id"].map(games).fillna(0)
    df["sb_att"] = df["SB"].fillna(0) + df["CS"].fillna(0)
    df["sb_pct"] = np.where(df["sb_att"] > 0, df["SB"].fillna(0) / df["sb_att"], np.nan)
    df["sb_att_pg"] = np.where(df["g"] > 0, df["sb_att"] / df["g"], np.nan)
    df["ubr_pg"] = np.where(df["g"] > 0, df["UBR"] / df["g"], np.nan)
    df["oob_pg"] = np.where(
        (df["g"] > 0) & df["OOB"].notna(),
        df["OOB"] / df["g"],
        np.nan,
    )
    lg_sb_pct = df["sb_pct"].mean(skipna=True)
    lg_sb_att_pg = df["sb_att_pg"].mean(skipna=True)
    lg_ubr_pg_mean = df["ubr_pg"].mean(skipna=True)
    lg_ubr_pg_std = df["ubr_pg"].std(skipna=True)
    lg_oob_pg = df["oob_pg"].mean(skipna=True) if df["oob_pg"].notna().any() else np.nan
    df["sb_pct_plus"] = df["sb_pct"] / lg_sb_pct if lg_sb_pct else np.nan
    df["sb_att_pg_plus"] = df["sb_att_pg"] / lg_sb_att_pg if lg_sb_att_pg else np.nan
    if lg_ubr_pg_std and not np.isclose(lg_ubr_pg_std, 0):
        df["ubr_pg_plus"] = 1 + (df["ubr_pg"] - lg_ubr_pg_mean) / lg_ubr_pg_std
    else:
        df["ubr_pg_plus"] = np.nan
    df["ubr_pg_plus"] = df["ubr_pg_plus"].clip(lower=0.1)
    df["oob_pg_plus"] = (
        df["oob_pg"] / lg_oob_pg if pd.notna(lg_oob_pg) else np.nan
    )
    def composite(row: pd.Series) -> float:
        components = [row.get("sb_att_pg_plus"), row.get("ubr_pg_plus")]
        vals = [v for v in components if pd.notna(v)]
        if not vals:
            return np.nan
        return sum(vals) / len(vals)
    df["pressure_index"] = df.apply(composite, axis=1)
    df["pressure_rating"] = df["pressure_index"].apply(rate_pressure)
    df["sb_pct"] = df["sb_pct"].round(3)
    df["sb_att_pg"] = df["sb_att_pg"].round(3)
    df["ubr_pg"] = df["ubr_pg"].round(3)
    df["oob_pg"] = df["oob_pg"].round(3)
    df["sb_pct_plus"] = df["sb_pct_plus"].round(3)
    df["sb_att_pg_plus"] = df["sb_att_pg_plus"].round(3)
    df["ubr_pg_plus"] = df["ubr_pg_plus"].round(3)
    df["pressure_index"] = df["pressure_index"].round(3)
    df = df[
        [
            "team_id",
            "team_display",
            "conf_div",
            "g",
            "SB",
            "CS",
            "sb_att",
            "sb_pct",
            "sb_att_pg",
            "UBR",
            "ubr_pg",
            "OOB",
            "oob_pg",
            "sb_pct_plus",
            "sb_att_pg_plus",
            "ubr_pg_plus",
            "pressure_index",
            "pressure_rating",
        ]
    ]
    return df


def rate_pressure(value: float) -> str:
    if pd.isna(value):
        return "NA"
    if value >= 1.20:
        return "Relentless"
    if value >= 1.05:
        return "Aggressive"
    if value <= 0.85:
        return "Passive"
    if value <= 0.95:
        return "Measured"
    return "Balanced"


def build_text_report(df: pd.DataFrame, limit: Optional[int] = None) -> str:
    lines = [
        "ABL Basepath Pressure",
        "=" * 26,
        "By Pressure Index - compares each club's stolen-base volume and UBR per game to league norms.",
        "Pressure Index = average of SB attempts per game plus and UBR per game plus; values above 1.00 mean above-average pressure.",
        "",
    ]
    subset = df if limit is None else df.head(limit)
    header = f"{'Team':<24} {'PI':>6} {'SB%':>7} {'SB Att/G':>10} {'UBR/G':>8} {'Rating':>12}"
    lines.append(header)
    lines.append("-" * len(header))
    for _, row in subset.iterrows():
        pi_txt = f"{row['pressure_index']:.3f}" if not pd.isna(row["pressure_index"]) else "NA "
        sb_txt = f"{row['sb_pct']:.3f}" if not pd.isna(row["sb_pct"]) else "NA "
        att_txt = f"{row['sb_att_pg']:.3f}" if not pd.isna(row["sb_att_pg"]) else "NA "
        ubr_txt = f"{row['ubr_pg']:.3f}" if not pd.isna(row["ubr_pg"]) else "NA "
        team_lbl = f"{row['team_display']} ({row['conf_div']})"
        lines.append(
            f"{team_lbl:<24} {pi_txt:>6} {sb_txt:>7} {att_txt:>10} {ubr_txt:>8} {row['pressure_rating']:>12}"
        )
    lines.append("")
    lines.append("Key:")
    lines.append("  Relentless -> pressure_index >= 1.20 (constant attack).")
    lines.append("  Aggressive -> pressure_index 1.05-1.19.")
    lines.append("  Balanced/Measured/Passive -> progressively less pressure.")
    lines.append("")
    lines.append("Definitions:")
    lines.append("  SB% = SB / (SB + CS); SB Att/G = attempts per game.")
    lines.append("  UBR/G = Ultimate Base Running runs per game (FanGraphs style).")
    lines.append("  Plus metrics compare performance to a 1.00 league average.")
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ABL Basepath Pressure report.")
    parser.add_argument("--base", type=str, default=".", help="Base directory.")
    parser.add_argument("--batting", type=str, help="Override batting/baserunning totals.")
    parser.add_argument("--record", type=str, help="Override season record file.")
    parser.add_argument("--logs", type=str, help="Override team logs for games.")
    parser.add_argument("--teams", type=str, help="Override team info file.")
    parser.add_argument(
        "--user-baserunning",
        type=str,
        help=f"Override user baserunning file (default: {USER_BASERUNNING_FILE}).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="out/csv_out/z_ABL_Basepath_Pressure.csv",
        help="Output CSV path (default: out/csv_out/z_ABL_Basepath_Pressure.csv).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    base_dir = Path(args.base).resolve()
    team_names, conf_map = load_team_info(base_dir, Path(args.teams) if args.teams else None)
    batting = load_batting_totals(base_dir, Path(args.batting) if args.batting else None)
    user_baserunning = load_user_baserunning(
        base_dir,
        Path(args.user_baserunning) if args.user_baserunning else None,
        team_names,
    )
    games = load_games_played(
        base_dir,
        Path(args.record) if args.record else None,
        Path(args.logs) if args.logs else None,
    )
    final_df = calc_rates(batting, user_baserunning, games, team_names, conf_map)
    final_df = final_df.sort_values("pressure_index", ascending=False, na_position="last").reset_index(drop=True)
    out_path = (base_dir / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_path, index=False)
    text_dir = base_dir / "out" / "txt_out"
    text_dir.mkdir(parents=True, exist_ok=True)
    text_path = text_dir / out_path.name.replace(".csv", ".txt")
    text_path.write_text(build_text_report(final_df, limit=None), encoding="utf-8")
    print("Basepath Pressure (top 12):")
    print(final_df.head(12).to_string(index=False))
    print(f"\nWrote {len(final_df)} rows to {out_path}.")


if __name__ == "__main__":
    main(sys.argv[1:])
