"""ABL Damage With RISP: compare hitter OPS in scoring spots vs overall."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

TEAM_MIN, TEAM_MAX = 1, 24

TOTALS_CANDIDATES = [
    "players_batting.csv",
    "player_batting_totals.csv",
    "batting_players.csv",
]
SPLIT_CANDIDATES = [
    "batting_splits.csv",
    "batting_splits_situational.csv",
    "players_batting_splits.csv",
]
TEAM_INFO_CANDIDATES = [
    "team_info.csv",
    "teams.csv",
    "standings.csv",
    "team_record.csv",
]
ROSTER_CANDIDATES = [
    "players.csv",
    "rosters.csv",
]
PLAYER_GAMES_FILE = "players_game_batting.csv"


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


def load_team_info(base: Path, override: Optional[Path]) -> Tuple[Dict[int, str], Dict[int, str]]:
    df = read_first(base, override, TEAM_INFO_CANDIDATES)
    if df is None:
        return {}, {}
    team_col = pick_column(df, "team_id", "teamid", "teamID")
    display_col = pick_column(df, "team_display", "team_name", "name", "TeamName")
    sub_col = pick_column(df, "sub_league_id", "conference_id", "subleague_id")
    div_col = pick_column(df, "division_id", "division")
    display_map: Dict[int, str] = {}
    conf_map: Dict[int, str] = {}
    conf_lookup = {0: "N", 1: "A"}
    div_lookup = {0: "E", 1: "C", 2: "W"}
    if not team_col:
        return display_map, conf_map
    for _, row in df.iterrows():
        team_val = row.get(team_col)
        if pd.isna(team_val):
            continue
        try:
            tid = int(team_val)
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


def load_roster(base: Path) -> Dict[int, str]:
    df = read_first(base, None, ROSTER_CANDIDATES)
    if df is None:
        return {}
    player_col = pick_column(df, "player_id", "playerid", "PlayerID")
    team_col = pick_column(df, "team_id", "teamid", "TeamID", "current_team_id")
    first_col = pick_column(df, "first_name", "firstname")
    last_col = pick_column(df, "last_name", "lastname")
    name_col = pick_column(df, "name_full", "name", "player_name")
    if not player_col:
        return {}
    df = df.copy()
    df["player_id"] = pd.to_numeric(df[player_col], errors="coerce").astype("Int64")
    df = df.dropna(subset=["player_id"])
    names = {}
    for _, row in df.iterrows():
        pid = int(row["player_id"])
        if first_col and last_col and pd.notna(row.get(first_col)) and pd.notna(row.get(last_col)):
            names[pid] = f"{row[first_col]} {row[last_col]}".strip()
        elif name_col and pd.notna(row.get(name_col)):
            names[pid] = str(row[name_col]).strip()
    return names


def load_totals(base: Path, override: Optional[Path]) -> pd.DataFrame:
    df = read_first(base, override, TOTALS_CANDIDATES)
    if df is not None:
        id_col = pick_column(df, "player_id", "playerid", "PlayerID")
        team_col = pick_column(df, "team_id", "teamid", "TeamID")
        pa_col = pick_column(df, "pa", "PA")
        obp_col = pick_column(df, "obp", "OBP")
        slg_col = pick_column(df, "slg", "SLG")
        ops_col = pick_column(df, "ops", "OPS")
        first_col = pick_column(df, "first_name", "firstname")
        last_col = pick_column(df, "last_name", "lastname")
        name_col = pick_column(df, "name", "name_full", "player_name")
        year_col = pick_column(df, "year", "season")
        if id_col and team_col and pa_col and (ops_col or (obp_col and slg_col)):
            totals = df.copy()
            totals["player_id"] = pd.to_numeric(totals[id_col], errors="coerce").astype("Int64")
            totals["team_id"] = pd.to_numeric(totals[team_col], errors="coerce").astype("Int64")
            totals["PA"] = pd.to_numeric(totals[pa_col], errors="coerce")
            if year_col:
                max_year = totals[year_col].max()
                totals = totals[totals[year_col] == max_year]
            totals = totals[(totals["team_id"] >= TEAM_MIN) & (totals["team_id"] <= TEAM_MAX)]
            totals["OBP_overall"] = pd.to_numeric(totals[obp_col], errors="coerce") if obp_col else np.nan
            totals["SLG_overall"] = pd.to_numeric(totals[slg_col], errors="coerce") if slg_col else np.nan
            if ops_col:
                totals["OPS_overall"] = pd.to_numeric(totals[ops_col], errors="coerce")
            else:
                totals["OPS_overall"] = totals["OBP_overall"] + totals["SLG_overall"]
            totals["player_name"] = ""
            if first_col and last_col:
                totals["player_name"] = (
                    totals[first_col].fillna("").astype(str).str.strip()
                    + " "
                    + totals[last_col].fillna("").astype(str).str.strip()
                ).str.strip()
            elif name_col:
                totals["player_name"] = totals[name_col].fillna("").astype(str).str.strip()
            return totals[
                [
                    "player_id",
                    "team_id",
                    "player_name",
                    "PA",
                    "OPS_overall",
                ]
            ]
    return aggregate_totals_from_games(base)


def aggregate_totals_from_games(base: Path) -> pd.DataFrame:
    path = base / PLAYER_GAMES_FILE
    if not path.exists():
        raise FileNotFoundError("Unable to derive batting totals; players_game_batting.csv not found.")
    df = pd.read_csv(path)
    required = {"player_id", "team_id", "pa", "ab", "h", "bb", "hp", "sf", "d", "t", "hr"}
    if not required.issubset(set(df.columns)):
        raise ValueError("players_game_batting.csv missing required columns for aggregation.")
    df = df.dropna(subset=["player_id", "team_id"])
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce").astype("Int64")
    df = df[(df["team_id"] >= TEAM_MIN) & (df["team_id"] <= TEAM_MAX)]
    sort_col = "game_id" if "game_id" in df.columns else "player_id"
    df = df.sort_values(sort_col)
    agg = df.groupby("player_id").agg(
        team_id=("team_id", lambda s: s.iloc[-1]),
        PA=("pa", "sum"),
        AB=("ab", "sum"),
        H=("h", "sum"),
        BB=("bb", "sum"),
        HBP=("hp", "sum"),
        SF=("sf", "sum"),
        doubles=("d", "sum"),
        triples=("t", "sum"),
        homers=("hr", "sum"),
    )
    agg = agg.reset_index()
    singles = agg["H"] - agg["doubles"] - agg["triples"] - agg["homers"]
    total_bases = singles + (2 * agg["doubles"]) + (3 * agg["triples"]) + (4 * agg["homers"])
    obp_denom = agg["AB"] + agg["BB"] + agg["HBP"] + agg["SF"]
    obp_num = agg["H"] + agg["BB"] + agg["HBP"]
    slg_denom = agg["AB"]
    obp = np.where(obp_denom > 0, obp_num / obp_denom, np.nan)
    slg = np.where(slg_denom > 0, total_bases / slg_denom, np.nan)
    agg["OPS_overall"] = obp + slg
    agg["player_name"] = ""
    return agg[
        [
            "player_id",
            "team_id",
            "player_name",
            "PA",
            "OPS_overall",
        ]
    ]


def load_risp_splits(base: Path, override: Optional[Path]) -> pd.DataFrame:
    df = read_first(base, override, SPLIT_CANDIDATES)
    if df is None:
        return pd.DataFrame(columns=["player_id", "team_id", "PA_RISP", "OPS_RISP"])
    df = df.copy()
    id_col = pick_column(df, "player_id", "playerid", "PlayerID")
    team_col = pick_column(df, "team_id", "teamid", "TeamID")
    if not id_col:
        return pd.DataFrame(columns=["player_id", "team_id", "PA_RISP", "OPS_RISP"])
    df["player_id"] = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")
    if team_col:
        df["team_id"] = pd.to_numeric(df[team_col], errors="coerce").astype("Int64")
    else:
        df["team_id"] = pd.NA

    split_col = pick_column(df, "split", "split_name", "situation", "category")
    subset = None
    if split_col:
        subset = df[df[split_col].astype(str).str.upper().str.contains("RISP", na=False)].copy()
    if subset is not None and not subset.empty:
        pa_col = pick_column(subset, "pa_risp", "PA_RISP", "pa", "PA")
        obp_col = pick_column(subset, "obp_risp", "OBP_RISP", "obp", "OBP")
        slg_col = pick_column(subset, "slg_risp", "SLG_RISP", "slg", "SLG")
        ops_col = pick_column(subset, "ops_risp", "OPS_RISP", "ops", "OPS")
        result = subset[
            [
                "player_id",
                "team_id",
            ]
        ].copy()
        result["PA_RISP"] = pd.to_numeric(subset[pa_col], errors="coerce") if pa_col else np.nan
        if ops_col:
            result["OPS_RISP"] = pd.to_numeric(subset[ops_col], errors="coerce")
        elif obp_col and slg_col:
            result["OPS_RISP"] = pd.to_numeric(subset[obp_col], errors="coerce") + pd.to_numeric(
                subset[slg_col], errors="coerce"
            )
        else:
            result["OPS_RISP"] = np.nan
        return result

    # dedicated column pattern fallback
    pa_col = pick_column(df, "pa_risp", "PA_RISP")
    ops_col = pick_column(df, "ops_risp", "OPS_RISP")
    obp_col = pick_column(df, "obp_risp", "OBP_RISP")
    slg_col = pick_column(df, "slg_risp", "SLG_RISP")
    if not pa_col:
        return pd.DataFrame(columns=["player_id", "team_id", "PA_RISP", "OPS_RISP"])
    result = df[
        [
            "player_id",
            "team_id",
        ]
    ].copy()
    result["PA_RISP"] = pd.to_numeric(df[pa_col], errors="coerce")
    if ops_col:
        result["OPS_RISP"] = pd.to_numeric(df[ops_col], errors="coerce")
    elif obp_col and slg_col:
        result["OPS_RISP"] = pd.to_numeric(df[obp_col], errors="coerce") + pd.to_numeric(df[slg_col], errors="coerce")
    else:
        result["OPS_RISP"] = np.nan
    return result


def rate_delta(delta: float) -> str:
    if pd.isna(delta):
        return "Unknown"
    if delta >= 0.150:
        return "Icewater"
    if delta >= 0.075:
        return "Clutch"
    if delta >= -0.025:
        return "Steady"
    if delta >= -0.100:
        return "Pressing"
    return "Chilled"


def build_text_report(df: pd.DataFrame, min_pa: int, min_pa_risp: int) -> str:
    lines = ["ABL Damage With RISP", "=" * 27, ""]
    header = f"{'Player':<22} {'Tm':<5} {'Rating':<8} {'OPS':>6} {'RISP OPS':>9} {'dOPS':>7} {'PA':>5} {'PA_R':>6}"
    lines.append(header)
    lines.append("-" * len(header))
    for _, row in df.iterrows():
        conf = row["conf_div"]
        conf_txt = conf if isinstance(conf, str) and conf else "--"
        player = f"{row['player_name']} ({conf_txt})"
        ops_txt = f"{row['OPS_overall']:.3f}" if not pd.isna(row["OPS_overall"]) else "NA "
        risp_txt = f"{row['OPS_RISP']:.3f}" if not pd.isna(row["OPS_RISP"]) else "NA "
        delta_txt = f"{row['delta_ops']:.3f}" if not pd.isna(row["delta_ops"]) else "NA "
        pa_txt = f"{int(row['PA'])}" if not pd.isna(row["PA"]) else "NA"
        pa_risp_txt = f"{int(row['PA_RISP'])}" if not pd.isna(row["PA_RISP"]) else "NA"
        team_str = row["team_display"]
        team_str = "" if pd.isna(team_str) else str(team_str)
        lines.append(
            f"{player:<22} {team_str[:4]:<5} {row['clutch_rating']:<8} "
            f"{ops_txt:>6} {risp_txt:>9} {delta_txt:>7} {pa_txt:>5} {pa_risp_txt:>6}"
        )
    if df.empty:
        lines.append("(No hitters met the qualification thresholds.)")
        lines.append("")
    lines.append(f"Thresholds: overall PA >= {min_pa}, RISP PA >= {min_pa_risp}.")
    lines.append("")
    lines.append("Key:")
    lines.append("  Icewater >= +0.150 delta OPS, Clutch 0.075-0.149, Steady within +/-0.025, Pressing -0.099 to -0.026, Chilled <= -0.100.")
    lines.append("")
    lines.append("Definitions:")
    lines.append("  OPS = season-wide on-base plus slugging.")
    lines.append("  RISP OPS = OPS limited to PA with runners in scoring position.")
    lines.append("  dOPS = OPS_RISP - OPS_overall (positive = better with ducks on the pond).")
    lines.append("  z_Score contextualizes dOPS relative to league distribution (>= thresholds).")
    lines.append("  League averages (OPS and RISP OPS) printed in CSV for reference.")
    return "\n".join(lines)


def print_top_table(df: pd.DataFrame) -> None:
    subset = df[
        [
            "player_name",
            "team_display",
            "conf_div",
            "clutch_rating",
            "OPS_overall",
            "OPS_RISP",
            "delta_ops",
            "PA",
            "PA_RISP",
        ]
    ].head(25)
    display_df = subset.copy()
    display_df = display_df.rename(
        columns={
            "player_name": "Player",
            "team_display": "Team",
            "conf_div": "ConfDiv",
            "clutch_rating": "Rating",
            "OPS_overall": "OPS",
            "OPS_RISP": "RISP OPS",
            "delta_ops": "dOPS",
            "PA": "PA",
            "PA_RISP": "PA RISP",
        }
    )
    for col in ["OPS", "RISP OPS", "dOPS"]:
        display_df[col] = display_df[col].map(lambda v: f"{v:.3f}" if not pd.isna(v) else "NA ")
    print(display_df.to_string(index=False))


def compute_stats(
    totals: pd.DataFrame,
    risp: pd.DataFrame,
    display_map: Dict[int, str],
    conf_map: Dict[int, str],
    names_map: Dict[int, str],
    min_pa: int,
    min_pa_risp: int,
) -> pd.DataFrame:
    merged = totals.merge(risp, on=["player_id", "team_id"], how="left")
    merged["PA_RISP"] = pd.to_numeric(merged["PA_RISP"], errors="coerce").fillna(0)
    merged["OPS_RISP"] = pd.to_numeric(merged["OPS_RISP"], errors="coerce")
    merged["delta_ops"] = merged["OPS_RISP"] - merged["OPS_overall"]
    merged["team_display"] = merged["team_id"].apply(lambda tid: display_map.get(tid, ""))
    merged["conf_div"] = merged["team_id"].apply(lambda tid: conf_map.get(tid, ""))
    merged["player_name"] = merged.apply(
        lambda row: row["player_name"] if row["player_name"] else names_map.get(int(row["player_id"]), f"Player {row['player_id']}"),
        axis=1,
    )
    qualified = merged[
        (merged["PA"] >= min_pa)
        & (merged["PA_RISP"] >= min_pa_risp)
        & merged["OPS_overall"].notna()
        & merged["OPS_RISP"].notna()
    ].copy()
    if qualified.empty:
        return qualified
    lg_ops_overall = qualified["OPS_overall"].mean()
    lg_ops_risp = qualified["OPS_RISP"].mean()
    delta_series = qualified["delta_ops"].dropna()
    if len(delta_series) >= 3:
        delta_mean = delta_series.mean()
        delta_std = delta_series.std(ddof=0)
        qualified["delta_ops_z"] = (qualified["delta_ops"] - delta_mean) / delta_std if delta_std > 0 else np.nan
    else:
        qualified["delta_ops_z"] = np.nan
    qualified["lg_ops_overall"] = round(lg_ops_overall, 3)
    qualified["lg_ops_risp"] = round(lg_ops_risp, 3)
    qualified["clutch_rating"] = qualified["delta_ops"].apply(rate_delta)
    return qualified


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ABL Damage With RISP report.")
    parser.add_argument("--base", type=str, default=".", help="Base directory for CSV files.")
    parser.add_argument("--totals", type=str, help="Override file for player batting totals.")
    parser.add_argument("--splits", type=str, help="Override file for situational splits.")
    parser.add_argument("--teams", type=str, help="Override file for team info.")
    parser.add_argument("--min_pa", type=int, default=80, help="Minimum PA to qualify.")
    parser.add_argument("--min_pa_risp", type=int, default=20, help="Minimum PA with RISP.")
    parser.add_argument("--out", type=str, default="out/z_ABL_Damage_With_RISP.csv", help="Output CSV path.")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    base_dir = Path(args.base).resolve()
    totals_override = resolve_optional_path(base_dir, args.totals)
    splits_override = resolve_optional_path(base_dir, args.splits)
    teams_override = resolve_optional_path(base_dir, args.teams)

    display_map, conf_map = load_team_info(base_dir, teams_override)
    names_map = load_roster(base_dir)
    totals = load_totals(base_dir, totals_override)
    risp = load_risp_splits(base_dir, splits_override)
    stats = compute_stats(
        totals,
        risp,
        display_map,
        conf_map,
        names_map,
        args.min_pa,
        args.min_pa_risp,
    )
    stats = stats.sort_values(by=["delta_ops", "OPS_RISP"], ascending=[False, False]).reset_index(drop=True)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = base_dir / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if stats.empty:
        csv_df = pd.DataFrame(columns=[
            "team_id",
            "team_display",
            "conf_div",
            "player_id",
            "player_name",
            "PA",
            "OPS_overall",
            "PA_RISP",
            "OPS_RISP",
            "delta_ops",
            "delta_ops_z",
            "lg_ops_overall",
            "lg_ops_risp",
            "clutch_rating",
        ])
    else:
        csv_df = stats.copy()
        for col in ["OPS_overall", "OPS_RISP", "delta_ops"]:
            csv_df[col] = csv_df[col].round(3)
        if "delta_ops_z" in csv_df.columns:
            csv_df["delta_ops_z"] = csv_df["delta_ops_z"].round(2)
    csv_columns = [
        "team_id",
        "team_display",
        "conf_div",
        "player_id",
        "player_name",
        "PA",
        "OPS_overall",
        "PA_RISP",
        "OPS_RISP",
        "delta_ops",
        "delta_ops_z",
        "lg_ops_overall",
        "lg_ops_risp",
        "clutch_rating",
    ]
    if stats.empty:
        csv_df = pd.DataFrame(columns=csv_columns)
    csv_df.to_csv(out_path, index=False)

    text_report = build_text_report(stats.head(25), args.min_pa, args.min_pa_risp)
    out_path.with_suffix(".txt").write_text(text_report, encoding="utf-8")

    if stats.empty:
        print("No hitters met the PA thresholds.")
    else:
        print_top_table(stats)


if __name__ == "__main__":
    main()
