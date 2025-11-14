"""ABL Platoon Assassins: spotlight hitters with huge opposite-hand splits."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

TEAM_MIN, TEAM_MAX = 1, 24

SPLIT_CANDIDATES = [
    "batting_splits_vs_hand.csv",
    "players_batting_splits_vs_hand.csv",
    "batting_splits_hand.csv",
]
ROSTER_CANDIDATES = [
    "players.csv",
    "rosters.csv",
    "players_batting.csv",
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
        if name.lower() in lowered:
            return lowered[name.lower()]
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
    team_col = pick_column(df, "team_id", "teamid", "TeamID")
    display_col = pick_column(df, "team_display", "team_name", "name", "TeamName")
    sub_col = pick_column(df, "sub_league_id", "subleague_id", "conference_id")
    div_col = pick_column(df, "division_id", "division")
    display_map: Dict[int, str] = {}
    conf_map: Dict[int, str] = {}
    conf_lookup = {0: "N", 1: "A"}
    div_lookup = {0: "E", 1: "C", 2: "W"}
    if not team_col:
        return display_map, conf_map
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


def load_roster_names(base: Path, override: Optional[Path]) -> Tuple[Dict[int, str], Dict[int, str]]:
    df = read_first(base, override, ROSTER_CANDIDATES)
    if df is None:
        return {}, {}
    id_col = pick_column(df, "player_id", "playerid", "PlayerID")
    team_col = pick_column(df, "team_id", "teamid", "TeamID")
    first_col = pick_column(df, "first_name", "firstname")
    last_col = pick_column(df, "last_name", "lastname")
    name_col = pick_column(df, "name_full", "name", "player_name")
    pos_col = pick_column(df, "position", "pos")
    names: Dict[int, str] = {}
    positions: Dict[int, str] = {}
    if not id_col:
        return names, positions
    df = df.copy()
    df["player_id"] = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")
    if team_col:
        df["team_id"] = pd.to_numeric(df[team_col], errors="coerce").astype("Int64")
    else:
        df["team_id"] = pd.NA
    for _, row in df.dropna(subset=["player_id"]).iterrows():
        pid = int(row["player_id"])
        if first_col and last_col and pd.notna(row.get(first_col)) and pd.notna(row.get(last_col)):
            names[pid] = f"{row[first_col]} {row[last_col]}".strip()
        elif name_col and pd.notna(row.get(name_col)):
            names[pid] = str(row[name_col]).strip()
        if pos_col and pd.notna(row.get(pos_col)):
            positions[pid] = str(row[pos_col]).strip().upper()
    return names, positions


def load_splits(base: Path, override: Optional[Path]) -> pd.DataFrame:
    df = read_first(base, override, SPLIT_CANDIDATES)
    if df is None:
        raise FileNotFoundError("Unable to locate splits vs hand data.")
    id_col = pick_column(df, "player_id", "playerid", "PlayerID")
    team_col = pick_column(df, "team_id", "teamid", "TeamID")
    bats_col = pick_column(df, "bats", "bat_hand", "handedness")
    pa_vr_col = pick_column(df, "pa_vr", "PA_vR", "PA_vs_RHP")
    pa_vl_col = pick_column(df, "pa_vl", "PA_vL", "PA_vs_LHP")
    ops_vr_col = pick_column(df, "ops_vr", "OPS_vR", "OPS_vs_RHP")
    ops_vl_col = pick_column(df, "ops_vl", "OPS_vL", "OPS_vs_LHP")
    if not all([id_col, team_col, bats_col, pa_vr_col, pa_vl_col, ops_vr_col, ops_vl_col]):
        raise ValueError("Splits file missing required columns.")
    data = df.copy()
    data["player_id"] = pd.to_numeric(data[id_col], errors="coerce").astype("Int64")
    data["team_id"] = pd.to_numeric(data[team_col], errors="coerce").astype("Int64")
    data = data[(data["team_id"] >= TEAM_MIN) & (data["team_id"] <= TEAM_MAX)]
    data["bats"] = data[bats_col].fillna("").astype(str).str.upper().str[0]
    data["PA_vR"] = pd.to_numeric(data[pa_vr_col], errors="coerce").fillna(0.0)
    data["PA_vL"] = pd.to_numeric(data[pa_vl_col], errors="coerce").fillna(0.0)
    data["OPS_vR"] = pd.to_numeric(data[ops_vr_col], errors="coerce")
    data["OPS_vL"] = pd.to_numeric(data[ops_vl_col], errors="coerce")
    return data[
        [
            "player_id",
            "team_id",
            "bats",
            "PA_vR",
            "OPS_vR",
            "PA_vL",
            "OPS_vL",
        ]
    ]


def expand_rows(base_df: pd.DataFrame, names: Dict[int, str], positions: Dict[int, str]) -> pd.DataFrame:
    rows = []
    for _, row in base_df.iterrows():
        pid = int(row["player_id"])
        team_id = int(row["team_id"])
        bats = row["bats"] or "R"
        name = names.get(pid, f"Player {pid}")
        pos = positions.get(pid, "")
        if pos == "P" and row["PA_vR"] + row["PA_vL"] < 10:
            continue
        record = {
            "player_id": pid,
            "team_id": team_id,
            "player_name": name,
            "bats": bats,
            "PA_vR": row["PA_vR"],
            "OPS_vR": row["OPS_vR"],
            "PA_vL": row["PA_vL"],
            "OPS_vL": row["OPS_vL"],
        }
        if bats in ("L", "S"):
            rows.append(
                {
                    **record,
                    "adv_context": "LHB vs RHP",
                    "adv_pa": row["PA_vR"],
                    "adv_ops": row["OPS_vR"],
                    "disadv_pa": row["PA_vL"],
                    "disadv_ops": row["OPS_vL"],
                }
            )
        if bats in ("R", "S"):
            rows.append(
                {
                    **record,
                    "adv_context": "RHB vs LHP",
                    "adv_pa": row["PA_vL"],
                    "adv_ops": row["OPS_vL"],
                    "disadv_pa": row["PA_vR"],
                    "disadv_ops": row["OPS_vR"],
                }
            )
    return pd.DataFrame(rows)


def rate_delta(delta: float) -> str:
    if pd.isna(delta):
        return "Unknown"
    if delta >= 0.200:
        return "Terminator"
    if delta >= 0.120:
        return "Assassin"
    if delta >= 0.060:
        return "Crusher"
    if delta >= 0.0:
        return "Leveraged"
    if delta >= -0.060:
        return "Neutralized"
    return "Exposed"


def compute_metrics(
    df: pd.DataFrame,
    display_map: Dict[int, str],
    conf_map: Dict[int, str],
    min_pa_both: int,
    min_pa_adv: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df, df
    df = df.copy()
    df["delta_ops"] = df["adv_ops"] - df["disadv_ops"]
    df["pa_balance_ratio"] = df.apply(
        lambda row: min(row["adv_pa"], row["disadv_pa"]) / max(row["adv_pa"], row["disadv_pa"])
        if row["adv_pa"] > 0 and row["disadv_pa"] > 0
        else np.nan,
        axis=1,
    )
    df["team_display"] = df["team_id"].apply(lambda tid: display_map.get(tid, ""))
    df["conf_div"] = df["team_id"].apply(lambda tid: conf_map.get(tid, ""))
    df["qualified"] = (
        (df["adv_pa"] >= max(min_pa_both, min_pa_adv))
        & (df["disadv_pa"] >= min_pa_both)
        & df["adv_ops"].notna()
        & df["disadv_ops"].notna()
    )
    qualified = df[df["qualified"]].copy()
    if not qualified.empty:
        lg_ops_adv = qualified["adv_ops"].mean()
        lg_ops_disadv = qualified["disadv_ops"].mean()
        lg_delta = qualified["delta_ops"].mean()
        delta_std = qualified["delta_ops"].std(ddof=0)
        if delta_std and delta_std > 0:
            qualified["delta_z"] = (qualified["delta_ops"] - lg_delta) / delta_std
        else:
            qualified["delta_z"] = np.nan
        qualified["lg_ops_adv"] = round(lg_ops_adv, 3)
        qualified["lg_ops_disadv"] = round(lg_ops_disadv, 3)
        qualified["lg_delta_ops"] = round(lg_delta, 3)
    else:
        qualified["delta_z"] = np.nan
        qualified["lg_ops_adv"] = np.nan
        qualified["lg_ops_disadv"] = np.nan
        qualified["lg_delta_ops"] = np.nan
    df["lg_ops_adv"] = qualified["lg_ops_adv"].iloc[0] if not qualified.empty else np.nan
    df["lg_ops_disadv"] = qualified["lg_ops_disadv"].iloc[0] if not qualified.empty else np.nan
    df["lg_delta_ops"] = qualified["lg_delta_ops"].iloc[0] if not qualified.empty else np.nan
    df = df.merge(
        qualified[["player_id", "adv_context", "delta_z"]],
        on=["player_id", "adv_context"],
        how="left",
        suffixes=("", "_qual"),
    )
    df["delta_z"] = df["delta_z"].fillna(df["delta_z_qual"])
    df = df.drop(columns=["delta_z_qual"])
    df["clutch_rating"] = df["delta_ops"].apply(rate_delta)
    return df, qualified


def build_text_report(df: pd.DataFrame, min_pa_both: int, min_pa_adv: int) -> str:
    lines = [
        "ABL Platoon Assassins",
        "=" * 25,
        "Spotlights hitters who torch the advantaged matchup (LHB vs RHP or RHB vs LHP) far beyond the norm.",
        "Great for lineup notes: lean on these bats when the platoon edge is in their favor.",
        "",
    ]
    header = (
        f"{'Player (Team)':<28} {'CD':<4} {'Context':<13} {'Rating':<11} "
        f"{'OPS_adv':>7} {'OPS_dis':>8} {'ΔOPS':>7} {'PA_adv':>7}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    if df.empty:
        lines.append("(No hitters met the qualification thresholds.)")
    for _, row in df.iterrows():
        player = f"{row['player_name']} ({row['team_display']})"
        conf = row["conf_div"] or "--"
        adv_ops_txt = f"{row['adv_ops']:.3f}" if not pd.isna(row["adv_ops"]) else "NA "
        disadv_ops_txt = f"{row['disadv_ops']:.3f}" if not pd.isna(row["disadv_ops"]) else "NA "
        delta_txt = f"{row['delta_ops']:.3f}" if not pd.isna(row["delta_ops"]) else "NA "
        lines.append(
            f"{player:<28} {conf:<4} {row['adv_context']:<13} {row['clutch_rating']:<11} "
            f"{adv_ops_txt:>7} {disadv_ops_txt:>8} {delta_txt:>7} {int(row['adv_pa']):>7}"
        )
    lines.append("")
    lines.append(f"Thresholds: adv PA >= {min_pa_adv}, both-hand PA >= {min_pa_both}.")
    lines.append("")
    lines.append("Key:")
    lines.append("  Terminator >= +0.200 deltas; Assassin 0.120-0.199; Crusher 0.060-0.119; Leveraged 0-0.059;")
    lines.append("  Neutralized -0.059~-0.001; Exposed <= -0.060 (opponents flip the platoon).")
    lines.append("")
    lines.append("Definitions:")
    lines.append("  adv_context identifies the matchup (LHB vs RHP or RHB vs LHP).")
    lines.append("  adv_ops/disadv_ops = OPS versus advantaged vs disadvantaged hand.")
    lines.append("  ΔOPS = advantaged OPS minus disadvantaged OPS.")
    lines.append("  PA balance ratio = exposure balance between both sides (1.0 = even).")
    lines.append("  League OPS figures and ΔOPS mean appear in the CSV for reference.")
    return "\n".join(lines)


def print_top_table(df: pd.DataFrame) -> None:
    subset = df[
        [
            "player_name",
            "team_display",
            "adv_context",
            "clutch_rating",
            "adv_ops",
            "disadv_ops",
            "delta_ops",
            "adv_pa",
        ]
    ].head(25)
    display_df = subset.copy()
    display_df = display_df.rename(
        columns={
            "player_name": "Player",
            "team_display": "Team",
            "adv_context": "Context",
            "clutch_rating": "Rating",
            "adv_ops": "OPS_adv",
            "disadv_ops": "OPS_dis",
            "delta_ops": "ΔOPS",
            "adv_pa": "PA_adv",
        }
    )
    for col in ["OPS_adv", "OPS_dis", "ΔOPS"]:
        display_df[col] = display_df[col].map(lambda v: f"{v:.3f}" if not pd.isna(v) else "NA ")
    print(display_df.to_string(index=False))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ABL Platoon Assassins report.")
    parser.add_argument("--base", type=str, default=".", help="Base directory.")
    parser.add_argument("--splits", type=str, help="Override path for splits vs hand.")
    parser.add_argument("--roster", type=str, help="Override path for roster/names.")
    parser.add_argument("--teams", type=str, help="Override path for team names.")
    parser.add_argument("--min_pa_both", type=int, default=60, help="Minimum PA vs each hand.")
    parser.add_argument("--min_pa_adv", type=int, default=40, help="Minimum advantaged PA.")
    parser.add_argument("--show_all", action="store_true", help="Include non-qualifiers.")
    parser.add_argument(
        "--out",
        type=str,
        default="out/csv_out/z_ABL_Platoon_Assassins.csv",
        help="Output CSV path (default inside out/csv_out).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    base_dir = Path(args.base).resolve()
    splits_override = resolve_optional_path(base_dir, args.splits)
    roster_override = resolve_optional_path(base_dir, args.roster)
    teams_override = resolve_optional_path(base_dir, args.teams)

    splits = load_splits(base_dir, splits_override)
    names_map, positions_map = load_roster_names(base_dir, roster_override)
    display_map, conf_map = load_team_info(base_dir, teams_override)

    expanded = expand_rows(splits, names_map, positions_map)
    stats, qualified = compute_metrics(expanded, display_map, conf_map, args.min_pa_both, args.min_pa_adv)
    final = stats if args.show_all else stats[stats["qualified"]]
    final = final.sort_values(
        by=["delta_ops", "adv_ops", "adv_pa"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = base_dir / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    csv_df = final.copy()
    for col in ["adv_ops", "disadv_ops", "delta_ops", "pa_balance_ratio", "lg_ops_adv", "lg_ops_disadv", "lg_delta_ops"]:
        if col in csv_df.columns:
            csv_df[col] = csv_df[col].round(3)
    if "delta_z" in csv_df.columns:
        csv_df["delta_z"] = csv_df["delta_z"].round(2)
    csv_columns = [
        "team_id",
        "team_display",
        "conf_div",
        "player_id",
        "player_name",
        "bats",
        "adv_context",
        "adv_pa",
        "adv_ops",
        "disadv_pa",
        "disadv_ops",
        "delta_ops",
        "pa_balance_ratio",
        "lg_ops_adv",
        "lg_ops_disadv",
        "lg_delta_ops",
        "delta_z",
        "clutch_rating",
    ]
    csv_df[csv_columns].to_csv(out_path, index=False)

    text_report = build_text_report(final.head(25), args.min_pa_both, args.min_pa_adv)
    text_filename = out_path.with_suffix(".txt").name
    if out_path.parent.name.lower() in {'csv_out'}:
        text_dir = out_path.parent.parent / "txt_out"
    else:
        text_dir = out_path.parent
    text_dir.mkdir(parents=True, exist_ok=True)
    (text_dir / text_filename).write_text(text_report, encoding="utf-8")

    if final.empty:
        print("No hitters met the qualification thresholds.")
    else:
        print_top_table(final)


if __name__ == "__main__":
    main()

